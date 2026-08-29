# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Point Transformer V3 — Pure PyTorch ("vanilla") implementation.

No spconv, no torch_scatter, no warpconvnet.  Replaces:
 - spconv.SubMConv3d  → HashSparseConv3d  (hash-table neighbor lookup)
 - torch_scatter.segment_csr → segment_csr_vanilla (torch.scatter_reduce_)
 - SparseConvTensor bookkeeping → removed entirely

The z-order (Morton) and Hilbert space-filling curve encoders used for
serialized attention are inlined below so this module has no sibling
dependencies.

Based on the Pointcept PTv3 architecture by Xiaoyang Wu.
"""

import math
from collections import OrderedDict
from functools import partial

import torch
from addict import Dict
from torch import nn
from torch.nn import functional as F


class DropPath(nn.Module):
    """Per-sample stochastic depth without the otherwise-unused timm dependency."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        keep = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            keep.div_(keep_prob)
        return x * keep


# ---------------------------------------------------------------------------
# z-order (Morton) coordinate encoding
#
# Adapted from "Octree-based Sparse Convolutional Neural Networks"
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>, MIT License.
# ---------------------------------------------------------------------------


class _ZOrderKeyLUT:
    def __init__(self):
        r256 = torch.arange(256, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        device = torch.device("cpu")
        self._encode = {
            device: (
                self._xyz2key_slow(r256, zero, zero, 8),
                self._xyz2key_slow(zero, r256, zero, 8),
                self._xyz2key_slow(zero, zero, r256, 8),
            )
        }

    def encode_lut(self, device=None):
        device = torch.device("cpu") if device is None else device
        if device not in self._encode:
            cpu = torch.device("cpu")
            self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
        return self._encode[device]

    @staticmethod
    def _xyz2key_slow(x, y, z, depth):
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (
                key
                | ((x & mask) << (2 * i + 2))
                | ((y & mask) << (2 * i + 1))
                | ((z & mask) << (2 * i + 0))
            )
        return key


_z_order_lut = _ZOrderKeyLUT()


def _z_order_xyz2key(x, y, z, depth=16):
    """Encode (x, y, z) integer coordinates to 64-bit Morton (z-order) keys."""
    EX, EY, EZ = _z_order_lut.encode_lut(x.device)
    x, y, z = x.long(), y.long(), z.long()

    mask = 255 if depth > 8 else (1 << depth) - 1
    key = EX[x & mask] | EY[y & mask] | EZ[z & mask]
    if depth > 8:
        mask = (1 << (depth - 8)) - 1
        key16 = EX[(x >> 8) & mask] | EY[(y >> 8) & mask] | EZ[(z >> 8) & mask]
        key = key16 << 24 | key
    return key


# ---------------------------------------------------------------------------
# Hilbert curve encoding
#
# Vectorized PyTorch port of Skilling's Hilbert curve (AIP 2004), adapted from
# https://github.com/PrincetonLIPS/numpy-hilbert-curve by Xiaoyang Wu et al.
# ---------------------------------------------------------------------------


def _right_shift(binary, k=1, axis=-1):
    """Right-shift a boolean/binary tensor along `axis`, zero-padding the head."""
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    return F.pad(binary[tuple(slicing)], (k, 0), mode="constant", value=0)


def _gray2binary(gray, axis=-1):
    """Convert Gray code bits back to plain binary along `axis`."""
    shift = 2 ** (torch.Tensor([gray.shape[axis]]).log2().ceil().int() - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, _right_shift(gray, shift))
        shift = torch.div(shift, 2, rounding_mode="floor")
    return gray


def _hilbert_encode(locs, num_dims, num_bits):
    """Encode integer hypercube coordinates to Hilbert-curve int64 indices."""
    orig_shape = locs.shape
    bitpack_mask = 1 << torch.arange(0, 8).to(locs.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    if orig_shape[-1] != num_dims:
        raise ValueError(
            f"locs last dim is {orig_shape[-1]} but num_dims={num_dims}; must match"
        )
    if num_dims * num_bits > 63:
        raise ValueError(
            f"num_dims={num_dims} * num_bits={num_bits} exceeds the int64 63-bit budget"
        )

    locs_uint8 = locs.long().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)
    gray = (
        locs_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[..., -num_bits:]
    )

    for bit in range(num_bits):
        for dim in range(num_dims):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))
    hh_bin = _gray2binary(gray)
    extra_dims = 64 - num_bits * num_dims
    padded = F.pad(hh_bin, (extra_dims, 0), "constant", 0)
    hh_uint8 = (
        (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask)
        .sum(2)
        .squeeze()
        .type(torch.uint8)
    )
    return hh_uint8.view(torch.int64).squeeze()


# ---------------------------------------------------------------------------
# Top-level serialization encoder used by VanillaPoint.serialization
# ---------------------------------------------------------------------------


@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z"):
    """Serialize integer grid coordinates into batched 64-bit curve codes."""
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        x, y, z = (
            grid_coord[:, 0].long(),
            grid_coord[:, 1].long(),
            grid_coord[:, 2].long(),
        )
        code = _z_order_xyz2key(x, y, z, depth=depth)
    elif order == "z-trans":
        x, y, z = (
            grid_coord[:, 1].long(),
            grid_coord[:, 0].long(),
            grid_coord[:, 2].long(),
        )
        code = _z_order_xyz2key(x, y, z, depth=depth)
    elif order == "hilbert":
        code = _hilbert_encode(grid_coord, num_dims=3, num_bits=depth)
    else:  # hilbert-trans
        code = _hilbert_encode(grid_coord[:, [1, 0, 2]], num_dims=3, num_bits=depth)

    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code
    return code


# ---------------------------------------------------------------------------
# Pure-PyTorch helpers (ported from ptv3.py, no external deps)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


def segment_csr_vanilla(src, indptr, reduce="mean"):
    """Pure-PyTorch replacement for torch_scatter.segment_csr.

    Args:
        src: (N, C) source features, assumed pre-sorted by segment.
        indptr: (S+1,) CSR index pointers.
        reduce: "mean", "sum", "max", or "min".
    Returns:
        (S, C) reduced features.
    """
    counts = indptr[1:] - indptr[:-1]
    num_segments = counts.shape[0]
    C = src.shape[1] if src.dim() > 1 else 1

    # Build per-element segment ids from CSR pointers
    seg_ids = torch.arange(num_segments, device=src.device).repeat_interleave(counts)

    if reduce in ("mean", "sum"):
        out = torch.zeros(num_segments, C, device=src.device, dtype=src.dtype)
        out.scatter_reduce_(
            0,
            seg_ids.unsqueeze(-1).expand_as(src),
            src,
            reduce=reduce,
            include_self=False,
        )
    elif reduce == "max":
        out = torch.full(
            (num_segments, C), float("-inf"), device=src.device, dtype=src.dtype
        )
        out.scatter_reduce_(
            0,
            seg_ids.unsqueeze(-1).expand_as(src),
            src,
            reduce="amax",
            include_self=False,
        )
    elif reduce == "min":
        out = torch.full(
            (num_segments, C), float("inf"), device=src.device, dtype=src.dtype
        )
        out.scatter_reduce_(
            0,
            seg_ids.unsqueeze(-1).expand_as(src),
            src,
            reduce="amin",
            include_self=False,
        )
    else:
        raise ValueError(f"Unsupported reduce: {reduce}")
    return out


def segment_ids_vanilla(src, seg_ids, num_segments, reduce="mean"):
    """Static-shape segment reduction used by captured one-object stages."""
    channels = src.shape[1] if src.dim() > 1 else 1
    expanded_ids = seg_ids.unsqueeze(-1).expand_as(src)
    if reduce in ("mean", "sum"):
        out = torch.zeros(num_segments, channels, device=src.device, dtype=src.dtype)
        out.scatter_reduce_(
            0,
            expanded_ids,
            src,
            reduce=reduce,
            include_self=False,
        )
    elif reduce == "max":
        out = torch.full(
            (num_segments, channels),
            float("-inf"),
            device=src.device,
            dtype=src.dtype,
        )
        out.scatter_reduce_(
            0,
            expanded_ids,
            src,
            reduce="amax",
            include_self=False,
        )
    elif reduce == "min":
        out = torch.full(
            (num_segments, channels),
            float("inf"),
            device=src.device,
            dtype=src.dtype,
        )
        out.scatter_reduce_(
            0,
            expanded_ids,
            src,
            reduce="amin",
            include_self=False,
        )
    else:
        raise ValueError(f"Unsupported reduce: {reduce}")
    return out


# ---------------------------------------------------------------------------
# HashSparseConv3d — submanifold sparse 3D conv via hash-table lookup
# ---------------------------------------------------------------------------


class HashSparseConv3d(nn.Module):
    """Submanifold sparse 3D convolution using hash-based neighbor lookup.

    For each active voxel, looks up K^3 potential neighbors in a hash table
    built from grid coordinates, gathers their features, applies per-offset
    learned weights, and sums.  Output has the same sparsity as input.
    """

    # Large primes for spatial hashing
    _P = (73856093, 19349669, 83492791, 334214467)

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        r = kernel_size // 2
        offsets = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    offsets.append([dx, dy, dz])
        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.long)
        )  # (K3, 3)
        num_offsets = self.offsets.shape[0]

        self.weight = nn.Parameter(torch.empty(num_offsets, in_channels, out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    @staticmethod
    def _hash(batch, grid_coord):
        """Hash (batch, x, y, z) → int64 key."""
        P = HashSparseConv3d._P
        return (
            batch.long() * P[3]
            + grid_coord[:, 0].long() * P[0]
            + grid_coord[:, 1].long() * P[1]
            + grid_coord[:, 2].long() * P[2]
        )

    def build_neighbor_map(
        self,
        grid_coord,
        batch,
    ):
        """Build the feature-independent neighbor topology with PyTorch."""
        N = grid_coord.shape[0]
        K3 = self.offsets.shape[0]
        keys = self._hash(batch, grid_coord)
        sorted_keys, sort_idx = keys.sort()
        neighbor_coords = grid_coord.unsqueeze(1) + self.offsets.unsqueeze(0)
        neighbor_batch = batch.unsqueeze(1).expand(-1, K3)
        neighbor_keys = self._hash(
            neighbor_batch.reshape(-1),
            neighbor_coords.reshape(-1, 3),
        )
        positions = torch.searchsorted(sorted_keys, neighbor_keys)
        positions = positions.clamp(max=N - 1)
        found_mask = (sorted_keys[positions] == neighbor_keys).view(N, K3)
        found_indices = sort_idx[positions].view(N, K3)
        return found_indices, found_mask

    def forward(
        self,
        feat,
        grid_coord,
        batch,
        neighbor_map=None,
        fused_weight=None,
        fused_bias=None,
    ):
        """
        Args:
            feat: (N, C_in) point features.
            grid_coord: (N, 3) integer grid coordinates.
            batch: (N,) batch index per point.
        Returns:
            (N, C_out) convolved features (same sparsity pattern).
        """
        if neighbor_map is None:
            neighbor_map = self.build_neighbor_map(grid_coord, batch)
        found_indices, found_mask = neighbor_map

        flat_weight = (
            self.weight.flatten(0, 1) if fused_weight is None else fused_weight
        )
        output_bias = self.bias if fused_bias is None else fused_bias
        # Gather neighbor features; zeros where no neighbor exists.
        # Fixed-shape formulation (export/TensorRT-friendly): `found_indices`
        # are already valid (positions clamped to [0, N-1]), so the gather is
        # always safe; multiplying by the found mask zeros out the entries that
        # had no real neighbor. Numerically identical to the masked-assignment
        # form `neighbor_feat[found_mask] = feat[found_indices[found_mask]]`.
        neighbor_feat = feat[found_indices] * found_mask.unsqueeze(-1)

        # This contraction is one ordinary GEMM when both K3 and Cin are
        # flattened. ``einsum("nki,kio->no")`` decomposes it into many small
        # offset-wise GEMMs, which leaves tensor cores underutilized.
        out = neighbor_feat.flatten(1) @ flat_weight
        if output_bias is not None:
            out = out + output_bias
        return out


# ---------------------------------------------------------------------------
# VanillaPoint — simplified Point dict (no spconv fields)
# ---------------------------------------------------------------------------


class VanillaPoint(Dict):
    """Simplified Point cloud container — no SparseConvTensor bookkeeping."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth
        assert depth * 3 + len(self.offset).bit_length() <= 63
        assert depth <= 16

        code = torch.stack(
            [
                encode(self.grid_coord, self.batch, depth, order=order_)
                for order_ in order
            ]
        )
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse


# ---------------------------------------------------------------------------
# VanillaPointModule / VanillaPointSequential
# ---------------------------------------------------------------------------


class VanillaPointModule(nn.Module):
    """Marker base class: subclasses receive the full VanillaPoint dict."""


class VanillaPointSequential(VanillaPointModule):
    """Sequential container that routes VanillaPoint through children."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f"index {idx} is out of range")
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for _ in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for module in self._modules.values():
            if isinstance(module, VanillaPointModule):
                input = module(input)
            else:
                # Standard nn.Module — apply to feat tensor
                if isinstance(input, VanillaPoint):
                    input.feat = module(input.feat)
                else:
                    input = module(input)
        return input


# ---------------------------------------------------------------------------
# Attention (pure PyTorch, no flash_attn)
# ---------------------------------------------------------------------------


class RPE(nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)
            + self.pos_bnd
            + torch.arange(3, device=coord.device) * self.rpe_num
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class VanillaSerializedAttention(VanillaPointModule):
    """Serialized patch attention — pure PyTorch, optional SDPA flash path."""

    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.enable_flash = enable_flash
        self.enable_rpe = enable_rpe

        if enable_flash:
            # SDPA handles upcasting and softmax internally
            self.upcast_attention = False
            self.upcast_softmax = False
            self.attn_drop_p = attn_drop
            self.attn_drop = nn.Identity()  # not used in flash path
        else:
            self.upcast_attention = upcast_attention
            self.upcast_softmax = upcast_softmax
            self.attn_drop_p = attn_drop
            self.attn_drop = nn.Dropout(attn_drop)

        self.patch_size_max = patch_size
        self.patch_size = 0

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.rpe = (
            RPE(patch_size, num_heads)
            if (self.enable_rpe and not enable_flash)
            else None
        )

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point:
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point
            or unpad_key not in point
            or cu_seqlens_key not in point
        ):
            # The deployment contract is one object per call.  Derive the
            # padded patch layout from static tensor shapes in that case so
            # no CUDA scalar is copied to the host in every attention block.
            # Keep the original multi-object path below for training/tests.
            if point.offset.numel() == 1:
                point_count = point.feat.shape[0]
                padded_count = point_count
                if point_count > self.patch_size:
                    padded_count = (
                        (point_count + self.patch_size - 1) // self.patch_size
                    ) * self.patch_size
                pad = torch.arange(padded_count, device=point.feat.device)
                if padded_count != point_count:
                    remainder = point_count % self.patch_size
                    pad[padded_count - self.patch_size + remainder :] = pad[
                        padded_count
                        - 2 * self.patch_size
                        + remainder : padded_count
                        - self.patch_size
                    ]
                point[pad_key] = pad
                point[unpad_key] = torch.arange(point_count, device=point.feat.device)
                point[cu_seqlens_key] = torch.arange(
                    0,
                    padded_count + 1,
                    step=self.patch_size,
                    dtype=torch.int32,
                    device=point.feat.device,
                )
                return (
                    point[pad_key],
                    point[unpad_key],
                    point[cu_seqlens_key],
                )
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = F.pad(offset, (1, 0))
            _offset_pad = F.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = F.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if "static_attention_order" in point:
            H = self.num_heads
            C = self.channels
            order = point.static_attention_order[self.order_index]
            inverse = point.static_attention_inverse[self.order_index]
            qkv = self.qkv(point.feat)[order]
            q, k, v = qkv.reshape(-1, 3, H, C // H).unbind(dim=1)
            input_dtype = q.dtype
            q, k, v = q.half(), k.half(), v.half()
            feat = torch.ops.aten._flash_attention_forward(
                q,
                k,
                v,
                point.static_attention_cu_seqlens,
                point.static_attention_cu_seqlens,
                point.static_attention_max_sequence,
                point.static_attention_max_sequence,
                self.attn_drop_p if self.training else 0.0,
                False,
                False,
                scale=self.scale,
                seqused_k=point.static_attention_used_keys,
            )[0]
            feat = feat.to(input_dtype).reshape(-1, C)
            feat = feat[inverse]
            feat = self.proj(feat)
            feat = self.proj_drop(feat)
            point.feat = feat
            return point

        if point.get("batched_attention", False):
            # A multi-object batch can have different voxel counts per object,
            # so forcing one rectangular SDPA patch size changes the larger
            # objects' attention partition. Varlen FlashAttention consumes the
            # shared canonical padding plan directly and preserves each
            # object's one-call patch boundaries.
            H = self.num_heads
            C = self.channels
            self.patch_size = point.attention_patch_size
            order = point.serialized_order[self.order_index][point.pad]
            inverse = point.unpad[point.serialized_inverse[self.order_index]]
            qkv = self.qkv(point.feat)[order]
            q, k, v = qkv.reshape(-1, 3, H, C // H).unbind(dim=1)
            input_dtype = q.dtype
            q, k, v = q.half(), k.half(), v.half()
            feat = torch.ops.aten._flash_attention_forward(
                q,
                k,
                v,
                point.cu_seqlens_key,
                point.cu_seqlens_key,
                self.patch_size,
                self.patch_size,
                self.attn_drop_p if self.training else 0.0,
                False,
                False,
                scale=self.scale,
            )[0]
            feat = feat.to(input_dtype).reshape(-1, C)
            feat = feat[inverse]
            feat = self.proj(feat)
            feat = self.proj_drop(feat)
            point.feat = feat
            return point

        if point.offset.numel() == 1:
            # ``shape[0]`` is host metadata, unlike ``Tensor.tolist()`` below.
            self.patch_size = min(point.feat.shape[0], self.patch_size_max)
        else:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, _cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        qkv = self.qkv(point.feat)[order]

        # (N_padded, K, 3, H, C//H) → 3 x (N_padded, H, K, C//H)
        q, k, v = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)

        if self.enable_flash:
            # Cast to fp16 to trigger flash/mem-efficient SDPA kernels
            # (SDPA falls back to math backend with fp32 inputs)
            input_dtype = q.dtype
            q, k, v = q.half(), k.half(), v.half()
            feat = F.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=self.scale,
                dropout_p=self.attn_drop_p if self.training else 0.0,
            )
            feat = feat.to(input_dtype).transpose(1, 2).reshape(-1, C)
        else:
            # Manual attention path
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)

        feat = feat[inverse]
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


# ---------------------------------------------------------------------------
# MLP (same as original)
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# VanillaEmbedding — stem layer (replaces spconv SubMConv3d k=5)
# ---------------------------------------------------------------------------


class VanillaEmbedding(VanillaPointModule):
    def __init__(self, in_channels, embed_channels, norm_layer=None, act_layer=None):
        super().__init__()
        self.conv = HashSparseConv3d(
            in_channels, embed_channels, kernel_size=5, bias=False
        )
        self.norm = norm_layer(embed_channels) if norm_layer is not None else None
        self.act = act_layer() if act_layer is not None else None

    def forward(self, point: VanillaPoint):
        topology_key = f"sparse_neighbor_map_k{self.conv.kernel_size}"
        neighbor_map = point.get(topology_key, None)
        point.feat = self.conv(
            point.feat,
            point.grid_coord,
            point.batch,
            neighbor_map=neighbor_map,
        )
        if self.norm is not None:
            point.feat = self.norm(point.feat)
        if self.act is not None:
            point.feat = self.act(point.feat)
        return point


# ---------------------------------------------------------------------------
# VanillaBlock — transformer block with HashSparseConv3d CPE
# ---------------------------------------------------------------------------


class VanillaBlock(VanillaPointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        # CPE: HashSparseConv3d replaces spconv.SubMConv3d
        self.cpe_conv = HashSparseConv3d(channels, channels, kernel_size=3, bias=True)
        self.cpe_linear = nn.Linear(channels, channels)
        self.cpe_norm = norm_layer(channels)

        self.norm1 = norm_layer(channels)
        self.attn = VanillaSerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = norm_layer(channels)
        self.mlp = MLP(
            in_channels=channels,
            hidden_channels=int(channels * mlp_ratio),
            out_channels=channels,
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self._folded_cpe_key = None
        self._folded_cpe_weight = None
        self._folded_cpe_bias = None

    def _get_folded_cpe(self):
        """Fold the consecutive sparse-conv and dense-linear transforms."""
        key = (
            self.cpe_conv.weight._version,
            self.cpe_conv.bias._version,
            self.cpe_linear.weight._version,
            self.cpe_linear.bias._version,
            self.cpe_conv.weight.device,
            self.cpe_conv.weight.dtype,
        )
        if key != self._folded_cpe_key:
            with torch.no_grad():
                linear_weight_t = self.cpe_linear.weight.t()
                self._folded_cpe_weight = (
                    self.cpe_conv.weight.flatten(0, 1) @ linear_weight_t
                ).contiguous()
                self._folded_cpe_bias = (
                    self.cpe_conv.bias @ linear_weight_t + self.cpe_linear.bias
                ).contiguous()
            self._folded_cpe_key = key
        return self._folded_cpe_weight, self._folded_cpe_bias

    def forward(self, point: VanillaPoint):
        # CPE
        shortcut = point.feat
        topology_key = f"sparse_neighbor_map_k{self.cpe_conv.kernel_size}"
        if topology_key not in point:
            point[topology_key] = self.cpe_conv.build_neighbor_map(
                point.grid_coord, point.batch
            )
        fold_cpe = (
            not torch.is_grad_enabled() and not self.training and point.feat.is_cuda
        )
        if fold_cpe:
            folded_weight, folded_bias = self._get_folded_cpe()
            cpe_out = self.cpe_conv(
                point.feat,
                point.grid_coord,
                point.batch,
                neighbor_map=point[topology_key],
                fused_weight=folded_weight,
                fused_bias=folded_bias,
            )
        else:
            cpe_out = self.cpe_conv(
                point.feat,
                point.grid_coord,
                point.batch,
                neighbor_map=point[topology_key],
            )
            cpe_out = self.cpe_linear(cpe_out)
        cpe_out = self.cpe_norm(cpe_out)
        point.feat = shortcut + cpe_out

        # Attention
        shortcut = point.feat
        if self.pre_norm:
            point.feat = self.norm1(point.feat)
        point = self.attn(point)
        point.feat = shortcut + self.drop_path(point.feat)

        # FFN
        shortcut = point.feat
        if self.pre_norm:
            point.feat = self.norm2(point.feat)
        point.feat = shortcut + self.drop_path(self.mlp(point.feat))

        return point


# ---------------------------------------------------------------------------
# VanillaSerializedPooling — downsampling via scatter_reduce_
# ---------------------------------------------------------------------------


class VanillaSerializedPooling(VanillaPointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = norm_layer(out_channels) if norm_layer is not None else None
        self.act = act_layer() if act_layer is not None else None

    def build_topology(self, point: VanillaPoint, *, shuffle_orders=None):
        """Build the feature-independent pooling plan for one stage."""
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(point.keys())

        code = point.serialized_code >> pooling_depth * 3
        if shuffle_orders is None:
            shuffle_orders = self.shuffle_orders
        if shuffle_orders is False:
            # In canonical one-object inference, the first serialized order is
            # already sorted by the full-resolution code. Right-shifting keeps
            # equal parent voxels consecutive, eliminating unique's inverse
            # construction and the subsequent cluster sort.
            indices = point.serialized_order[0]
            sorted_parent_code = code[0, indices]
            _, counts = torch.unique_consecutive(sorted_parent_code, return_counts=True)
            idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
            segment_ids = torch.searchsorted(
                idx_ptr[1:],
                torch.arange(indices.shape[0], device=indices.device),
                right=True,
            )
            cluster = torch.empty_like(indices).scatter_(0, indices, segment_ids)
        else:
            _, cluster, counts = torch.unique(
                code[0],
                sorted=True,
                return_inverse=True,
                return_counts=True,
            )
            _, indices = torch.sort(cluster)
            idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
            segment_ids = torch.arange(
                counts.shape[0], device=counts.device
            ).repeat_interleave(counts)
        head_indices = indices[idx_ptr[:-1]]

        pooled_coord = segment_ids_vanilla(
            point.coord[indices],
            segment_ids,
            counts.shape[0],
            reduce="mean",
        )

        # Generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        point_dict = Dict(
            coord=pooled_coord,
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point:
            point_dict["condition"] = point.condition
        if "context" in point:
            point_dict["context"] = point.context

        return point_dict, indices, idx_ptr, cluster, segment_ids

    def forward(self, point: VanillaPoint):
        topology_stages = point.get("_topology_stages", None)
        if topology_stages is None:
            point_dict, indices, idx_ptr, cluster, _ = self.build_topology(point)
        else:
            stage_index = point["_topology_stage_index"]
            current_topology = topology_stages[stage_index]
            indices = current_topology["pool_indices"]
            idx_ptr = current_topology["pool_idx_ptr"]
            cluster = current_topology["pool_cluster"]
            segment_ids = current_topology["pool_segment_ids"]
            num_segments = current_topology["pool_num_segments"]
            point_dict = Dict(topology_stages[stage_index + 1])
            point_dict["_topology_stages"] = topology_stages
            point_dict["_topology_stage_index"] = stage_index + 1

        # Projection is branch-specific; the sort, segment pointers, and all
        # output geometry above are shared by the two independently weighted
        # encoders.
        proj_feat = self.proj(point.feat)
        sorted_feat = proj_feat[indices]
        if topology_stages is None:
            point_dict["feat"] = segment_csr_vanilla(
                sorted_feat, idx_ptr, reduce=self.reduce
            )
        else:
            point_dict["feat"] = segment_ids_vanilla(
                sorted_feat,
                segment_ids,
                num_segments,
                reduce=self.reduce,
            )

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point

        point = VanillaPoint(point_dict)
        if self.norm is not None:
            point.feat = self.norm(point.feat)
        if self.act is not None:
            point.feat = self.act(point.feat)
        return point


# ---------------------------------------------------------------------------
# PointTransformerV3Vanilla — main encoder-only model
# ---------------------------------------------------------------------------


class PointTransformerV3Vanilla(nn.Module):
    """Pure-PyTorch Point Transformer V3 (encoder-only, cls_mode).

    Drop-in replacement for the spconv-based PointTransformerV3 with
    cls_mode=True.  Accepts the same data_dict format from
    convert_to_ptv3_pc_format().

    Args:
        in_channels: Input feature dimension (3 for xyz).
        output_dim: Final embedding dimension. If different from
            enc_channels[-1], a linear projection is added.
        grid_size: Voxel grid size (used in convert_to_ptv3_pc_format,
            stored here for convenience but not used in forward).
        order: Serialization orders.
        stride: Pooling strides between encoder stages.
        enc_depths: Number of blocks per encoder stage.
        enc_channels: Channel width per encoder stage.
        enc_num_head: Number of attention heads per stage.
        enc_patch_size: Patch size for serialized attention per stage.
        mlp_ratio: FFN expansion ratio.
        drop_path: Stochastic depth rate.
        pre_norm: Use pre-normalization in blocks.
        shuffle_orders: Shuffle serialization orders.
        enable_rpe: Enable relative position encoding in attention.
    """

    def __init__(
        self,
        in_channels=3,
        output_dim=512,
        grid_size=0.01,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders
        self.grid_size = grid_size

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)

        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        # Stem embedding
        self.embedding = VanillaEmbedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # Encoder stages
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = VanillaPointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = VanillaPointSequential()
            if s > 0:
                enc.add(
                    VanillaSerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    VanillaBlock(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # Output projection
        bottleneck_dim = enc_channels[-1]
        if bottleneck_dim != output_dim:
            self.projection = nn.Linear(bottleneck_dim, output_dim)
        else:
            self.projection = nn.Identity()

    @torch.inference_mode()
    def build_single_object_topology(self, data_dict):
        """Build a canonical geometry plan shared by both encoders.

        PTV3 serialization, voxel pooling, sparse-neighbor maps, and attention
        patch layouts depend only on object coordinates.  The released model
        redundantly builds all of them for the generator and discriminator.
        This plan uses the canonical serialization ordering (no inference-time
        order shuffle) and can be consumed by either set of weights. The
        deployed native path remains single-object; this dynamic eager plan also
        supports a true multi-object batch.
        """
        point = VanillaPoint(data_dict)
        if point.offset.numel() < 1:
            raise ValueError("shared PTV3 topology requires at least one object")
        object_count = point.offset.numel()
        point.serialization(order=self.order, shuffle_orders=False)
        topology_stages = []

        for stage_index in range(self.num_stages):
            enc_stage = self.enc._modules[f"enc{stage_index}"]
            first_block = enc_stage._modules["block0"]

            topology = {
                key: point[key]
                for key in (
                    "coord",
                    "grid_coord",
                    "serialized_code",
                    "serialized_order",
                    "serialized_inverse",
                    "serialized_depth",
                    "batch",
                    "offset",
                )
            }
            if stage_index == 0:
                topology["sparse_neighbor_map_k5"] = (
                    self.embedding.conv.build_neighbor_map(
                        point.grid_coord,
                        point.batch,
                    )
                )
            topology["sparse_neighbor_map_k3"] = (
                first_block.cpe_conv.build_neighbor_map(
                    point.grid_coord,
                    point.batch,
                )
            )

            # All blocks in a stage share point count and maximum patch size,
            # so their padding maps are identical.
            if object_count == 1:
                first_block.attn.patch_size = min(
                    point.feat.shape[0], first_block.attn.patch_size_max
                )
            else:
                first_block.attn.patch_size = min(
                    int(offset2bincount(point.offset).max().item()),
                    first_block.attn.patch_size_max,
                )
            pad, unpad, cu_seqlens = first_block.attn.get_padding_and_inverse(point)
            topology["pad"] = pad
            topology["unpad"] = unpad
            topology["cu_seqlens_key"] = cu_seqlens
            if object_count > 1:
                topology["batched_attention"] = True
                topology["attention_patch_size"] = first_block.attn.patch_size
            topology_stages.append(topology)

            if stage_index + 1 < self.num_stages:
                next_stage = self.enc._modules[f"enc{stage_index + 1}"]
                pooling = next_stage._modules["down"]
                (
                    point_dict,
                    indices,
                    idx_ptr,
                    cluster,
                    segment_ids,
                ) = pooling.build_topology(point, shuffle_orders=False)
                topology["pool_indices"] = indices
                topology["pool_idx_ptr"] = idx_ptr
                topology["pool_cluster"] = cluster
                topology["pool_segment_ids"] = segment_ids
                topology["pool_num_segments"] = idx_ptr.shape[0] - 1
                if object_count == 1:
                    point_count = point_dict.serialized_code.shape[1]
                    point_dict["offset"] = torch.full(
                        (1,),
                        point_count,
                        dtype=torch.long,
                        device=point.grid_coord.device,
                    )
                else:
                    point_dict["offset"] = batch2offset(point_dict.batch)
                # Only point-count metadata is needed before the next stage's
                # branch-specific projected feature tensor exists.
                point_dict["feat"] = point_dict.coord
                point = VanillaPoint(point_dict)

        return tuple(topology_stages)

    def forward(self, data_dict, topology_stages=None):
        """
        Args:
            data_dict: Dictionary with keys "feat", "coord", "grid_size",
                and "offset" (or "batch").  Produced by
                convert_to_ptv3_pc_format().
        Returns:
            Tensor of shape (B, output_dim).
        """
        if topology_stages is None:
            point = VanillaPoint(data_dict)
            point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        else:
            if len(topology_stages) != self.num_stages:
                raise ValueError("PTV3 topology stage count does not match encoder")
            point_dict = Dict(topology_stages[0])
            point_dict["feat"] = data_dict["feat"]
            point_dict["_topology_stages"] = topology_stages
            point_dict["_topology_stage_index"] = 0
            point = VanillaPoint(point_dict)

        point = self.embedding(point)
        point = self.enc(point)

        # Global mean pool per batch item → (B, enc_channels[-1])
        if topology_stages is not None:
            if topology_stages[-1]["offset"].numel() == 1:
                pooled = point.feat.mean(dim=0, keepdim=True)
            else:
                pooled = segment_csr_vanilla(
                    point.feat,
                    F.pad(topology_stages[-1]["offset"], (1, 0)),
                    reduce="mean",
                )
        else:
            pooled = segment_csr_vanilla(
                point.feat,
                F.pad(point.offset, (1, 0)),
                reduce="mean",
            )

        return self.projection(pooled)
