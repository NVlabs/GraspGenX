# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the pure-PyTorch PTv3 backbone (ptv3_vanilla)."""

import pytest
import torch

from graspgenx.models.ptv3.ptv3_vanilla import (
    HashSparseConv3d,
    PointTransformerV3Vanilla,
    VanillaPoint,
    segment_csr_vanilla,
)
from graspgenx.models.model_utils import convert_to_ptv3_pc_format

# Small config for fast tests
SMALL_CFG = {
    "in_channels": 3,
    "output_dim": 64,
    "enc_depths": (1, 1, 1, 2, 1),
    "enc_channels": (16, 32, 64, 128, 256),
    "enc_num_head": (2, 4, 8, 8, 16),
    "enc_patch_size": (128, 128, 128, 128, 128),
    "drop_path": 0.0,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── HashSparseConv3d ─────────────────────────────────────────────────


class TestHashSparseConv3d:
    def test_output_shape(self):
        conv = HashSparseConv3d(3, 16, kernel_size=3).to(DEVICE)
        N = 100
        feat = torch.randn(N, 3, device=DEVICE)
        grid_coord = torch.randint(0, 20, (N, 3), device=DEVICE)
        batch = torch.zeros(N, dtype=torch.long, device=DEVICE)
        out = conv(feat, grid_coord, batch)
        assert out.shape == (N, 16)

    def test_kernel_size_5(self):
        conv = HashSparseConv3d(8, 8, kernel_size=5).to(DEVICE)
        N = 50
        feat = torch.randn(N, 8, device=DEVICE)
        grid_coord = torch.randint(0, 10, (N, 3), device=DEVICE)
        batch = torch.zeros(N, dtype=torch.long, device=DEVICE)
        out = conv(feat, grid_coord, batch)
        assert out.shape == (N, 8)

    def test_gradient_flow(self):
        conv = HashSparseConv3d(4, 4, kernel_size=3).to(DEVICE)
        feat = torch.randn(20, 4, device=DEVICE, requires_grad=True)
        grid_coord = torch.randint(0, 10, (20, 3), device=DEVICE)
        batch = torch.zeros(20, dtype=torch.long, device=DEVICE)
        out = conv(feat, grid_coord, batch)
        loss = out.sum()
        loss.backward()
        assert feat.grad is not None
        assert feat.grad.shape == feat.shape

    def test_multi_batch(self):
        conv = HashSparseConv3d(3, 8, kernel_size=3).to(DEVICE)
        N = 60
        feat = torch.randn(N, 3, device=DEVICE)
        grid_coord = torch.randint(0, 15, (N, 3), device=DEVICE)
        batch = torch.cat(
            [torch.zeros(30, dtype=torch.long), torch.ones(30, dtype=torch.long)]
        ).to(DEVICE)
        out = conv(feat, grid_coord, batch)
        assert out.shape == (N, 8)


# ── segment_csr_vanilla ──────────────────────────────────────────────


class TestSegmentCSR:
    def test_mean(self):
        src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        indptr = torch.tensor([0, 2, 4])
        out = segment_csr_vanilla(src, indptr, reduce="mean")
        expected = torch.tensor([[2.0, 3.0], [6.0, 7.0]])
        assert torch.allclose(out, expected, atol=1e-5)

    def test_sum(self):
        src = torch.tensor([[1.0], [2.0], [3.0]])
        indptr = torch.tensor([0, 1, 3])
        out = segment_csr_vanilla(src, indptr, reduce="sum")
        expected = torch.tensor([[1.0], [5.0]])
        assert torch.allclose(out, expected, atol=1e-5)

    def test_max(self):
        src = torch.tensor([[1.0, 5.0], [3.0, 2.0], [4.0, 1.0]])
        indptr = torch.tensor([0, 2, 3])
        out = segment_csr_vanilla(src, indptr, reduce="max")
        expected = torch.tensor([[3.0, 5.0], [4.0, 1.0]])
        assert torch.allclose(out, expected, atol=1e-5)


# ── VanillaPoint serialization ────────────────────────────────────────


class TestVanillaPoint:
    def test_serialization(self):
        N = 64
        point = VanillaPoint(
            coord=torch.randn(N, 3),
            feat=torch.randn(N, 3),
            grid_size=0.01,
            offset=torch.tensor([N]),
        )
        point.serialization(order=["z", "hilbert"], shuffle_orders=False)
        assert "serialized_code" in point
        assert "serialized_order" in point
        assert "serialized_inverse" in point
        assert point.serialized_code.shape[0] == 2  # two orders
        assert point.serialized_code.shape[1] == N

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_neighbor_map_on_cuda_with_duplicates(self):
        torch.manual_seed(9)
        conv = HashSparseConv3d(3, 8, kernel_size=3).cuda()
        grid = torch.randint(0, 20, (400, 3), dtype=torch.int32, device="cuda")
        grid[1] = grid[0]
        batch = torch.randint(0, 3, (400,), dtype=torch.long, device="cuda")
        actual_indices, actual_mask = conv.build_neighbor_map(grid, batch)

        keys = conv._hash(batch, grid)
        sorted_keys, sort_indices = keys.sort()
        neighbor_grid = grid[:, None] + conv.offsets[None]
        neighbor_batch = batch[:, None].expand(-1, len(conv.offsets))
        neighbor_keys = conv._hash(
            neighbor_batch.reshape(-1), neighbor_grid.reshape(-1, 3)
        )
        positions = torch.searchsorted(sorted_keys, neighbor_keys).clamp(
            max=len(grid) - 1
        )
        expected_mask = (sorted_keys[positions] == neighbor_keys).view_as(actual_mask)
        expected_indices = sort_indices[positions].view_as(actual_indices)
        assert torch.equal(actual_mask, expected_mask)
        assert torch.equal(actual_indices, expected_indices)


# ── Full model ────────────────────────────────────────────────────────


class TestPointTransformerV3Vanilla:
    @pytest.fixture
    def model(self):
        return PointTransformerV3Vanilla(**SMALL_CFG).to(DEVICE)

    def test_output_shape(self, model):
        B, N = 2, 512
        pc = torch.randn(B, N, 3, device=DEVICE)
        data_dict = convert_to_ptv3_pc_format(pc, grid_size=0.01)
        out = model(data_dict)
        assert out.shape == (B, SMALL_CFG["output_dim"])

    def test_gradient_flow(self, model):
        B, N = 2, 256
        pc = torch.randn(B, N, 3, device=DEVICE)
        data_dict = convert_to_ptv3_pc_format(pc, grid_size=0.01)
        # Need feat to require grad for backward
        data_dict["feat"] = data_dict["feat"].clone().requires_grad_(True)
        out = model(data_dict)
        loss = out.sum()
        loss.backward()
        # Check that model parameters received gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, "No gradients flowed to model parameters"

    def test_single_batch(self, model):
        B, N = 1, 128
        pc = torch.randn(B, N, 3, device=DEVICE)
        data_dict = convert_to_ptv3_pc_format(pc, grid_size=0.01)
        out = model(data_dict)
        assert out.shape == (1, SMALL_CFG["output_dim"])

    def test_deterministic_eval(self, model):
        """Deterministic when seeded (shuffle_orders uses torch.randperm)."""
        B, N = 2, 256
        pc = torch.randn(B, N, 3, device=DEVICE)
        # Warmup pass to stabilize BN running stats
        data_warmup = convert_to_ptv3_pc_format(pc, grid_size=0.01)
        with torch.no_grad():
            model(data_warmup)
        model.eval()
        # Same seed → same shuffles → same result
        torch.manual_seed(0)
        if DEVICE == "cuda":
            torch.cuda.manual_seed(0)
        data_dict1 = convert_to_ptv3_pc_format(pc, grid_size=0.01)
        with torch.no_grad():
            out1 = model(data_dict1)
        torch.manual_seed(0)
        if DEVICE == "cuda":
            torch.cuda.manual_seed(0)
        data_dict2 = convert_to_ptv3_pc_format(pc, grid_size=0.01)
        with torch.no_grad():
            out2 = model(data_dict2)
        assert torch.allclose(out1, out2, atol=1e-5)
