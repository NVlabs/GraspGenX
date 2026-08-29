#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export fixed-capacity PTV3 learned stages with topology as explicit inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

from graspgenx.models.grasp_gen import GraspGen
from graspgenx.utils.checkpoint_io import load_model_cfg

STAGE_CAPACITIES = (3500, 1536, 512, 128, 64)
ORDER_COUNT = 4


def _affine_batch_norm_parameters(batch_norm):
    scale = batch_norm.weight / torch.sqrt(batch_norm.running_var + batch_norm.eps)
    shift = batch_norm.bias - batch_norm.running_mean * scale
    return scale.detach(), shift.detach()


class Ptv3StageGraph(nn.Module):
    """One learned PTV3 stage between native topology/pooling boundaries."""

    def __init__(self, encoder, stage_index):
        super().__init__()
        self.stage_index = int(stage_index)
        self.capacity = STAGE_CAPACITIES[self.stage_index]
        self.stage = encoder.enc._modules[f"enc{self.stage_index}"]
        self.blocks = nn.ModuleList(
            module
            for name, module in self.stage._modules.items()
            if name.startswith("block")
        )
        self.patch_size = min(self.capacity, int(self.blocks[0].attn.patch_size_max))
        self.padded_count = (
            (self.capacity + self.patch_size - 1) // self.patch_size
        ) * self.patch_size

        if self.stage_index == 0:
            self.embedding = encoder.embedding
            scale, shift = _affine_batch_norm_parameters(self.embedding.norm)
            self.register_buffer("input_norm_scale", scale)
            self.register_buffer("input_norm_shift", shift)
        else:
            scale, shift = _affine_batch_norm_parameters(self.stage.down.norm)
            self.register_buffer("input_norm_scale", scale)
            self.register_buffer("input_norm_shift", shift)

        if self.stage_index + 1 < len(STAGE_CAPACITIES):
            next_stage = encoder.enc._modules[f"enc{self.stage_index + 1}"]
            self.pool_projection = next_stage.down.proj
        else:
            self.pool_projection = None

        for block_index, block in enumerate(self.blocks):
            linear_weight_t = block.cpe_linear.weight.t()
            folded_weight = (
                block.cpe_conv.weight.flatten(0, 1) @ linear_weight_t
            ).detach()
            folded_bias = (
                block.cpe_conv.bias @ linear_weight_t + block.cpe_linear.bias
            ).detach()
            self.register_buffer(f"cpe_weight_{block_index}", folded_weight)
            self.register_buffer(f"cpe_bias_{block_index}", folded_bias)

    @staticmethod
    def _gather_rows(features, indices):
        """Gather feature rows for an arbitrarily shaped index tensor."""
        return functional.embedding(indices, features)

    def _sparse_linear(self, features, indices, found_mask, weight, bias=None):
        gathered = self._gather_rows(features, indices)
        gathered = gathered * found_mask.unsqueeze(-1).to(features.dtype)
        return functional.linear(gathered.flatten(1), weight.t(), bias)

    def _attention(
        self,
        features,
        block,
        order,
        inverse,
        valid_key_counts,
    ):
        channels = int(block.channels)
        heads = int(block.attn.num_heads)
        qkv = self._gather_rows(block.attn.qkv(features), order)
        q, key, value = (
            qkv.reshape(
                -1,
                self.patch_size,
                3,
                heads,
                channels // heads,
            )
            .permute(2, 0, 3, 1, 4)
            .unbind(dim=0)
        )
        scores = torch.matmul(q * float(block.attn.scale), key.transpose(-2, -1))
        key_positions = torch.arange(self.patch_size, device=features.device)
        valid_keys = key_positions[None, :] < valid_key_counts[:, None]
        scores = scores.masked_fill(~valid_keys[:, None, None, :], -1.0e4)
        probabilities = torch.softmax(scores, dim=-1)
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).reshape(-1, channels)
        return block.attn.proj(self._gather_rows(attended, inverse))

    def forward(
        self,
        features,
        neighbor_indices_k3,
        neighbor_mask_k3,
        attention_order,
        attention_inverse,
        valid_key_counts,
        neighbor_indices_k5=None,
        neighbor_mask_k5=None,
    ):
        if self.stage_index == 0:
            if neighbor_indices_k5 is None or neighbor_mask_k5 is None:
                raise ValueError("stage 0 requires K5 topology")
            features = self._sparse_linear(
                features,
                neighbor_indices_k5,
                neighbor_mask_k5,
                self.embedding.conv.weight.flatten(0, 1),
            )
            features = features * self.input_norm_scale + self.input_norm_shift
            features = functional.gelu(features, approximate="none")
        else:
            features = features * self.input_norm_scale + self.input_norm_shift
            features = functional.gelu(features, approximate="none")

        for block_index, block in enumerate(self.blocks):
            shortcut = features
            cpe = self._sparse_linear(
                features,
                neighbor_indices_k3,
                neighbor_mask_k3,
                getattr(self, f"cpe_weight_{block_index}"),
                getattr(self, f"cpe_bias_{block_index}"),
            )
            cpe = functional.layer_norm(
                cpe,
                (int(block.channels),),
                block.cpe_norm.weight,
                block.cpe_norm.bias,
                block.cpe_norm.eps,
            )
            features = shortcut + cpe

            shortcut = features
            normalized = functional.layer_norm(
                features,
                (int(block.channels),),
                block.norm1.weight,
                block.norm1.bias,
                block.norm1.eps,
            )
            order_index = block_index % ORDER_COUNT
            features = shortcut + self._attention(
                normalized,
                block,
                attention_order[order_index],
                attention_inverse[order_index],
                valid_key_counts,
            )

            shortcut = features
            features = functional.layer_norm(
                features,
                (int(block.channels),),
                block.norm2.weight,
                block.norm2.bias,
                block.norm2.eps,
            )
            features = block.mlp.fc1(features)
            features = functional.gelu(features, approximate="none")
            features = shortcut + block.mlp.fc2(features)

        if self.pool_projection is not None:
            return self.pool_projection(features)
        return features


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=root / "ext/graspgenx_checkpoints/GraspGenXModel/release",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "cpp_runtime/artifacts/onnx/ptv3",
    )
    return parser.parse_args()


def load_model(release_root):
    cfg = load_model_cfg(release_root / "gen", release_root / "dis")
    model = GraspGen.from_config(cfg.diffusion, cfg.discriminator)
    model.load_state_dict(cfg.eval.gen_checkpoint, cfg.eval.dis_checkpoint)
    return model.eval()


def synthetic_inputs(module):
    capacity = module.capacity
    channels = 3 if module.stage_index == 0 else int(module.blocks[0].channels)
    features = torch.randn(capacity, channels, dtype=torch.float16)
    indices_k3 = torch.arange(capacity, dtype=torch.int64)[:, None].expand(capacity, 27)
    mask_k3 = torch.zeros(capacity, 27, dtype=torch.bool)
    mask_k3[:, 13] = True
    padded = module.padded_count
    order = torch.arange(padded, dtype=torch.int64).remainder(capacity)
    orders = order[None, :].expand(ORDER_COUNT, -1).contiguous()
    inverses = (
        torch.arange(capacity, dtype=torch.int64)[None, :]
        .expand(ORDER_COUNT, -1)
        .contiguous()
    )
    valid_keys = torch.full(
        (padded // module.patch_size,), module.patch_size, dtype=torch.int32
    )
    valid_keys[-1] = capacity - (padded - module.patch_size)
    if module.stage_index == 0:
        indices_k5 = torch.arange(capacity, dtype=torch.int64)[:, None].expand(
            capacity, 125
        )
        mask_k5 = torch.zeros(capacity, 125, dtype=torch.bool)
        mask_k5[:, 62] = True
    else:
        return (
            features,
            indices_k3,
            mask_k3,
            orders,
            inverses,
            valid_keys,
        )
    return (
        features,
        indices_k3,
        mask_k3,
        orders,
        inverses,
        valid_keys,
        indices_k5,
        mask_k5,
    )


def export_stage(module, inputs, path):
    import onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        expected = module(*inputs)
        input_names = (
            "features",
            "neighbor_indices_k3",
            "neighbor_mask_k3",
            "attention_order",
            "attention_inverse",
            "valid_key_counts",
        )
        if module.stage_index == 0:
            input_names += ("neighbor_indices_k5", "neighbor_mask_k5")
        torch.onnx.export(
            module,
            inputs,
            path,
            input_names=input_names,
            output_names=("features_out",),
            opset_version=18,
            dynamo=True,
            external_data=False,
            optimize=True,
            verify=False,
        )
    graph = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(graph)
    gather_count = sum(node.op_type == "Gather" for node in graph.graph.node)
    gather_nd_count = sum(node.op_type == "GatherND" for node in graph.graph.node)
    if gather_nd_count:
        raise RuntimeError(
            f"{path} contains {gather_nd_count} GatherND nodes; "
            "PTV3 row lookup must export as Gather"
        )
    return {
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "onnx_nodes": len(graph.graph.node),
        "onnx_gather_nodes": gather_count,
        "onnx_gather_nd_nodes": gather_nd_count,
        "output_shape": list(expected.shape),
        "output_finite": bool(torch.isfinite(expected).all()),
    }


def main():
    args = parse_args()
    torch.manual_seed(0)
    model = load_model(args.release_root)
    encoders = {
        "generator": model.grasp_generator.object_encoder,
        "discriminator": model.grasp_discriminator.object_encoder,
    }
    report = {"precision": "fp16", "graphs": {}}
    for encoder_name, encoder in encoders.items():
        encoder.half()
        for stage_index in range(5):
            name = f"ptv3_{encoder_name}_stage{stage_index}"
            module = Ptv3StageGraph(encoder, stage_index).eval()
            report["graphs"][name] = export_stage(
                module,
                synthetic_inputs(module),
                args.output_dir / f"{name}.onnx",
            )
    report_path = args.output_dir / "export_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
