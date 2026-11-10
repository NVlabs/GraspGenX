#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the two dense ONNX graphs used by the native C++ runtime."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import onnx
import torch
from artifact_tools import fold_attention
from torch import nn
from torch.nn import functional
from artifact_tools import layer_norm_fp32, gelu_fp32

from graspgenx.models.grasp_gen import GraspGen
from graspgenx.utils.checkpoint_io import load_model_cfg


class FoldedDenoiserCore(nn.Module):
    """TensorRT denoiser region between native sample and DDPM boundaries."""

    def __init__(self, head):
        super().__init__()
        self.attention_norms = nn.ModuleList()
        self.ffns = nn.ModuleList(head.ffn_layers)
        for index, (weight, bias, norm) in enumerate(fold_attention(head)):
            if index == 0:
                sample_projection = head.sample_encoder[2]
                sample_weight = weight[:, :512]
                weight = sample_weight @ sample_projection.weight
                bias = functional.linear(sample_projection.bias, sample_weight, None)
            self.register_buffer(f"attention_weight_{index}", weight)
            self.register_buffer(f"attention_bias_{index}", bias)
            self.attention_norms.append(norm)
        self.prediction_first = head.prediction_head[0]
        self.prediction_second = head.prediction_head[2]

    def forward(self, sample_embedding, conditioning_projection):
        hidden = sample_embedding
        for index, (attention_norm, ffn) in enumerate(
            zip(self.attention_norms, self.ffns)
        ):
            weight = getattr(self, f"attention_weight_{index}")
            bias = getattr(self, f"attention_bias_{index}")
            if index == 0:
                hidden = functional.linear(hidden, weight, bias)
                hidden = hidden + conditioning_projection
            else:
                hidden = functional.linear(hidden, weight, bias)
            hidden = layer_norm_fp32(
                hidden,
                (hidden.shape[-1],),
                attention_norm.weight,
                attention_norm.bias,
                attention_norm.eps,
            )
            residual = hidden
            hidden = functional.linear(hidden, ffn.ff[0].weight, ffn.ff[0].bias)
            hidden = gelu_fp32(hidden, approximate="none")
            hidden = functional.linear(hidden, ffn.ff[2].weight, ffn.ff[2].bias)
            hidden = residual + hidden
            hidden = layer_norm_fp32(
                hidden,
                (hidden.shape[-1],),
                ffn.norm.weight,
                ffn.norm.bias,
                ffn.norm.eps,
            )
        hidden = functional.relu(self.prediction_first(hidden))
        return functional.relu(self.prediction_second(hidden))


class ScorerCoreGraph(nn.Module):
    """Gripper-independent scorer accepting an encoded gripper feature."""

    def __init__(self, discriminator):
        super().__init__()
        self.sample_encoder = discriminator.sample_encoder
        self.prediction_head = discriminator.prediction_head

    def forward(self, canonical_rt, object_embedding, gripper_embedding):
        sample_embedding = self.sample_encoder(canonical_rt)
        conditioning = torch.cat((object_embedding, gripper_embedding), dim=-1)
        first = self.prediction_head[0]
        hidden = functional.linear(
            sample_embedding, first.weight[:, : sample_embedding.shape[-1]], None
        )
        hidden = hidden + functional.linear(
            conditioning,
            first.weight[:, sample_embedding.shape[-1] :],
            first.bias,
        )
        hidden = functional.relu(hidden)
        hidden = functional.relu(self.prediction_head[2](hidden))
        logits = self.prediction_head[4](hidden)
        return logits, torch.sigmoid(logits)


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
        default=root / "cpp_runtime/artifacts/onnx",
    )
    parser.add_argument("--diffusion-grasps", type=int, default=100)
    parser.add_argument("--obb-grasps", type=int, default=50)
    return parser.parse_args()


def load_model(release_root):
    cfg = load_model_cfg(release_root / "gen", release_root / "dis")
    model = GraspGen.from_config(cfg.diffusion, cfg.discriminator)
    model.load_state_dict(cfg.eval.gen_checkpoint, cfg.eval.dis_checkpoint)
    return model.eval(), cfg


def export_graph(module, args, path, input_names, output_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    with torch.inference_mode():
        expected = module(*args)
        torch.onnx.export(
            module,
            args,
            path,
            input_names=input_names,
            output_names=output_names,
            opset_version=18,
            dynamo=True,
            external_data=False,
            optimize=True,
            verify=False,
            report=True,
            artifacts_dir=str(path.parent / f"{path.stem}_export_report"),
        )
    return expected


def onnx_summary(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    histogram = Counter(node.op_type for node in model.graph.node)
    initializer_bytes = sum(
        len(initializer.raw_data) for initializer in model.graph.initializer
    )
    return {
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "initializer_raw_bytes": initializer_bytes,
        "op_histogram": dict(sorted(histogram.items())),
        "inputs": [value.name for value in model.graph.input],
        "outputs": [value.name for value in model.graph.output],
    }


def tensorrt_parse_summary(path):
    if platform.machine() == "aarch64":
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        for package_path in reversed(
            (
                Path(f"/usr/lib/python{version}/dist-packages"),
                Path("/usr/lib/python3/dist-packages"),
            )
        ):
            if package_path.is_dir() and str(package_path) not in sys.path:
                sys.path.insert(0, str(package_path))
    try:
        import tensorrt as trt
    except ModuleNotFoundError:
        try:
            import tensorrt_bindings as trt
        except ModuleNotFoundError:
            return {
                "available": False,
                "parsed": False,
                "errors": ["TensorRT missing"],
            }
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    parsed = bool(parser.parse_from_file(str(path)))
    errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
    histogram = Counter(
        str(network.get_layer(index).type).split(".")[-1]
        for index in range(network.num_layers)
    )
    return {
        "available": True,
        "parsed": parsed,
        "errors": errors,
        "network_layers": network.num_layers,
        "layer_type_histogram": dict(sorted(histogram.items())),
    }


def output_difference(left, right):
    if isinstance(left, torch.Tensor):
        left = (left,)
    if isinstance(right, torch.Tensor):
        right = (right,)
    return [
        {
            "max_abs": float((actual.float() - expected.float()).abs().max()),
            "rmse": float(
                torch.sqrt(torch.mean((actual.float() - expected.float()) ** 2))
            ),
        }
        for actual, expected in zip(left, right)
    ]


def main():
    args = parse_args()
    if args.diffusion_grasps < 1:
        raise ValueError("--diffusion-grasps must be positive")
    if args.obb_grasps < 0:
        raise ValueError("--obb-grasps must be non-negative")
    scorer_count = args.diffusion_grasps + args.obb_grasps
    if scorer_count > 4096:
        raise ValueError("combined candidate count must not exceed 4096")
    torch.manual_seed(0)
    model, cfg = load_model(args.release_root)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    head = model.grasp_generator.diffusion_head
    discriminator = model.grasp_discriminator
    report = {
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
        },
        "contract": {
            "points": int(cfg.data.num_points),
            "diffusion_grasps": args.diffusion_grasps,
            "obb_grasps": args.obb_grasps,
            "scorer_candidates": scorer_count,
        },
        "graphs": {},
    }

    denoiser_core = FoldedDenoiserCore(head).half().eval()
    core_sample = torch.randn(args.diffusion_grasps, 512, dtype=torch.float16)
    core_conditioning = torch.randn(1, 2048, dtype=torch.float16)
    denoiser_core_path = output_dir / f"denoiser_core_d{args.diffusion_grasps}.onnx"
    export_graph(
        denoiser_core,
        (core_sample, core_conditioning),
        denoiser_core_path,
        ("sample_embedding", "conditioning_projection"),
        ("prediction_hidden",),
    )
    report["graphs"]["denoiser_core"] = {
        "onnx": onnx_summary(denoiser_core_path),
        "tensorrt": tensorrt_parse_summary(denoiser_core_path),
    }

    canonical = torch.randn(scorer_count, 6, dtype=torch.float32)
    object_embedding = torch.randn(1, 512, dtype=torch.float32)
    gripper = torch.randn(1, 12, dtype=torch.float32)
    with torch.inference_mode():
        sample_embedding = discriminator.sample_encoder(canonical)
        gripper_embedding = discriminator.gripper_encoder(gripper)
        conditioning = torch.cat((object_embedding, gripper_embedding), dim=-1)
        released_logits = discriminator.prediction_head(
            torch.cat(
                (sample_embedding, conditioning.expand(scorer_count, -1)),
                dim=-1,
            )
        )
        released_scorer = (released_logits, torch.sigmoid(released_logits))

    scorer_core = ScorerCoreGraph(discriminator)
    scorer_core_path = output_dir / f"scorer_core_c{scorer_count}.onnx"
    scorer_core_output = export_graph(
        scorer_core,
        (canonical, object_embedding, gripper_embedding),
        scorer_core_path,
        ("canonical_rt", "object_embedding", "gripper_embedding"),
        ("logits", "grasp_confidence"),
    )
    report["graphs"]["scorer_core"] = {
        "onnx": onnx_summary(scorer_core_path),
        "tensorrt": tensorrt_parse_summary(scorer_core_path),
        "optimized_vs_released": output_difference(scorer_core_output, released_scorer),
    }

    report_path = output_dir / (
        f"export_report_d{args.diffusion_grasps}_o{args.obb_grasps}.json"
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
