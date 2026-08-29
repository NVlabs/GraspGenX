#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build all fixed GraspGenX ONNX graphs into TensorRT plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import tensorrt as trt
except ModuleNotFoundError:
    import tensorrt_bindings as trt

GRAPH_PRECISIONS = {
    "denoiser_core": "fp16",
    "scorer_core": "fp32",
}
WORKSPACE_BYTES = 8 * (1024**3)
TIMING_ITERATIONS = 8
DEFAULT_OPTIMIZATION_LEVEL = 2


def parse_args(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=root / "cpp_runtime/artifacts/onnx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "cpp_runtime/artifacts/engines",
    )
    parser.add_argument("--diffusion-grasps", type=int, default=100)
    parser.add_argument("--obb-grasps", type=int, default=50)
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=range(6),
        default=DEFAULT_OPTIMIZATION_LEVEL,
    )
    return parser.parse_args(argv)


def build_engine(
    onnx_path,
    plan_path,
    precision,
    workspace_bytes,
    timing_iterations,
    optimization_level,
):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    strongly_typed = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
    if explicit_batch is not None:
        network_flags = 1 << int(explicit_batch)
    else:
        network_flags = 0 if strongly_typed is None else 1 << int(strongly_typed)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    config.builder_optimization_level = optimization_level
    config.avg_timing_iterations = timing_iterations
    config.max_aux_streams = 0
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if precision == "fp16":
        fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
        if fp16_flag is not None:
            config.set_flag(fp16_flag)
    elif precision == "fp32":
        config.clear_flag(trt.BuilderFlag.TF32)
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {onnx_path}")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_bytes(bytes(serialized))
    return {
        "onnx": str(onnx_path),
        "plan": str(plan_path),
        "precision": precision,
        "network_layers_before_build": network.num_layers,
        "serialized_bytes": plan_path.stat().st_size,
    }


def main():
    args = parse_args()
    if args.diffusion_grasps < 1:
        raise ValueError("--diffusion-grasps must be positive")
    if args.obb_grasps < 0:
        raise ValueError("--obb-grasps must be non-negative")
    scorer_count = args.diffusion_grasps + args.obb_grasps
    if scorer_count > 4096:
        raise ValueError("combined candidate count must not exceed 4096")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "tensorrt": trt.__version__,
        "workspace_bytes": WORKSPACE_BYTES,
        "timing_iterations": TIMING_ITERATIONS,
        "builder_optimization_level": args.optimization_level,
        "engines": {},
    }
    ptv3_graphs = [
        path.stem for path in sorted((args.onnx_dir / "ptv3").glob("ptv3_*.onnx"))
    ]
    if not ptv3_graphs:
        raise RuntimeError(f"No PTV3 ONNX graphs found under {args.onnx_dir / 'ptv3'}")
    graphs = [*GRAPH_PRECISIONS, *ptv3_graphs]
    for graph in graphs:
        if graph in GRAPH_PRECISIONS:
            precision = GRAPH_PRECISIONS[graph]
            suffix = (
                f"d{args.diffusion_grasps}"
                if graph == "denoiser_core"
                else f"c{scorer_count}"
            )
            onnx_path = args.onnx_dir / f"{graph}_{suffix}.onnx"
            plan_path = args.output_dir / f"{graph}_{suffix}_{precision}.plan"
        elif graph.startswith("ptv3_"):
            precision = "fp16"
            onnx_path = args.onnx_dir / "ptv3" / f"{graph}.onnx"
            plan_path = args.output_dir / "ptv3" / f"{graph}_fp16.plan"
        else:
            raise ValueError(f"Unknown graph {graph!r}")
        if not onnx_path.is_file():
            raise FileNotFoundError(onnx_path)
        print(f"Building {graph}...", flush=True)
        report["engines"][graph] = build_engine(
            onnx_path,
            plan_path,
            precision,
            WORKSPACE_BYTES,
            TIMING_ITERATIONS,
            args.optimization_level,
        )
    report_path = args.output_dir / (
        f"build_report_d{args.diffusion_grasps}_o{args.obb_grasps}.json"
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
