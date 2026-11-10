#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build all fixed GraspGenX ONNX graphs into TensorRT plans."""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import sys
from pathlib import Path

if platform.machine() == "aarch64":
    _version = f"{sys.version_info.major}.{sys.version_info.minor}"
    for _path in reversed(
        (
            Path(f"/usr/lib/python{_version}/dist-packages"),
            Path("/usr/lib/python3/dist-packages"),
        )
    ):
        if _path.is_dir() and str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

try:
    import tensorrt as trt
except ModuleNotFoundError:
    import tensorrt_bindings as trt

from graspgenx.fast_inference._profiles import (
    PROFILE_FILENAME,
    read_stage_capacities,
    validate_stage_capacities,
    write_stage_capacities,
)

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
    parser.add_argument("--plugin-library", type=Path, required=True)
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
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

    input_shapes = {
        network.get_input(i).name: list(network.get_input(i).shape)
        for i in range(network.num_inputs)
    }
    if "features" in input_shapes:
        shape = input_shapes["features"]
        minimum = 100
        if len(shape) != 2 or not minimum <= shape[0] <= 3500:
            raise ValueError(f"Unsupported PTV3 features shape in {onnx_path}: {shape}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    config.builder_optimization_level = optimization_level
    config.avg_timing_iterations = timing_iterations
    config.max_aux_streams = 0
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if precision not in ("fp16", "fp32"):
        raise ValueError(f"Unsupported precision: {precision}")
    # Types are explicit in ONNX; precision-enabling flags are invalid with strong typing.
    tf32 = getattr(trt.BuilderFlag, "TF32", None)
    if precision == "fp32" and tf32 is not None:
        config.clear_flag(tf32)
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
        "input_shapes": input_shapes,
    }


def main():
    args = parse_args()
    plugin_handle = ctypes.CDLL(
        str(args.plugin_library.resolve()), mode=ctypes.RTLD_GLOBAL
    )
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
    encoder_graphs = ["ptv3_generator", "ptv3_discriminator"]
    exported_capacities = read_stage_capacities(args.onnx_dir / "ptv3")
    # Invalidate the previous cache marker before replacing any plan. A failed
    # build must not leave an apparently complete profile for the installer.
    (args.output_dir / PROFILE_FILENAME).write_text(
        json.dumps({"stage_capacities": [], "status": "building"})
    )
    graphs = [*GRAPH_PRECISIONS, *encoder_graphs]
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
    capacities = validate_stage_capacities(
        report["engines"]["ptv3_generator"]["input_shapes"][
            f"stage{i}_neighbor_indices_k3"
        ][0]
        for i in range(5)
    )
    discriminator_capacities = tuple(
        report["engines"]["ptv3_discriminator"]["input_shapes"][
            f"stage{i}_neighbor_indices_k3"
        ][0]
        for i in range(5)
    )
    if capacities != discriminator_capacities or (
        exported_capacities is not None and capacities != exported_capacities
    ):
        raise ValueError("PTV3 generator/discriminator/export profile mismatch")
    report["stage_capacities"] = capacities
    write_stage_capacities(args.output_dir, capacities)
    report_path = args.output_dir / (
        f"build_report_d{args.diffusion_grasps}_o{args.obb_grasps}.json"
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
