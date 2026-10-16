#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the Python boundary of the native C++ GraspGenX runtime.

The timed path passes PyTorch CUDA pointers and the current stream through the
stable C ABI. TensorRT contexts, fixed buffers, internal streams, and the
captured inference graph are all owned by ``libgraspgenx_runtime``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

from graspgenx.fast_inference.cpp_runtime import (
    NativeCppRuntime,
    NativeGripperEmbeddings,
)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=os.environ.get("GRASPGENX_CPP_RUNTIME_ARTIFACTS"),
        help="C++ runtime artifact directory (or GRASPGENX_CPP_RUNTIME_ARTIFACTS)",
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=os.environ.get("GRASPGENX_CPP_RUNTIME_LIBRARY"),
        help="libgraspgenx_runtime.so (or GRASPGENX_CPP_RUNTIME_LIBRARY)",
    )
    parser.add_argument(
        "--generator-embedding",
        type=Path,
        help="512-element FP32 raw generator gripper embedding",
    )
    parser.add_argument(
        "--discriminator-embedding",
        type=Path,
        help="512-element FP32 raw discriminator gripper embedding",
    )
    parser.add_argument("--gripper-depth", type=float, default=0.136)
    parser.add_argument("--diffusion-grasps", type=int, default=100)
    parser.add_argument("--obb-grasps", type=int, default=50)
    parser.add_argument("--denoiser", choices=("ddpm",), default="ddpm")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--cycles", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.diffusion_grasps < 1:
        parser.error("diffusion grasps must be positive")
    if args.obb_grasps < 0:
        parser.error("OBB grasps must be non-negative")
    if args.diffusion_grasps + args.obb_grasps > 4096:
        parser.error("combined candidate count must not exceed 4096")
    if args.warmup < 1 or args.cycles < 1:
        parser.error("warmup and cycles must be positive")
    args.root = root
    return args


def _load_clouds(root: Path):
    clouds = []
    pc_root = root / "assets" / "sample_data" / "object_pc"
    for path in sorted(pc_root.glob("*.json")):
        with path.open() as stream:
            points = np.asarray(json.load(stream)["pc"], dtype=np.float32)
        clouds.append((path.stem, torch.from_numpy(points).cuda()))

    for path in sorted((root / "assets" / "sample_data").glob("**/*.obj")):
        mesh = trimesh.load(path, force="mesh")
        np.random.seed(len(clouds) + 11)
        points, _ = trimesh.sample.sample_surface(mesh, 5000)
        points = points.astype(np.float32)
        if float(np.max(mesh.extents)) > 10.0:
            points *= 0.001
        clouds.append((path.stem, torch.from_numpy(points).cuda()))
    if len(clouds) < 10:
        raise RuntimeError(f"Need ten distinct geometries, found {len(clouds)}")
    return clouds[:10]


def _load_raw_embedding(path: Path, *, dtype: torch.dtype):
    value = np.fromfile(path, dtype=np.float32)
    if value.shape != (512,):
        raise ValueError(f"Expected 512 FP32 values in {path}, found {value.shape}")
    return torch.from_numpy(value).cuda().to(dtype=dtype).contiguous()


def _resolve_embeddings(args, artifacts: Path):
    generator_path = args.generator_embedding or (
        artifacts / "robotiq_2f_85_embeddings.generator_embedding.fp32.bin"
    )
    discriminator_path = args.discriminator_embedding or (
        artifacts / "robotiq_2f_85_embeddings.discriminator_embedding.fp32.bin"
    )
    return NativeGripperEmbeddings(
        generator=_load_raw_embedding(generator_path, dtype=torch.float16),
        discriminator=_load_raw_embedding(discriminator_path, dtype=torch.float32),
        depth=args.gripper_depth,
    )


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def _measure_cuda(runtime, clouds, gripper, warmup, cycles):
    for _ in range(warmup):
        for _, cloud in clouds:
            runtime.infer(cloud, gripper)
    torch.cuda.synchronize(runtime.device)

    per_object = [[] for _ in clouds]
    with torch.inference_mode():
        for _ in range(cycles):
            for index, (_, cloud) in enumerate(clouds):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                runtime.infer(cloud, gripper)
                stop.record()
                stop.synchronize()
                per_object[index].append(start.elapsed_time(stop))

    values = np.asarray(per_object, dtype=np.float64)
    result = _summary(values)
    result["per_object_mean_ms"] = [float(value) for value in values.mean(axis=1)]
    return result


def _measure_dispatch(runtime, clouds, gripper, cycles):
    """Measure steady-state graph submission with one synchronization per batch."""
    torch.cuda.synchronize(runtime.device)
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    wall_start = time.perf_counter()
    for _ in range(cycles):
        for _, cloud in clouds:
            runtime.infer(cloud, gripper)
    stop_event.record()
    stop_event.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    calls = cycles * len(clouds)
    return {
        "cuda_ms_per_call": start_event.elapsed_time(stop_event) / calls,
        "wall_ms_per_call": wall_ms / calls,
        "calls": calls,
    }


def _check_output(runtime, cloud, gripper):
    first = runtime.infer(cloud, gripper, seed=7, sequence_offset=11)
    torch.cuda.synchronize(runtime.device)
    first_grasps = first.grasps.clone()
    first_confidence = first.confidence.clone()
    second = runtime.infer(cloud, gripper, seed=7, sequence_offset=11)
    torch.cuda.synchronize(runtime.device)
    count = runtime.grasp_count
    return {
        "shape": list(second.grasps.shape),
        "finite": bool(torch.isfinite(second.grasps).all()),
        "confidence_sorted_descending": bool(
            torch.all(second.confidence[:-1] >= second.confidence[1:])
        ),
        "source_indices_are_permutation": bool(
            torch.equal(
                torch.sort(second.source_indices).values,
                torch.arange(count, device=runtime.device, dtype=torch.int32),
            )
        ),
        "deterministic_replay": bool(
            torch.equal(first_grasps, second.grasps)
            and torch.equal(first_confidence, second.confidence)
        ),
        "ptv3_stage_counts": second.ptv3_stage_counts.cpu().tolist(),
        "ptv3_topology_overflow": second.ptv3_topology_overflow.cpu().tolist(),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    runtime = NativeCppRuntime(
        artifact_directory=args.artifacts,
        library_path=args.library,
        diffusion_grasps=args.diffusion_grasps,
        obb_grasps=args.obb_grasps,
        denoiser=args.denoiser,
    )
    try:
        gripper = _resolve_embeddings(args, runtime.artifact_directory)
        clouds = _load_clouds(args.root)
        latency = _measure_cuda(runtime, clouds, gripper, args.warmup, args.cycles)
        dispatch = _measure_dispatch(runtime, clouds, gripper, args.cycles)
        checks = _check_output(runtime, clouds[0][1], gripper)
        result = {
            "gpu": torch.cuda.get_device_name(runtime.device),
            "runtime_library": str(runtime.library_path),
            "artifact_directory": str(runtime.artifact_directory),
            "execution": "Python CUDA pointers -> C ABI -> C++ CUDA Graph runtime",
            "objects": [name for name, _ in clouds],
            "denoising_steps": runtime.denoising_steps,
            "denoiser": args.denoiser,
            "diffusion_grasps": args.diffusion_grasps,
            "obb_grasps": args.obb_grasps,
            "warmup": args.warmup,
            "cycles": args.cycles,
            "cuda_latency": latency,
            "batched_graph_dispatch": dispatch,
            "output_checks": checks,
        }
    finally:
        runtime.close()

    output = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
