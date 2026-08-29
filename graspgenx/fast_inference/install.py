# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-command source installation for native GraspGenX inference."""

from __future__ import annotations

import argparse
import importlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from graspgenx._setup_dependencies import get_checkpoints_version_dir
from graspgenx.fast_inference._constants import DEFAULT_DDPM_INFERENCE_STEPS
from graspgenx.fast_inference.cpp_runtime import required_runtime_artifacts

_ROOT = Path(__file__).resolve().parents[2]
_CPP_RUNTIME = _ROOT / "cpp_runtime"


def _default_release_root() -> Path:
    resolved = get_checkpoints_version_dir()
    if (resolved / "gen/config.yaml").is_file():
        return resolved
    nested = resolved.parent / "GraspGenXModel" / resolved.name
    return nested if (nested / "gen/config.yaml").is_file() else resolved


def _default_nvcc() -> Path | None:
    candidates = []
    if os.environ.get("CUDACXX"):
        candidates.append(Path(os.environ["CUDACXX"]))
    if os.environ.get("CUDAToolkit_ROOT"):
        candidates.append(Path(os.environ["CUDAToolkit_ROOT"]) / "bin/nvcc")
    # Prefer the canonical toolkit symlink over a potentially older distro
    # compiler in /usr/bin (important for Blackwell / sm_120 builds).
    candidates.append(Path("/usr/local/cuda/bin/nvcc"))
    discovered = shutil.which("nvcc")
    if discovered:
        candidates.append(Path(discovered))
    return next((path.resolve() for path in candidates if path.is_file()), None)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=_default_release_root())
    parser.add_argument(
        "--artifact-directory", type=Path, default=_CPP_RUNTIME / "artifacts"
    )
    parser.add_argument(
        "--build-directory", type=Path, default=_CPP_RUNTIME / "build-native"
    )
    parser.add_argument(
        "--library-directory",
        type=Path,
        default=_ROOT / "graspgenx/fast_inference/lib",
        help="output directory searched automatically by the Python wrapper",
    )
    parser.add_argument("--cuda-compiler", type=Path, default=_default_nvcc())
    parser.add_argument("--cuda-root", type=Path)
    parser.add_argument("--tensorrt-root", type=Path)
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        choices=(5, 10, 20),
        default=DEFAULT_DDPM_INFERENCE_STEPS,
    )
    parser.add_argument("--diffusion-grasps", type=int, default=100)
    parser.add_argument("--obb-grasps", type=int, default=50)
    parser.add_argument(
        "--tensorrt-optimization-level",
        type=int,
        choices=range(6),
        default=2,
        help="TensorRT builder optimization level (default: 2; use 3 or 5 for longer builds)",
    )
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument(
        "--force-artifacts",
        action="store_true",
        help="regenerate ONNX, weights, and TensorRT plans even when complete",
    )
    parser.add_argument(
        "--skip-artifacts",
        action="store_true",
        help="require and reuse existing runtime artifacts",
    )
    parser.add_argument(
        "--skip-native-build",
        action="store_true",
        help="require and reuse the existing package-local shared library",
    )
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="stream commands and child-process output instead of hiding it",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print commands without running them"
    )
    args = parser.parse_args(argv)
    if args.diffusion_grasps < 1:
        parser.error("diffusion grasps must be positive")
    if args.obb_grasps < 0:
        parser.error("OBB grasps must be non-negative")
    if args.diffusion_grasps + args.obb_grasps > 4096:
        parser.error("combined candidate count must not exceed 4096")
    if args.jobs < 1:
        parser.error("jobs must be positive")
    if args.force_artifacts and args.skip_artifacts:
        parser.error("--force-artifacts and --skip-artifacts are mutually exclusive")
    return args


def _run(command: list[str], *, dry_run: bool, verbose: bool = False):
    if dry_run or verbose:
        print(f"+ {shlex.join(command)}", flush=True)
    if dry_run:
        return
    if verbose:
        subprocess.run(command, cwd=_ROOT, check=True)
        return

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as output:
        completed = subprocess.run(
            command,
            cwd=_ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0:
            output.seek(0)
            print(f"Command failed: {shlex.join(command)}", file=sys.stderr)
            diagnostics = output.read()
            if diagnostics:
                print(diagnostics, file=sys.stderr, end="")
            completed.check_returncode()


def _artifact_commands(args) -> list[list[str]]:
    python = sys.executable
    release = str(args.release_root.resolve())
    artifacts = args.artifact_directory.resolve()
    counts = [
        "--diffusion-grasps",
        str(args.diffusion_grasps),
        "--obb-grasps",
        str(args.obb_grasps),
    ]
    return [
        [
            python,
            str(_CPP_RUNTIME / "tools/export_onnx.py"),
            "--release-root",
            release,
            "--output-dir",
            str(artifacts / "onnx"),
            *counts,
        ],
        [
            python,
            str(_CPP_RUNTIME / "tools/export_ptv3_stages.py"),
            "--release-root",
            release,
            "--output-dir",
            str(artifacts / "onnx/ptv3"),
        ],
        [
            python,
            str(_CPP_RUNTIME / "tools/export_runtime_weights.py"),
            "--release-root",
            release,
            "--output-dir",
            str(artifacts / "weights"),
            "--diffusion-grasps",
            str(args.diffusion_grasps),
        ],
        [
            python,
            str(_CPP_RUNTIME / "tools/build_engines.py"),
            "--onnx-dir",
            str(artifacts / "onnx"),
            "--output-dir",
            str(artifacts / "engines"),
            *counts,
            "--optimization-level",
            str(args.tensorrt_optimization_level),
        ],
    ]


def _cmake_commands(args, cmake: str) -> list[list[str]]:
    configure = [
        cmake,
        "-S",
        str(_CPP_RUNTIME),
        "-B",
        str(args.build_directory.resolve()),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={args.library_directory.resolve()}",
    ]
    if not (args.build_directory / "CMakeCache.txt").is_file() and shutil.which(
        "ninja"
    ):
        configure[1:1] = ["-G", "Ninja"]
    if args.cuda_compiler is not None:
        compiler = args.cuda_compiler.resolve()
        configure.append(f"-DCMAKE_CUDA_COMPILER={compiler}")
        cuda_root = args.cuda_root or compiler.parents[1]
        configure.append(f"-DCUDAToolkit_ROOT={cuda_root.resolve()}")
    elif args.cuda_root is not None:
        configure.append(f"-DCUDAToolkit_ROOT={args.cuda_root.resolve()}")
    tensorrt_root = args.tensorrt_root or (
        Path(os.environ["TENSORRT_ROOT"]) if os.environ.get("TENSORRT_ROOT") else None
    )
    if tensorrt_root is not None:
        configure.append(f"-DTENSORRT_ROOT={tensorrt_root.resolve()}")
    build = [
        cmake,
        "--build",
        str(args.build_directory.resolve()),
        "--parallel",
        str(args.jobs),
    ]
    return [configure, build]


def _import_build_dependencies():
    missing = []
    for module in ("onnx", "onnxscript"):
        try:
            importlib.import_module(module)
        except Exception as error:  # noqa: BLE001 - report binary import failures
            missing.append(f"{module} ({error})")
    tensorrt_error = None
    for module in ("tensorrt", "tensorrt_bindings"):
        try:
            importlib.import_module(module)
            break
        except Exception as error:  # noqa: BLE001 - try standalone bindings
            tensorrt_error = error
    else:
        missing.append(f"TensorRT Python bindings ({tensorrt_error})")
    if missing:
        raise RuntimeError(
            "Missing native artifact-build dependencies: "
            + ", ".join(missing)
            + ". Run this command through `uv run --extra fast-inference`."
        )


def _missing_artifacts(args) -> list[Path]:
    required = required_runtime_artifacts(
        args.artifact_directory.resolve(),
        diffusion_steps=args.diffusion_steps,
        diffusion_grasps=args.diffusion_grasps,
        obb_grasps=args.obb_grasps,
    )
    return [path for path in required if not path.is_file()]


def _verify(args, library: Path):
    import torch

    from graspgenx.fast_inference.cpp_runtime import (
        NativeCppRuntime,
        NativeGripperEmbeddings,
    )

    runtime = NativeCppRuntime(
        artifact_directory=args.artifact_directory,
        library_path=library,
        diffusion_steps=args.diffusion_steps,
        diffusion_grasps=args.diffusion_grasps,
        obb_grasps=args.obb_grasps,
        seed=0,
    )
    try:
        coordinate = torch.linspace(-0.05, 0.05, 3500, device=runtime.device)
        points = torch.stack(
            (coordinate, 0.02 * torch.sin(97 * coordinate), coordinate.square()),
            dim=-1,
        )
        gripper = NativeGripperEmbeddings(
            generator=torch.zeros(512, device=runtime.device, dtype=torch.float16),
            discriminator=torch.zeros(512, device=runtime.device, dtype=torch.float32),
            depth=0.1,
        )
        result = runtime.infer(points, gripper, seed=0, sequence_offset=0)
        torch.cuda.synchronize(runtime.device)
        if not torch.isfinite(result.grasps).all():
            raise RuntimeError("native smoke test produced non-finite grasps")
        if result.ptv3_topology_overflow.any():
            raise RuntimeError("native smoke test overflowed PTV3 capacities")
        if not torch.all(result.confidence[:-1] >= result.confidence[1:]):
            raise RuntimeError("native smoke-test scores are not sorted")
    finally:
        runtime.close()


def main(argv=None):
    args = parse_args(argv)
    if not (_CPP_RUNTIME / "CMakeLists.txt").is_file():
        raise RuntimeError(
            "the fast-inference installer requires a GraspGenX source checkout; "
            "run it from an editable/source installation"
        )
    args.release_root = args.release_root.expanduser().resolve()
    args.artifact_directory = args.artifact_directory.expanduser().resolve()
    args.build_directory = args.build_directory.expanduser().resolve()
    args.library_directory = args.library_directory.expanduser().resolve()
    library = args.library_directory / "libgraspgenx_runtime.so"
    missing = _missing_artifacts(args)

    if args.skip_artifacts and missing:
        raise RuntimeError(
            "--skip-artifacts requested, but runtime files are missing: "
            + ", ".join(str(path) for path in missing)
        )
    if not args.skip_artifacts and (args.force_artifacts or missing):
        if not args.dry_run:
            if (
                not (args.release_root / "gen").is_dir()
                or not (args.release_root / "dis").is_dir()
            ):
                raise FileNotFoundError(
                    f"released generator/discriminator not found under {args.release_root}"
                )
            _import_build_dependencies()
        print("Generating GPU-specific native runtime artifacts...", flush=True)
        for command in _artifact_commands(args):
            _run(command, dry_run=args.dry_run, verbose=args.verbose)
    else:
        print(f"Reusing complete artifacts: {args.artifact_directory}", flush=True)

    if args.skip_native_build:
        if not args.dry_run and not library.is_file():
            raise FileNotFoundError(
                f"--skip-native-build requested, but {library} does not exist"
            )
    else:
        cmake = shutil.which("cmake")
        if cmake is None:
            raise RuntimeError(
                "CMake 3.24 or newer is required; run through "
                "`uv run --extra fast-inference`."
            )
        args.library_directory.mkdir(parents=True, exist_ok=True)
        print("Building the native C++ runtime...", flush=True)
        for command in _cmake_commands(args, cmake):
            _run(command, dry_run=args.dry_run, verbose=args.verbose)

    if args.dry_run:
        print("Dry run complete.")
        return 0
    missing = _missing_artifacts(args)
    if missing:
        raise RuntimeError(
            "artifact generation completed without required files: "
            + ", ".join(str(path) for path in missing)
        )
    if not library.is_file():
        raise RuntimeError(f"native build did not produce {library}")
    if not args.no_verify:
        print("Running Python-to-C++ smoke test...", flush=True)
        try:
            _verify(args, library)
        except Exception as error:
            raise RuntimeError(
                "native verification failed. TensorRT plans are GPU- and "
                "version-specific; rerun with --force-artifacts after checking "
                "the CUDA/TensorRT installations"
            ) from error
    print("Fast inference is ready.")
    print(f"  library:  {library}")
    print(f"  artifacts: {args.artifact_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
