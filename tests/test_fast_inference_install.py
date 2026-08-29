# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

from graspgenx.fast_inference.cpp_runtime import required_runtime_artifacts
from graspgenx.fast_inference.install import (
    _artifact_commands,
    _cmake_commands,
    main,
    parse_args,
)


def test_native_runtime_has_one_public_namespace():
    from graspgenx import fast_inference

    assert hasattr(fast_inference, "install_fast_runtime")
    assert importlib.util.find_spec("graspgenx.inference") is None


def _touch_artifacts(directory: Path, steps=20, diffusion=100, obb=50):
    paths = required_runtime_artifacts(
        directory,
        diffusion_steps=steps,
        diffusion_grasps=diffusion,
        obb_grasps=obb,
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return paths


def test_required_runtime_artifacts_cover_full_native_contract(tmp_path):
    paths = required_runtime_artifacts(
        tmp_path,
        diffusion_steps=5,
        diffusion_grasps=17,
        obb_grasps=7,
    )
    assert len(paths) == 20
    assert tmp_path / "engines/denoiser_core_d17_fp16.plan" in paths
    assert tmp_path / "engines/scorer_core_c24_fp32.plan" in paths
    assert tmp_path / "weights/schedules/5/ddpm_coefficients.bin" in paths
    assert sum("ptv3_" in path.name for path in paths) == 10


def test_installer_routes_every_generated_file_to_selected_artifact_root(tmp_path):
    artifacts = tmp_path / "runtime-artifacts"
    args = parse_args(
        [
            "--artifact-directory",
            str(artifacts),
            "--diffusion-grasps",
            "17",
            "--obb-grasps",
            "7",
        ]
    )
    commands = _artifact_commands(args)
    assert len(commands) == 4
    combined = " ".join(" ".join(command) for command in commands)
    assert str(artifacts.resolve() / "onnx") in combined
    assert str(artifacts.resolve() / "onnx/ptv3") in combined
    assert str(artifacts.resolve() / "weights") in combined
    assert str(artifacts.resolve() / "engines") in combined
    assert "--diffusion-grasps 17 --obb-grasps 7" in combined


def test_cmake_build_places_library_in_python_discovery_directory(tmp_path):
    compiler = tmp_path / "cuda/bin/nvcc"
    output = tmp_path / "python-lib"
    args = parse_args(
        [
            "--build-directory",
            str(tmp_path / "build"),
            "--library-directory",
            str(output),
            "--cuda-compiler",
            str(compiler),
            "--tensorrt-root",
            str(tmp_path / "tensorrt"),
        ]
    )
    configure, build = _cmake_commands(args, "/usr/bin/cmake")
    assert f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output.resolve()}" in configure
    assert f"-DCMAKE_CUDA_COMPILER={compiler.resolve()}" in configure
    assert f"-DTENSORRT_ROOT={(tmp_path / 'tensorrt').resolve()}" in configure
    assert "--parallel" in build


def test_complete_dry_run_needs_no_checkpoint_or_build_dependencies(tmp_path):
    artifacts = tmp_path / "artifacts"
    _touch_artifacts(artifacts)
    assert (
        main(
            [
                "--artifact-directory",
                str(artifacts),
                "--library-directory",
                str(tmp_path / "lib"),
                "--skip-artifacts",
                "--skip-native-build",
                "--dry-run",
            ]
        )
        == 0
    )
