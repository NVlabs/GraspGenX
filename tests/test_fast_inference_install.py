# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10, supplied by pytest.

import pytest

from graspgenx.fast_inference.cpp_runtime import required_runtime_artifacts
from graspgenx.fast_inference._profiles import (
    DEFAULT_STAGE_CAPACITIES,
    write_stage_capacities,
)
from graspgenx.fast_inference.install import (
    _artifact_commands,
    _cmake_commands,
    _run,
    main,
    parse_args,
)


def test_native_runtime_has_one_public_namespace():
    from graspgenx import fast_inference

    assert hasattr(fast_inference, "install_fast_runtime")
    assert importlib.util.find_spec("graspgenx.inference") is None


def test_fast_inference_selects_cuda13_dependencies():
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()

    assert 'name = "pytorch-cu130"' in pyproject
    assert 'url = "https://download.pytorch.org/whl/cu130"' in pyproject
    assert '"torch>=2.9"' in pyproject
    assert '{ index = "pytorch-cu130", extra = "fast-inference" }' in pyproject
    assert (
        "\"tensorrt-cu13-bindings==10.13.3.9; platform_machine != 'aarch64'\""
        in pyproject
    )
    assert (
        "\"tensorrt-cu13-libs==10.13.3.9; platform_machine != 'aarch64'\"" in pyproject
    )
    assert "tensorrt-cu13-bindings>=10.16" not in pyproject
    assert "tensorrt-cu13-libs>=10.16" not in pyproject
    assert '"nvidia-cuda-runtime-cu13"' in pyproject
    assert '"torchvision>=0.16"' in pyproject


def test_visualization_keeps_usd_in_a_separate_extra():
    pyproject = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text()
    )
    extras = pyproject["project"]["optional-dependencies"]

    assert "usd-core" not in extras["visualization"]
    assert "scene-synthesizer" not in extras["visualization"]
    assert extras["usd"] == ["scene-synthesizer", "usd-core"]


def _touch_artifacts(directory: Path, diffusion=100, obb=50, denoiser="ddpm"):
    paths = required_runtime_artifacts(
        directory,
        diffusion_grasps=diffusion,
        obb_grasps=obb,
        denoiser=denoiser,
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    write_stage_capacities(directory / "engines", DEFAULT_STAGE_CAPACITIES)
    return paths


def test_required_runtime_artifacts_cover_full_native_contract(tmp_path):
    paths = required_runtime_artifacts(
        tmp_path,
        diffusion_grasps=17,
        obb_grasps=7,
    )
    assert len(paths) == 12
    assert tmp_path / "engines/denoiser_core_d17_fp16.plan" in paths
    assert tmp_path / "engines/scorer_core_c24_fp32.plan" in paths
    assert tmp_path / "weights/schedules/20/denoising_coefficients.bin" in paths
    assert sum("ptv3_" in path.name for path in paths) == 2


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
    assert commands[-1][-2:] == ["--optimization-level", "2"]

    optimized = parse_args(["--tensorrt-optimization-level", "5"])
    assert _artifact_commands(optimized)[-1][-2:] == ["--optimization-level", "5"]


def test_installer_only_supports_ddpm20():
    assert parse_args([]).artifact_directory.name == "ddpm20-p3500-2048-1024-512-512"
    with pytest.raises(SystemExit):
        parse_args(["--denoiser", "unknown"])


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


def test_installer_hides_successful_child_output_by_default(capsys):
    _run(
        [sys.executable, "-c", "print('hidden exporter noise')"],
        dry_run=False,
    )
    captured = capsys.readouterr()
    assert "hidden exporter noise" not in captured.out
    assert "hidden exporter noise" not in captured.err


def test_installer_replays_hidden_output_when_child_fails(capsys):
    with pytest.raises(subprocess.CalledProcessError):
        _run(
            [
                sys.executable,
                "-c",
                "print('actionable failure'); raise SystemExit(2)",
            ],
            dry_run=False,
        )
    assert "actionable failure" in capsys.readouterr().err


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
