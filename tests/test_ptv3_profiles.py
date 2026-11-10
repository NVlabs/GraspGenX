# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from graspgenx.fast_inference._profiles import (
    DEFAULT_STAGE_CAPACITIES,
    PROFILE_FILENAME,
    read_stage_capacities,
    validate_stage_capacities,
    write_stage_capacities,
)
from graspgenx.fast_inference.cpp_runtime import required_runtime_artifacts
from graspgenx.fast_inference.install import (
    _artifact_commands,
    _missing_artifacts,
    main,
    parse_args,
)
from graspgenx.fast_inference.runtime import install_fast_runtime


@pytest.mark.parametrize(
    "capacities",
    [
        DEFAULT_STAGE_CAPACITIES,
        (3500,) * 5,
        (100, 1, 1, 1, 1),
        (100, 3500, 777, 333, 77),
    ],
)
def test_supported_profile_bounds_and_independent_stage_sizes(capacities):
    assert validate_stage_capacities(capacities) == capacities


@pytest.mark.parametrize(
    "capacities",
    [
        (3500,) * 4,
        (3500,) * 6,
        (99, 512, 512, 512, 512),
        (3501, 512, 512, 512, 512),
        (3500, 0, 512, 512, 512),
        (3500, 3501, 512, 512, 512),
        (3500, 512, 512, 512, -1),
        (3500, 512.5, 512, 512, 512),
        (3500, True, 512, 512, 512),
    ],
)
def test_unsupported_profile_rejected(capacities):
    with pytest.raises(ValueError):
        validate_stage_capacities(capacities)


def test_installer_namespaces_custom_profiles_and_passes_export_shape():
    original = parse_args([])
    custom = parse_args(["--ptv3-capacities", *map(str, (3500, 2048, 1024, 512, 512))])
    assert original.ptv3_capacities == (3500, 2048, 1024, 512, 512)
    assert original.artifact_directory.name == "ddpm20-p3500-2048-1024-512-512"
    assert custom.artifact_directory.name == "ddpm20-p3500-2048-1024-512-512"
    exporter = _artifact_commands(custom)[1]
    assert exporter[-6:] == ["--ptv3-capacities", "3500", "2048", "1024", "512", "512"]
    with pytest.raises(SystemExit):
        parse_args(["--ptv3-capacities", "3500", "3501", "512", "512", "512"])


def test_default_runtime_resolves_the_installer_profile(tmp_path, monkeypatch):
    from graspgenx.fast_inference import cpp_runtime
    from graspgenx.fast_inference._profiles import profile_artifact_subdirectory

    monkeypatch.delenv("GRASPGENX_CPP_RUNTIME_ARTIFACTS", raising=False)
    monkeypatch.setattr(cpp_runtime, "_ROOT", tmp_path)
    selected = (
        tmp_path
        / "cpp_runtime/artifacts"
        / profile_artifact_subdirectory("ddpm", DEFAULT_STAGE_CAPACITIES)
    )
    _complete_artifacts(selected)
    assert (
        cpp_runtime.resolve_artifact_directory(
            None, diffusion_grasps=100, obb_grasps=50
        )
        == selected
    )


def _complete_artifacts(directory):
    for path in required_runtime_artifacts(
        directory, diffusion_grasps=100, obb_grasps=50
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_cache_requires_matching_completed_profile(tmp_path):
    _complete_artifacts(tmp_path)
    default = parse_args(
        [
            "--artifact-directory",
            str(tmp_path),
            "--ptv3-capacities",
            *map(str, DEFAULT_STAGE_CAPACITIES),
        ]
    )
    marker = tmp_path / "engines" / PROFILE_FILENAME
    assert _missing_artifacts(default) == [marker]
    write_stage_capacities(marker.parent, DEFAULT_STAGE_CAPACITIES)
    assert not _missing_artifacts(default)
    custom = parse_args(
        ["--artifact-directory", str(tmp_path), "--ptv3-capacities", *["3500"] * 5]
    )
    marker = tmp_path / "engines" / PROFILE_FILENAME
    assert _missing_artifacts(custom) == [marker]
    with pytest.raises(ValueError, match="profile mismatch"):
        main(
            [
                "--artifact-directory",
                str(tmp_path),
                "--ptv3-capacities",
                *["3500"] * 5,
                "--dry-run",
                "--skip-native-build",
            ]
        )
    write_stage_capacities(marker.parent, (3500,) * 5)
    assert read_stage_capacities(marker.parent) == (3500,) * 5
    assert not _missing_artifacts(custom)
    assert _missing_artifacts(default) == [marker]
    marker.write_text(json.dumps({"stage_capacities": [], "status": "building"}))
    assert _missing_artifacts(default) == [marker]
    with pytest.raises(ValueError, match="incomplete"):
        read_stage_capacities(marker.parent)
    assert (
        main(
            [
                "--artifact-directory",
                str(tmp_path),
                "--ptv3-capacities",
                *["3500"] * 5,
                "--dry-run",
                "--skip-native-build",
            ]
        )
        == 0
    )


def test_installer_does_not_silently_replace_a_different_profile(tmp_path):
    write_stage_capacities(tmp_path / "engines", (3500,) * 5)
    with pytest.raises(ValueError, match="profile mismatch"):
        main(["--artifact-directory", str(tmp_path), "--dry-run"])


def test_shared_model_cannot_silently_reuse_another_profile(tmp_path):
    active = tmp_path / "active"
    runtime = SimpleNamespace(
        denoiser="ddpm", diffusion_grasps=100, obb_grasps=50, artifact_directory=active
    )
    sampler = SimpleNamespace(model=SimpleNamespace(native_cpp_runtime=runtime))
    assert install_fast_runtime(sampler, artifact_directory=active)
    with pytest.raises(ValueError, match="different artifact directory"):
        install_fast_runtime(sampler, artifact_directory=tmp_path / "other")
