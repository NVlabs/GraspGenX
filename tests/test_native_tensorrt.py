# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the Python-to-native-C++ optimized inference boundary."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cpp_runtime.tools.artifact_tools import (
    inference_timesteps,
    normal_sobol_samples,
)
from graspgenx._constants import MIN_OBJECT_POINT_COUNT
from graspgenx.fast_inference import DEFAULT_DDPM_INFERENCE_STEPS
from graspgenx.fast_inference.cpp_runtime import (
    NativeCppRuntime,
    NativeGripperEmbeddings,
    resolve_artifact_directory,
)
from graspgenx.grasp_server import GraspGenXSampler

pytestmark = pytest.mark.tensorrt

_RUN_NATIVE = os.environ.get("GRASPGENX_RUN_CPP_RUNTIME_TESTS") == "1"
_NATIVE_LIBRARY = os.environ.get("GRASPGENX_CPP_RUNTIME_LIBRARY")
_NATIVE_ARTIFACTS = os.environ.get("GRASPGENX_CPP_RUNTIME_ARTIFACTS")


def test_default_inference_uses_released_twenty_step_schedule():
    assert DEFAULT_DDPM_INFERENCE_STEPS == 20


def test_respaced_ddpm_presets_remain_offline_artifact_contracts():
    generator = SimpleNamespace(num_diffusion_iters=20, num_diffusion_iters_eval=10)
    assert inference_timesteps(generator).tolist() == [
        19,
        17,
        15,
        13,
        11,
        8,
        6,
        4,
        2,
        0,
    ]
    assert inference_timesteps(generator, 5).tolist() == [19, 8, 5, 3, 0]
    assert inference_timesteps(generator, 20).tolist() == list(range(19, -1, -1))
    with pytest.raises(ValueError, match="one of 5, 10, 20"):
        inference_timesteps(generator, 6)


def test_normal_sobol_artifacts_are_finite_nested_prefixes():
    samples_100 = normal_sobol_samples(100)
    samples_500 = normal_sobol_samples(500)
    assert samples_100.shape == (100, 6)
    assert samples_500.shape == (500, 6)
    assert torch.isfinite(samples_500).all()
    torch.testing.assert_close(samples_100, samples_500[:100], atol=0, rtol=0)


def test_cpp_artifact_resolver_reports_count_specific_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="denoiser_core_d17_fp16.plan"):
        resolve_artifact_directory(tmp_path, diffusion_grasps=17, obb_grasps=7)


def test_failed_cpp_install_does_not_publish_partial_runtime(monkeypatch):
    import graspgenx.fast_inference.runtime as install_module

    gripper_encoder = torch.nn.Linear(12, 512)
    model = SimpleNamespace(
        grasp_generator=SimpleNamespace(gripper_encoder=gripper_encoder),
        grasp_discriminator=SimpleNamespace(gripper_encoder=torch.nn.Linear(12, 512)),
    )

    def fail(**_kwargs):
        raise RuntimeError("native construction failed")

    monkeypatch.setattr(install_module, "NativeCppRuntime", fail)
    installed = install_module.install_fast_runtime(SimpleNamespace(model=model))
    assert not installed
    assert not hasattr(model, "native_cpp_runtime")


def test_shared_cpp_runtime_rejects_shape_reconfiguration():
    from graspgenx.fast_inference.runtime import install_fast_runtime

    model = SimpleNamespace(
        native_cpp_runtime=SimpleNamespace(
            diffusion_steps=20, diffusion_grasps=100, obb_grasps=50
        )
    )
    with pytest.raises(ValueError, match="already has native runtime"):
        install_fast_runtime(
            SimpleNamespace(model=model), diffusion_grasps=200, obb_grasps=75
        )


def test_cpp_wrapper_rejects_small_cloud_before_native_call():
    runtime = object.__new__(NativeCppRuntime)
    with pytest.raises(ValueError, match=f"at least {MIN_OBJECT_POINT_COUNT}"):
        runtime.infer(
            torch.zeros(MIN_OBJECT_POINT_COUNT - 1, 3),
            SimpleNamespace(generator=None, discriminator=None, depth=0.1),
        )


def test_optimized_helpers_reject_multiple_objects():
    sampler = SimpleNamespace(using_fast_inference=True)
    with pytest.raises(ValueError, match="exactly one object per call"):
        GraspGenXSampler.run_inference_batch(
            [torch.zeros(3500, 3), torch.zeros(3500, 3)], sampler
        )


@pytest.mark.skipif(
    not (
        _RUN_NATIVE
        and torch.cuda.is_available()
        and _NATIVE_LIBRARY
        and _NATIVE_ARTIFACTS
    ),
    reason=("set GRASPGENX_RUN_CPP_RUNTIME_TESTS=1 plus native library/artifact paths"),
)
def test_python_boundary_replays_cpp_runtime_deterministically():
    artifacts = Path(_NATIVE_ARTIFACTS)
    generator = (
        torch.from_numpy(
            np.fromfile(
                artifacts / "robotiq_2f_85_embeddings.generator_embedding.fp32.bin",
                dtype=np.float32,
            )
        )
        .cuda()
        .half()
    )
    discriminator = torch.from_numpy(
        np.fromfile(
            artifacts / "robotiq_2f_85_embeddings.discriminator_embedding.fp32.bin",
            dtype=np.float32,
        )
    ).cuda()
    gripper = NativeGripperEmbeddings(generator, discriminator, 0.136)
    runtime = NativeCppRuntime(
        artifact_directory=artifacts,
        library_path=_NATIVE_LIBRARY,
        diffusion_steps=20,
        diffusion_grasps=100,
        obb_grasps=50,
        seed=7,
    )
    coordinate = torch.linspace(-0.05, 0.05, 3500, device="cuda")
    points = torch.stack(
        (coordinate, 0.02 * torch.sin(97 * coordinate), coordinate.square()), dim=-1
    )
    first_result = runtime.infer(points, gripper, seed=7, sequence_offset=0)
    first = (
        first_result.grasps.clone(),
        first_result.confidence.clone(),
        first_result.source_indices.clone(),
    )
    second_result = runtime.infer(points.clone(), gripper, seed=7, sequence_offset=0)
    torch.cuda.synchronize()

    torch.testing.assert_close(second_result.grasps, first[0], atol=0, rtol=0)
    torch.testing.assert_close(second_result.confidence, first[1], atol=0, rtol=0)
    torch.testing.assert_close(second_result.source_indices, first[2], atol=0, rtol=0)
    assert second_result.grasps.shape == (150, 4, 4)
    assert torch.isfinite(second_result.grasps).all()
    assert torch.all(second_result.confidence[:-1] >= second_result.confidence[1:])
    assert sorted(second_result.source_indices.cpu().tolist()) == list(range(150))
    assert not second_result.ptv3_topology_overflow.any()
    runtime.close()
