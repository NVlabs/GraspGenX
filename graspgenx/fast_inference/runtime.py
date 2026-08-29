# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public installation boundary for native C++ GraspGenX inference."""

from __future__ import annotations

import os

import torch

from graspgenx.fast_inference._constants import DEFAULT_DDPM_INFERENCE_STEPS
from graspgenx.fast_inference.cpp_runtime import NativeCppRuntime
from graspgenx.utils.logging_config import get_logger

logger = get_logger(__name__)


def _cuda_bytes(modules) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for module in modules
        for parameter in module.parameters()
        if parameter.is_cuda
    )


def _offload_native_replacements(model) -> int:
    """Offload only modules replaced by C++; retain both gripper encoders."""
    modules = (
        model.grasp_generator.object_encoder,
        model.grasp_generator.diffusion_head,
        model.grasp_discriminator.object_encoder,
        model.grasp_discriminator.sample_encoder,
        model.grasp_discriminator.prediction_head,
    )
    released = _cuda_bytes(modules)
    for module in modules:
        module.cpu()
    return released


def install_fast_runtime(
    sampler,
    *,
    diffusion_steps: int = DEFAULT_DDPM_INFERENCE_STEPS,
    diffusion_grasps: int = 100,
    obb_grasps: int = 50,
    artifact_directory: str | os.PathLike[str] | None = None,
    library_path: str | os.PathLike[str] | None = None,
) -> bool:
    """Attach the native fast-inference implementation.

    ``sampler`` only needs a ``model`` attribute. This permits a serving process
    to construct one native runtime on its shared model before per-gripper
    samplers are created. Gripper embeddings are retained by each sampler and
    copied into the shared runtime on the caller stream.
    """
    model = sampler.model
    diffusion_steps = int(diffusion_steps)
    diffusion_grasps = int(diffusion_grasps)
    obb_grasps = int(obb_grasps)
    existing = getattr(model, "native_cpp_runtime", None)
    if existing is not None:
        requested = (diffusion_steps, diffusion_grasps, obb_grasps)
        active = (
            existing.diffusion_steps,
            existing.diffusion_grasps,
            existing.obb_grasps,
        )
        if requested != active:
            raise ValueError(
                f"shared model already has native runtime {active}; requested {requested}"
            )
        return True

    generator_device = next(model.grasp_generator.gripper_encoder.parameters()).device
    try:
        runtime = NativeCppRuntime(
            artifact_directory=artifact_directory,
            library_path=library_path,
            diffusion_steps=diffusion_steps,
            diffusion_grasps=diffusion_grasps,
            obb_grasps=obb_grasps,
            device=generator_device,
        )
    except Exception as error:  # noqa: BLE001 - eager fallback remains intact
        logger.warning("Native C++ inference runtime was not applied: %s", error)
        return False

    model.native_cpp_runtime = runtime
    model.optimized_diffusion_grasps = diffusion_grasps
    model.optimized_obb_grasps = obb_grasps
    model.grasp_generator.num_diffusion_iters_eval = diffusion_steps
    model.grasp_generator.num_grasps_per_object = diffusion_grasps
    try:
        released_bytes = _offload_native_replacements(model)
    except Exception as error:  # noqa: BLE001  # pragma: no cover
        released_bytes = 0
        logger.warning("Could not offload C++-replaced PyTorch weights: %s", error)
    torch.cuda.empty_cache()
    logger.info(
        "Native C++ TensorRT/CUDA runtime enabled (%d DDPM steps, %d diffusion "
        "+ %d OBB grasps, %.1f MiB PyTorch weights offloaded)",
        diffusion_steps,
        diffusion_grasps,
        obb_grasps,
        released_bytes / (1 << 20),
    )
    return True
