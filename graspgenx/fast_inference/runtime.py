# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public installation boundary for native C++ GraspGenX inference."""

from __future__ import annotations

import os
from pathlib import Path

import torch

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
    diffusion_grasps: int = 100,
    obb_grasps: int = 50,
    artifact_directory: str | os.PathLike[str] | None = None,
    library_path: str | os.PathLike[str] | None = None,
    denoiser: str = "ddpm",
) -> bool:
    """Attach the native fast-inference implementation.

    ``sampler`` only needs a ``model`` attribute. This permits a serving process
    to construct one native runtime on its shared model before per-gripper
    samplers are created. Gripper embeddings are retained by each sampler and
    copied into the shared runtime on the caller stream.
    """
    model = sampler.model
    diffusion_grasps = int(diffusion_grasps)
    obb_grasps = int(obb_grasps)
    existing = getattr(model, "native_cpp_runtime", None)
    if existing is not None:
        requested = (denoiser, diffusion_grasps, obb_grasps)
        active = (
            getattr(existing, "denoiser", "ddpm"),
            existing.diffusion_grasps,
            existing.obb_grasps,
        )
        if requested != active:
            raise ValueError(
                f"shared model already has native runtime {active}; requested {requested}"
            )
        requested_artifacts = artifact_directory or os.environ.get(
            "GRASPGENX_CPP_RUNTIME_ARTIFACTS"
        )
        if (
            requested_artifacts is not None
            and Path(requested_artifacts).expanduser().resolve()
            != existing.artifact_directory
        ):
            raise ValueError(
                "shared model already has a native runtime with a different artifact directory/PTV3 profile"
            )
        return True

    generator_device = next(model.grasp_generator.gripper_encoder.parameters()).device
    runtime = NativeCppRuntime(
        artifact_directory=artifact_directory,
        library_path=library_path,
        diffusion_grasps=diffusion_grasps,
        obb_grasps=obb_grasps,
        denoiser=denoiser,
        device=generator_device,
    )

    model.native_cpp_runtime = runtime
    model.optimized_diffusion_grasps = diffusion_grasps
    model.optimized_obb_grasps = obb_grasps
    model.grasp_generator.num_diffusion_iters_eval = runtime.denoising_steps
    model.grasp_generator.num_grasps_per_object = diffusion_grasps
    try:
        released_bytes = _offload_native_replacements(model)
    except Exception as error:  # noqa: BLE001  # pragma: no cover
        released_bytes = 0
        logger.warning("Could not offload C++-replaced PyTorch weights: %s", error)
    torch.cuda.empty_cache()
    logger.info(
        "Native C++ TensorRT/CUDA runtime enabled (%s%d, %d diffusion "
        "+ %d OBB grasps, %.1f MiB PyTorch weights offloaded)",
        denoiser.upper(),
        runtime.denoising_steps,
        diffusion_grasps,
        obb_grasps,
        released_bytes / (1 << 20),
    )
    return True
