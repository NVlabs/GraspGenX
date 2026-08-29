# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python boundary for the native C++ fast-inference runtime."""

from graspgenx.fast_inference._constants import DEFAULT_DDPM_INFERENCE_STEPS


def install_fast_runtime(
    sampler,
    *,
    diffusion_steps: int = DEFAULT_DDPM_INFERENCE_STEPS,
    diffusion_grasps: int = 100,
    obb_grasps: int = 50,
    artifact_directory=None,
    library_path=None,
) -> bool:
    """Attach the native runtime without loading it at package import."""
    from graspgenx.fast_inference.runtime import (
        install_fast_runtime as _install_fast_runtime,
    )

    return _install_fast_runtime(
        sampler,
        diffusion_steps=diffusion_steps,
        diffusion_grasps=diffusion_grasps,
        obb_grasps=obb_grasps,
        artifact_directory=artifact_directory,
        library_path=library_path,
    )


__all__ = ["DEFAULT_DDPM_INFERENCE_STEPS", "install_fast_runtime"]
