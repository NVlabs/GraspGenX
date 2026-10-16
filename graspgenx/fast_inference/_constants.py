# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared defaults for the native fast-inference runtime."""

DEFAULT_DDPM_INFERENCE_STEPS = 20

_DENOISER_CONFIGS = {
    "ddpm": (DEFAULT_DDPM_INFERENCE_STEPS, "ddpm20"),
}


def _denoiser_config(denoiser: str) -> tuple[int, str]:
    try:
        return _DENOISER_CONFIGS[str(denoiser)]
    except KeyError as error:
        raise ValueError("denoiser must be 'ddpm'") from error


def denoising_steps(denoiser: str) -> int:
    """Return the fixed recurrent-step count for a deployed denoiser."""
    return _denoiser_config(denoiser)[0]


def artifact_subdirectory(denoiser: str) -> str:
    """Return the non-overlapping artifact namespace for a denoiser."""
    return _denoiser_config(denoiser)[1]
