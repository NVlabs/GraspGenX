# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep released DDPM sampling and native constants across Diffusers upgrades."""

import hashlib
from types import SimpleNamespace

import pytest
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from cpp_runtime.tools.artifact_tools import scheduler_coefficients
from graspgenx.models.generator import _set_inference_timesteps


def make_scheduler(schedule):
    return DDPMScheduler(
        num_train_timesteps=20,
        beta_schedule=schedule,
        clip_sample=True,
        prediction_type="epsilon",
    )


@pytest.mark.parametrize(
    "steps,expected",
    [
        (20, list(range(19, -1, -1))),
        (10, list(range(18, -1, -2))),
        (7, list(range(18, -1, -2))),
        (100, list(range(19, -1, -1))),
    ],
)
@pytest.mark.parametrize("schedule", ["scaled_linear", "squaredcos_cap_v2"])
def test_eager_evaluation_grid_keeps_adjacent_ddpm_transitions(
    steps, expected, schedule
):
    scheduler = make_scheduler(schedule)
    timesteps = _set_inference_timesteps(scheduler, steps)
    assert timesteps.tolist() == expected
    for timestep in timesteps:
        assert int(scheduler.previous_timestep(timestep)) == int(timestep) - 1


@pytest.mark.parametrize("steps", [0, -1])
def test_eager_rejects_nonpositive_step_count(steps):
    with pytest.raises(ValueError, match="must be positive"):
        _set_inference_timesteps(make_scheduler("scaled_linear"), steps)


# First three samples from Diffusers 0.11.1, captured before the upgrade with
# torch.manual_seed(739), sample/noise shape (64, 3), and step RNG seed 42.
_RELEASED_OUTPUTS = {
    ("scaled_linear", 20): [
        [-0.9998340607, 0.4794590771, -0.7814799547],
        [-0.7319393754, -0.9945096970, -0.9998340607],
        [-0.8938209414, 0.0248985067, -0.0267913584],
    ],
    ("scaled_linear", 10): [
        [-0.6260927916, 0.6876760721, -0.9998340607],
        [-0.9998340607, -0.9998340607, -0.9612661600],
        [-0.9790135622, 0.1056902185, 0.5113198161],
    ],
    ("squaredcos_cap_v2", 20): [
        [-0.9999970794, -0.9507608414, 0.9999970794],
        [0.9999970794, 0.1821766943, -0.9999970794],
        [0.6724145412, -0.5962047577, -0.9999970794],
    ],
    ("squaredcos_cap_v2", 10): [
        [0.2689199448, -0.9999970794, 0.4155944586],
        [-0.9999970794, -0.5922738314, -0.8404662013],
        [-0.8216452599, -0.3099769652, -0.9999970794],
    ],
}


@pytest.mark.parametrize("steps", [20, 10, 7, 100])
@pytest.mark.parametrize("schedule", ["scaled_linear", "squaredcos_cap_v2"])
def test_seeded_eager_denoising_matches_released_samples(schedule, steps):
    scheduler = make_scheduler(schedule)
    timesteps = _set_inference_timesteps(scheduler, steps)
    torch.manual_seed(739)
    sample = torch.randn(64, 3)
    noise = torch.randn_like(sample)
    rng = torch.Generator().manual_seed(42)
    for timestep in timesteps:
        sample = scheduler.step(noise, timestep, sample, generator=rng).prev_sample
    reference_steps = 20 if steps >= 20 else 10
    expected = torch.tensor(_RELEASED_OUTPUTS[schedule, reference_steps])
    # New Diffusers reconstructs beta from rounded alpha products. For the
    # scaled-linear beta_start=1e-4, that changes the final coefficient from
    # 0.999834 to 1.0; allow this FP32 rounding difference, not skipped steps.
    tolerance = 2e-4 if schedule == "scaled_linear" else 5e-6
    torch.testing.assert_close(sample[:3], expected, atol=tolerance, rtol=0)


@pytest.mark.parametrize("eager_steps", [None, 10, 7])
def test_native_coefficients_keep_released_bytes_after_eager_configuration(eager_steps):
    pos = make_scheduler("scaled_linear")
    rot = make_scheduler("squaredcos_cap_v2")
    if eager_steps is not None:
        pos.set_timesteps(eager_steps)
        rot.set_timesteps(eager_steps)
    model = SimpleNamespace(
        num_diffusion_iters=20,
        compositional_schedular=True,
        output_dim=6,
        noise_scheduler_pos=pos,
        noise_scheduler_rot=rot,
    )
    timesteps, coefficients, _ = scheduler_coefficients(model)
    assert timesteps.tolist() == list(range(19, -1, -1))
    # SHA-256 of the released Diffusers 0.11.1 FP16 export, not a new reference
    # computed from the implementation under test.
    assert hashlib.sha256(coefficients.half().numpy().tobytes()).hexdigest() == (
        "4057756725d83a016a9012ddeb161f38cbb8cae79d4e6c4f9523e41fd9a3a244"
    )
