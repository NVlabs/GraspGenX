# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-PyTorch helpers used only to export native runtime artifacts."""

from __future__ import annotations

import torch

DDPM_TIMESTEP_PRESETS = {
    5: (19, 8, 5, 3, 0),
    10: (19, 17, 15, 13, 11, 8, 6, 4, 2, 0),
    20: tuple(range(19, -1, -1)),
}

_COEFFICIENT_NAMES = (
    "sample_to_original",
    "noise_to_original",
    "original_to_previous",
    "sample_to_previous",
    "variance_scale",
)


def normal_sobol_samples(
    grasp_count: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a deterministic nested Sobol prefix with normal marginals."""
    if grasp_count <= 0:
        raise ValueError("Sobol sample count must be positive")
    draw_count = 1 << (grasp_count - 1).bit_length()
    uniform = torch.quasirandom.SobolEngine(
        dimension=6, scramble=True, seed=0
    ).draw_base2(draw_count.bit_length() - 1)
    uniform = uniform[:grasp_count].clamp_(1.0e-7, 1.0 - 1.0e-7)
    samples = torch.erfinv(uniform.mul_(2.0).sub_(1.0)).mul_(2.0**0.5)
    return samples.to(device=device, dtype=dtype)


def fold_attention(head: torch.nn.Module):
    """Fold sequence-length-one self-attention into exact affine layers."""
    embed_dim = int(head.embed_dim)
    query_position = head.query_pos_enc.weight[0].detach().float().cpu()
    folded = []
    for layer in head.self_attention_layers:
        attention = layer.attn
        value_weight = attention.in_proj_weight[2 * embed_dim :].detach().float().cpu()
        value_bias = attention.in_proj_bias[2 * embed_dim :].detach().float().cpu()
        output_weight = attention.out_proj.weight.detach().float().cpu()
        output_bias = attention.out_proj.bias.detach().float().cpu()

        weight = output_weight @ value_weight
        weight.diagonal().add_(1.0)
        bias = (
            output_weight @ (value_weight @ query_position + value_bias) + output_bias
        )
        folded.append((weight.contiguous(), bias.contiguous(), layer.norm))
    return folded


def inference_timesteps(generator: torch.nn.Module, inference_steps: int | None = None):
    """Return one of the validated deployment schedules."""
    train_steps = int(generator.num_diffusion_iters)
    inference_steps = int(
        generator.num_diffusion_iters_eval
        if inference_steps is None
        else inference_steps
    )
    preset = DDPM_TIMESTEP_PRESETS.get(inference_steps)
    if preset is None or preset[0] != train_steps - 1:
        choices = ", ".join(str(value) for value in DDPM_TIMESTEP_PRESETS)
        raise ValueError(
            f"inference_steps must be one of {choices}; got {inference_steps}"
        )
    return torch.tensor(preset, dtype=torch.int64)


def scheduler_coefficients(
    generator: torch.nn.Module, inference_steps: int | None = None
):
    """Return timestep and generalized-DDPM transition coefficient tables."""
    if not generator.compositional_schedular:
        raise ValueError("native artifacts require compositional_schedular=True")
    if generator.output_dim != 6:
        raise ValueError("native artifacts support the released 6D latent")

    pos = generator.noise_scheduler_pos
    rot = generator.noise_scheduler_rot
    timesteps = inference_timesteps(generator, inference_steps)
    rows = []
    likelihood_betas = []
    for step_index, timestep_tensor in enumerate(timesteps):
        timestep = int(timestep_tensor.item())
        previous_timestep = (
            int(timesteps[step_index + 1].item())
            if step_index + 1 < len(timesteps)
            else -1
        )
        per_scheduler = []
        for scheduler in (pos, rot):
            alpha_product = scheduler.alphas_cumprod[timestep]
            alpha_product_previous = (
                scheduler.alphas_cumprod[previous_timestep]
                if previous_timestep >= 0
                else scheduler.one
            )
            beta_product = 1 - alpha_product
            beta_product_previous = 1 - alpha_product_previous
            if previous_timestep == timestep - 1:
                predicted_original_scale = (
                    alpha_product_previous.sqrt()
                    * scheduler.betas[timestep]
                    / beta_product
                )
                current_sample_scale = (
                    scheduler.alphas[timestep].sqrt()
                    * beta_product_previous
                    / beta_product
                )
                variance_scale = (
                    scheduler._get_variance(timestep).sqrt()
                    if timestep > 0
                    else torch.zeros_like(alpha_product)
                )
                effective_beta = scheduler.betas[timestep]
            else:
                alpha_ratio = alpha_product / alpha_product_previous
                effective_beta = 1 - alpha_ratio
                predicted_original_scale = (
                    alpha_product_previous.sqrt() * effective_beta / beta_product
                )
                current_sample_scale = (
                    alpha_ratio.sqrt() * beta_product_previous / beta_product
                )
                variance_scale = (
                    (beta_product_previous / beta_product * effective_beta)
                    .clamp_min(0)
                    .sqrt()
                )
            per_scheduler.append(
                (
                    alpha_product.rsqrt(),
                    -beta_product.sqrt() * alpha_product.rsqrt(),
                    predicted_original_scale,
                    current_sample_scale,
                    variance_scale,
                    effective_beta,
                )
            )
        coefficient_row = []
        for coefficient_index in range(len(_COEFFICIENT_NAMES)):
            coefficient_row.append(
                torch.stack(
                    (
                        per_scheduler[0][coefficient_index].repeat(3),
                        per_scheduler[1][coefficient_index].repeat(3),
                    )
                ).reshape(6)
            )
        rows.append(torch.stack(coefficient_row))
        likelihood_betas.append(
            torch.cat((per_scheduler[0][5].repeat(3), per_scheduler[1][5].repeat(3)))
        )

    return (
        timesteps,
        torch.stack(rows).float(),
        torch.stack(likelihood_betas).float(),
    )


__all__ = [
    "DDPM_TIMESTEP_PRESETS",
    "fold_attention",
    "inference_timesteps",
    "normal_sobol_samples",
    "scheduler_coefficients",
]
