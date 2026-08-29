#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the small native-CUDA constants that are not embedded in TRT plans."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from artifact_tools import (
    DDPM_TIMESTEP_PRESETS,
    fold_attention,
    normal_sobol_samples,
    scheduler_coefficients,
)
from torch.nn import functional

from graspgenx.models.grasp_gen import GraspGen
from graspgenx.utils.checkpoint_io import load_model_cfg


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=root / "ext/graspgenx_checkpoints/GraspGenXModel/release",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "cpp_runtime/artifacts/weights",
    )
    parser.add_argument("--diffusion-grasps", type=int, default=100)
    return parser.parse_args()


def load_model(release_root):
    cfg = load_model_cfg(release_root / "gen", release_root / "dis")
    model = GraspGen.from_config(cfg.diffusion, cfg.discriminator)
    model.load_state_dict(cfg.eval.gen_checkpoint, cfg.eval.dis_checkpoint)
    return model.eval()


def main():
    args = parse_args()
    if args.diffusion_grasps < 1:
        raise ValueError("--diffusion-grasps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diffusion_steps = tuple(DDPM_TIMESTEP_PRESETS)
    model = load_model(args.release_root)
    generator = model.grasp_generator
    head = generator.diffusion_head
    with torch.inference_mode():
        first_weight, first_bias, _ = fold_attention(head)[0]
        tensors = {
            "observation_weight": first_weight[:, 1024:].half(),
            "sample_encoder_weight": head.sample_encoder[0].weight.half(),
            "sample_encoder_bias": head.sample_encoder[0].bias.half(),
            "prediction_weight": head.prediction_head[4].weight.half(),
            "prediction_bias": head.prediction_head[4].bias.half(),
            "sobol_initial_sample": normal_sobol_samples(args.diffusion_grasps).half(),
        }
    report = {
        "format": "contiguous little-endian raw tensor files",
        "kappa": float(generator.kappa),
        "grid_size": float(generator.grid_size),
        "diffusion_presets": list(diffusion_steps),
        "diffusion_grasps": args.diffusion_grasps,
        "tensors": {},
        "schedules": {},
    }

    def write_tensor(name, value, directory):
        value = value.detach().cpu().contiguous()
        payload = value.numpy().tobytes()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.bin"
        path.write_bytes(payload)
        return {
            "path": str(path.relative_to(args.output_dir)),
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    for name, value in tensors.items():
        output_name = (
            f"sobol_initial_sample_d{args.diffusion_grasps}"
            if name == "sobol_initial_sample"
            else name
        )
        report["tensors"][name] = write_tensor(output_name, value, args.output_dir)

    for step_count in diffusion_steps:
        timesteps, coefficients, _ = scheduler_coefficients(generator, step_count)
        with torch.inference_mode():
            timestep_embeddings = head.diffusion_step_encoder(timesteps).float()
            timestep_conditioning = functional.linear(
                timestep_embeddings, first_weight[:, 512:1024], first_bias
            ).half()
        schedule_directory = args.output_dir / "schedules" / str(step_count)
        report["schedules"][str(step_count)] = {
            "timesteps": timesteps.tolist(),
            "tensors": {
                "timestep_conditioning": write_tensor(
                    "timestep_conditioning",
                    timestep_conditioning,
                    schedule_directory,
                ),
                "ddpm_coefficients": write_tensor(
                    "ddpm_coefficients", coefficients.half(), schedule_directory
                ),
            },
        }
    manifest = args.output_dir / f"manifest_d{args.diffusion_grasps}.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
