#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encode one gripper descriptor for the gripper-independent C++ plans."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--gripper-name",
        help="Named gripper under the gripper-descriptions asset directory",
    )
    source.add_argument("--input-npy", type=Path, help="FP32 descriptor with 12 values")
    source.add_argument("--values", nargs=12, type=float, metavar="X")
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=(
            root / "ext/gripper_descriptions/gripper_descriptions/assets/x_grippers"
        ),
    )
    parser.add_argument(
        "--gripper-depth",
        type=float,
        help="Fingertip depth in meters; inferred for --gripper-name",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    gripper_depth = args.gripper_depth
    if args.gripper_name is not None:
        config_path = args.assets_dir / args.gripper_name / "config.json"
        with config_path.open() as stream:
            config = json.load(stream)
        sweep = config["sweep_volume"]
        descriptor = np.concatenate(
            (
                sweep["extents"],
                sweep["offset"],
                sweep["extents2"],
                sweep["offset2"],
            )
        )
        if gripper_depth is None:
            gripper_depth = float(config["fingertip"][-1])
    elif args.input_npy is not None:
        descriptor = np.load(args.input_npy)
    else:
        descriptor = np.asarray(args.values, dtype=np.float32)
    descriptor = np.asarray(descriptor, dtype=np.float32).reshape(1, 12)
    if gripper_depth is not None and (
        not np.isfinite(gripper_depth) or gripper_depth <= 0.0
    ):
        raise ValueError("gripper depth must be a positive finite value")
    cfg = load_model_cfg(args.release_root / "gen", args.release_root / "dis")
    model = GraspGen.from_config(cfg.diffusion, cfg.discriminator)
    model.load_state_dict(cfg.eval.gen_checkpoint, cfg.eval.dis_checkpoint)
    model.eval()
    value = torch.from_numpy(descriptor)
    with torch.inference_mode():
        generator = model.grasp_generator.gripper_encoder(value).float().numpy()
        discriminator = model.grasp_discriminator.gripper_encoder(value).float().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        descriptor=descriptor,
        generator_embedding=generator,
        discriminator_embedding=discriminator,
        gripper_depth=np.asarray(
            np.nan if gripper_depth is None else gripper_depth, dtype=np.float32
        ),
    )
    raw_files = {}
    for name, array in (
        ("descriptor", descriptor),
        ("generator_embedding", generator),
        ("discriminator_embedding", discriminator),
    ):
        path = args.output.with_suffix(f".{name}.fp32.bin")
        contiguous = np.ascontiguousarray(array, dtype="<f4")
        contiguous.tofile(path)
        raw_files[name] = {
            "path": path.name,
            "shape": list(contiguous.shape),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "format": "little-endian contiguous FP32",
        "gripper_name": args.gripper_name,
        "gripper_depth": gripper_depth,
        "npz": args.output.name,
        "raw_files": raw_files,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"wrote {args.output}: generator={generator.shape}, "
        f"discriminator={discriminator.shape}; manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
