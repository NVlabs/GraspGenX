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


def validate_grasp_volume(bounds):
    bounds = np.asarray(bounds, dtype=np.float32).reshape(2, 3)
    if not np.isfinite(bounds).all() or not (bounds[0] < bounds[1]).all():
        raise ValueError("grasp volume requires finite minimum < maximum")
    return bounds


def grasp_volume_from_config(config):
    """Match x_grippers.load_gripper's capture bounds, including standoff."""
    sweep = config["sweep_volume"]
    extents = np.asarray(sweep["extents"], dtype=np.float64).reshape(3)
    offset = np.asarray(sweep["offset"], dtype=np.float64).reshape(3)
    standoff = np.asarray(config["standoff"], dtype=np.float64).reshape(2)
    if not np.isfinite(extents).all() or not (extents > 0).all():
        raise ValueError("sweep volume extents must be positive and finite")
    return validate_grasp_volume([
        offset - extents / 2 + [0, 0, standoff[0]],
        offset + extents / 2 + [0, 0, standoff[1]],
    ])


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
    parser.add_argument(
        "--grasp-volume", nargs=6, type=float, metavar="B",
        help="Explicit xmin ymin zmin xmax ymax zmax for raw descriptors; named grippers use config.json standoff",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gripper_depth = args.gripper_depth
    grasp_volume = None
    if args.gripper_name is not None:
        if args.grasp_volume is not None:
            raise ValueError("named grippers use the grasp volume from config.json")
        config_path = args.assets_dir / args.gripper_name / "config.json"
        with config_path.open() as stream:
            config = json.load(stream)
        sweep = config["sweep_volume"]
        grasp_volume = grasp_volume_from_config(config)
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
    if args.grasp_volume is not None:
        grasp_volume = validate_grasp_volume(args.grasp_volume)
    descriptor = np.asarray(descriptor, dtype=np.float32).reshape(1, 12)
    if gripper_depth is not None and (
        not np.isfinite(gripper_depth) or gripper_depth <= 0.0
    ):
        raise ValueError("gripper depth must be a positive finite value")
    import torch
    from graspgenx.models.grasp_gen import GraspGen
    from graspgenx.utils.checkpoint_io import load_model_cfg

    cfg = load_model_cfg(args.release_root / "gen", args.release_root / "dis")
    model = GraspGen.from_config(cfg.diffusion, cfg.discriminator)
    model.load_state_dict(cfg.eval.gen_checkpoint, cfg.eval.dis_checkpoint)
    model.eval()
    value = torch.from_numpy(descriptor)
    with torch.inference_mode():
        generator = model.grasp_generator.gripper_encoder(value).float().numpy()
        discriminator = model.grasp_discriminator.gripper_encoder(value).float().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    values = dict(
        descriptor=descriptor, generator_embedding=generator,
        discriminator_embedding=discriminator, gripper_depth=np.asarray(
            np.nan if gripper_depth is None else gripper_depth, dtype=np.float32
        ),
    )
    if grasp_volume is not None:
        values["grasp_volume"] = grasp_volume
    np.savez(args.output, **values)
    raw_files = {}
    raw_arrays = [
        ("descriptor", descriptor),
        ("generator_embedding", generator),
        ("discriminator_embedding", discriminator),
    ]
    if grasp_volume is not None:
        raw_arrays.append(("grasp_volume", grasp_volume))
    for name, array in raw_arrays:
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
        "grasp_volume": None if grasp_volume is None else grasp_volume.tolist(),
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
