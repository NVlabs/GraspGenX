# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small tensor batching helpers shared by training and inference."""

from __future__ import annotations

import torch

from graspgenx.utils.logging_config import get_logger

logger = get_logger(__name__)

_STACK_KEYS = {
    "inputs",
    "points",
    "seg",
    "z_offset",
    "onehot",
    "sweep_volume",
    "sweep_volume_open_and_mid",
    "object_inputs",
    "gripper_open_ptc",
    "gripper_close_ptc",
    "gripper_selected_ptc",
    "gripper_selected_open_ptc",
    "gripper_selected_mid_ptc",
    "gripper_selected_close_ptc",
    "gripper_vol_tsdf",
    "gripper_pointnet_repr",
    "bottom_center",
    "cam_pose",
    "ee_pose",
    "placement_masks",
    "placement_region",
}
_CAT_KEYS = {"contact_dirs", "approach_dirs"}


def collate_batch_keys(samples: list[dict]) -> dict | None:
    """Stack the tensor fields used by GraspGenX model batches."""
    if not samples:
        return None

    batch = {key: [sample[key] for sample in samples] for key in samples[0]}
    if "task" in batch:
        tasks = batch.pop("task")
        batch["task_is_pick"] = torch.stack(
            [torch.tensor(task == "pick") for task in tasks]
        )
        batch["task_is_place"] = torch.stack(
            [torch.tensor(task == "place") for task in tasks]
        )

    for key in batch:
        if key in _STACK_KEYS:
            batch[key] = torch.stack(batch[key])
        elif key in _CAT_KEYS:
            batch[key] = torch.cat(batch[key])
    return batch


def collate(samples: list[dict]) -> dict | None:
    """Drop invalid samples and collate the remaining model inputs."""
    valid_samples = [sample for sample in samples if not sample.get("invalid", False)]
    dropped = len(samples) - len(valid_samples)
    if dropped:
        logger.warning("[COLLATE] Dropped %d/%d invalid samples", dropped, len(samples))
    return collate_batch_keys(valid_samples)


__all__ = ["collate", "collate_batch_keys"]
