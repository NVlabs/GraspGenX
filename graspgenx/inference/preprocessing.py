# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Point-cloud preprocessing for eager PyTorch inference."""

from __future__ import annotations

import torch

from graspgenx.utils.point_cloud import (
    point_cloud_outlier_removal,
    resample_point_cloud_fixed,
)

MIN_OBJECT_POINT_COUNT = 100


def filter_and_resample_point_cloud(
    obj_pc: torch.Tensor,
    *,
    num_points: int = 3500,
    threshold: float = 0.014,
) -> torch.Tensor:
    """Filter outliers and return exactly ``num_points`` rows.

    This path deliberately uses PyTorch operations on either CPU or CUDA. The
    native runtime owns its separately compiled fixed-shape CUDA preprocessing.
    """
    if obj_pc.ndim != 2 or obj_pc.shape[1] != 3:
        raise ValueError(f"expected point cloud [N,3], got {tuple(obj_pc.shape)}")
    if obj_pc.shape[0] < MIN_OBJECT_POINT_COUNT:
        raise ValueError(
            f"GraspGenX requires at least {MIN_OBJECT_POINT_COUNT} object "
            f"points; received {obj_pc.shape[0]}"
        )
    obj_pc = obj_pc.float().contiguous()
    filtered, _ = point_cloud_outlier_removal(obj_pc, threshold=threshold)
    if len(filtered) < 10:
        filtered = obj_pc
    return resample_point_cloud_fixed(filtered, num_points)
