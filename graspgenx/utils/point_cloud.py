# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import trimesh
import trimesh.transformations as tra
from tqdm import tqdm

from graspgenx.utils.logging_config import get_logger

logger = get_logger(__name__)


def knn_points(X: torch.Tensor, K: int, norm: int):
    """
    Computes the K-nearest neighbors for each point in the point cloud X.

    Args:
        X: (N, 3) tensor representing the point cloud.
        K: Number of nearest neighbors.

    Returns:
        dists: (N, K) tensor containing squared Euclidean distances to the K nearest neighbors.
        idxs: (N, K) tensor containing indices of the K nearest neighbors.
    """
    N, _ = X.shape

    # Compute pairwise squared Euclidean distances
    dist_matrix = torch.cdist(X, X, p=norm)  # (N, N)

    # Ignore self-distance (optional, but avoids trivial zero distance)
    self_mask = torch.eye(N, device=X.device, dtype=torch.bool)
    dist_matrix.masked_fill_(self_mask, float("inf"))  # Set self-distances to inf

    # Get the indices of the K-nearest neighbors
    dists, idxs = torch.topk(dist_matrix, K, dim=1, largest=False)

    return dists, idxs


def point_cloud_outlier_removal(
    obj_pc: torch.Tensor, threshold: float = 0.014
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove outliers from a point cloud. K-nearest neighbors is used to compute the distance to the nearest neighbor for each point.
    If the distance is greater than a threshold, the point is considered an outlier and removed.

    RANSAC can also be used.

    Args:
        obj_pc (torch.Tensor): (N, 3) tensor representing the point cloud.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing filtered and removed point clouds.
    """
    obj_pc = obj_pc.float()
    obj_pc = obj_pc.unsqueeze(0)

    nn_dists, _ = knn_points(obj_pc[0], K=20, norm=1)

    mask = nn_dists.mean(1) < threshold
    filtered_pc = obj_pc[0, mask]
    removed_pc = obj_pc[0][~mask]
    filtered_pc = filtered_pc.view(-1, 3)
    removed_pc = removed_pc.view(-1, 3)

    logger.info(
        f"Removed {obj_pc.shape[1] - filtered_pc.shape[0]} points from point cloud"
    )
    return filtered_pc, removed_pc


def resample_point_cloud_fixed(
    obj_pc: torch.Tensor, num_points: int = 3500
) -> torch.Tensor:
    """Deterministically select or repeat rows to a fixed point count."""
    if obj_pc.ndim != 2 or obj_pc.shape[1] != 3:
        raise ValueError(f"expected point cloud [N,3], got {tuple(obj_pc.shape)}")
    if len(obj_pc) == 0:
        raise ValueError("cannot resample an empty point cloud")
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    count = len(obj_pc)
    output_indices = torch.arange(num_points, device=obj_pc.device)
    if count >= num_points:
        input_indices = torch.div(
            output_indices * count, num_points, rounding_mode="floor"
        )
    else:
        input_indices = output_indices.remainder(count)
    return obj_pc[input_indices]


def depth2points(
    depth: np.ndarray,
    fx: int,
    fy: int,
    cx: int,
    cy: int,
    xmap: np.ndarray = None,
    ymap: np.ndarray = None,
    rgb: np.ndarray = None,
    seg: np.ndarray = None,
    mask: np.ndarray = None,
) -> Dict:
    """Back-project a depth image into a point cloud."""
    if rgb is not None:
        assert rgb.shape[0] == depth.shape[0] and rgb.shape[1] == depth.shape[1]
    if xmap is not None:
        assert xmap.shape[0] == depth.shape[0] and xmap.shape[1] == depth.shape[1]
    if ymap is not None:
        assert ymap.shape[0] == depth.shape[0] and ymap.shape[1] == depth.shape[1]

    im_height, im_width = depth.shape[0], depth.shape[1]

    if xmap is None or ymap is None:
        ww = np.linspace(0, im_width - 1, im_width)
        hh = np.linspace(0, im_height - 1, im_height)
        xmap, ymap = np.meshgrid(ww, hh)

    pt2 = depth
    pt0 = (xmap - cx) * pt2 / fx
    pt1 = (ymap - cy) * pt2 / fy

    mask_depth = np.ma.getmaskarray(np.ma.masked_greater(pt2, 0))
    if mask is None:
        mask = mask_depth
    else:
        mask_semantic = np.ma.getmaskarray(np.ma.masked_equal(mask, 1))
        mask = mask_depth * mask_semantic

    index = mask.flatten().nonzero()[0]

    pt2_valid = pt2.flatten()[:, np.newaxis].astype(np.float32)
    pt0_valid = pt0.flatten()[:, np.newaxis].astype(np.float32)
    pt1_valid = pt1.flatten()[:, np.newaxis].astype(np.float32)
    pc_xyz = np.concatenate((pt0_valid, pt1_valid, pt2_valid), axis=1)
    if rgb is not None:
        r = rgb[:, :, 0].flatten()[:, np.newaxis]
        g = rgb[:, :, 1].flatten()[:, np.newaxis]
        b = rgb[:, :, 2].flatten()[:, np.newaxis]
        pc_rgb = np.concatenate((r, g, b), axis=1)
    else:
        pc_rgb = None

    if seg is not None:
        pc_seg = seg.flatten()[:, np.newaxis]
    else:
        pc_seg = None

    return {"xyz": pc_xyz, "rgb": pc_rgb, "seg": pc_seg, "index": index}


def depth_and_segmentation_to_point_clouds(
    depth_image: np.ndarray,
    segmentation_mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rgb_image: np.ndarray = None,
    target_object_id: int = 1,
    remove_object_from_scene: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a depth + segmentation image into scene and target-object point clouds."""
    unique_ids = np.unique(segmentation_mask)
    if target_object_id not in unique_ids:
        raise ValueError(
            f"Target object ID {target_object_id} not found in segmentation mask. "
            f"Available IDs: {unique_ids}"
        )

    non_background_ids = unique_ids[unique_ids != 0]
    if len(non_background_ids) > 1:
        raise ValueError(
            f"Multiple objects detected in segmentation mask: {non_background_ids}. "
            f"Please ensure only one object is present."
        )

    pts_data = depth2points(
        depth=depth_image,
        fx=int(fx),
        fy=int(fy),
        cx=int(cx),
        cy=int(cy),
        rgb=rgb_image,
        seg=segmentation_mask,
    )

    xyz = pts_data["xyz"]
    rgb = pts_data["rgb"]
    seg = pts_data["seg"]
    index = pts_data["index"]

    xyz_valid = xyz[index]
    seg_valid = seg[index] if seg is not None else None
    rgb_valid = rgb[index] if rgb is not None else None

    scene_pc = xyz_valid
    scene_colors = rgb_valid

    if seg_valid is None:
        raise ValueError("Segmentation data not available from depth2points")

    object_mask = seg_valid.flatten() == target_object_id
    object_pc = xyz_valid[object_mask]
    object_colors = rgb_valid[object_mask] if rgb_valid is not None else None

    if remove_object_from_scene:
        scene_mask = ~object_mask
        scene_pc = xyz_valid[scene_mask]
        scene_colors = rgb_valid[scene_mask] if rgb_valid is not None else None
        logger.info(
            f"Removed {np.sum(object_mask)} object points from scene point cloud"
        )

    if len(object_pc) == 0:
        raise ValueError(f"No points found for target object ID {target_object_id}")

    logger.info(f"Scene point cloud: {len(scene_pc)} points")
    logger.info(f"Object point cloud: {len(object_pc)} points")

    return scene_pc, object_pc, scene_colors, object_colors


def filter_colliding_grasps(
    scene_pc: np.ndarray,
    grasp_poses: np.ndarray,
    gripper_collision_mesh: trimesh.Trimesh,
    collision_threshold: float = 0.002,
    num_collision_samples: int = 2000,
) -> np.ndarray:
    """Return a K-length boolean mask marking collision-free grasps.

    A grasp is collision-free if no surface-sample point of the transformed
    gripper mesh is within `collision_threshold` meters of any scene point.
    """
    gripper_surface_points, _ = trimesh.sample.sample_surface(
        gripper_collision_mesh, num_collision_samples
    )
    gripper_surface_points = np.array(gripper_surface_points)

    scene_pc_torch = torch.from_numpy(scene_pc).float()
    collision_free_mask = []

    logger.info(
        f"Checking collision for {len(grasp_poses)} grasps against "
        f"{len(scene_pc)} scene points..."
    )

    for grasp_pose in tqdm(
        grasp_poses, total=len(grasp_poses), desc="Collision checking"
    ):
        gripper_points_transformed = tra.transform_points(
            gripper_surface_points, grasp_pose
        )
        gripper_points_torch = torch.from_numpy(gripper_points_transformed).float()

        min_distances = []
        batch_size = 100
        for j in range(0, len(gripper_points_torch), batch_size):
            batch_gripper_points = gripper_points_torch[j : j + batch_size]
            distances = torch.cdist(batch_gripper_points, scene_pc_torch, p=2)
            batch_min_distances = torch.min(distances, dim=1)[0]
            min_distances.append(batch_min_distances)

        all_min_distances = torch.cat(min_distances)
        collision_detected = torch.any(all_min_distances < collision_threshold)
        collision_free_mask.append(not collision_detected.item())

    collision_free_mask = np.array(collision_free_mask)
    logger.info(
        f"Found {np.sum(collision_free_mask)}/{len(grasp_poses)} collision-free grasps"
    )
    return collision_free_mask
