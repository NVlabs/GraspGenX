# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GraspMoE: GraspGenX diffusion sampler outputs union OBB-swept top-down
# candidates, every grasp scored by the GraspGenX discriminator.
#
# GraspGenX's cross-embodiment conventions (per-gripper depth/width pulled
# from XGripperInfo) and with a pure-numpy/scipy OBB implementation in place
# of cv2.

# The OBB implementation is inspired from Berkeley AUTOLab's Cap-X paper and discussions with Ken Goldberg, Shuangyu Xie and Eric Chen.
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import torch
from scipy.spatial import ConvexHull, cKDTree

from graspgenx.batching import collate
from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.utils.logging_config import get_logger
from graspgenx.utils.point_cloud import point_cloud_outlier_removal

logger = get_logger(__name__)

# Pure-PyTorch planner defaults. The C++ runtime owns its corresponding fixed
# OBB shape and receives the requested count through the fast-runtime config.
OBB_YAW_COUNT = 10
OBB_Z_OFFSETS_CM = (-8.0, -6.0, -4.0, -2.0, 0.0)

# GPU preprocessing flag. Controls (a) the OBB statistical-outlier-removal kNN
# (torch.cdist instead of scipy cKDTree, double precision -> exact parity) and
# (b) the diffusion-path point_cloud_outlier_removal device. Numerically
# equivalent to the CPU paths.
#
# Default = CPU (scipy). GRASPGENX_GPU_OBB=1 explicitly enables the eager
# PyTorch/CUDA implementation; the native runtime does not use this code.
_GPU_OBB_ENV = os.environ.get("GRASPGENX_GPU_OBB")  # None | "0" | "1"
_GPU_OBB = _GPU_OBB_ENV == "1"

# Preprocessing optimizations (default on; exposed for ablation/benchmarking).
# _REUSE_OBJ_EMBED: reuse the diffusion-pass discriminator object embedding for
#   OBB-candidate scoring instead of re-encoding per object.
# _VECTORIZE_CAND: build OBB face candidates with vectorized numpy instead of a
#   triple Python loop (bit-identical output).
_REUSE_OBJ_EMBED = True
_VECTORIZE_CAND = True


def _sor_keep_mask_torch(pts: np.ndarray, k: int, std_ratio: float) -> np.ndarray:
    """kNN-based statistical-outlier-removal keep-mask, computed on GPU.

    Mirrors the scipy path exactly (double precision, mean of the k nearest
    non-self distances, population std). Chunked over query rows to bound the
    distance-matrix memory for large clouds.
    """
    t = torch.from_numpy(np.ascontiguousarray(pts)).double().cuda()
    n = t.shape[0]
    mean_d = torch.empty(n, dtype=torch.float64, device=t.device)
    chunk = 4096
    for i in range(0, n, chunk):
        d = torch.cdist(t[i : i + chunk], t)  # (chunk, N)
        knn = d.topk(k + 1, dim=1, largest=False).values  # includes self (0)
        mean_d[i : i + chunk] = knn[:, 1:].mean(dim=1)
    thresh = mean_d.mean() + std_ratio * mean_d.std(unbiased=False)
    return (mean_d < thresh).cpu().numpy()


# ---------------------------------------------------------------------------
# Oriented bounding box (numpy/scipy only, no cv2)
# ---------------------------------------------------------------------------
def _statistical_outlier_removal(
    pts: np.ndarray, k: int = 20, std_ratio: float = 2.0
) -> np.ndarray:
    if len(pts) <= k + 1:
        return pts
    if _GPU_OBB and torch.cuda.is_available():
        try:
            return pts[_sor_keep_mask_torch(pts, k, std_ratio)]
        except Exception as e:  # pragma: no cover - fall back to CPU
            logger.debug(f"[graspmoe] GPU SOR failed ({e}); using scipy cKDTree.")
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=k + 1)
    mean_d = d[:, 1:].mean(axis=1)
    keep = mean_d < (mean_d.mean() + std_ratio * mean_d.std())
    return pts[keep]


def _min_area_rect_xy(pts_xy: np.ndarray) -> float:
    """Rotating-calipers min-area rectangle on the 2D convex hull.

    Returns the rotation angle (radians) of the rectangle's principal axis from
    world-X. Equivalent to cv2.minAreaRect's angle output, modulo sign/convention.
    """
    if len(pts_xy) < 3:
        raise ValueError(f"Need >=3 points for min-area rect, got {len(pts_xy)}")
    hull = ConvexHull(pts_xy)
    hull_pts = pts_xy[hull.vertices]
    n = len(hull_pts)
    best_area = np.inf
    best_angle = 0.0
    for i in range(n):
        p0, p1 = hull_pts[i], hull_pts[(i + 1) % n]
        edge = p1 - p0
        if np.linalg.norm(edge) < 1e-9:
            continue
        angle = float(np.arctan2(edge[1], edge[0]))
        c, s = np.cos(-angle), np.sin(-angle)
        R = np.array([[c, -s], [s, c]])
        rotated = hull_pts @ R.T
        xmin, ymin = rotated.min(axis=0)
        xmax, ymax = rotated.max(axis=0)
        area = (xmax - xmin) * (ymax - ymin)
        if area < best_area:
            best_area = area
            best_angle = angle
    return best_angle


def _obb_from_angle(
    object_pc: np.ndarray, angle: float, lo: float = 2.0, hi: float = 98.0
):
    """Given a chosen XY rotation angle, compute (center, half_extent, R) using
    `lo`/`hi` percentile extents (matches the production helper's robust bounds)."""
    pts_xy = object_pc[:, :2]
    z_vals = object_pc[:, 2]
    c, s = np.cos(-angle), np.sin(-angle)
    R2d = np.array([[c, -s], [s, c]])
    rotated_xy = pts_xy @ R2d.T
    mins = np.array(
        [
            np.percentile(rotated_xy[:, 0], lo),
            np.percentile(rotated_xy[:, 1], lo),
            np.percentile(z_vals, lo),
        ]
    )
    maxs = np.array(
        [
            np.percentile(rotated_xy[:, 0], hi),
            np.percentile(rotated_xy[:, 1], hi),
            np.percentile(z_vals, hi),
        ]
    )
    extent = maxs - mins
    center_local = (mins + maxs) / 2.0
    R = np.eye(3)
    R[:2, :2] = R2d.T
    center = R @ center_local
    half_extent = extent / 2.0
    return center, half_extent, R


def _compute_obb(object_pc: np.ndarray, mode: str = "advanced"):
    """Oriented bounding box of an XY-projected 3D cloud.

    mode="advanced": SOR -> convex hull -> rotating-calipers min-area rect
                     -> 2nd/98th percentile extents. Pure numpy/scipy.
    mode="pca":      classic PCA on XY (fallback).
    """
    if mode == "advanced" and object_pc.shape[0] >= 4:
        try:
            pts = object_pc.astype(np.float64) + np.random.normal(
                0.0, 1e-4, object_pc.shape
            )
            clean = _statistical_outlier_removal(pts, k=20, std_ratio=2.0)
            if len(clean) < 4:
                raise RuntimeError(f"too few points after SOR: {len(clean)}")
            angle = _min_area_rect_xy(clean[:, :2])
            return _obb_from_angle(clean, angle)
        except Exception as e:
            logger.debug(f"[graspmoe] advanced OBB failed ({e}); falling back to PCA")

    if object_pc.shape[0] < 3:
        raise ValueError(f"Need >=3 points for OBB, got {object_pc.shape[0]}")
    pts_xy = object_pc[:, :2]
    cov = np.cov(pts_xy, rowvar=False)
    _, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]
    angle = float(np.arctan2(principal[1], principal[0]))
    return _obb_from_angle(object_pc, angle)


def _world_aligned_top_down_grasp(
    center: np.ndarray, half_extent: np.ndarray, R: np.ndarray, z_offset: float = 0.0
) -> np.ndarray:
    """Top-down grasp pose centered above the OBB. Gripper Z points -world_Z."""
    world_z_half = float(np.sum(np.abs(R[2, :]) * half_extent))
    top_z = center[2] + world_z_half
    raw_z = max(top_z + z_offset, -0.05)
    R_grasp = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    T = np.eye(4)
    T[:3, :3] = R_grasp
    T[:3, 3] = [center[0], center[1], raw_z]
    return T


# ---------------------------------------------------------------------------
# Discriminator scoring
# ---------------------------------------------------------------------------
def _score_grasps_with_discriminator(
    grasps_world: np.ndarray,
    pc_centered: torch.Tensor,
    pc_center: np.ndarray,
    grasp_sampler: GraspGenXSampler,
    object_embedding: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Discriminator-only inference on a batch of world-frame grasps.

    If ``object_embedding`` (the object's precomputed discriminator embedding,
    shape [num_object_dim]) is given, it is injected so the discriminator skips
    re-encoding the object point cloud.
    """
    if len(grasps_world) == 0:
        return np.zeros((0,), dtype=np.float32)

    device = next(grasp_sampler.model.parameters()).device

    grasps_centered = grasps_world.copy().astype(np.float32)
    grasps_centered[:, :3, 3] -= pc_center.astype(np.float32)[None, :3]
    grasps_t = torch.from_numpy(grasps_centered).to(device)

    obj_pts_color = torch.zeros_like(pc_centered)
    data = {}
    data["task"] = "pick"
    data["inputs"] = torch.cat([pc_centered, obj_pts_color[:, :3]], dim=-1).float()
    data["points"] = pc_centered
    data = grasp_sampler.load_gripper_input(data)
    data["grasps"] = grasps_t
    data["grasp_key"] = "grasps"

    data_batch = collate([data])
    data_batch["grasp_key"] = "grasps"
    if object_embedding is not None:
        # [num_object_dim] -> [1, num_object_dim] (one object in this batch)
        data_batch["object_embedding"] = object_embedding.reshape(1, -1).to(device)
    with torch.inference_mode():
        out_data, _, _ = grasp_sampler.model.grasp_discriminator.infer(data_batch)
    return (
        out_data["grasp_confidence"][0, :, 0].detach().cpu().numpy().astype(np.float32)
    )


def _interior_positions(half: float, spacing_m: float) -> np.ndarray:
    """Signed offsets ``arange(-half + spacing, half - spacing + eps, spacing)``;
    falls back to ``[0.0]`` when the axis is too short for a single interior
    sample."""
    if half <= spacing_m or spacing_m <= 0.0:
        return np.array([0.0], dtype=np.float64)
    eps = 1e-9
    out = np.arange(-half + spacing_m, half - spacing_m + eps, spacing_m)
    if len(out) == 0:
        out = np.array([0.0], dtype=np.float64)
    return out.astype(np.float64)


def _long_axis_positions(
    half_extent: np.ndarray, spacing_m: float
) -> tuple[int, np.ndarray]:
    """Discretized positions along the OBB's longer XY axis.

    Returns ``(long_axis_idx, positions)`` where ``long_axis_idx`` is 0 or 1
    selecting the longer of the two horizontal axes.
    """
    long_idx = 0 if half_extent[0] >= half_extent[1] else 1
    return long_idx, _interior_positions(float(half_extent[long_idx]), spacing_m)


def _build_face_candidates(
    face_origin_world: np.ndarray,
    approach_dir_world: np.ndarray,
    in_plane_axis_world: np.ndarray,
    positions_local: np.ndarray,
    yaws: np.ndarray,
    z_offsets_m: np.ndarray,
    gripper_depth_m: float,
) -> np.ndarray:
    """Build (P*Y*Z, 4, 4) world-frame grasp poses approaching one OBB face.

    Conventions:
      - gripper Z (closing direction) = -approach_dir_world (points into OBB)
      - gripper X = in_plane_axis_world (re-orthogonalized against gripper Z)
      - yaw rotates about local gripper Z (= the approach axis)
      - positions_local: signed offsets along in_plane_axis_world (m)
      - z_offsets_m: signed offsets along approach_dir_world (m);
        positive = further from face, negative = into the OBB

    Returns poses anchored at the gripper base (tip-frame shifted by
    -gripper_depth_m along local Z).
    """
    n = approach_dir_world / max(float(np.linalg.norm(approach_dir_world)), 1e-12)
    gz = -n
    gx = in_plane_axis_world - in_plane_axis_world.dot(gz) * gz
    nrm = float(np.linalg.norm(gx))
    if nrm < 1e-9:
        fallback = (
            np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        )
        gx = fallback - fallback.dot(gz) * gz
        nrm = float(np.linalg.norm(gx))
    gx = gx / nrm
    gy = np.cross(gz, gx)
    base_R = np.column_stack([gx, gy, gz])

    base_offset = np.eye(4)
    base_offset[2, 3] = -gripper_depth_m

    if not _VECTORIZE_CAND:
        candidates = []
        for p in positions_local:
            anchor = face_origin_world + float(p) * in_plane_axis_world
            for yaw in yaws:
                c, s = float(np.cos(yaw)), float(np.sin(yaw))
                R_yaw_local = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                R_grasp = base_R @ R_yaw_local
                for z_off in z_offsets_m:
                    T = np.eye(4)
                    T[:3, :3] = R_grasp
                    T[:3, 3] = anchor + float(z_off) * n
                    candidates.append(T)
        grasps_tip = np.stack(candidates, axis=0).astype(np.float32)
        return (grasps_tip @ base_offset).astype(np.float32)

    # Vectorized over (positions P, yaws Y, z_offsets Z) — same nested order as
    # `for p: for yaw: for z_off`, i.e. C-order reshape p-outer, yaw, z-inner.
    positions_local = np.asarray(positions_local, dtype=np.float64)
    yaws = np.asarray(yaws, dtype=np.float64)
    z_offsets_m = np.asarray(z_offsets_m, dtype=np.float64)
    P, Y, Z = len(positions_local), len(yaws), len(z_offsets_m)

    cy, sy = np.cos(yaws), np.sin(yaws)
    R_yaw = np.zeros((Y, 3, 3))
    R_yaw[:, 0, 0] = cy
    R_yaw[:, 0, 1] = -sy
    R_yaw[:, 1, 0] = sy
    R_yaw[:, 1, 1] = cy
    R_yaw[:, 2, 2] = 1.0
    R_grasp = base_R[None] @ R_yaw  # (Y, 3, 3)

    anchors = (
        face_origin_world[None, :]
        + positions_local[:, None] * in_plane_axis_world[None, :]
    )  # (P, 3)
    pos = anchors[:, None, :] + z_offsets_m[None, :, None] * n[None, None, :]  # (P,Z,3)

    T = np.zeros((P, Y, Z, 4, 4), dtype=np.float64)
    T[..., 3, 3] = 1.0
    T[:, :, :, :3, :3] = R_grasp[None, :, None, :, :]
    T[:, :, :, :3, 3] = pos[:, None, :, :]
    grasps_tip = T.reshape(P * Y * Z, 4, 4).astype(np.float32)
    return (grasps_tip @ base_offset).astype(np.float32)


def _run_obb_branch(
    pc_filtered: np.ndarray,
    pc_filtered_centered: torch.Tensor,
    pc_center: np.ndarray,
    grasp_sampler: GraspGenXSampler,
    num_yaws: int,
    z_offsets_cm: Sequence[float],
    obb_mode: str,
    gripper_width_m: float,
    gripper_depth_m: float,
    skip_obb_rule: str,
    obb_density: str = "sparse",
    obb_position_spacing_m: float = 0.01,
    object_embedding: Optional[torch.Tensor] = None,
):
    try:
        center, half_extent, R_obb = _compute_obb(pc_filtered, mode=obb_mode)
    except Exception as e:
        logger.warning(f"[graspmoe] OBB compute failed: {e}; skipping OBB branch")
        return (
            np.zeros((0, 4, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            None,
            True,
        )

    obb_dict = {
        "center": center.astype(np.float64),
        "half_extent": half_extent.astype(np.float64),
        "R": R_obb.astype(np.float64),
    }

    full_extent = 2.0 * half_extent
    if skip_obb_rule == "auto" and np.all(full_extent > gripper_width_m):
        logger.info(
            f"[graspmoe] OBB sweep skipped: extents {full_extent.round(3).tolist()} "
            f"all > gripper width {gripper_width_m:.5f} m"
        )
        return (
            np.zeros((0, 4, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            obb_dict,
            True,
        )

    world_z_half = float(np.sum(np.abs(R_obb[2, :]) * half_extent))
    top_z = center[2] + world_z_half

    yaws = np.linspace(0.0, 2.0 * np.pi, int(num_yaws), endpoint=False)
    z_offsets_m = np.asarray(z_offsets_cm, dtype=np.float64) / 100.0

    if obb_density == "dense-topandside":
        # 5 OBB faces: top (+z_obb) plus the four horizontal sides (±x_obb,
        # ±y_obb). Bottom skipped — the object sits on the table.
        x_axis = R_obb[:, 0]
        y_axis = R_obb[:, 1]
        z_world = np.array([0.0, 0.0, 1.0])
        face_blocks: list[np.ndarray] = []
        face_log: list[tuple[str, int]] = []

        top_long_idx, top_positions = _long_axis_positions(
            half_extent, obb_position_spacing_m
        )
        face_blocks.append(
            _build_face_candidates(
                face_origin_world=np.array([center[0], center[1], top_z]),
                approach_dir_world=z_world,
                in_plane_axis_world=R_obb[:, top_long_idx],
                positions_local=top_positions,
                yaws=yaws,
                z_offsets_m=z_offsets_m,
                gripper_depth_m=gripper_depth_m,
            )
        )
        face_log.append(("top", len(face_blocks[-1])))

        positions_x_face = _interior_positions(
            float(half_extent[1]), obb_position_spacing_m
        )
        for sign, tag in ((+1.0, "+x"), (-1.0, "-x")):
            face_origin = center + sign * float(half_extent[0]) * x_axis
            face_origin[2] = center[2]
            face_blocks.append(
                _build_face_candidates(
                    face_origin_world=face_origin,
                    approach_dir_world=sign * x_axis,
                    in_plane_axis_world=y_axis,
                    positions_local=positions_x_face,
                    yaws=yaws,
                    z_offsets_m=z_offsets_m,
                    gripper_depth_m=gripper_depth_m,
                )
            )
            face_log.append((tag, len(face_blocks[-1])))

        positions_y_face = _interior_positions(
            float(half_extent[0]), obb_position_spacing_m
        )
        for sign, tag in ((+1.0, "+y"), (-1.0, "-y")):
            face_origin = center + sign * float(half_extent[1]) * y_axis
            face_origin[2] = center[2]
            face_blocks.append(
                _build_face_candidates(
                    face_origin_world=face_origin,
                    approach_dir_world=sign * y_axis,
                    in_plane_axis_world=x_axis,
                    positions_local=positions_y_face,
                    yaws=yaws,
                    z_offsets_m=z_offsets_m,
                    gripper_depth_m=gripper_depth_m,
                )
            )
            face_log.append((tag, len(face_blocks[-1])))

        grasps_world = np.concatenate(face_blocks, axis=0).astype(np.float32)
        per_face_str = ", ".join(f"{tag}={n}" for tag, n in face_log)
        logger.info(
            f"[graspmoe] generated {len(grasps_world)} OBB candidates "
            f"({per_face_str}; density=dense-topandside, spacing="
            f"{obb_position_spacing_m * 100:.1f}cm)"
        )
    else:
        base = _world_aligned_top_down_grasp(center, half_extent, R_obb, z_offset=0.0)
        base_R = base[:3, :3]

        if obb_density == "dense":
            long_idx, positions_local = _long_axis_positions(
                half_extent, obb_position_spacing_m
            )
            long_axis_world = R_obb[:, long_idx]
        else:
            long_idx = 0 if half_extent[0] >= half_extent[1] else 1
            positions_local = np.array([0.0], dtype=np.float64)
            long_axis_world = R_obb[:, long_idx]

        candidates = []
        for p in positions_local:
            cx = center[0] + p * long_axis_world[0]
            cy = center[1] + p * long_axis_world[1]
            for yaw in yaws:
                c, s = float(np.cos(yaw)), float(np.sin(yaw))
                R_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                R_grasp = R_yaw @ base_R
                for z_off in z_offsets_m:
                    T = np.eye(4)
                    T[:3, :3] = R_grasp
                    T[:3, 3] = [cx, cy, top_z + float(z_off)]
                    candidates.append(T)
        grasps_world = np.stack(candidates, axis=0).astype(np.float32)

        base_offset = np.eye(4)
        base_offset[2, 3] = -gripper_depth_m
        grasps_world = (grasps_world @ base_offset).astype(np.float32)

        if obb_density == "dense":
            logger.info(
                f"[graspmoe] generated {len(grasps_world)} OBB candidates "
                f"({len(positions_local)} positions x {num_yaws} yaws x "
                f"{len(z_offsets_m)} Zs, density=dense, spacing="
                f"{obb_position_spacing_m * 100:.1f}cm, axis={'X' if long_idx == 0 else 'Y'})"
            )
        else:
            logger.info(
                f"[graspmoe] generated {len(grasps_world)} OBB candidates "
                f"({num_yaws} yaws x {len(z_offsets_m)} Zs, density=sparse)"
            )

    scores = _score_grasps_with_discriminator(
        grasps_world,
        pc_filtered_centered,
        pc_center,
        grasp_sampler,
        object_embedding=object_embedding,
    )
    return grasps_world, scores, obb_dict, False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _run_native_graspmoe(
    object_pc: np.ndarray | torch.Tensor,
    grasp_sampler: GraspGenXSampler,
    grasp_threshold: float,
    num_grasps: int,
    topk_num_grasps: int,
) -> dict:
    """One-object fixed-shape C++ fast-inference path."""
    diff, scores_diff, obb, scores_obb, bounds = grasp_sampler.sample_moe(
        object_pc,
        threshold=grasp_threshold,
        num_grasps=num_grasps,
        remove_outliers=True,
    )
    if topk_num_grasps is not None and topk_num_grasps > 0:
        scores = torch.cat((scores_diff, scores_obb))
        if len(scores) > topk_num_grasps:
            _, order = torch.topk(scores, topk_num_grasps, largest=True, sorted=False)
            diff_count = len(scores_diff)
            diff_order = order[order < diff_count]
            obb_order = order[order >= diff_count] - diff_count
            diff = diff[diff_order]
            scores_diff = scores_diff[diff_order]
            obb = obb[obb_order]
            scores_obb = scores_obb[obb_order]

    grasps_diff = diff.cpu().numpy().astype(np.float32)
    scores_diff_np = scores_diff.cpu().numpy().astype(np.float32)
    grasps_obb = obb.cpu().numpy().astype(np.float32)
    scores_obb_np = scores_obb.cpu().numpy().astype(np.float32)
    obb_parameters = bounds.cpu().numpy().astype(np.float64)
    cosine, sine = obb_parameters[6:]
    obb_dict = {
        "center": obb_parameters[:3],
        "half_extent": obb_parameters[3:6],
        "R": np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
    }
    return {
        "grasps_diff": grasps_diff,
        "scores_diff": scores_diff_np,
        "grasps_obb": grasps_obb,
        "scores_obb": scores_obb_np,
        "pc_removed": np.zeros((0, 3), dtype=np.float32),
        "obb": obb_dict,
        "skipped_obb": False,
    }


def run_graspmoe(
    object_pc: np.ndarray,
    grasp_sampler: GraspGenXSampler,
    grasp_threshold: float = -1.0,
    num_grasps: int | None = None,
    topk_num_grasps: int = -1,
    num_yaws: int | None = None,
    z_offsets_cm: Sequence[float] = (-8, -6, -4, -2, 0),
    outlier_threshold: float = 0.014,
    outlier_k: int = 20,  # currently a no-op: graspgenx's point_cloud_outlier_removal hardcodes K=20
    obb_mode: str = "pca",
    skip_obb_rule: str = "never",
    obb_density: str = "sparse",
    obb_position_spacing_m: float = 0.01,
) -> dict:
    """Diffusion grasps union OBB-swept top-down candidates, all scored by the
    GraspGenX discriminator.

    Args:
        object_pc: (N, 3) segmented object point cloud in world frame.
        grasp_sampler: Initialized GraspGenXSampler.
        grasp_threshold: Discriminator score threshold; applied to both branches.
                         Use -1.0 to keep all and rely on top-k.
        num_grasps: Diffusion samples to draw per pass.
        topk_num_grasps: Top-k cap applied to the union after both branches run.
                         -1 means "keep all".
        num_yaws / z_offsets_cm: OBB sweep dimensions. ``None`` selects the
                                 configured count in the optimized runtime and
                                 10 yaws in the eager implementation.
        outlier_threshold / outlier_k: outlier-removal hyperparameters (run once
                                       externally so both branches see the same cloud).
        obb_mode: "advanced" (SOR + hull + rotating calipers) or "pca".
        skip_obb_rule: "auto" skips OBB branch when every OBB extent > gripper
                       width; "never" always runs it.

    Returns:
        Dict with keys:
            grasps_diff, scores_diff,
            grasps_obb,  scores_obb,
            pc_removed,  obb (dict|None), skipped_obb.
    """
    if num_grasps is None:
        num_grasps = grasp_sampler.diffusion_grasps
    native_obb_count = int(
        getattr(getattr(grasp_sampler, "native_runtime", None), "obb_grasps", -1)
    )
    native_fixed_shape = (
        grasp_sampler.using_fast_inference
        and (num_yaws is None or int(num_yaws) * len(z_offsets_cm) == native_obb_count)
        and tuple(float(value) for value in z_offsets_cm) == OBB_Z_OFFSETS_CM
        and obb_mode == "pca"
        and obb_density == "sparse"
        and skip_obb_rule == "never"
    )
    if native_fixed_shape:
        return _run_native_graspmoe(
            object_pc,
            grasp_sampler,
            grasp_threshold,
            num_grasps,
            topk_num_grasps,
        )
    if grasp_sampler.using_fast_inference:
        raise ValueError(
            "the native C++ runtime supports its fixed PCA/sparse OBB plan only; "
            "construct an eager sampler for custom OBB settings"
        )
    if num_yaws is None:
        num_yaws = OBB_YAW_COUNT

    gripper = grasp_sampler.gripper
    gripper_depth_m = float(gripper.depth)
    # Use the sweep-volume X extent (= jaw opening width) for the skip-OBB
    # rule, not the outer bbox width which includes the linkage and is much
    # larger than the actual fingertip aperture. gripper.sweep_volume is
    # [extents_xyz, offset_xyz] (see x_grippers.py:203).
    gripper_width_m = float(gripper.sweep_volume[0])
    if gripper_width_m <= 0.0:
        raise RuntimeError(
            f"Gripper '{gripper.gripper_name}' has zero jaw width "
            f"(sweep_volume[0]={gripper.sweep_volume[0]})."
        )

    # 1. Outlier removal — run once externally so both branches see the same cloud.
    object_pc_t = torch.from_numpy(object_pc.astype(np.float32))
    pc_filtered_t, pc_removed_t = point_cloud_outlier_removal(
        object_pc_t,
        threshold=outlier_threshold,
    )
    pc_filtered = pc_filtered_t.cpu().numpy().astype(np.float32)
    pc_removed = pc_removed_t.cpu().numpy().astype(np.float32)
    logger.info(
        f"[graspmoe] outlier removal: {len(pc_filtered)} kept, "
        f"{len(pc_removed)} removed (thresh={outlier_threshold}, k={outlier_k})"
    )
    if len(pc_filtered) < 10:
        logger.warning(
            "[graspmoe] too few points after outlier removal; returning empty"
        )
        return {
            "grasps_diff": np.zeros((0, 4, 4), dtype=np.float32),
            "scores_diff": np.zeros((0,), dtype=np.float32),
            "grasps_obb": np.zeros((0, 4, 4), dtype=np.float32),
            "scores_obb": np.zeros((0,), dtype=np.float32),
            "pc_removed": pc_removed,
            "obb": None,
            "skipped_obb": True,
        }

    # 2. Diffusion branch — keep every generated grasp here (topk=num_grasps
    # avoids run_inference's implicit 100-cap when grasp_threshold == -1);
    # the global top-k across the union is applied later.
    grasps_diff_t, scores_diff_t = GraspGenXSampler.run_inference(
        pc_filtered,
        grasp_sampler,
        grasp_threshold=grasp_threshold,
        num_grasps=num_grasps,
        topk_num_grasps=num_grasps,
        remove_outliers=False,
    )
    if len(grasps_diff_t) > 0:
        grasps_diff = grasps_diff_t.cpu().numpy().astype(np.float32)
        scores_diff = scores_diff_t.cpu().numpy().astype(np.float32)
        grasps_diff[:, 3, 3] = 1.0
    else:
        grasps_diff = np.zeros((0, 4, 4), dtype=np.float32)
        scores_diff = np.zeros((0,), dtype=np.float32)

    # 3. OBB branch.
    pc_center = pc_filtered.mean(axis=0).astype(np.float64)
    device = next(grasp_sampler.model.parameters()).device
    pc_filtered_centered_t = (
        torch.from_numpy(pc_filtered) - torch.from_numpy(pc_center.astype(np.float32))
    ).to(device)
    grasps_obb, scores_obb, obb_dict, skipped = _run_obb_branch(
        pc_filtered=pc_filtered,
        pc_filtered_centered=pc_filtered_centered_t,
        pc_center=pc_center,
        grasp_sampler=grasp_sampler,
        num_yaws=num_yaws,
        z_offsets_cm=tuple(z_offsets_cm),
        obb_mode=obb_mode,
        gripper_width_m=gripper_width_m,
        gripper_depth_m=gripper_depth_m,
        skip_obb_rule=skip_obb_rule,
        obb_density=obb_density,
        obb_position_spacing_m=obb_position_spacing_m,
    )

    # 4. Apply the same threshold to the OBB side that GraspGenX applies
    # internally to diffusion samples (so both branches obey one threshold).
    if grasp_threshold > 0.0 and len(scores_obb) > 0:
        keep = scores_obb >= float(grasp_threshold)
        grasps_obb = grasps_obb[keep]
        scores_obb = scores_obb[keep]

    # 5. Optional global top-k across the union (caller's responsibility to
    # re-route the union; we still return both subsets separately).
    if topk_num_grasps is not None and topk_num_grasps > 0:
        all_scores = np.concatenate([scores_diff, scores_obb])
        if len(all_scores) > topk_num_grasps:
            kth = np.partition(all_scores, -topk_num_grasps)[-topk_num_grasps]
            keep_diff = scores_diff >= kth
            keep_obb = scores_obb >= kth
            grasps_diff = grasps_diff[keep_diff]
            scores_diff = scores_diff[keep_diff]
            grasps_obb = grasps_obb[keep_obb]
            scores_obb = scores_obb[keep_obb]

    n_total = len(grasps_diff) + len(grasps_obb)
    if n_total > 0:
        cat = np.concatenate([scores_diff, scores_obb])
        score_lo, score_hi = float(cat.min()), float(cat.max())
    else:
        score_lo = score_hi = 0.0
    logger.info(
        f"[graspmoe] {n_total} total grasps (diffusion={len(grasps_diff)}, "
        f"OBB={len(grasps_obb)}, skipped_obb={skipped}); "
        f"score range {score_lo:.3f}..{score_hi:.3f}"
    )

    return {
        "grasps_diff": grasps_diff,
        "scores_diff": scores_diff,
        "grasps_obb": grasps_obb,
        "scores_obb": scores_obb,
        "pc_removed": pc_removed,
        "obb": obb_dict,
        "skipped_obb": skipped,
    }


def run_graspmoe_batch(
    object_pcs: list,
    grasp_sampler: GraspGenXSampler,
    grasp_threshold: float = -1.0,
    num_grasps: int | None = None,
    topk_num_grasps: int = -1,
    num_yaws: int | None = None,
    z_offsets_cm: Sequence[float] = (-8, -6, -4, -2, 0),
    outlier_threshold: float = 0.014,
    outlier_k: int = 20,
    obb_mode: str = "pca",
    skip_obb_rule: str = "never",
    obb_density: str = "sparse",
    obb_position_spacing_m: float = 0.01,
) -> list:
    """Batched :func:`run_graspmoe`: shares one diffusion forward pass across
    all input objects. The OBB branch and its discriminator scoring remain
    per-object (each object has a different OBB and may skip the branch).
    Returns one dict per input PC, same keys as :func:`run_graspmoe`.
    """
    n = len(object_pcs)
    if n == 0:
        return []
    if num_grasps is None:
        num_grasps = grasp_sampler.diffusion_grasps

    native_obb_count = int(
        getattr(getattr(grasp_sampler, "native_runtime", None), "obb_grasps", -1)
    )
    native_fixed_shape = (
        grasp_sampler.using_fast_inference
        and (num_yaws is None or int(num_yaws) * len(z_offsets_cm) == native_obb_count)
        and tuple(float(value) for value in z_offsets_cm) == OBB_Z_OFFSETS_CM
        and obb_mode == "pca"
        and obb_density == "sparse"
        and skip_obb_rule == "never"
    )
    if native_fixed_shape:
        return [
            _run_native_graspmoe(
                point_cloud,
                grasp_sampler,
                grasp_threshold,
                num_grasps,
                topk_num_grasps,
            )
            for point_cloud in object_pcs
        ]
    if num_yaws is None:
        num_yaws = OBB_YAW_COUNT
    if grasp_sampler.using_fast_inference:
        raise ValueError(
            "the native C++ runtime supports its fixed PCA/sparse OBB plan only; "
            "construct an eager sampler for custom OBB settings"
        )

    gripper = grasp_sampler.gripper
    gripper_depth_m = float(gripper.depth)
    gripper_width_m = float(gripper.sweep_volume[0])
    if gripper_width_m <= 0.0:
        raise RuntimeError(
            f"Gripper '{gripper.gripper_name}' has zero jaw width "
            f"(sweep_volume[0]={gripper.sweep_volume[0]})."
        )

    # NOTE: this L1 kNN outlier removal stays on CPU. Measured on the RTX 3090,
    # the GPU path is ~3x slower per object — L1 (p=1) cdist has no matmul
    # fast-path on GPU, so for a single ~3500-pt cloud the launch/transfer
    # overhead dominates. (The OBB SOR is GPU-accelerated because it uses L2.)
    pc_filtered_list: list = []
    pc_removed_list: list = []
    for pc in object_pcs:
        pc_t = (
            torch.from_numpy(pc.astype(np.float32))
            if isinstance(pc, np.ndarray)
            else pc.float()
        )
        f_t, r_t = point_cloud_outlier_removal(pc_t, threshold=outlier_threshold)
        pc_filtered_list.append(f_t.cpu().numpy().astype(np.float32))
        pc_removed_list.append(r_t.cpu().numpy().astype(np.float32))

    # Reuse the per-object discriminator embedding computed during diffusion-grasp
    # scoring for the OBB-candidate scoring, so each object is encoded once
    # instead of being re-encoded per object in the OBB branch.
    # topk=num_grasps keeps every generated grasp here (avoids
    # run_inference_batch's implicit 100-cap when grasp_threshold == -1);
    # per-object thresholding + global top-k are applied afterwards.
    diff_results, obj_embeddings = GraspGenXSampler.run_inference_batch(
        [pc for pc in pc_filtered_list],
        grasp_sampler,
        grasp_threshold=-1.0,
        num_grasps=num_grasps,
        topk_num_grasps=num_grasps,
        remove_outliers=False,
        return_object_embedding=True,
    )

    device = next(grasp_sampler.model.parameters()).device
    outputs: list = []
    for i, pc_filtered in enumerate(pc_filtered_list):
        pc_removed = pc_removed_list[i]

        if len(pc_filtered) < 10:
            logger.warning(
                f"[graspmoe_batch] obj {i}: too few points after outlier removal; "
                f"returning empty"
            )
            outputs.append(
                {
                    "grasps_diff": np.zeros((0, 4, 4), dtype=np.float32),
                    "scores_diff": np.zeros((0,), dtype=np.float32),
                    "grasps_obb": np.zeros((0, 4, 4), dtype=np.float32),
                    "scores_obb": np.zeros((0,), dtype=np.float32),
                    "pc_removed": pc_removed,
                    "obb": None,
                    "skipped_obb": True,
                }
            )
            continue

        grasps_diff_t, scores_diff_t = diff_results[i]
        if len(grasps_diff_t) > 0:
            grasps_diff = grasps_diff_t.cpu().numpy().astype(np.float32)
            scores_diff = scores_diff_t.cpu().numpy().astype(np.float32)
            grasps_diff[:, 3, 3] = 1.0
        else:
            grasps_diff = np.zeros((0, 4, 4), dtype=np.float32)
            scores_diff = np.zeros((0,), dtype=np.float32)

        pc_center = pc_filtered.mean(axis=0).astype(np.float64)
        pc_filtered_centered_t = (
            torch.from_numpy(pc_filtered)
            - torch.from_numpy(pc_center.astype(np.float32))
        ).to(device)
        grasps_obb, scores_obb, obb_dict, skipped = _run_obb_branch(
            pc_filtered=pc_filtered,
            pc_filtered_centered=pc_filtered_centered_t,
            pc_center=pc_center,
            grasp_sampler=grasp_sampler,
            num_yaws=num_yaws,
            z_offsets_cm=tuple(z_offsets_cm),
            obb_mode=obb_mode,
            gripper_width_m=gripper_width_m,
            gripper_depth_m=gripper_depth_m,
            skip_obb_rule=skip_obb_rule,
            obb_density=obb_density,
            obb_position_spacing_m=obb_position_spacing_m,
            object_embedding=(
                obj_embeddings[i]
                if (_REUSE_OBJ_EMBED and obj_embeddings is not None)
                else None
            ),
        )

        if grasp_threshold > 0.0:
            if len(scores_diff) > 0:
                keep = scores_diff >= float(grasp_threshold)
                grasps_diff = grasps_diff[keep]
                scores_diff = scores_diff[keep]
            if len(scores_obb) > 0:
                keep = scores_obb >= float(grasp_threshold)
                grasps_obb = grasps_obb[keep]
                scores_obb = scores_obb[keep]

        if topk_num_grasps is not None and topk_num_grasps > 0:
            all_scores = np.concatenate([scores_diff, scores_obb])
            if len(all_scores) > topk_num_grasps:
                kth = np.partition(all_scores, -topk_num_grasps)[-topk_num_grasps]
                keep_diff = scores_diff >= kth
                keep_obb = scores_obb >= kth
                grasps_diff = grasps_diff[keep_diff]
                scores_diff = scores_diff[keep_diff]
                grasps_obb = grasps_obb[keep_obb]
                scores_obb = scores_obb[keep_obb]

        outputs.append(
            {
                "grasps_diff": grasps_diff,
                "scores_diff": scores_diff,
                "grasps_obb": grasps_obb,
                "scores_obb": scores_obb,
                "pc_removed": pc_removed,
                "obb": obb_dict,
                "skipped_obb": skipped,
            }
        )

    logger.info(
        f"[graspmoe_batch] {n} objects: "
        + ", ".join(
            f"obj{i}={len(o['grasps_diff']) + len(o['grasps_obb'])}"
            for i, o in enumerate(outputs)
        )
    )
    return outputs
