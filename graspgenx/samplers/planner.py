# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Planner dispatch: chooses between plain diffusion+discriminator and the
# GraspMoE (diffusion union OBB-swept top-down) planner. Shared between the
# object-PC and scene-PC demo scripts.

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.samplers.graspmoe import run_graspmoe, run_graspmoe_batch
from graspgenx.utils.point_cloud import point_cloud_outlier_removal


def run_planner_on_object(
    obj_pc: np.ndarray,
    grasp_sampler: GraspGenXSampler,
    *,
    planner: str = "graspmoe",
    grasp_threshold: float = -1.0,
    num_grasps: int | None = None,
    topk_num_grasps: int = -1,
    moe_num_yaws: int | None = None,
    moe_z_offsets_cm: Sequence[float] = (-8, -6, -4, -2, 0),
    moe_outlier_threshold: float = 0.014,
    moe_outlier_k: int = 20,
    moe_obb_mode: str = "pca",
    moe_skip_obb_rule: str = "never",
    moe_obb_density: str = "sparse",
    moe_obb_position_spacing_cm: float = 1.0,
):
    """Run the configured planner on a single object PC (world frame).

    Returns:
        (grasps_world, grasp_conf, branch_tags, obb_dict_or_None)

        - grasps_world: (K, 4, 4) float32
        - grasp_conf:   (K,)     float32  discriminator confidences in [0, 1]
        - branch_tags:  list of "diff" | "obb" per grasp
        - obb_dict:     {"center", "half_extent", "R"} or None
    """
    if planner == "graspmoe":
        moe = run_graspmoe(
            obj_pc,
            grasp_sampler,
            grasp_threshold=grasp_threshold,
            num_grasps=num_grasps,
            topk_num_grasps=topk_num_grasps,
            num_yaws=moe_num_yaws,
            z_offsets_cm=tuple(moe_z_offsets_cm),
            outlier_threshold=moe_outlier_threshold,
            outlier_k=moe_outlier_k,
            obb_mode=moe_obb_mode,
            skip_obb_rule=moe_skip_obb_rule,
            obb_density=moe_obb_density,
            obb_position_spacing_m=float(moe_obb_position_spacing_cm) / 100.0,
        )
        grasps = np.concatenate([moe["grasps_diff"], moe["grasps_obb"]], axis=0)
        conf = np.concatenate([moe["scores_diff"], moe["scores_obb"]], axis=0)
        tags = ["diff"] * len(moe["grasps_diff"]) + ["obb"] * len(moe["grasps_obb"])
        return grasps, conf, tags, moe["obb"]

    # diffusion baseline
    obj_pc_t = torch.from_numpy(obj_pc.astype(np.float32))
    obj_pc_filtered_t, _ = point_cloud_outlier_removal(obj_pc_t)
    if len(obj_pc_filtered_t) < 10:
        # Outlier removal nuked (nearly) everything — fall back to the raw PC.
        obj_pc_filtered_t = obj_pc_t
    obj_pc_filtered = obj_pc_filtered_t.cpu().numpy()
    grasps_t, conf_t = GraspGenXSampler.run_inference(
        obj_pc_filtered,
        grasp_sampler,
        grasp_threshold=grasp_threshold,
        num_grasps=num_grasps,
        topk_num_grasps=topk_num_grasps,
    )
    if len(grasps_t) == 0:
        return (
            np.zeros((0, 4, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            [],
            None,
        )
    grasps = grasps_t.cpu().numpy()
    conf = conf_t.cpu().numpy()
    grasps[:, 3, 3] = 1
    return grasps, conf, ["diff"] * len(grasps), None


def run_planner_on_batch(
    object_pcs: list,
    grasp_sampler: GraspGenXSampler,
    *,
    planner: str = "graspmoe",
    grasp_threshold: float = -1.0,
    num_grasps: int | None = None,
    topk_num_grasps: int = -1,
    moe_num_yaws: int | None = None,
    moe_z_offsets_cm: Sequence[float] = (-8, -6, -4, -2, 0),
    moe_outlier_threshold: float = 0.014,
    moe_outlier_k: int = 20,
    moe_obb_mode: str = "pca",
    moe_skip_obb_rule: str = "never",
    moe_obb_density: str = "sparse",
    moe_obb_position_spacing_cm: float = 1.0,
    object_batch_size: int | None = None,
) -> list:
    """Batched form of :func:`run_planner_on_object`. Returns one
    ``(grasps_world, grasp_conf, branch_tags, obb_dict_or_None)`` tuple per
    input PC, in input order.

    Eager PyTorch may fold the GraspMoE diffusion branch into a batched call;
    the optimized runtime loops over objects because each object has its own
    fixed-shape CUDA Graph.

    The optimized diffusion runtime always uses one-object calls in a loop.
    Eager PyTorch can still batch multiple objects through
    :meth:`GraspGenXSampler.run_inference_batch`.
    """
    n = len(object_pcs)
    if n == 0:
        return []

    if planner == "graspmoe":
        moe_results = run_graspmoe_batch(
            object_pcs,
            grasp_sampler,
            grasp_threshold=grasp_threshold,
            num_grasps=num_grasps,
            topk_num_grasps=topk_num_grasps,
            num_yaws=moe_num_yaws,
            z_offsets_cm=tuple(moe_z_offsets_cm),
            outlier_threshold=moe_outlier_threshold,
            outlier_k=moe_outlier_k,
            obb_mode=moe_obb_mode,
            skip_obb_rule=moe_skip_obb_rule,
            obb_density=moe_obb_density,
            obb_position_spacing_m=float(moe_obb_position_spacing_cm) / 100.0,
        )
        results: list = []
        for moe in moe_results:
            grasps = np.concatenate([moe["grasps_diff"], moe["grasps_obb"]], axis=0)
            conf = np.concatenate([moe["scores_diff"], moe["scores_obb"]], axis=0)
            tags = ["diff"] * len(moe["grasps_diff"]) + ["obb"] * len(moe["grasps_obb"])
            results.append((grasps, conf, tags, moe["obb"]))
        return results

    # Diffusion-only path: move each variable-length cloud to CUDA and apply
    # the exact fused filter/resampler there. The C++ runtime uses chunks of one;
    # eager PyTorch may use a larger object batch.
    if grasp_sampler.using_fast_inference:
        if object_batch_size not in (None, 1):
            raise ValueError(
                "The native C++ runtime supports object_batch_size=1 "
                "only; loop over objects for larger batches."
            )
        object_batch_size = 1
    elif object_batch_size is None or object_batch_size == 0:
        object_batch_size = n
    if object_batch_size < 1:
        raise ValueError("object_batch_size must be positive or zero for all")
    diff_results = []
    for start in range(0, n, object_batch_size):
        diff_results.extend(
            GraspGenXSampler.run_inference_batch(
                object_pcs[start : start + object_batch_size],
                grasp_sampler,
                grasp_threshold=grasp_threshold,
                num_grasps=num_grasps,
                topk_num_grasps=topk_num_grasps,
                remove_outliers=True,
            )
        )
    results = []
    for grasps_t, conf_t in diff_results:
        if len(grasps_t) == 0:
            results.append(
                (
                    np.zeros((0, 4, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    [],
                    None,
                )
            )
            continue
        grasps = grasps_t.cpu().numpy()
        conf = conf_t.cpu().numpy()
        grasps[:, 3, 3] = 1
        results.append((grasps, conf, ["diff"] * len(grasps), None))
    return results
