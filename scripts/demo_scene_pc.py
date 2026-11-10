# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import time
from pathlib import Path

import numpy as np
import trimesh
from demo_object_pc import _resolve_default_checkpoints

from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.samplers import run_planner_on_batch
from graspgenx.utils.checkpoint_io import load_model_cfg
from graspgenx.utils.collision_filter import filter_colliding_grasps
from graspgenx.utils.scene_loaders import (
    build_scene_pc_excluding_object,
    collect_scene_items,
    load_graspgenx_json_scene,
    load_realworld_scene,
)
from graspgenx.utils.viser_utils import (
    create_visualizer,
    get_color_from_score,
    visualize_bbox,
    visualize_mesh,
    visualize_pointcloud,
    visualize_x_grasp,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GraspGenX inference on scene point clouds with per-object segmentation. "
        "Supports two sample_data_dir formats: (1) GraspGenX JSON scenes, "
        "(2) M2T2 real_world/<NN>/{depth.npy,rgb.png,seg.png,meta_data.json}."
    )
    parser.add_argument("--sample_data_dir", type=str, required=True)
    parser.add_argument("--gen_pth", type=str, default=None)
    parser.add_argument("--dis_pth", type=str, default=None)
    parser.add_argument("--gripper_name", type=str, required=True)
    parser.add_argument("--assets_dir", type=str, default=None)
    parser.add_argument("--grasp_threshold", type=float, default=0.7)
    parser.add_argument("--num_grasps", type=int, default=200)
    parser.add_argument("--return_topk", action="store_true")
    parser.add_argument("--topk_num_grasps", type=int, default=-1)
    parser.add_argument("--plot_mesh", action="store_true")
    parser.add_argument(
        "--plot_top_mesh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render the top-ranked grasp's collision mesh. Disable with --no-plot_top_mesh.",
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="graspmoe",
        choices=["diffusion", "graspmoe"],
    )
    parser.add_argument("--moe_num_yaws", type=int, default=None)
    parser.add_argument("--moe_z_offsets_cm", type=str, default="-8,-6,-4,-2,0")
    parser.add_argument("--moe_outlier_threshold", type=float, default=0.014)
    parser.add_argument("--moe_outlier_k", type=int, default=20)
    parser.add_argument(
        "--moe_obb_mode", type=str, default="pca", choices=["advanced", "pca"]
    )
    parser.add_argument(
        "--moe_skip_obb_rule", type=str, default="never", choices=["auto", "never"]
    )
    parser.add_argument(
        "--moe_obb_density",
        type=str,
        default="sparse",
        choices=["sparse", "dense", "dense-topandside"],
        help="OBB candidate placement. 'sparse' (default): single (x,y) at the OBB "
        "centroid, top-down only. 'dense': sweep positions along the OBB's longer "
        "XY axis, top-down only. 'dense-topandside': dense sweep on the top face "
        "and on each of the 4 horizontal side faces (gripper approaches "
        "horizontally on the sides).",
    )
    parser.add_argument(
        "--moe_obb_position_spacing_cm",
        type=float,
        default=1.0,
        help="[dense mode] Spacing (cm) of candidate positions along the OBB's long axis.",
    )
    parser.add_argument("--moe_no_obb_wireframe", action="store_true")
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Real-world format only: restrict to a single scene <NN> (e.g. '00').",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Path to a checkpoint root containing 'gen/' and 'dis/' subdirectories "
        "(each with config.yaml and .pth files). If omitted, defaults to the "
        "current release under $GRASPGENX_CHECKPOINT_DIR or the auto-cloned "
        "<repo>/ext/graspgenx_checkpoints/<version>/.",
    )
    parser.add_argument(
        "--min_obj_points",
        type=int,
        default=100,
        help="Real-world format only: skip objects whose segmented PC has fewer points.",
    )
    parser.add_argument(
        "--fast-inference",
        action="store_true",
        help="Use the one-object native C++ fast-inference runtime "
        "(requires the 'fast-inference' extra).",
    )
    parser.add_argument(
        "--filter_collisions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop grasps whose gripper mesh would intersect the scene PC "
        "(target object's own pixels are excluded). "
        "Disable with --no-filter_collisions.",
    )
    parser.add_argument(
        "--collision_threshold",
        type=float,
        default=0.02,
        help="Distance (meters) under which a gripper surface sample counts as "
        "colliding with the scene PC.",
    )
    parser.add_argument(
        "--max_scene_points",
        type=int,
        default=8192,
        help="Random-downsample the scene PC to at most this many points before "
        "running collision check.",
    )
    parser.add_argument(
        "--num_collision_samples",
        type=int,
        default=2000,
        help="Surface samples drawn from the gripper collision mesh per check.",
    )
    return parser.parse_args()


def visualize_grasps_for_object(
    vis,
    label: str,
    grasps: np.ndarray,
    grasp_conf: np.ndarray,
    branch_tags: list,
    obb_dict,
    gripper,
    args,
):
    """Render all grasps (and optionally OBB wireframe + top-mesh) under
    obj/<label>/... in world frame."""
    if len(grasps) == 0:
        return

    scores = get_color_from_score(grasp_conf, use_255_scale=True)
    ns_prefix = f"obj/{label}"

    if (
        args.planner == "graspmoe"
        and obb_dict is not None
        and not args.moe_no_obb_wireframe
    ):
        T = np.eye(4)
        T[:3, :3] = obb_dict["R"]
        T[:3, 3] = obb_dict["center"]
        visualize_bbox(
            vis,
            f"{ns_prefix}/moe_obb",
            2.0 * obb_dict["half_extent"],
            T=T,
            color=[255, 130, 0],
        )

    best_idx = int(grasp_conf.argmax())
    diff_count = 0
    obb_count = 0
    for j, grasp in enumerate(grasps):
        tag = branch_tags[j] if j < len(branch_tags) else "diff"
        if tag == "obb":
            ns = f"{ns_prefix}/pred_grasps/obb_gen/grasp_{obb_count:03d}"
            mesh_ns = f"{ns_prefix}/pred_meshes/obb_gen/grasp_{j:03d}"
            obb_count += 1
        else:
            ns = f"{ns_prefix}/pred_grasps/diff_gen/grasp_{diff_count:03d}"
            mesh_ns = f"{ns_prefix}/pred_meshes/diff_gen/grasp_{j:03d}"
            diff_count += 1
        color = [0, 100, 255] if j == best_idx else scores[j]
        lw = 5.0 if j == best_idx else 1.5
        visualize_x_grasp(
            vis,
            ns,
            grasp,
            color=color,
            gripper_info=gripper,
            linewidth=lw,
        )
        if j < 5 and args.plot_mesh:
            visualize_mesh(
                vis,
                mesh_ns,
                gripper.visual_mesh,
                color=scores[j],
                transform=grasp,
            )

    if args.plot_top_mesh:
        visualize_mesh(
            vis,
            f"{ns_prefix}/top_grasp_mesh",
            gripper.visual_mesh,
            color=[0, 100, 255],
            transform=grasps[best_idx],
        )


def main():
    args = parse_args()

    if not os.path.exists(args.sample_data_dir):
        raise FileNotFoundError(
            f"sample_data_dir {args.sample_data_dir} does not exist"
        )

    if args.return_topk and args.topk_num_grasps == -1:
        args.topk_num_grasps = 100

    repo_root = Path(__file__).resolve().parent.parent
    if args.assets_dir is None:
        args.assets_dir = str(repo_root / "assets")

    checkpoint_root = args.checkpoints or _resolve_default_checkpoints()
    print(f"Using checkpoints under: {checkpoint_root}")
    model_cfg = load_model_cfg(
        os.path.join(checkpoint_root, "gen"),
        os.path.join(checkpoint_root, "dis"),
        args.gen_pth,
        args.dis_pth,
    )
    items = collect_scene_items(args.sample_data_dir, scene_filter=args.scene)
    print(f"[scene_pc] {len(items)} scene(s) to process")

    print(f"Loading gripper: {args.gripper_name}")
    print(f"Assets directory: {args.assets_dir}")
    z_offsets_cm = tuple(float(x) for x in args.moe_z_offsets_cm.split(","))
    grasp_sampler = GraspGenXSampler(
        model_cfg,
        args.gripper_name,
        assets_dir=args.assets_dir,
        use_fast_inference=args.fast_inference,
        diffusion_grasps=args.num_grasps,
        obb_grasps=(args.moe_num_yaws or 10) * len(z_offsets_cm),
    )
    if args.fast_inference and not grasp_sampler.using_fast_inference:
        raise RuntimeError(
            "--fast-inference was requested, but the native optimized backend could "
            "not be enabled; see the preceding warning for the root cause."
        )
    backend = (
        "native C++ fast runtime" if grasp_sampler.using_fast_inference else "PyTorch"
    )
    print(f"Inference backend: {backend}")
    gripper = grasp_sampler.get_gripper_info()

    collision_mesh_available = (
        gripper.collision_mesh is not None and len(gripper.collision_mesh.faces) > 0
    )
    visual_mesh_available = (
        gripper.visual_mesh is not None and len(gripper.visual_mesh.faces) > 0
    )
    if not collision_mesh_available and args.filter_collisions:
        print("Gripper collision mesh is empty; disabling collision filtering.")
        args.filter_collisions = False
    if not visual_mesh_available and (args.plot_mesh or args.plot_top_mesh):
        print(
            "Gripper visual mesh is empty; disabling mesh overlays. Grasp "
            "frames will still be visualized."
        )
        args.plot_mesh = False
        args.plot_top_mesh = False

    vis = create_visualizer()

    # Sample the gripper collision mesh once and reuse for every collision
    # check (same mesh, same random sampling reasonable across the run).
    if args.filter_collisions:
        sampled_pts, _ = trimesh.sample.sample_surface(
            gripper.collision_mesh, args.num_collision_samples
        )
        gripper_surface_points = np.asarray(sampled_pts, dtype=np.float32)
    else:
        gripper_surface_points = None

    for fmt_tag, path in items:
        print(f"\n========== Processing {os.path.basename(path)} ==========")
        vis.scene.reset()

        if fmt_tag == "realworld":
            scene = load_realworld_scene(path, min_obj_points=args.min_obj_points)
        else:
            scene = load_graspgenx_json_scene(path)

        # Render the full scene point cloud (world frame).
        VIZ_BOUNDS = [[-1.5, -1.25, -0.15], [1.5, 1.25, 2.0]]
        xyz_scene = scene["scene_xyz"]
        rgb_scene = scene["scene_rgb"]
        m = np.all((xyz_scene > VIZ_BOUNDS[0]) & (xyz_scene < VIZ_BOUNDS[1]), axis=1)
        xyz_scene, rgb_scene = xyz_scene[m], rgb_scene[m]
        visualize_pointcloud(vis, "pc_scene", xyz_scene, rgb_scene, size=0.0025)

        if not scene["objects"]:
            print("  No segmented objects found; skipping.")
            input("Press Enter to continue to next scene...")
            continue

        # Render all object point clouds up front so the user sees the
        # segmentation while the (batched) planner runs.
        labels = list(scene["objects"].keys())
        obj_pcs = [scene["objects"][lab]["pc"] for lab in labels]
        for lab, obj_pc in zip(labels, obj_pcs):
            print(f"\n  --- {lab} ({len(obj_pc)} pts) ---")
            visualize_pointcloud(
                vis,
                f"obj/{lab}/pc",
                obj_pc,
                scene["objects"][lab]["rgb"],
                size=0.004,
            )

        planner_kwargs = {
            "planner": args.planner,
            "grasp_threshold": args.grasp_threshold,
            "num_grasps": args.num_grasps,
            "topk_num_grasps": args.topk_num_grasps,
            "moe_num_yaws": args.moe_num_yaws,
            "moe_z_offsets_cm": z_offsets_cm,
            "moe_outlier_threshold": args.moe_outlier_threshold,
            "moe_outlier_k": args.moe_outlier_k,
            "moe_obb_mode": args.moe_obb_mode,
            "moe_skip_obb_rule": args.moe_skip_obb_rule,
            "moe_obb_density": args.moe_obb_density,
            "moe_obb_position_spacing_cm": args.moe_obb_position_spacing_cm,
            "object_batch_size": (
                1
                if grasp_sampler.using_fast_inference and args.planner == "diffusion"
                else None
            ),
        }
        if grasp_sampler.using_fast_inference:
            # Engine deserialization and CUDA-Graph capture are process startup
            # costs. Warm the one-object plan once so the displayed
            # number is steady-state inference, matching the benchmark contract.
            warmup_count = min(1, len(obj_pcs))
            print("  Warming native one-object runtime...")
            run_planner_on_batch(
                obj_pcs[:warmup_count], grasp_sampler, **planner_kwargs
            )

        inference_start = time.perf_counter()
        batch_results = run_planner_on_batch(
            obj_pcs,
            grasp_sampler,
            **planner_kwargs,
        )
        inference_seconds = time.perf_counter() - inference_start
        print(
            f"  Steady-state optimized {args.planner} inference: "
            f"{inference_seconds * 1e3:.2f} ms total, "
            f"{inference_seconds * 1e3 / len(labels):.2f} ms/object amortized "
            f"({len(labels)} objects x {args.num_grasps} grasps)"
        )

        for label, (grasps, conf, tags, obb_dict) in zip(labels, batch_results):
            if len(grasps) == 0:
                print(f"  [{label}] no grasps; skipping viz")
                continue

            if args.filter_collisions:
                scene_pc_no_target = build_scene_pc_excluding_object(scene, label)
                if len(scene_pc_no_target) > args.max_scene_points:
                    idx = np.random.choice(
                        len(scene_pc_no_target),
                        args.max_scene_points,
                        replace=False,
                    )
                    scene_pc_no_target = scene_pc_no_target[idx]
                cf_mask = filter_colliding_grasps(
                    scene_pc=scene_pc_no_target,
                    grasp_poses=grasps,
                    collision_threshold=args.collision_threshold,
                    gripper_surface_points=gripper_surface_points,
                )
                n_before = len(grasps)
                grasps = grasps[cf_mask]
                conf = conf[cf_mask]
                tags = [t for t, keep in zip(tags, cf_mask) if keep]
                print(
                    f"  [{label}] collision filter: {len(grasps)} free / "
                    f"{n_before - len(grasps)} colliding "
                    f"(thr={args.collision_threshold:.3f}m, "
                    f"scene_pts={len(scene_pc_no_target)})"
                )
                if len(grasps) == 0:
                    print(f"  [{label}] all grasps colliding; nothing to viz")
                    continue

            print(
                f"  [{label}] {len(grasps)} grasps "
                f"(diff={tags.count('diff')}, obb={tags.count('obb')}); "
                f"score range {conf.min():.3f}..{conf.max():.3f}"
            )
            visualize_grasps_for_object(
                vis, label, grasps, conf, tags, obb_dict, gripper, args
            )

        input("\nPress Enter to continue to next scene...")


if __name__ == "__main__":
    main()
