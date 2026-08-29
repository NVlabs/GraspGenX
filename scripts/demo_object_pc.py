# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import select
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh.transformations as tra
import yourdfpy

from graspgenx import get_checkpoints_version_dir
from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.samplers import run_planner_on_object
from graspgenx.utils.checkpoint_io import load_model_cfg
from graspgenx.utils.scene_loaders import collect_object_items
from graspgenx.utils.viser_utils import (
    create_visualizer,
    get_color_from_score,
    visualize_bbox,
    visualize_x_grasp,
    visualize_mesh,
    visualize_pointcloud,
)
from graspgenx.utils.point_cloud import point_cloud_outlier_removal


def _resolve_default_checkpoints() -> str:
    """Resolve the default --checkpoints location at runtime.

    Returns the version subdir under $GRASPGENX_CHECKPOINT_DIR or the
    auto-cloned <repo>/ext/graspgenx_checkpoints/.
    """
    resolved = get_checkpoints_version_dir()
    if (resolved / "gen" / "config.yaml").is_file():
        return str(resolved)
    # Some local clones retain the Git repository's top-level
    # ``GraspGenXModel`` directory under ext/graspgenx_checkpoints.
    nested = resolved.parent / "GraspGenXModel" / resolved.name
    if (nested / "gen" / "config.yaml").is_file():
        return str(nested)
    return str(resolved)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GraspGenX inference on segmented object point clouds"
    )
    parser.add_argument(
        "--sample_data_dir",
        type=str,
        required=True,
        help="Directory containing JSON files with point cloud data",
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
        "--gen_pth",
        type=str,
        default=None,
        help="Generator .pth filename (default: auto-detect latest epoch_*.pth)",
    )
    parser.add_argument(
        "--dis_pth",
        type=str,
        default=None,
        help="Discriminator .pth filename (default: auto-detect latest epoch_*.pth)",
    )
    parser.add_argument(
        "--gripper_name",
        type=str,
        nargs="+",
        required=True,
        help="One or more gripper names (from assets/x_grippers/ or assets/proc_grippers/)",
    )
    parser.add_argument(
        "--assets_dir",
        type=str,
        default=None,
        help="Path to the assets directory containing x_grippers/ and proc_grippers/ "
        "(default: <repo_root>/assets)",
    )
    parser.add_argument(
        "--grasp_threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for grasps. Use -1.0 to return top-k instead.",
    )
    parser.add_argument(
        "--num_grasps",
        type=int,
        default=200,
        help="Number of grasps to generate per inference pass",
    )
    parser.add_argument(
        "--return_topk",
        action="store_true",
        help="Return only the top-k grasps ranked by confidence",
    )
    parser.add_argument(
        "--topk_num_grasps",
        type=int,
        default=-1,
        help="Number of top grasps to return (default: 100 when --return_topk is set)",
    )
    parser.add_argument(
        "--plot_mesh",
        action="store_true",
        help="Also render the gripper collision mesh for the top 5 grasps",
    )
    parser.add_argument(
        "--plot_top_mesh",
        action="store_true",
        help="Render the gripper collision mesh at the single top-ranked grasp",
    )
    parser.add_argument(
        "--vis-top-grasp-meshes",
        action="store_true",
        help="Render gripper collision meshes for the top-N grasps (by confidence) "
        "from the set filtered by --grasp_threshold",
    )
    parser.add_argument(
        "--num-top-grasp-meshes",
        type=int,
        default=5,
        help="Number of top grasps to render meshes for when --vis-top-grasp-meshes is set (default: 5)",
    )
    parser.add_argument(
        "--interactive_threshold_tuner",
        action="store_true",
        help="Add a viser GUI slider to interactively threshold predicted grasps by confidence",
    )
    parser.add_argument(
        "--plot_gripper_sweep_volume_animation",
        action="store_true",
        help="Show an animated gripper open/close cycle to the right of the scene, "
        "with sweep-volume bbox overlay",
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="graspmoe",
        choices=["diffusion", "graspmoe"],
        help="Grasp planner: 'graspmoe' (default) = diffusion union OBB-swept top-down "
        "candidates, all scored by the discriminator. 'diffusion' = GraspGenX "
        "diffusion+discriminator only.",
    )
    parser.add_argument(
        "--moe_num_yaws",
        type=int,
        default=None,
        help=(
            "[graspmoe] World-Z yaw samples; defaults to the configured CUDA "
            "candidate count or 10 yaws in eager mode."
        ),
    )
    parser.add_argument(
        "--moe_z_offsets_cm",
        type=str,
        default="-8,-6,-4,-2,0",
        help="[graspmoe] Comma-separated Z offsets (cm) relative to OBB top.",
    )
    parser.add_argument(
        "--moe_outlier_threshold",
        type=float,
        default=0.014,
        help="[graspmoe] Outlier removal distance threshold.",
    )
    parser.add_argument(
        "--moe_outlier_k",
        type=int,
        default=20,
        help="[graspmoe] Outlier removal k-NN.",
    )
    parser.add_argument(
        "--moe_obb_mode",
        type=str,
        default="pca",
        choices=["advanced", "pca"],
        help="[graspmoe] OBB algorithm: 'advanced' (SOR + hull + rotating calipers) "
        "or 'pca' fallback.",
    )
    parser.add_argument(
        "--moe_skip_obb_rule",
        type=str,
        default="never",
        choices=["auto", "never"],
        help="[graspmoe] Skip OBB branch when every OBB extent > gripper width ('auto'), "
        "or always run it ('never').",
    )
    parser.add_argument(
        "--moe_obb_density",
        type=str,
        default="sparse",
        choices=["sparse", "dense", "dense-topandside"],
        help="[graspmoe] OBB candidate placement. 'sparse' puts every candidate at "
        "the OBB centroid (top-down, yaw + Z sweep only). 'dense' sweeps "
        "positions along the OBB's longer XY axis (top-down). "
        "'dense-topandside' also adds horizontal-approach grasps on the 4 "
        "side faces of the OBB.",
    )
    parser.add_argument(
        "--moe_obb_position_spacing_cm",
        type=float,
        default=1.0,
        help="[graspmoe, dense] Spacing (cm) of candidate positions along the OBB's long axis.",
    )
    parser.add_argument(
        "--moe_no_obb_wireframe",
        action="store_true",
        help="[graspmoe] Disable rendering of the OBB wireframe.",
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
        help="Use the native C++ fast-inference runtime through its C ABI. "
        "Install it with `graspgenx-install-fast-inference`; custom locations "
        "can use GRASPGENX_CPP_RUNTIME_LIBRARY and "
        "GRASPGENX_CPP_RUNTIME_ARTIFACTS. Falls back to eager PyTorch if "
        "unavailable.",
    )
    return parser.parse_args()


def process_point_cloud(pc, grasps, grasp_conf):
    """Center a point cloud and its associated grasps around the PC centroid."""
    scores = get_color_from_score(grasp_conf, use_255_scale=True)
    print(
        f"Stored-grasp scores: min {grasp_conf.min():.3f}, max {grasp_conf.max():.3f}"
    )

    grasps[:, 3, 3] = 1
    T_subtract_pc_mean = tra.translation_matrix(-pc.mean(axis=0))
    pc_centered = tra.transform_points(pc, T_subtract_pc_mean)
    grasps_centered = np.array(
        [T_subtract_pc_mean @ np.array(g) for g in grasps.tolist()]
    )

    return pc_centered, grasps_centered, scores


def load_gripper_urdf(
    asset_path: str, gripper_name: str
) -> Tuple[Optional[yourdfpy.URDF], Optional[Dict]]:
    """Load gripper URDF and config for animation."""
    urdf_path = os.path.join(asset_path, gripper_name, "gripper.urdf")
    config_path = os.path.join(asset_path, gripper_name, "config.json")

    if not os.path.exists(urdf_path) or not os.path.exists(config_path):
        print(f"Warning: URDF or config not found for {gripper_name}")
        return None, None

    try:
        robot = yourdfpy.URDF.load(
            urdf_path,
            build_scene_graph=True,
            load_meshes=True,
            build_collision_scene_graph=False,
            load_collision_meshes=False,
        )
        with open(config_path, "r") as f:
            config = json.load(f)
        return robot, config
    except Exception as e:
        print(f"Warning: Could not load gripper URDF: {e}")
        return None, None


def get_traj_js(gripper_config: Dict, num_steps: int = 10) -> List[Dict]:
    """Generate trajectory of joint states from open to close."""
    gripper_open = gripper_config["open"]
    gripper_close = gripper_config["close"]

    trajs = []
    for s in range(num_steps + 1):
        js = dict()
        for k in gripper_open.keys():
            open_js = gripper_open[k]
            close_js = gripper_close[k]
            js[k] = open_js + (close_js - open_js) * s / num_steps
        trajs.append(js)

    return trajs


def visualize_gripper_at_pose(
    vis,
    robot: yourdfpy.URDF,
    js_cfg: Dict,
    grasp_pose: np.ndarray,
    name_prefix: str = "gripper",
    color: List[int] = [80, 80, 80],
):
    """Visualize gripper URDF meshes at a grasp pose."""
    robot.update_cfg(js_cfg)
    scene = robot.scene
    geometry_names = list(scene.geometry.keys())

    for i, geom_name in enumerate(geometry_names):
        mesh = scene.geometry[geom_name]
        local_transform = scene.graph.get(geom_name)[0]
        world_transform = grasp_pose @ local_transform

        transformed_mesh = mesh.copy()
        transformed_mesh.apply_transform(world_transform)

        visualize_mesh(
            vis,
            f"{name_prefix}/link_{i}",
            transformed_mesh,
            color=color,
        )


class AnimationThread:
    """Background thread for gripper open/close animation."""

    def __init__(
        self, vis, robot, gripper_config, pose, animation_steps=15, animation_delay=0.08
    ):
        self.vis = vis
        self.robot = robot
        self.traj = get_traj_js(gripper_config, num_steps=animation_steps)
        self.pose = pose
        self.animation_delay = animation_delay
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._animate_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _animate_loop(self):
        color = [255, 100, 50]  # Orange for visibility
        frame_count = 0
        print("  [Animation] Loop started")
        while self.running:
            for js in self.traj:
                if not self.running:
                    break
                try:
                    visualize_gripper_at_pose(
                        self.vis,
                        self.robot,
                        js,
                        self.pose,
                        name_prefix="animated_gripper",
                        color=color,
                    )
                    frame_count += 1
                    if frame_count == 1:
                        print(f"  [Animation] First frame rendered")
                except Exception as e:
                    import traceback

                    print(f"  [Animation] Error: {e}")
                    traceback.print_exc()
                    return
                time.sleep(self.animation_delay)
            for js in reversed(self.traj):
                if not self.running:
                    break
                try:
                    visualize_gripper_at_pose(
                        self.vis,
                        self.robot,
                        js,
                        self.pose,
                        name_prefix="animated_gripper",
                        color=color,
                    )
                except Exception as e:
                    print(f"  [Animation] Error: {e}")
                    return
                time.sleep(self.animation_delay)


if __name__ == "__main__":
    args = parse_args()

    if args.return_topk and args.topk_num_grasps == -1:
        args.topk_num_grasps = 100

    repo_root = Path(__file__).resolve().parent.parent
    if args.assets_dir is None:
        args.assets_dir = str(repo_root / "assets")

    checkpoint_root = args.checkpoints or str(_resolve_default_checkpoints())
    print(f"Using checkpoints under: {checkpoint_root}")
    model_cfg = load_model_cfg(
        os.path.join(checkpoint_root, "gen"),
        os.path.join(checkpoint_root, "dis"),
        args.gen_pth,
        args.dis_pth,
    )

    items = collect_object_items(
        args.sample_data_dir, min_obj_points=args.min_obj_points
    )
    if not items:
        raise FileNotFoundError(
            f"No object samples found in {args.sample_data_dir}. "
            f"Expected either *.json files with a top-level 'pc' field, "
            f"or <NN>/meta_data.json real-world scene dirs."
        )
    print(f"Found {len(items)} object sample(s)")
    print(f"Grippers to run: {args.gripper_name}")

    z_offsets_cm = tuple(float(x) for x in args.moe_z_offsets_cm.split(","))

    vis = create_visualizer()

    for gripper_name in args.gripper_name:
        print(f"\n{'=' * 60}")
        print(f"Loading gripper: {gripper_name}")
        print(f"Assets directory: {args.assets_dir}")
        grasp_sampler = GraspGenXSampler(
            model_cfg,
            gripper_name,
            assets_dir=args.assets_dir,
            use_fast_inference=args.fast_inference,
            diffusion_grasps=args.num_grasps,
            obb_grasps=(args.moe_num_yaws or 10) * len(z_offsets_cm),
        )
        gripper = grasp_sampler.get_gripper_info()

        robot = None
        gripper_config = None
        animate = args.plot_gripper_sweep_volume_animation
        if animate:
            for subdir in ("x_grippers", "proc_grippers"):
                gripper_asset_dir = os.path.join(args.assets_dir, subdir)
                urdf_candidate = os.path.join(
                    gripper_asset_dir, gripper_name, "gripper.urdf"
                )
                if os.path.exists(urdf_candidate):
                    print(
                        f"Loading gripper URDF for animation from {gripper_asset_dir}..."
                    )
                    robot, gripper_config = load_gripper_urdf(
                        gripper_asset_dir, gripper_name
                    )
                    break
            if robot is None:
                print(
                    "Warning: Could not load gripper URDF, animation will be disabled"
                )
                animate = False

        for item in items:
            print(f"\nProcessing {item['name']}")
            animation_thread = None
            vis.scene.reset()

            pc = item["pc"]
            pc_color = item["pc_color"]

            if (
                item["stored_grasps"] is not None
                and item["stored_grasp_conf"] is not None
            ):
                grasps = item["stored_grasps"]
                grasp_conf = item["stored_grasp_conf"]
                pc_centered, grasps_centered, scores = process_point_cloud(
                    pc, grasps, grasp_conf
                )
            else:
                T_subtract_pc_mean = tra.translation_matrix(-pc.mean(axis=0))
                pc_centered = tra.transform_points(pc, T_subtract_pc_mean)

            print(pc_centered.max())
            visualize_pointcloud(vis, "pc", pc_centered, pc_color, size=0.002)

            grasps_inferred, grasp_conf_inferred, branch_tags, moe_obb_dict = (
                run_planner_on_object(
                    pc_centered,
                    grasp_sampler,
                    planner=args.planner,
                    grasp_threshold=args.grasp_threshold,
                    num_grasps=args.num_grasps,
                    topk_num_grasps=args.topk_num_grasps,
                    moe_num_yaws=args.moe_num_yaws,
                    moe_z_offsets_cm=z_offsets_cm,
                    moe_outlier_threshold=args.moe_outlier_threshold,
                    moe_outlier_k=args.moe_outlier_k,
                    moe_obb_mode=args.moe_obb_mode,
                    moe_skip_obb_rule=args.moe_skip_obb_rule,
                    moe_obb_density=args.moe_obb_density,
                    moe_obb_position_spacing_cm=args.moe_obb_position_spacing_cm,
                )
            )
            if args.planner == "graspmoe":
                n_diff = branch_tags.count("diff")
                n_obb = branch_tags.count("obb")
                print(f"[graspmoe] diffusion={n_diff} OBB={n_obb}")

            if (
                args.planner == "graspmoe"
                and moe_obb_dict is not None
                and not args.moe_no_obb_wireframe
            ):
                T_obb = np.eye(4)
                T_obb[:3, :3] = moe_obb_dict["R"]
                T_obb[:3, 3] = moe_obb_dict["center"]
                visualize_bbox(
                    vis,
                    "moe_obb",
                    2.0 * moe_obb_dict["half_extent"],
                    T=T_obb,
                    color=[255, 130, 0],
                )

            if len(grasps_inferred) > 0:
                scores_inferred = get_color_from_score(
                    grasp_conf_inferred, use_255_scale=True
                )
                print(
                    f"Inferred {len(grasps_inferred)} grasps, scores: "
                    f"{grasp_conf_inferred.min():.3f} — {grasp_conf_inferred.max():.3f}"
                )

                best_idx = int(grasp_conf_inferred.argmax())
                if args.vis_top_grasp_meshes:
                    n_top = min(args.num_top_grasp_meshes, len(grasps_inferred))
                    top_grasp_mesh_indices = set(
                        np.argsort(-grasp_conf_inferred)[:n_top].tolist()
                    )
                    print(
                        f"Rendering collision meshes for top {n_top} grasps "
                        f"(by confidence) from {len(grasps_inferred)} thresholded grasps"
                    )
                else:
                    top_grasp_mesh_indices = set()

                grasp_handles = []  # list of (conf, [line_handles], [mesh_handles])
                diff_count = 0
                obb_count = 0
                for j, grasp in enumerate(grasps_inferred):
                    tag = branch_tags[j] if j < len(branch_tags) else "diff"
                    if tag == "obb":
                        ns = f"pred_grasps/obb_gen/grasp_{obb_count:03d}"
                        obb_count += 1
                    else:
                        ns = f"pred_grasps/diff_gen/grasp_{diff_count:03d}"
                        diff_count += 1
                    base_color = scores_inferred[j]
                    color = [0, 100, 255] if j == best_idx else base_color
                    lw = 5.0 if j == best_idx else 3.0
                    line_hs = visualize_x_grasp(
                        vis,
                        ns,
                        grasp,
                        color=color,
                        gripper_info=gripper,
                        linewidth=lw,
                    )
                    mesh_hs = []
                    if j < 5 and args.plot_mesh:
                        if tag == "obb":
                            mesh_ns = f"pred_meshes/obb_gen/grasp_{j:03d}"
                        else:
                            mesh_ns = f"pred_meshes/diff_gen/grasp_{j:03d}"
                        mh = visualize_mesh(
                            vis,
                            mesh_ns,
                            gripper.collision_mesh,
                            color=base_color,
                            transform=grasp,
                        )
                        if mh is not None:
                            mesh_hs.append(mh)
                    if j in top_grasp_mesh_indices:
                        mh = visualize_mesh(
                            vis,
                            f"top_grasp_meshes/grasp_{j:03d}",
                            gripper.collision_mesh,
                            color=scores_inferred[j],
                            transform=grasp,
                        )
                        if mh is not None:
                            mesh_hs.append(mh)
                    grasp_handles.append((grasp_conf_inferred[j], line_hs, mesh_hs))

                top_mesh_handle = None
                if args.plot_top_mesh:
                    print(
                        f"Top grasp: index {best_idx}, "
                        f"confidence {grasp_conf_inferred[best_idx]:.3f}"
                    )
                    top_mesh_handle = visualize_mesh(
                        vis,
                        "top_grasp_mesh",
                        gripper.collision_mesh,
                        color=[0, 100, 255],
                        transform=grasps_inferred[best_idx],
                    )
                if animate and robot is not None:
                    # Place animated gripper to the left (-X) of the point cloud,
                    # offset by 3x the PC extent so it's clearly separated but visible
                    pc_x_range = pc_centered[:, 0].max() - pc_centered[:, 0].min()
                    side_x = pc_centered[:, 0].min() - max(pc_x_range * 3, 0.3)
                    side_pose = np.array(
                        [
                            [1, 0, 0, side_x],
                            [0, 1, 0, 0.00],
                            [0, 0, 1, 0.00],
                            [0, 0, 0, 1.00],
                        ],
                        dtype=np.float64,
                    )
                    print(f"  Animation side pose: x={side_x:.3f}")

                    sweep_volume = gripper_config.get("sweep_volume", None)

                    # Draw grasp reference lines at the side pose
                    visualize_x_grasp(
                        vis,
                        "animation_grasp_reference",
                        side_pose,
                        color=[255, 255, 0],
                        sweep_volume=sweep_volume,
                        linewidth=4.0,
                    )

                    # Draw sweep volume bbox
                    if sweep_volume is not None:
                        sv_extents = np.array(sweep_volume["extents"])
                        sv_offset = np.array(sweep_volume["offset"])
                        offset_transform = np.eye(4)
                        offset_transform[:3, 3] = sv_offset
                        sv_world_transform = side_pose @ offset_transform
                        visualize_bbox(
                            vis,
                            "animated_sweep_volume",
                            sv_extents,
                            T=sv_world_transform,
                            color=[0, 100, 255],
                        )
                        print(f"  Sweep volume: extents={sv_extents.tolist()}")

                    # Render one static frame first to verify meshes load
                    js_open = gripper_config["open"]
                    visualize_gripper_at_pose(
                        vis,
                        robot,
                        js_open,
                        side_pose,
                        name_prefix="animated_gripper",
                        color=[255, 100, 50],
                    )
                    print(f"  Static gripper frame rendered at side pose (x=-1.0)")

                    # Start animation thread
                    animation_thread = AnimationThread(
                        vis,
                        robot,
                        gripper_config,
                        side_pose,
                        animation_steps=15,
                        animation_delay=0.08,
                    )
                    animation_thread.start()
                    print(f"  Gripper animation started (1m to the left of scene)")
                # Interactive threshold tuner
                threshold_gui = None
                count_md = None
                if args.interactive_threshold_tuner and len(grasp_handles) > 0:
                    with vis.gui.add_folder("Threshold Tuner"):
                        count_md = vis.gui.add_markdown(
                            f"**Visible: {len(grasp_handles)} / {len(grasp_handles)}**"
                        )
                        threshold_gui = vis.gui.add_slider(
                            "Confidence threshold",
                            min=0.0,
                            max=1.0,
                            step=0.01,
                            initial_value=0.0,
                        )

                    @threshold_gui.on_update
                    def _on_threshold_change(_):
                        thresh = threshold_gui.value
                        n_visible = 0
                        for conf, line_hs, mesh_hs in grasp_handles:
                            vis_flag = bool(conf >= thresh)
                            if vis_flag:
                                n_visible += 1
                            for h in line_hs:
                                h.visible = vis_flag
                            for h in mesh_hs:
                                h.visible = vis_flag
                        if top_mesh_handle is not None:
                            top_mesh_handle.visible = bool(
                                grasp_conf_inferred[best_idx] >= thresh
                            )
                        count_md.content = (
                            f"**Visible: {n_visible} / {len(grasp_handles)}**"
                        )

            else:
                print("No grasps found from inference! Skipping to next object...")

            advance_event = threading.Event()
            next_btn = vis.gui.add_button("Next Object")

            @next_btn.on_click
            def _on_next_click(_):
                advance_event.set()

            print(
                f"[{gripper_name}] Press 'Next Object' in the GUI or Enter in the terminal to continue..."
            )
            while not advance_event.is_set():
                if advance_event.wait(timeout=0.2):
                    break
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    sys.stdin.readline()
                    break

            next_btn.remove()

            # Cleanup GUI elements before next object
            if args.interactive_threshold_tuner:
                if threshold_gui is not None:
                    threshold_gui.remove()
                if count_md is not None:
                    count_md.remove()

            if animation_thread is not None:
                animation_thread.stop()
