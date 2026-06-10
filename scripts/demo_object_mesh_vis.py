# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import glob
import os
import select
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import trimesh
import trimesh.transformations as tra

from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.utils.viser_utils import (
    create_visualizer,
    get_color_from_score,
    visualize_x_grasp,
    visualize_mesh,
    visualize_pointcloud,
)
from graspgenx.dataset.dataset_utils import sample_points
from graspgenx.dataset.eval_utils import save_to_isaac_grasp_format
from demo_object_pc import _resolve_default_checkpoints, load_model_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="GraspGenX inference on a single object mesh (visual mesh rendering)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--mesh_file",
        type=str,
        help="Path to a single mesh file (obj, stl, ply, usd, usda, usdc, or usdz)",
    )
    group.add_argument(
        "--sample_data_dir",
        type=str,
        help="Directory containing mesh files to cycle through",
    )
    parser.add_argument(
        "--mesh_scale",
        type=float,
        default=1.0,
        help="Scale factor to apply to the mesh",
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
        required=True,
        help="Gripper name (must match an entry in assets/x_grippers/ or assets/proc_grippers/)",
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
        default=-1.0,
        help="Confidence threshold for grasps. Use -1.0 to return top-k instead.",
    )
    parser.add_argument(
        "--num_grasps",
        type=int,
        default=400,
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
        "--num_sample_points",
        type=int,
        default=3500,
        help="Number of points to sample from the mesh surface",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="Path to save the output grasps in YAML format. If empty, grasps are not saved.",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Disable viser visualization",
    )
    parser.add_argument(
        "--plot_mesh",
        action="store_true",
        help="Also render the gripper visual mesh for the top 5 grasps",
    )
    parser.add_argument(
        "--plot_top_mesh",
        action="store_true",
        help="Render the gripper visual mesh at the single top-ranked grasp",
    )
    parser.add_argument(
        "--interactive_threshold_tuner",
        action="store_true",
        help="Add a viser GUI slider to interactively threshold predicted grasps by confidence",
    )
    return parser.parse_args()


def load_mesh_data(mesh_file, scale, num_sample_points):
    """Load mesh data and sample points from the surface."""
    if mesh_file.endswith(".ply"):
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(mesh_file)
        xyz = np.array(pcd.points).astype(np.float32)
        pt_idx = sample_points(xyz, num_sample_points)
        xyz = xyz[pt_idx]
        obj = None
    elif mesh_file.endswith((".usd", ".usda", ".usdc", ".usdz")):
        import scene_synthesizer as synth

        asset = synth.Asset(mesh_file)
        obj = asset.mesh()
        obj.apply_scale(scale)
        xyz, _ = trimesh.sample.sample_surface(obj, num_sample_points)
        xyz = np.array(xyz)
    else:
        obj = trimesh.load(mesh_file)
        obj.apply_scale(scale)
        xyz, _ = trimesh.sample.sample_surface(obj, num_sample_points)
        xyz = np.array(xyz)

    T_subtract_pc_mean = tra.translation_matrix(-xyz.mean(axis=0))
    xyz = tra.transform_points(xyz, T_subtract_pc_mean)
    if obj is not None:
        obj.apply_transform(T_subtract_pc_mean)

    rgb = np.ones((len(xyz), 3)) * 255

    return xyz, rgb, obj, T_subtract_pc_mean


def process_single_mesh(args, mesh_file, grasp_sampler, gripper, vis):
    """Run inference on a single mesh file and visualize results.

    Returns (grasp_handles, top_mesh_handle, grasp_conf_inferred, best_idx)
    for use by the threshold tuner, or Nones if no grasps found.
    """
    valid_extensions = (".stl", ".obj", ".ply", ".usd", ".usda", ".usdc", ".usdz")
    if not mesh_file.endswith(valid_extensions):
        print(f"Skipping {mesh_file} — not a supported mesh format")
        return None, None, None, None

    print(f"\nProcessing mesh file: {mesh_file}")
    pc, pc_color, obj_mesh, T_subtract_pc_mean = load_mesh_data(
        mesh_file, args.mesh_scale, args.num_sample_points
    )

    if vis is not None and obj_mesh is not None:
        visualize_mesh(vis, "object_mesh", obj_mesh, color=[169, 169, 169])
    if vis is not None:
        visualize_pointcloud(vis, "pc", pc, pc_color, size=0.0025)

    grasps_inferred, grasp_conf_inferred = GraspGenXSampler.run_inference(
        pc,
        grasp_sampler,
        grasp_threshold=args.grasp_threshold,
        num_grasps=args.num_grasps,
        topk_num_grasps=args.topk_num_grasps,
        remove_outliers=False,
    )

    if len(grasps_inferred) == 0:
        print("No grasps found from inference!")
        return None, None, None, None

    grasp_conf_inferred = grasp_conf_inferred.cpu().numpy()
    grasps_inferred = grasps_inferred.cpu().numpy()
    grasps_inferred[:, 3, 3] = 1
    scores_inferred = get_color_from_score(grasp_conf_inferred, use_255_scale=True)
    print(
        f"Inferred {len(grasps_inferred)} grasps, scores: "
        f"{grasp_conf_inferred.min():.3f} — {grasp_conf_inferred.max():.3f}"
    )

    grasp_handles = []
    top_mesh_handle = None
    best_idx = int(grasp_conf_inferred.argmax())

    if vis is not None:
        for j, grasp in enumerate(grasps_inferred):
            color = [0, 100, 255] if j == best_idx else scores_inferred[j]
            lw = 5.0 if j == best_idx else 3.0
            line_hs = visualize_x_grasp(
                vis,
                f"pred_grasps/grasp_{j:03d}",
                grasp,
                color=color,
                gripper_info=gripper,
                linewidth=lw,
            )
            mesh_hs = []
            if j < 5 and args.plot_mesh:
                mesh = gripper.visual_mesh if gripper.visual_mesh is not None else gripper.collision_mesh
                mh = visualize_mesh(
                    vis,
                    f"pred_meshes/grasp_{j:03d}",
                    mesh,
                    color=scores_inferred[j],
                    transform=grasp,
                )
                if mh is not None:
                    mesh_hs.append(mh)
            grasp_handles.append((grasp_conf_inferred[j], line_hs, mesh_hs))

        if args.plot_top_mesh:
            print(
                f"Top grasp: index {best_idx}, "
                f"confidence {grasp_conf_inferred[best_idx]:.3f}"
            )
            mesh = gripper.visual_mesh if gripper.visual_mesh is not None else gripper.collision_mesh
            top_mesh_handle = visualize_mesh(
                vis,
                "top_grasp_mesh",
                mesh,
                color=[0, 100, 255],
                transform=grasps_inferred[best_idx],
            )

    grasps_original_frame = np.array(
        [tra.inverse_matrix(T_subtract_pc_mean) @ g for g in grasps_inferred]
    )

    if args.output_file != "":
        print(f"Saving predicted grasps to {args.output_file}")
        save_to_isaac_grasp_format(
            grasps_original_frame, grasp_conf_inferred, args.output_file
        )

    return grasp_handles, top_mesh_handle, grasp_conf_inferred, best_idx


def add_threshold_tuner(
    vis, grasp_handles, top_mesh_handle, grasp_conf_inferred, best_idx
):
    """Add interactive confidence threshold slider to viser GUI."""
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
            top_mesh_handle.visible = bool(grasp_conf_inferred[best_idx] >= thresh)
        count_md.content = f"**Visible: {n_visible} / {len(grasp_handles)}**"

    return threshold_gui, count_md


def wait_for_next(vis, label=""):
    """Block until the user clicks 'Next Object' in the GUI or presses Enter."""
    advance_event = threading.Event()
    next_btn = vis.gui.add_button("Next Object")

    @next_btn.on_click
    def _on_next_click(_):
        advance_event.set()

    print(
        f"[{label}] Press 'Next Object' in the GUI or Enter in the terminal to continue..."
    )
    while not advance_event.is_set():
        if advance_event.wait(timeout=0.2):
            break
        if select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.readline()
            break

    next_btn.remove()


if __name__ == "__main__":
    args = parse_args()

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

    print(f"Loading gripper: {args.gripper_name}")
    print(f"Assets directory: {args.assets_dir}")
    grasp_sampler = GraspGenXSampler(
        model_cfg, args.gripper_name, assets_dir=args.assets_dir
    )
    gripper = grasp_sampler.get_gripper_info()

    vis = None if getattr(args, "no_visualization", False) else create_visualizer()

    valid_extensions = (".stl", ".obj", ".ply", ".usd", ".usda", ".usdc", ".usdz")
    if args.mesh_file:
        mesh_files = [args.mesh_file]
    else:
        mesh_files = sorted(
            f
            for f in glob.glob(os.path.join(args.sample_data_dir, "*"))
            if f.endswith(valid_extensions)
        )
        if not mesh_files:
            raise FileNotFoundError(
                f"No mesh files ({', '.join(valid_extensions)}) found in {args.sample_data_dir}"
            )
    print(f"Found {len(mesh_files)} mesh file(s)")

    for mesh_file in mesh_files:
        if vis is not None:
            vis.scene.reset()

        grasp_handles, top_mesh_handle, grasp_conf_inferred, best_idx = (
            process_single_mesh(args, mesh_file, grasp_sampler, gripper, vis)
        )

        threshold_gui = None
        count_md = None
        if (
            vis is not None
            and args.interactive_threshold_tuner
            and grasp_handles is not None
            and len(grasp_handles) > 0
        ):
            threshold_gui, count_md = add_threshold_tuner(
                vis, grasp_handles, top_mesh_handle, grasp_conf_inferred, best_idx
            )

        if vis is not None:
            if len(mesh_files) > 1:
                wait_for_next(vis, label=os.path.basename(mesh_file))
            else:
                print(
                    "Visualization ready — open http://localhost:8080. Press Enter to exit."
                )
                input()

        if threshold_gui is not None:
            threshold_gui.remove()
        if count_md is not None:
            count_md.remove()
