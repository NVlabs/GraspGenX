# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Modules to compute gripper poses from contact masks and parameters.
"""

import json
from dataclasses import dataclass

import numpy as np
import os
import torch
import trimesh
import trimesh.transformations as tra

from graspgenx.robot import get_canonical_gripper_control_points


@dataclass
class XGripperInfo:
    """A dataclass containing information about a gripper configuration.

    This class stores all the necessary information about a gripper including its
    meshes, transforms, control points, and other properties.

    Attributes:
        gripper_name (str): Name identifier for the gripper.
        collision_mesh (trimesh.base.Trimesh): Mesh used for collision detection.
        visual_mesh (trimesh.base.Trimesh): Mesh used for visualization.
        depth (float, optional): Depth of the gripper. Defaults to None.
        symmetric (float): Whether the gripper is symmetric. Defaults to False. For antipodal grippers only.
        control_points (np.ndarray): Control points used for applying learning losses and computing metrics.
        control_points_visualization (np.ndarray, optional): Points used for visualization. Defaults to None.
        tool_tcp_transform (np.ndarray, optional): Transform from the gripper base link to the tool TCP (Tool Center Point). Defaults to None.
    """

    gripper_name: str
    gripper_type: int
    collision_mesh: trimesh.base.Trimesh
    visual_mesh: trimesh.base.Trimesh
    depth: float
    symmetric: float
    sweep_volume: np.ndarray
    sweep_volume_mid: np.ndarray
    grasp_volume: np.ndarray
    open_pointcloud: np.ndarray
    close_pointcloud: np.ndarray
    vol_tsdf: np.ndarray
    pointnet_vae: dict
    control_points: np.ndarray
    control_points_visualization: np.ndarray
    tool_tcp_transform: np.ndarray
    width: float = 0.0


def load_control_points(w, d):
    """
    Load the control points for the gripper, used for training.
    Returns a tensor of shape (4, N) where N is the number of control points.
    """
    control_points = get_canonical_gripper_control_points(w, d)
    control_points = np.vstack([control_points, np.zeros(3)])
    control_points = np.hstack([control_points, np.ones([len(control_points), 1])])
    return control_points.T


def sample_gripper_pointclouds(gripper: XGripperInfo, num_points: int):
    """Resample the open and closed gripper point clouds to ``num_points`` each.

    Points are drawn with replacement, each from its own state's cloud.
    Returns ``(open_points, close_points)`` as float32 arrays of shape
    ``(num_points, 3)``.
    """
    sampled = []
    for points in (gripper.open_pointcloud, gripper.close_pointcloud):
        mask = np.random.randint(0, points.shape[0], (num_points,))
        sampled.append(points[mask].astype(np.float32))
    return sampled[0], sampled[1]


def load_control_points_for_visualization(ctrl_pts):

    control_points = ctrl_pts.T

    mid_point = (control_points[0] + control_points[1]) / 2

    control_points = [
        control_points[-3],
        control_points[0],
        mid_point,
        [0, 0, 0, 1],
        mid_point,
        control_points[1],
        control_points[-2],
    ]
    return [
        control_points,
    ]


def get_gripper_info(asset_path, name: str) -> XGripperInfo:
    """Get comprehensive information about a specified gripper.

    This function orchestrates the loading of the gripper configuration, model,
    and gripper-specific information from Python definitions.

    Args:
        name (str): Name of the gripper to get information for.

    Returns:
        GripperInfo: Object containing all information about the gripper.

    Raises:
        ValueError: If the gripper is not registered.
        NotImplementedError: If required functions are not implemented in the gripper module.
    """
    with open(f"{asset_path}/{name}/config.json", "r") as f:
        config = json.load(f)

    points_path = f"{asset_path}/{name}/points.json"
    if os.path.exists(points_path):
        with open(points_path, "r") as f:
            points = json.load(f)
    else:
        print(f"[WARNING] points.json not found for {name}, using dummy values.")
        points = {
            "open": [[0.0, 0.0, 0.0]] * 10500,
            "close": [[0.0, 0.0, 0.0]] * 10500,
        }

    vae_repr_path = f"{asset_path}/{name}/proc_gripper_only_pointnet_vae_repr.json"
    if os.path.exists(vae_repr_path):
        with open(vae_repr_path, "r") as f:
            pointnet_vae_repr = json.load(f)
    else:
        print(
            f"[WARNING] proc_gripper_only_pointnet_vae_repr.json not found for {name}, using dummy values."
        )
        pointnet_vae_repr = {
            "open": [0.0] * 64,
            "close": [0.0] * 64,
            "half": [0.0] * 64,
        }

    tsdf_path = f"{asset_path}/{name}/tsdf.npy"
    if os.path.exists(tsdf_path):
        vol_tsdf = np.load(tsdf_path, allow_pickle=True).tolist()
    else:
        print(f"[WARNING] tsdf.npy not found for {name}, using dummy values.")
        vol_tsdf = {
            "open_tsdf": np.zeros((64, 32, 64), dtype=np.float16),
            "close_tsdf": np.zeros((64, 32, 64), dtype=np.float16),
        }

    coll_mesh_path = f"{asset_path}/{name}/coll_mesh.obj"
    if os.path.exists(coll_mesh_path):
        coll_mesh = trimesh.load(coll_mesh_path, force="mesh")
    else:
        print(f"[WARNING] coll_mesh.obj not found for {name}, using dummy mesh.")
        coll_mesh = trimesh.primitives.Box(extents=[0.01, 0.01, 0.01])

    vis_mesh_path = f"{asset_path}/{name}/vis_mesh.obj"
    if os.path.exists(vis_mesh_path):
        vis_mesh = trimesh.load(vis_mesh_path, force="mesh")
    else:
        print(f"[WARNING] vis_mesh.obj not found for {name}, using dummy mesh.")
        vis_mesh = trimesh.primitives.Box(extents=[0.01, 0.01, 0.01])

    fingertip_depth = config["fingertip"][-1]

    gripper_width = config["bbox"][1][0] - config["bbox"][0][0]
    gripper_depth = config["bbox"][1][-1]

    ctrl_pts = load_control_points(gripper_width, gripper_depth)
    ctrl_vis_pts = load_control_points_for_visualization(ctrl_pts)

    fingertip_pose = tra.translation_matrix(config["fingertip"])
    grasp_volume_extents = np.array(config["sweep_volume"]["extents"])
    grasp_volume_center = np.array(config["sweep_volume"]["offset"])
    grasp_volume_standoff = np.array(config["standoff"])

    grasp_volume_extents_mid = np.array(config["sweep_volume"]["extents2"])
    grasp_volume_center_mid = np.array(config["sweep_volume"]["offset2"])

    grasp_volume = [
        grasp_volume_center
        - grasp_volume_extents / 2.0
        + np.array([0, 0, grasp_volume_standoff[0]]),
        grasp_volume_center
        + grasp_volume_extents / 2.0
        + np.array([0, 0, grasp_volume_standoff[1]]),
    ]

    gripper_type_map = {"parallel_2f": 0, "revolute_2f": 1, "revolute_3f": 2}
    gripper_type_idx = gripper_type_map[config["type"]]

    gripper_info = XGripperInfo(
        gripper_name=name,
        gripper_type=gripper_type_idx,
        collision_mesh=coll_mesh,
        visual_mesh=vis_mesh,
        depth=fingertip_depth,
        symmetric=config["symmetric"],
        sweep_volume=np.concatenate(
            [grasp_volume_extents, grasp_volume_center], axis=0
        ),
        sweep_volume_mid=np.concatenate(
            [grasp_volume_extents_mid, grasp_volume_center_mid], axis=0
        ),
        grasp_volume=grasp_volume,
        open_pointcloud=np.array(points["open"]),
        close_pointcloud=np.array(points["close"]),
        vol_tsdf=vol_tsdf,
        pointnet_vae=pointnet_vae_repr,
        control_points=ctrl_pts,
        control_points_visualization=ctrl_vis_pts,
        tool_tcp_transform=fingertip_pose,
        width=gripper_width,
    )
    return gripper_info


def make_sweep_volume_gripper_info(
    extents_open,
    offset_open,
    extents_mid,
    offset_mid,
    gripper_type: int = 0,
    fingertip_depth: float = None,
    name: str = "custom_sweep_volume",
) -> XGripperInfo:
    """Build an XGripperInfo from raw sweep-volume parameters — no assets.

    Fields the sweep_volume_v2 model conditioning does not consume (gripper
    point clouds, TSDF, pointnet-VAE repr, meshes) are filled with the same
    dummy placeholders :func:`get_gripper_info` uses when the corresponding
    files are missing. The fingertip depth defaults to the top plane of the
    open sweep box.
    """
    extents_open = np.asarray(extents_open, dtype=np.float32).reshape(3)
    offset_open = np.asarray(offset_open, dtype=np.float32).reshape(3)
    extents_mid = np.asarray(extents_mid, dtype=np.float32).reshape(3)
    offset_mid = np.asarray(offset_mid, dtype=np.float32).reshape(3)

    if fingertip_depth is None:
        fingertip_depth = float(offset_open[2] + extents_open[2] / 2.0)
    fingertip_depth = float(fingertip_depth)
    gripper_width = float(extents_open[0])

    ctrl_pts = load_control_points(gripper_width, fingertip_depth)
    ctrl_vis_pts = load_control_points_for_visualization(ctrl_pts)

    grasp_volume = [
        offset_open - extents_open / 2.0,
        offset_open + extents_open / 2.0,
    ]

    dummy_points = np.zeros((10500, 3), dtype=np.float32)
    dummy_tsdf = {
        "open_tsdf": np.zeros((64, 32, 64), dtype=np.float16),
        "close_tsdf": np.zeros((64, 32, 64), dtype=np.float16),
    }
    dummy_vae = {
        "open": [0.0] * 64,
        "close": [0.0] * 64,
        "half": [0.0] * 64,
    }
    dummy_mesh = trimesh.primitives.Box(extents=[0.01, 0.01, 0.01])

    return XGripperInfo(
        gripper_name=name,
        gripper_type=int(gripper_type),
        collision_mesh=dummy_mesh,
        visual_mesh=dummy_mesh,
        depth=fingertip_depth,
        symmetric=False,
        sweep_volume=np.concatenate([extents_open, offset_open], axis=0),
        sweep_volume_mid=np.concatenate([extents_mid, offset_mid], axis=0),
        grasp_volume=grasp_volume,
        open_pointcloud=dummy_points,
        close_pointcloud=dummy_points,
        vol_tsdf=dummy_tsdf,
        pointnet_vae=dummy_vae,
        control_points=ctrl_pts,
        control_points_visualization=ctrl_vis_pts,
        tool_tcp_transform=tra.translation_matrix([0.0, 0.0, fingertip_depth]),
        width=gripper_width,
    )


def _list_available_grippers(assets_dir: str) -> set:
    """Scan all gripper sources and return available names."""
    available = set()
    try:
        import gripper_descriptions

        available.update(gripper_descriptions.list_grippers())
    except ImportError:
        pass
    for subdir in ("x_grippers", "proc_grippers"):
        d = os.path.join(assets_dir, subdir)
        if os.path.isdir(d):
            available.update(
                name
                for name in os.listdir(d)
                if os.path.isdir(os.path.join(d, name)) and name != "utils"
            )
    return available


def resolve_gripper_asset_dir(gripper_name: str, assets_dir: str = None) -> str:
    """Resolve a gripper name to its asset directory path.

    Same lookup order as resolve_gripper_info() but returns the directory path
    instead of loading the full XGripperInfo.
    """
    if assets_dir is None:
        assets_dir = "/code/assets"

    try:
        import gripper_descriptions

        gd_path = os.path.join(gripper_descriptions.get_assets_path(), gripper_name)
        if os.path.isdir(gd_path):
            return gd_path
    except ImportError:
        pass

    x_path = os.path.join(assets_dir, "x_grippers", gripper_name)
    if os.path.isdir(x_path):
        return x_path

    proc_path = os.path.join(assets_dir, "proc_grippers", gripper_name)
    if os.path.isdir(proc_path):
        return proc_path

    available = _list_available_grippers(assets_dir)
    raise ValueError(
        f"Unknown gripper '{gripper_name}'. Not found in gripper_descriptions package, "
        f"assets/x_grippers/, or assets/proc_grippers/.\n"
        f"Available grippers: {', '.join(sorted(available))}"
    )


def resolve_gripper_info(gripper_name: str, assets_dir: str = None) -> XGripperInfo:
    """Resolve a gripper by name from any available source.

    Checks locations in order:
      1. gripper_descriptions package (if installed)
      2. assets/x_grippers/<name>/
      3. assets/proc_grippers/<name>/

    Args:
        gripper_name: Gripper folder name.
        assets_dir: Root directory containing x_grippers/ and proc_grippers/.
                    Defaults to /code/assets (Docker path).

    Returns:
        XGripperInfo for the resolved gripper.

    Raises:
        ValueError: If the gripper is not found in any location.
    """
    if assets_dir is None:
        assets_dir = "/code/assets"

    # 1. gripper_descriptions checkout (resolved via $GRASPGENX_GRIPPER_CFG_DIR
    #    or the auto-cloned <repo>/ext/gripper_descriptions/, set up by
    #    ensure_gripper_descriptions() in graspgenx/__init__.py). This avoids
    #    needing the `gripper_descriptions` Python package on sys.path.
    try:
        from graspgenx._setup_dependencies import get_gripper_descriptions_assets

        gd_assets = str(get_gripper_descriptions_assets())
        if os.path.isdir(os.path.join(gd_assets, gripper_name)):
            return get_gripper_info(gd_assets, gripper_name)
    except (ImportError, FileNotFoundError):
        pass

    # 2. assets/x_grippers/
    x_path = os.path.join(assets_dir, "x_grippers", gripper_name)
    if os.path.isdir(x_path):
        return get_gripper_info(os.path.join(assets_dir, "x_grippers"), gripper_name)

    # 3. assets/proc_grippers/
    proc_path = os.path.join(assets_dir, "proc_grippers", gripper_name)
    if os.path.isdir(proc_path):
        return get_gripper_info(os.path.join(assets_dir, "proc_grippers"), gripper_name)

    # Not found
    available = _list_available_grippers(assets_dir)
    raise ValueError(
        f"Unknown gripper '{gripper_name}'. Not found in gripper_descriptions package, "
        f"assets/x_grippers/, or assets/proc_grippers/.\n"
        f"Available grippers: {', '.join(sorted(available))}"
    )
