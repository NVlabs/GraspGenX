# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Modules to compute gripper poses from contact masks and parameters.
"""

import glob
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import trimesh
import trimesh.transformations as tra
import yaml



@dataclass
class GripperInfo:
    """A dataclass containing information about a gripper configuration.

    This class stores all the necessary information about a gripper including its
    meshes, transforms, control points, and other properties.

    Attributes:
        gripper_name (str): Name identifier for the gripper.
        collision_mesh (trimesh.base.Trimesh): Mesh used for collision detection.
        visual_mesh (trimesh.base.Trimesh): Mesh used for visualization.
        offset_transform (trimesh.base.Trimesh): This is the transform offset from converting a gripper from its original axis to the GraspGen convention
        depth (float, optional): Depth of the gripper. Defaults to None.
        symmetric (float): Whether the gripper is symmetric. Defaults to False. For antipodal grippers only.
        control_points (np.ndarray): Control points used for applying learning losses and computing metrics.
        control_points_visualization (np.ndarray, optional): Points used for visualization. Defaults to None.
        transform_from_base_link_to_tool_tcp (np.ndarray, optional): Transform from the gripper base link to the tool TCP (Tool Center Point). Defaults to None.
    """

    gripper_name: str
    collision_mesh: trimesh.base.Trimesh
    visual_mesh: trimesh.base.Trimesh
    offset_transform: trimesh.base.Trimesh
    control_points: np.ndarray
    depth: float = None
    symmetric: float = False
    control_points: np.ndarray = None
    control_points_visualization: np.ndarray = None
    transform_from_base_link_to_tool_tcp: np.ndarray = None


def get_canonical_gripper_control_points(w, d):
    """Generate canonical control points for a gripper.

    Convention assumes approach direction is z-positive and contact direction is along x.
    The control points are arranged in a rectangular pattern.

    Args:
        w (float): Width of the gripper.
        d (float): Depth of the gripper.

    Returns:
        np.ndarray: Array of shape (4, 3) containing the control points in 3D space.
                    Points are arranged as [right_front, left_front, right_back, left_back].
    """
    control_points = np.array(
        [[w / 2, -0, d / 2], [-w / 2, 0, d / 2], [w / 2, -0, d], [-w / 2, 0, d]]
    )
    return control_points


def get_gripper_depth(gripper_name) -> float:
    """Get the depth of a specified gripper.

    This gives the distance from the gripper root/base link frame to the gripper tool tip frame.

    Args:
        gripper_name (str): Name of the gripper to get depth for.

    Returns:
        float: The depth of the gripper in meters.
    """
    gripper_info = get_gripper_info(gripper_name)
    return gripper_info.depth


def load_control_points_core(gripper_config: Dict):
    """Load control points from a gripper configuration.

    This function handles loading control points either from explicit control points
    in the config or by generating them from width and depth parameters.

    Args:
        gripper_config (Dict): Dictionary containing gripper configuration.

    Returns:
        np.ndarray: Array of control points for the gripper.

    Raises:
        NotImplementedError: If neither control_points nor width and depth are specified.
    """
    if "control_points" in gripper_config:
        control_points = gripper_config["control_points"]
    elif "width" in gripper_config and "depth" in gripper_config:
        width = gripper_config["width"]
        depth = gripper_config["depth"]
        control_points = get_canonical_gripper_control_points(width, depth)
    else:
        raise NotImplementedError(
            "Unable to load control points: neither 'control_points' nor "
            "both 'width' and 'depth' are specified in gripper_config."
        )
    return control_points


def load_control_points_for_visualization(gripper_name: str):
    """Load control points used for visualization of a gripper.

    Args:
        gripper_name (str): Name of the gripper to load visualization points for.

    Returns:
        np.ndarray: Array of control points used for visualization.
    """
    gripper_info = get_gripper_info(gripper_name)
    control_points = gripper_info.control_points_visualization
    return control_points


def load_control_points(gripper_name: str) -> torch.Tensor:
    """Load control points for a specified gripper.

    Args:
        gripper_name (str): Name of the gripper to load control points for.

    Returns:
        torch.Tensor: Tensor containing the control points for the gripper.
    """
    gripper_info = get_gripper_info(gripper_name)
    control_points = gripper_info.control_points
    return control_points


def generate_circle_points(center, radius=0.007, N=30):
    """Generate a set of points that form a circle in 2D space.

    This function creates N equally spaced points along the circumference of a circle
    with the specified center and radius. The points are returned as a 2D array of
    x,y coordinates.

    Args:
        center (np.ndarray or list): The [x, y] coordinates of the circle's center.
        radius (float, optional): The radius of the circle in meters. Defaults to 0.007.
        N (int, optional): Number of points to generate along the circle. Defaults to 30.

    Returns:
        np.ndarray: A 2D array of shape (N, 2) containing the x,y coordinates of the
                   generated points. Each row represents one point [x, y].

    Example:
        >>> center = [0.0, 0.0]
        >>> points = generate_circle_points(center, radius=0.01, N=8)
        >>> print(points.shape)
        (8, 2)
    """
    # Generate N equally spaced angles between 0 and 2π
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # Calculate the x and y coordinates of the points
    x_points = center[0] + radius * np.cos(angles)
    y_points = center[1] + radius * np.sin(angles)

    # Stack x and y points into a 2D array
    points = np.stack((x_points, y_points), axis=1)

    return points


def load_visualize_control_points_multi_suction(
    list_of_suction_center_points=List[List[float]],
):
    """Generate visualization points for multiple suction cups.

    This function creates visualization points for multiple suction cups by generating
    circular patterns around each suction center point.

    Args:
        list_of_suction_center_points (List[List[float]]): List of [x, y, z] coordinates
            for each suction cup center. The z-axis value is obtained from the first
            suction center point.

    Returns:
        np.ndarray: Array of shape (N, M, 3) containing the x,y,z coordinates for
                   visualization points of all suction cups, where N is the number of
                   suction cups and M is the number of points per suction cup.

    Note:
        The z-axis value is obtained from the first suction center point and applied
        to all points.
    """
    assert len(list_of_suction_center_points) > 0
    for pt in list_of_suction_center_points:
        assert len(pt) == 3

    z = list_of_suction_center_points[0][2]
    xy = [
        generate_circle_points(c[0:2], radius=0.005)
        for c in list_of_suction_center_points
    ]
    xy = np.stack(xy)
    ptsz = z * np.ones([xy.shape[0], xy.shape[1], 1])
    xyz = np.concatenate([xy, ptsz], axis=2)
    return xyz


def load_gripper_yaml_file(yaml_path: Path):
    """Load gripper configuration from a YAML file.

    Args:
        yaml_path (Path): Path to the YAML configuration file.

    Returns:
        Dict: Dictionary containing the gripper configuration.
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def import_module_from_path(path):
    """Import a Python module from a file path.

    Args:
        path (Path): Path to the Python module file.

    Returns:
        module: The imported Python module.

    Raises:
        ImportError: If the module cannot be imported.
    """
    path = Path(path)
    module_name = path.stem

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None:
        raise ImportError(f"Could not create a spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_default_gripper_config(gripper_name: str) -> Dict:
    """Load the default configuration for a specified gripper.

    Args:
        gripper_name (str): Name of the gripper to load configuration for.

    Returns:
        Dict: Dictionary containing the gripper's default configuration.
    """
    conf_path = (
        Path(__file__).parent.parent / "config" / "grippers" / f"{gripper_name}.yaml"
    )
    config = load_gripper_yaml_file(conf_path)
    return config


def parse_offset_transform_from_yaml(offset_transform: List[List[float]]) -> np.ndarray:
    """Parse offset transform from YAML format to 4x4 transformation matrix.

    Args:
        offset_transform (List[List[float]]): List containing [translation (xyz), quaternion (xyzw)].

    Returns:
        np.ndarray: 4x4 transformation matrix.

    Raises:
        AssertionError: If the input format is incorrect.
    """
    assert len(offset_transform) == 2
    assert len(offset_transform[0]) == 3
    assert (
        len(offset_transform[1]) == 4
    ), f"Quaternion must be in xyzw format. Got {offset_transform[1]}"

    trans = offset_transform[0]
    quat = offset_transform[1]
    offset_transform_mat = tra.quaternion_matrix(
        [
            quat[-1],
        ]
        + quat[:-1]
    )
    offset_transform_mat[:3, 3] = trans
    return offset_transform_mat


def get_gripper_info(name: str) -> GripperInfo:
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
    import glob

    registered_grippers = glob.glob(
        f"{Path(__file__).parent.parent}/config/grippers/*.yaml"
    )
    registered_grippers = [Path(gripper).stem for gripper in registered_grippers]
    gripper_name = name

    if gripper_name not in registered_grippers:
        raise ValueError(
            f"Gripper {gripper_name} not registered yet. Available grippers are: {registered_grippers}"
        )

    gripper_module = import_module_from_path(
        f"{Path(__file__).parent.parent}/config/grippers/{gripper_name}.py"
    )
    gripper_config = load_default_gripper_config(gripper_name)

    gripper_model = gripper_module.GripperModel()
    collision_mesh = gripper_model.get_gripper_collision_mesh()
    visual_mesh = gripper_model.get_gripper_visual_mesh()

    if "transform_offset_from_asset_to_graspgen_convention" in gripper_config:
        offset_transform = parse_offset_transform_from_yaml(
            gripper_config["transform_offset_from_asset_to_graspgen_convention"]
        )
    else:
        offset_transform = np.eye(4)

    depth = gripper_config["depth"]

    if "symmetric_antipodal" in gripper_config:
        symmetric_antipodal = gripper_config["symmetric_antipodal"]
    else:
        symmetric_antipodal = False

    if hasattr(gripper_module, "load_control_points"):
        control_points = gripper_module.load_control_points()
    else:
        raise NotImplementedError(
            f"Please implement a load_control_points function for gripper {gripper_name}."
        )

    if hasattr(gripper_module, "load_control_points_for_visualization"):
        control_points_visualization = (
            gripper_module.load_control_points_for_visualization()
        )
    else:
        raise NotImplementedError(
            f"Please implement a load_control_points_for_visualization function for gripper {gripper_name}."
        )

    if hasattr(gripper_module, "get_transform_from_base_link_to_tool_tcp"):
        transform_from_base_link_to_tool_tcp = (
            gripper_module.get_transform_from_base_link_to_tool_tcp()
        )
    else:
        transform_from_base_link_to_tool_tcp = tra.translation_matrix(
            [0, 0, np.abs(depth)]
        )

    collision_mesh.apply_transform(offset_transform)
    visual_mesh.apply_transform(offset_transform)

    gripper_info = GripperInfo(
        gripper_name=gripper_name,
        collision_mesh=collision_mesh,
        visual_mesh=visual_mesh,
        offset_transform=offset_transform,
        depth=depth,
        control_points=control_points,
        control_points_visualization=control_points_visualization,
        symmetric=symmetric_antipodal,
        transform_from_base_link_to_tool_tcp=transform_from_base_link_to_tool_tcp,
    )
    return gripper_info
