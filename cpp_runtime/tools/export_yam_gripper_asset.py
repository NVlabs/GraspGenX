#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the deployed YAM gripper in its cuRobo planning frame."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from yourdfpy import URDF


ROOT_LINK = "gripper"
REFERENCE_FRAME = "franka_gripper"
GRIPPER_LINKS = (
    ROOT_LINK,
    "tip_left",
    "tip_right",
    "right_finger_set",
    "left_finger_set",
    "right_finger_flex",
    "left_finger_flex",
    REFERENCE_FRAME,
)
GRIPPER_JOINTS = (
    "joint7",
    "joint8",
    "right_finger_tip",
    "left_finger_tip",
    "right_finger_flexible",
    "left_finger_flexible",
    "franka_gripper_joint",
)
PAD_LINKS = ("right_finger_flex", "left_finger_flex")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_element(tree: ET.ElementTree, kind: str, name: str) -> ET.Element:
    element = tree.find(f"{kind}[@name='{name}']")
    if element is None:
        raise ValueError(f"Source URDF has no {kind} named {name}")
    return element


def stage_urdf(source_urdf: Path, mesh_root: Path, output: Path) -> dict[str, str]:
    source = ET.parse(source_urdf)
    robot = ET.Element("robot", {"name": "yam_flexible"})
    for name in GRIPPER_LINKS:
        robot.append(copy.deepcopy(require_element(source, "link", name)))
    for name in GRIPPER_JOINTS:
        robot.append(copy.deepcopy(require_element(source, "joint", name)))

    mesh_output = output / "meshes"
    mesh_output.mkdir(parents=True)
    copied: dict[str, str] = {}
    destinations: dict[str, str] = {}
    for mesh in robot.iter("mesh"):
        filename = mesh.get("filename")
        if not filename or "://" in filename:
            raise ValueError(f"Unsupported YAM mesh path: {filename!r}")
        source_mesh = Path(filename)
        if not source_mesh.is_absolute():
            source_mesh = mesh_root / source_mesh
        source_mesh = source_mesh.resolve()
        if not source_mesh.is_file():
            raise FileNotFoundError(source_mesh)
        digest = sha256(source_mesh)
        destination = mesh_output / source_mesh.name
        previous = destinations.get(destination.name)
        if previous is not None and previous != digest:
            raise ValueError(f"Conflicting YAM mesh basenames: {destination.name}")
        if not destination.exists():
            shutil.copyfile(source_mesh, destination)
        destinations[destination.name] = digest
        copied[filename] = digest
        mesh.set("filename", f"meshes/{destination.name}")

    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(
        output / "gripper.urdf", encoding="utf-8", xml_declaration=True
    )
    return copied


def geometry_links(robot: URDF) -> dict[str, str]:
    parents = robot.scene.graph.transforms.parents
    links = set(GRIPPER_LINKS)
    result: dict[str, str] = {}
    for geometry in robot.scene.graph.nodes_geometry:
        node = parents[geometry]
        while node not in links and node in parents:
            node = parents[node]
        if node not in links:
            raise ValueError(f"Cannot resolve link for geometry {geometry}")
        result[geometry] = node
    return result


def merged_mesh(
    robot: URDF, links: dict[str, str], reference_from_root: np.ndarray
) -> trimesh.Trimesh:
    meshes = []
    for geometry in robot.scene.graph.nodes_geometry:
        if links[geometry] == REFERENCE_FRAME:
            continue
        mesh = robot.scene.geometry[geometry].copy()
        mesh.apply_transform(reference_from_root @ robot.scene.graph.get(geometry)[0])
        meshes.append(mesh)
    if not meshes:
        raise ValueError("YAM gripper URDF contains no visual geometry")
    return trimesh.util.concatenate(meshes)


def link_bounds(
    robot: URDF,
    links: dict[str, str],
    reference_from_root: np.ndarray,
    link: str,
) -> np.ndarray:
    vertices = []
    for geometry, geometry_link in links.items():
        if geometry_link != link:
            continue
        mesh = robot.scene.geometry[geometry].copy()
        mesh.apply_transform(reference_from_root @ robot.scene.graph.get(geometry)[0])
        vertices.append(mesh.vertices)
    if not vertices:
        raise ValueError(f"YAM gripper link {link} has no visual geometry")
    points = np.concatenate(vertices)
    return np.stack((points.min(axis=0), points.max(axis=0)))


def sweep_box(
    robot: URDF,
    links: dict[str, str],
    reference_from_root: np.ndarray,
    *,
    base_trim: float,
    tip_trim: float,
) -> tuple[list[float], list[float], float]:
    pads = [
        link_bounds(robot, links, reference_from_root, link) for link in PAD_LINKS
    ]
    pads.sort(key=lambda bounds: float(bounds[:, 0].mean()))
    low = np.asarray(
        [
            pads[0][1, 0],
            min(pad[0, 1] for pad in pads),
            min(pad[0, 2] for pad in pads) + base_trim,
        ]
    )
    high = np.asarray(
        [
            pads[1][0, 0],
            max(pad[1, 1] for pad in pads),
            max(pad[1, 2] for pad in pads) - tip_trim,
        ]
    )
    if np.any(high <= low):
        raise ValueError(f"Invalid YAM sweep box: low={low}, high={high}")
    fingertip_depth = max(float(pad[1, 2]) for pad in pads)
    return (high - low).tolist(), ((high + low) * 0.5).tolist(), fingertip_depth


def set_gripper_position(robot: URDF, position: float) -> None:
    robot.update_cfg({"joint7": position, "joint8": position})


def export_asset(args: argparse.Namespace) -> None:
    source_urdf = args.robot_urdf.resolve()
    mesh_root = args.mesh_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    source_meshes = stage_urdf(source_urdf, mesh_root, output)
    robot = URDF.load(
        str(output / "gripper.urdf"),
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
        force_mesh=False,
        force_collision_mesh=False,
    )
    links = geometry_links(robot)
    root_from_reference = robot.get_transform(REFERENCE_FRAME, ROOT_LINK)
    reference_from_root = np.linalg.inv(root_from_reference)

    set_gripper_position(robot, args.open)
    open_extents, open_offset, fingertip_depth = sweep_box(
        robot,
        links,
        reference_from_root,
        base_trim=args.base_trim,
        tip_trim=args.tip_trim,
    )
    open_mesh = merged_mesh(robot, links, reference_from_root)
    bbox = open_mesh.bounds.tolist()
    open_mesh.export(output / "vis_mesh.obj")
    open_mesh.export(output / "coll_mesh.obj")

    set_gripper_position(robot, 0.5 * (args.open + args.close))
    mid_extents, mid_offset, _ = sweep_box(
        robot,
        links,
        reference_from_root,
        base_trim=args.base_trim,
        tip_trim=args.tip_trim,
    )

    rng = np.random.default_rng(args.seed)
    open_points, _ = trimesh.sample.sample_surface(
        open_mesh, args.surface_points, seed=rng
    )
    set_gripper_position(robot, args.close)
    close_mesh = merged_mesh(robot, links, reference_from_root)
    close_points, _ = trimesh.sample.sample_surface(
        close_mesh, args.surface_points, seed=rng
    )
    (output / "points.json").write_text(
        json.dumps({"open": open_points.tolist(), "close": close_points.tolist()})
        + "\n"
    )

    config = {
        "open": {"joint7": args.open, "joint8": args.open},
        "close": {"joint7": args.close, "joint8": args.close},
        "sim": {"base_joint": "joint7", "mimic_joints": {"joint8": 1.0}},
        "reference_frame": REFERENCE_FRAME,
        "root_link": ROOT_LINK,
        "fingertip": [0.0, 0.0, fingertip_depth],
        "sweep_volume": {
            "extents": open_extents,
            "offset": open_offset,
            "extents2": mid_extents,
            "offset2": mid_offset,
            "tip_trim": args.tip_trim,
            "base_trim": args.base_trim,
        },
        "links": list(GRIPPER_LINKS[:-1]),
        "standoff": [0.0, open_extents[2] * 0.5],
        "symmetric": True,
        "type": "parallel_2f",
        "bbox": bbox,
        "base_rotation": reference_from_root.tolist(),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    artifact_files = (
        "gripper.urdf",
        "config.json",
        "vis_mesh.obj",
        "coll_mesh.obj",
        "points.json",
    )
    manifest = {
        "schema_version": 1,
        "source_robot_urdf": str(source_urdf),
        "source_robot_urdf_sha256": sha256(source_urdf),
        "reference_frame": REFERENCE_FRAME,
        "root_link": ROOT_LINK,
        "reference_from_root": reference_from_root.tolist(),
        "source_mesh_sha256": source_meshes,
        "artifact_sha256": {
            name: sha256(output / name) for name in artifact_files
        },
    }
    (output / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"Exported {output}: reference={REFERENCE_FRAME}, "
        f"open_sweep={open_extents}, fingertip_depth={fingertip_depth:.9f} m"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-urdf", type=Path, required=True)
    parser.add_argument(
        "--mesh-root",
        type=Path,
        required=True,
        help="Directory against which relative URDF mesh paths are resolved",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--open", type=float, default=-0.0475)
    parser.add_argument("--close", type=float, default=0.0)
    parser.add_argument("--base-trim", type=float, default=0.01)
    parser.add_argument("--tip-trim", type=float, default=0.03)
    parser.add_argument("--surface-points", type=int, default=10500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    export_asset(parse_args())
