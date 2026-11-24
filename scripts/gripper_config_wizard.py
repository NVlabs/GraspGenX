# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Interactive GUI tool for creating a GraspGenX config.json for a new gripper.

This script launches a Viser-based web GUI that walks you through configuring
a new gripper for use with GraspGenX. It produces a config.json (with open/close
joint states, sweep volumes, fingertip, bounding box, and gripper type) and copies
the URDF + meshes into assets/x_grippers/<name>/.

Workflow (6 steps in the GUI at http://localhost:<port>):

  Step 1 — Align Base Frame
      Use the ±90° rotate buttons (about world X/Y/Z) to orient the gripper so
      that +Z is the approach direction (parallel to the long dimension of the
      fingers) and +X is the closing direction. The chosen rotation is stored
      as a 4x4 `base_rotation` and applied to every downstream computation. If
      the URDF is already aligned, click "Confirm Alignment" to keep identity.

  Step 2 — Confirm Open Configuration
      The URDF is loaded and displayed. Adjust joint sliders to set the
      fully-open (fingers extended) pose, then click "Confirm Open Configuration".
      If the URDF is not at the correct open pose, exit and edit the URDF first.

  Step 3 — Annotate Open Sweep Volume
      Use extent (x/y/z) and offset (x/y/z) sliders to position a blue wireframe
      bounding box that encloses the inner volume between the fingertips at the
      open state. An initial guess is derived from the moving (finger) link
      geometry. Click "Confirm Open Sweep Volume" when satisfied.

  Step 4 — Annotate Half-Open Sweep Volume
      First, adjust joint sliders to define the fully-closed pose and click
      "Set Close Config & Show Half-Open". The gripper is then displayed at
      the half-open (mid) state. Adjust the orange bounding box to enclose
      the half-open inner volume (the blue open-state box is shown as reference).
      Click "Confirm Half-Open Sweep Volume".

  Step 5 — Review Animation
      The gripper animates continuously from open to close and back, with both
      sweep volume boxes (blue = open, orange = half-open) overlaid. Use
      "Go Back (Re-annotate)" to return to Step 3, or "Looks Good — Proceed
      to Save" to continue.

  Step 6 — Confirm and Save Config
      A JSON preview of the config is shown. Adjust the gripper type dropdown
      (parallel_2f / revolute_2f / revolute_3f) and symmetry checkbox if needed.
      Click "Confirm and Save Config" to write everything to disk.

On save, the following files are written to assets/x_grippers/<name>/:
    gripper.urdf      — copied from the source URDF
    meshes/           — copied mesh directory (if present alongside the URDF)
    config.json       — generated configuration file
    vis_mesh.obj      — merged visual mesh in the open configuration

After saving, visualize the result with:
    python scripts/vis_gripper.py --gripper <name>
    python scripts/vis_gripper.py --gripper <name> --show-sweep-volume --animate

Usage:
    # Basic — will prompt for gripper name interactively
    python scripts/gripper_config_wizard.py --urdf /path/to/gripper.urdf

    # Provide name on the command line
    python scripts/gripper_config_wizard.py --urdf /path/to/gripper.urdf --name my_gripper

    # Custom port
    python scripts/gripper_config_wizard.py --urdf /path/to/gripper.urdf --name my_gripper --port 8081
"""

import argparse
import copy
import json
import os
import shutil
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import yourdfpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graspgenx import get_gripper_descriptions_assets
from graspgenx.utils.viser_utils import (
    create_visualizer,
    make_frame,
    visualize_bbox,
    visualize_mesh,
)

# Last-resort fallback if the gripper_descriptions resolver fails. Normal
# operation goes through ``get_gripper_descriptions_assets()``.
X_GRIPPERS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "x_grippers",
)


def _default_output_root() -> str:
    """Resolve the wizard's default output directory.

    Returns the gripper_descriptions ``assets/x_grippers`` path (resolved via
    ``$GRASPGENX_GRIPPER_CFG_DIR`` or auto-clone), falling back to the repo's
    own ``assets/x_grippers/`` if the resolver raises.
    """
    try:
        return str(get_gripper_descriptions_assets())
    except Exception as exc:  # noqa: BLE001
        print(
            f"[warn] could not resolve gripper_descriptions assets dir ({exc}); "
            f"falling back to {X_GRIPPERS_PATH}"
        )
        return X_GRIPPERS_PATH


# ---------------------------------------------------------------------------
# URDF helpers
# ---------------------------------------------------------------------------


def load_urdf(urdf_path: str) -> yourdfpy.URDF:
    return yourdfpy.URDF.load(
        urdf_path,
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
        force_mesh=False,
        force_collision_mesh=False,
    )


def get_joint_names(robot: yourdfpy.URDF) -> List[str]:
    """Return actuated joint names (non-fixed and not mimic).

    Mimic joints are driven by their master joint: yourdfpy recomputes them
    on every ``update_cfg`` call, so they must not get their own sliders or
    entries in the saved open/close joint states.
    """
    return list(robot.actuated_joint_names)


def _top_level_moving_joints(robot: yourdfpy.URDF, include_mimic: bool) -> List[str]:
    """Moving joints that have no moving joint between them and the root.

    Each such joint starts an independent kinematic branch off the palm,
    i.e. roughly one finger (or one side of a linkage).
    """
    joints = [j for j in robot.robot.joints if j.type != "fixed"]
    if not include_mimic:
        joints = [j for j in joints if j.mimic is None]
    moving = {j.name for j in joints}
    parent_joint = {j.child: j for j in robot.robot.joints}

    top_level = []
    for j in joints:
        link = j.parent
        has_moving_ancestor = False
        while link in parent_joint:
            ancestor = parent_joint[link]
            if ancestor.name in moving:
                has_moving_ancestor = True
                break
            link = ancestor.parent
        if not has_moving_ancestor:
            top_level.append(j.name)
    return top_level


def infer_gripper_type(robot: yourdfpy.URDF) -> Tuple[str, bool]:
    """Infer the GraspGenX gripper type and symmetry from the URDF kinematics.

    Returns:
        (gripper_type, symmetric), where gripper_type is one of
        "parallel_2f", "revolute_2f" or "revolute_3f". The result is only the
        initial value of the Step 6 dropdown/checkbox; users can override it.
    """
    actuated = [robot.joint_map[n] for n in robot.actuated_joint_names]
    if not actuated:
        return "parallel_2f", True

    if all(j.type == "prismatic" for j in actuated):
        gtype = "parallel_2f"
    elif len(_top_level_moving_joints(robot, include_mimic=False)) <= 2:
        gtype = "revolute_2f"
    else:
        gtype = "revolute_3f"

    # Two-finger grippers are symmetric when both jaws move (e.g. via a mimic
    # joint); a single moving jaw against a fixed one (e.g. Spot) is not.
    symmetric = (
        gtype != "revolute_3f"
        and len(_top_level_moving_joints(robot, include_mimic=True)) >= 2
    )
    return gtype, symmetric


def get_link_names(robot: yourdfpy.URDF) -> List[str]:
    return [l.name for l in robot.robot.links]


def get_link_colors(gripper_name: str, num_links: int) -> List[List[int]]:
    base_color = [80, 80, 80]
    finger_color = [50, 180, 50]
    default_color = [120, 120, 120]
    colors = []
    for i in range(num_links):
        if i <= 1:
            colors.append(base_color)
        elif gripper_name.startswith("parallel") or gripper_name.startswith("revolute"):
            colors.append(finger_color if i >= 2 else default_color)
        else:
            colors.append(default_color)
    return colors


def visualize_gripper(
    vis, robot, js_cfg, gripper_name, name_prefix="gripper", base_T=None
):
    """Render gripper with given joint config.

    If `base_T` is provided (4x4), it is left-multiplied onto every link's
    world transform so the entire gripper appears rotated by `base_T` about
    the world origin.
    """
    robot.update_cfg(js_cfg)
    scene = robot.scene
    geometry_names = list(scene.geometry.keys())
    colors = get_link_colors(gripper_name, len(geometry_names))
    if base_T is None:
        base_T = np.eye(4)

    for i, geom_name in enumerate(geometry_names):
        mesh = scene.geometry[geom_name]
        local_tf = scene.graph.get(geom_name)[0]
        world_tf = base_T @ local_tf
        transformed = mesh.copy()
        transformed.apply_transform(world_tf)
        visualize_mesh(
            vis,
            f"{name_prefix}/link_{i}",
            transformed,
            color=colors[i] if i < len(colors) else [120, 120, 120],
        )


def compute_gripper_bbox(
    robot: yourdfpy.URDF, js_cfg: Dict, base_T: Optional[np.ndarray] = None
) -> Tuple[List, List]:
    """Compute axis-aligned bounding box of the gripper in a given joint config.

    `base_T` (optional 4x4) is left-multiplied onto link transforms so the
    bbox is reported in the rotated frame.
    """
    robot.update_cfg(js_cfg)
    scene = robot.scene
    if base_T is None:
        base_T = np.eye(4)
    all_verts = []
    for geom_name in scene.geometry:
        mesh = scene.geometry[geom_name]
        tf = base_T @ scene.graph.get(geom_name)[0]
        transformed = mesh.copy()
        transformed.apply_transform(tf)
        all_verts.append(transformed.vertices)
    all_verts = np.concatenate(all_verts, axis=0)
    bbox_min = all_verts.min(axis=0).tolist()
    bbox_max = all_verts.max(axis=0).tolist()
    return bbox_min, bbox_max


def detect_finger_geoms(
    robot: yourdfpy.URDF, base_js: Dict, eps: float = 0.01
) -> List[str]:
    """Identify scene geometries that move when any actuated joint is perturbed.

    Returns the list of geometry names corresponding to moving (finger) links.
    Falls back to all geometries if nothing appears to move.
    """
    robot.update_cfg(base_js)
    scene = robot.scene
    base_poses = {g: scene.graph.get(g)[0].copy() for g in scene.geometry}

    finger_geoms: set = set()
    joint_names = get_joint_names(robot)
    for jn in joint_names:
        joint_obj = next(j for j in robot.robot.joints if j.name == jn)
        lo = joint_obj.limit.lower if joint_obj.limit else -3.14
        hi = joint_obj.limit.upper if joint_obj.limit else 3.14
        cur = base_js.get(jn, 0.0)
        # Pick a perturbation that stays in-range
        delta = eps if cur + eps <= hi else -eps if cur - eps >= lo else (hi - lo) * 0.1
        if delta == 0.0:
            continue
        perturbed = dict(base_js)
        perturbed[jn] = cur + delta
        robot.update_cfg(perturbed)
        for g in scene.geometry:
            new_pose = robot.scene.graph.get(g)[0]
            if not np.allclose(base_poses[g], new_pose, atol=1e-6):
                finger_geoms.add(g)
    # Restore base config
    robot.update_cfg(base_js)
    if not finger_geoms:
        return list(scene.geometry.keys())
    return sorted(finger_geoms)


def compute_geom_bbox(
    robot: yourdfpy.URDF, geom_name: str, base_T: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Bounding box of a single scene geometry under the current joint config."""
    scene = robot.scene
    mesh = scene.geometry[geom_name]
    tf = scene.graph.get(geom_name)[0]
    if base_T is not None:
        tf = base_T @ tf
    verts = trimesh.transformations.transform_points(mesh.vertices, tf)
    return verts.min(axis=0), verts.max(axis=0)


def estimate_inner_sweep_volume(
    robot: yourdfpy.URDF,
    js_cfg: Dict,
    finger_geoms: List[str],
    base_T: Optional[np.ndarray] = None,
) -> Tuple[List[float], List[float], int]:
    """Estimate the inner sweep volume between the fingers at the given config.

    Returns (extents, offset, closing_axis):
        - closing_axis = axis index (0/1/2) along which fingers separate the most
        - closing extent = gap between innermost finger surfaces
        - other extents = union of finger bboxes along non-closing axes
        - offset = center of that inner box
    """
    robot.update_cfg(js_cfg)
    bboxes = [compute_geom_bbox(robot, g, base_T=base_T) for g in finger_geoms]

    centroids = np.array([(b[0] + b[1]) / 2 for b in bboxes])
    spread = centroids.max(axis=0) - centroids.min(axis=0)
    closing_axis = int(np.argmax(spread))

    # Sort fingers by centroid along the closing axis
    order = np.argsort(centroids[:, closing_axis])
    left_bbox = bboxes[order[0]]  # smallest centroid → "left" finger
    right_bbox = bboxes[order[-1]]  # largest centroid → "right" finger

    inner_min = float(left_bbox[1][closing_axis])  # max of the left finger
    inner_max = float(right_bbox[0][closing_axis])  # min of the right finger
    if inner_max <= inner_min:
        # Fingers overlap (no gap to enclose): fall back to centroid spread
        inner_min = float(centroids[order[0], closing_axis])
        inner_max = float(centroids[order[-1], closing_axis])

    # Union of finger bboxes for the non-closing axes
    union_min = np.min([b[0] for b in bboxes], axis=0)
    union_max = np.max([b[1] for b in bboxes], axis=0)

    sv_min = union_min.copy()
    sv_max = union_max.copy()
    sv_min[closing_axis] = inner_min
    sv_max[closing_axis] = inner_max

    extents = (sv_max - sv_min).tolist()
    offset = ((sv_min + sv_max) / 2).tolist()
    return extents, offset, closing_axis


def interpolate_joint_states(open_js: Dict, close_js: Dict, alpha: float) -> Dict:
    js = {}
    for k in open_js:
        o = open_js[k]
        c = close_js[k]
        js[k] = o + (c - o) * alpha
    return js


def get_traj_js(open_js: Dict, close_js: Dict, num_steps: int = 20) -> List[Dict]:
    return [
        interpolate_joint_states(open_js, close_js, s / num_steps)
        for s in range(num_steps + 1)
    ]


def export_merged_mesh(
    robot: yourdfpy.URDF,
    js_cfg: Dict,
    out_path: str,
    base_T: Optional[np.ndarray] = None,
):
    """Merge all visual geometry into a single OBJ.

    If `base_T` is given, every link transform is left-multiplied so the
    exported mesh is in the rotated (canonical) frame.
    """
    robot.update_cfg(js_cfg)
    scene = robot.scene
    if base_T is None:
        base_T = np.eye(4)
    meshes = []
    for geom_name in scene.geometry:
        mesh = scene.geometry[geom_name]
        tf = base_T @ scene.graph.get(geom_name)[0]
        m = mesh.copy()
        m.apply_transform(tf)
        meshes.append(m)
    merged = trimesh.util.concatenate(meshes)
    merged.export(out_path)


# ---------------------------------------------------------------------------
# Main wizard
# ---------------------------------------------------------------------------


class ConfigWizard:
    """Step-by-step wizard state machine."""

    STEPS = [
        "align_base",  # Step 1: rotate the base link so +Z = approach, +X = closing
        "open_confirm",  # Step 2: confirm open configuration
        "annotate_open_sv",  # Step 3: annotate open sweep volume
        "annotate_half_sv",  # Step 4: annotate half-open sweep volume
        "animate_review",  # Step 5: review animation
        "save",  # Step 6: save
    ]

    def __init__(
        self,
        vis,
        robot,
        urdf_path,
        gripper_name,
        port,
        output_root: Optional[str] = None,
    ):
        self.vis = vis
        self.robot = robot
        self.urdf_path = urdf_path
        self.gripper_name = gripper_name
        self.port = port
        # Parent directory under which `<gripper_name>/` will be created. If
        # not specified, resolves to the gripper_descriptions assets dir
        # ($GRASPGENX_GRIPPER_CFG_DIR or auto-cloned ext/), falling back to
        # the repo's assets/x_grippers/ only if that resolver fails.
        if output_root:
            self.output_root = os.path.abspath(output_root)
        else:
            self.output_root = os.path.abspath(_default_output_root())

        self.joint_names = get_joint_names(robot)
        self.link_names = get_link_names(robot)

        # Initialise joint configs to the extremes of each actuated joint's
        # limits — open at the upper limit, close at the lower limit. This
        # matches the typical convention for prismatic and most revolute
        # grippers (open = fingers extended = upper joint value). Joints with
        # no limits fall back to 0.
        def _limit(j_name: str, which: str) -> float:
            joint = next(j for j in robot.robot.joints if j.name == j_name)
            if joint.limit is None:
                return 0.0
            if which == "upper":
                return joint.limit.upper if joint.limit.upper is not None else 0.0
            return joint.limit.lower if joint.limit.lower is not None else 0.0

        self.open_js = {j: _limit(j, "upper") for j in self.joint_names}
        self.close_js = {j: _limit(j, "lower") for j in self.joint_names}

        # Sweep volume parameters (will be set interactively)
        self.sv_extents = [0.08, 0.02, 0.04]
        self.sv_offset = [0.0, 0.0, 0.10]
        self.sv2_extents = [0.04, 0.02, 0.04]
        self.sv2_offset = [0.0, 0.0, 0.10]

        # Closing axis (0=x, 1=y, 2=z) — set when annotating the open SV
        self.closing_axis = 0

        # Cache of geometries that move with joints (computed lazily)
        self._finger_geoms: Optional[List[str]] = None

        # Cache for loaded reference grippers (franka_panda, robotiq_2f_85, robotiq_3f, surge_hand)
        self._ref_grippers: Optional[List] = None

        # Base-link rotation offset (applied as world-frame pre-multiply on all
        # gripper transforms). Identity by default; the user adjusts it in
        # Step 1 to align +Z with the approach axis (parallel to the fingers'
        # long dimension) and +X with the closing axis.
        self.base_rotation: np.ndarray = np.eye(4)

        # Fingertip (derived from sweep volume offset)
        self.fingertip = [0.0, 0.0, 0.10]

        self.step_idx = 0
        self._animating = False
        self._gui_handles = []
        self._gizmo_handle = None
        self._box_handle = None

    @property
    def current_step(self):
        return self.STEPS[self.step_idx]

    # ------------------------------------------------------------------
    # GUI helpers
    # ------------------------------------------------------------------

    def _clear_gui(self):
        for h in self._gui_handles:
            h.remove()
        self._gui_handles.clear()
        # Re-add the persistent wizard title at the top of every step's panel
        title = self.vis.gui.add_markdown("# 🛠 Gripper Wizard")
        self._gui_handles.append(title)

    def _add_status(self, text: str):
        md = self.vis.gui.add_markdown(f"### {text}")
        self._gui_handles.append(md)
        return md

    def _add_info(self, text: str):
        md = self.vis.gui.add_markdown(text)
        self._gui_handles.append(md)
        return md

    def _refresh_gripper(self, js=None, prefix="gripper"):
        if js is None:
            js = self.open_js
        visualize_gripper(
            self.vis,
            self.robot,
            js,
            self.gripper_name,
            name_prefix=prefix,
            base_T=self.base_rotation,
        )

    def _add_prev_button(self, target_step: str) -> None:
        """Add a '← Previous Step' button that jumps to the named step."""
        btn = self.vis.gui.add_button("← Previous Step", color="yellow")
        self._gui_handles.append(btn)

        @btn.on_click
        def _on_prev(_ev, _target=target_step):
            self._animating = False  # stop any running review animation
            self.step_idx = self.STEPS.index(_target)
            self._run_step()

    # ------------------------------------------------------------------
    # Step 1: Align base frame
    # ------------------------------------------------------------------

    def _rotation_axis_matrix(self, axis: str, deg: float) -> np.ndarray:
        """4x4 rotation matrix around the world axis ('x'/'y'/'z') by `deg` degrees."""
        direction = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}[
            axis
        ]
        return trimesh.transformations.rotation_matrix(np.deg2rad(deg), direction)

    def _draw_world_axis_labels(self, h: float = 0.20) -> None:
        """Annotate the world axes drawn by `make_frame` with +X / +Y / +Z text tags."""
        offset = h + 0.02
        for name, pos in (
            ("+X", (offset, 0.0, 0.0)),
            ("+Y", (0.0, offset, 0.0)),
            ("+Z", (0.0, 0.0, offset)),
        ):
            try:
                self.vis.scene.add_label(
                    f"world/label_{name}",
                    text=name,
                    wxyz=(1.0, 0.0, 0.0, 0.0),
                    position=pos,
                    font_screen_scale=4.0,
                    anchor="bottom-center",
                )
            except Exception:
                try:
                    self.vis.scene.add_label(
                        f"world/label_{name}",
                        text=name,
                        wxyz=(1.0, 0.0, 0.0, 0.0),
                        position=pos,
                    )
                except Exception:
                    pass

    def _resolve_ref_grippers_root(self) -> Optional[str]:
        """Find the x_grippers directory containing the 4 reference grippers."""
        if os.path.isdir(os.path.join(self.output_root, "franka_panda")):
            return self.output_root
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.normpath(
            os.path.join(
                repo_root,
                "..",
                "gripper_descriptions",
                "gripper_descriptions",
                "assets",
                "x_grippers",
            )
        )
        if os.path.isdir(os.path.join(candidate, "franka_panda")):
            return candidate
        return None

    def _load_reference_grippers(self) -> List:
        """Load the 4 reference grippers lazily (cached after first call).

        Each entry is a dict with keys:
          name, robot, open_js, close_js, base_rot,
          sv_open_extents, sv_open_offset, sv_half_extents, sv_half_offset
        """
        if self._ref_grippers is not None:
            return self._ref_grippers
        self._ref_grippers = []
        root = self._resolve_ref_grippers_root()
        if root is None:
            print(
                "[warn] Reference grippers not found; skipping side-by-side reference display."
            )
            return self._ref_grippers
        for name in ["franka_panda", "robotiq_2f_85", "robotiq_3f", "surge_hand"]:
            urdf_path = os.path.join(root, name, "gripper.urdf")
            cfg_path = os.path.join(root, name, "config.json")
            if not os.path.isfile(urdf_path) or not os.path.isfile(cfg_path):
                print(
                    f"[warn] Reference gripper '{name}' not found at {root}; skipping."
                )
                continue
            try:
                ref_robot = load_urdf(urdf_path)
                with open(cfg_path) as f:
                    cfg = json.load(f)
                sv = cfg.get("sweep_volume", {})
                base_rot = (
                    np.array(cfg["base_rotation"])
                    if "base_rotation" in cfg
                    else np.eye(4)
                )
                self._ref_grippers.append(
                    {
                        "name": name,
                        "robot": ref_robot,
                        "open_js": cfg.get("open", {}),
                        "close_js": cfg.get("close", {}),
                        "base_rot": base_rot,
                        "sv_open_extents": sv.get("extents"),
                        "sv_open_offset": sv.get("offset"),
                        "sv_half_extents": sv.get("extents2"),
                        "sv_half_offset": sv.get("offset2"),
                    }
                )
                print(f"[ref] Loaded reference gripper: {name}")
            except Exception as exc:
                print(f"[warn] Failed to load reference gripper '{name}': {exc}")
        return self._ref_grippers

    def _draw_reference_grippers(self, mode: str = "open") -> None:
        """Draw the 4 reference grippers side-by-side beside the user's gripper.

        mode: 'open'  — fully open joint config + open SV box (blue)
              'close' — fully closed joint config, no SV box
              'half'  — half-open joint config + half SV box (orange)
        """
        refs = self._load_reference_grippers()
        x_positions = [-0.45, -0.15, 0.15, 0.45]
        y_offset = -0.40
        mode_label = {
            "open": "fully open",
            "close": "fully closed",
            "half": "half open",
        }.get(mode, mode)

        for idx, ref in enumerate(refs):
            name = ref["name"]
            ref_robot = ref["robot"]
            open_js = ref["open_js"]
            close_js = ref["close_js"]
            base_rot = ref["base_rot"]

            x_pos = x_positions[idx] if idx < len(x_positions) else idx * 0.30 - 0.45
            ref_offset = np.array([x_pos, y_offset, 0.0])

            if mode == "open":
                js = open_js
            elif mode == "close":
                js = close_js
            else:
                js = interpolate_joint_states(open_js, close_js, 0.5)

            T_translate = np.eye(4)
            T_translate[:3, 3] = ref_offset
            T = T_translate @ base_rot

            prefix = f"ref_{name}"
            try:
                visualize_gripper(
                    self.vis, ref_robot, js, name, name_prefix=prefix, base_T=T
                )
            except Exception as exc:
                print(f"[warn] Could not render reference gripper '{name}': {exc}")
                continue

            # Draw sweep volume box for open and half modes
            if mode == "open":
                sv_extents = ref.get("sv_open_extents")
                sv_offset = ref.get("sv_open_offset")
                sv_color = [0, 100, 255]
            elif mode == "half":
                sv_extents = ref.get("sv_half_extents")
                sv_offset = ref.get("sv_half_offset")
                sv_color = [255, 165, 0]
            else:
                sv_extents = None
                sv_offset = None

            if sv_extents is not None and sv_offset is not None:
                # sv_offset is in the canonical (base_rot-applied) frame; translate to ref position
                world_sv_center = ref_offset + base_rot[:3, :3] @ np.array(sv_offset)
                T_sv = np.eye(4)
                T_sv[:3, 3] = world_sv_center
                visualize_bbox(
                    self.vis,
                    f"{prefix}/sv_box",
                    np.array(sv_extents),
                    T=T_sv,
                    color=sv_color,
                )

            label_text = f"{name}\n({mode_label})"
            try:
                self.vis.scene.add_label(
                    f"{prefix}/label",
                    text=label_text,
                    wxyz=(1.0, 0.0, 0.0, 0.0),
                    position=(float(x_pos), float(y_offset), 0.25),
                    font_screen_scale=1.4,
                    anchor="bottom-center",
                )
            except Exception:
                try:
                    self.vis.scene.add_label(
                        f"{prefix}/label",
                        text=name,
                        wxyz=(1.0, 0.0, 0.0, 0.0),
                        position=(float(x_pos), float(y_offset), 0.25),
                    )
                except Exception:
                    pass

    def _step_align_base(self):
        self._clear_gui()
        self.vis.scene.reset()
        # Big world frame so the user can read off the axes while rotating
        make_frame(self.vis, "world", h=0.20, radius=0.003)
        self._draw_world_axis_labels(h=0.20)

        self._add_status("Step 1/6: Align Base Frame")
        self._add_info(
            "Rotate the gripper so that:\n"
            "- **+Z** points along the **approach** direction (parallel to the long dimension of the fingers)\n"
            "- **+X** points along the **closing** direction\n\n"
            "Use the buttons below to apply ±90° rotations about the world X/Y/Z axes, "
            "or use the **Custom Rotation** folder to apply an arbitrary angle. "
            "The world axes shown are red=X, green=Y, blue=Z.\n\n"
            "**Reference grippers are shown on the side** — find the one closest to your "
            "gripper's type (parallel jaw, 3-finger, dexterous hand) and aim for a similar "
            "base-frame alignment.\n\n"
            "Click **Confirm Alignment** when done. If the URDF is already aligned, just confirm — "
            "the rotation defaults to identity."
        )

        def _rerender():
            # Clear previous gripper geometry and redraw with the current base_rotation
            self.vis.scene.reset()
            make_frame(self.vis, "world", h=0.20, radius=0.003)
            self._draw_world_axis_labels(h=0.20)
            self._draw_reference_grippers("open")
            self._refresh_gripper(self.open_js)

        def _make_rot_button(label: str, axis: str, deg: float):
            btn = self.vis.gui.add_button(label)
            self._gui_handles.append(btn)

            @btn.on_click
            def _on_click(_ev, _ax=axis, _d=deg):
                R = self._rotation_axis_matrix(_ax, _d)
                # World-frame rotation: left-multiply
                self.base_rotation = R @ self.base_rotation
                _rerender()

        with self.vis.gui.add_folder("Rotate ±90° about world axes") as folder:
            self._gui_handles.append(folder)
            _make_rot_button("Rotate +90° about X", "x", 90.0)
            _make_rot_button("Rotate -90° about X", "x", -90.0)
            _make_rot_button("Rotate +90° about Y", "y", 90.0)
            _make_rot_button("Rotate -90° about Y", "y", -90.0)
            _make_rot_button("Rotate +90° about Z", "z", 90.0)
            _make_rot_button("Rotate -90° about Z", "z", -90.0)

        with self.vis.gui.add_folder("Custom Rotation") as folder:
            self._gui_handles.append(folder)
            custom_axis = self.vis.gui.add_dropdown(
                "Axis", options=["X", "Y", "Z"], initial_value="X"
            )
            self._gui_handles.append(custom_axis)
            custom_deg = self.vis.gui.add_slider(
                "Angle (degrees)", min=-180.0, max=180.0, step=1.0, initial_value=90.0
            )
            self._gui_handles.append(custom_deg)
            apply_custom_btn = self.vis.gui.add_button("Apply Custom Rotation")
            self._gui_handles.append(apply_custom_btn)

            @apply_custom_btn.on_click
            def _on_custom_rot(_ev):
                R = self._rotation_axis_matrix(
                    custom_axis.value.lower(), custom_deg.value
                )
                self.base_rotation = R @ self.base_rotation
                _rerender()

        reset_btn = self.vis.gui.add_button("Reset to Identity", color="yellow")
        self._gui_handles.append(reset_btn)

        @reset_btn.on_click
        def _on_reset(_ev):
            self.base_rotation = np.eye(4)
            _rerender()

        confirm_btn = self.vis.gui.add_button("Confirm Alignment", color="green")
        self._gui_handles.append(confirm_btn)

        @confirm_btn.on_click
        def _on_confirm(_ev):
            print(f"[Step 1] Base rotation confirmed:\n{self.base_rotation}")
            self._next_step()

        # Initial render (reference + user's gripper)
        self._draw_reference_grippers("open")
        self._refresh_gripper(self.open_js)

    # ------------------------------------------------------------------
    # Step 2: Confirm open configuration
    # ------------------------------------------------------------------

    def _step_open_confirm(self):
        self._clear_gui()
        self.vis.scene.reset()
        make_frame(self.vis, "world", h=0.10, radius=0.002)
        self._draw_reference_grippers("open")

        self._add_status("Step 2/6: Confirm Open Configuration")
        self._add_info(
            f"Gripper: **{self.gripper_name}**\n\n"
            f"Joints: {', '.join(self.joint_names)}\n\n"
            "Adjust joint sliders to set the **fully-open** position.\n\n"
            "The gripper should have fingers fully extended / spread apart."
        )

        # Joint sliders
        self._joint_sliders = {}
        with self.vis.gui.add_folder("Joint Configuration (Open)") as folder:
            self._gui_handles.append(folder)
            for jname in self.joint_names:
                joint_obj = next(j for j in self.robot.robot.joints if j.name == jname)
                lo = joint_obj.limit.lower if joint_obj.limit else -3.14
                hi = joint_obj.limit.upper if joint_obj.limit else 3.14
                slider = self.vis.gui.add_slider(
                    jname,
                    min=lo,
                    max=hi,
                    step=0.001,
                    initial_value=self.open_js.get(jname, 0.0),
                )
                self._joint_sliders[jname] = slider

                @slider.on_update
                def _on_slider(_ev, _jn=jname, _sl=slider):
                    self.open_js[_jn] = _sl.value
                    self._refresh_gripper(self.open_js)

        self._refresh_gripper(self.open_js)

        btn = self.vis.gui.add_button("Confirm Open Configuration", color="green")
        self._gui_handles.append(btn)

        @btn.on_click
        def _on_confirm(_ev):
            # Capture open joint state
            for jn, sl in self._joint_sliders.items():
                self.open_js[jn] = sl.value
            print(f"[Step 2] Open joint state confirmed: {self.open_js}")
            self._next_step()

        self._add_prev_button("align_base")

    # ------------------------------------------------------------------
    # Step 2: Annotate open sweep volume
    # ------------------------------------------------------------------

    def _step_annotate_open_sv(self):
        self._clear_gui()
        self.vis.scene.reset()
        make_frame(self.vis, "world", h=0.10, radius=0.002)
        self._draw_reference_grippers("open")
        self._refresh_gripper(self.open_js)

        self._add_status("Step 3/6: Annotate Open Sweep Volume")
        self._add_info(
            "Use the **sliders** to position and size a 3D bounding box that encloses "
            "the inner volume swept by the fingers in the **open** configuration.\n\n"
            "The box should tightly wrap the space between the fingertips."
        )

        # Initial guess: derive from where the moving (finger) links live in the
        # open configuration so the box already wraps the inter-finger volume.
        if self._finger_geoms is None:
            self._finger_geoms = detect_finger_geoms(self.robot, self.open_js)
        try:
            extents, offset, closing_axis = estimate_inner_sweep_volume(
                self.robot,
                self.open_js,
                self._finger_geoms,
                base_T=self.base_rotation,
            )
            self.sv_extents = extents
            self.sv_offset = offset
            self.closing_axis = closing_axis
            print(
                f"[Step 3] Initial open SV from finger geometry — "
                f"closing_axis={closing_axis}, extents={extents}, offset={offset}"
            )
        except Exception as e:
            # Fallback to whole-gripper bbox if finger detection fails
            print(
                f"[Step 3] Finger-based init failed ({e}); falling back to gripper bbox."
            )
            bbox_min, bbox_max = compute_gripper_bbox(
                self.robot, self.open_js, base_T=self.base_rotation
            )
            self.sv_offset = [(bbox_min[i] + bbox_max[i]) / 2 for i in range(3)]
            self.sv_extents = [bbox_max[i] - bbox_min[i] for i in range(3)]
            self.sv_extents[2] *= 0.4
            self.closing_axis = int(np.argmax(self.sv_extents))

        with self.vis.gui.add_folder("Sweep Volume (Open) — Extents") as folder:
            self._gui_handles.append(folder)
            ext_x = self.vis.gui.add_slider(
                "extent_x",
                min=0.001,
                max=0.5,
                step=0.001,
                initial_value=self.sv_extents[0],
            )
            ext_y = self.vis.gui.add_slider(
                "extent_y",
                min=0.001,
                max=0.5,
                step=0.001,
                initial_value=self.sv_extents[1],
            )
            ext_z = self.vis.gui.add_slider(
                "extent_z",
                min=0.001,
                max=0.5,
                step=0.001,
                initial_value=self.sv_extents[2],
            )

        with self.vis.gui.add_folder("Sweep Volume (Open) — Offset") as folder:
            self._gui_handles.append(folder)
            off_x = self.vis.gui.add_slider(
                "offset_x",
                min=-0.3,
                max=0.3,
                step=0.001,
                initial_value=self.sv_offset[0],
            )
            off_y = self.vis.gui.add_slider(
                "offset_y",
                min=-0.3,
                max=0.3,
                step=0.001,
                initial_value=self.sv_offset[1],
            )
            off_z = self.vis.gui.add_slider(
                "offset_z",
                min=-0.3,
                max=0.3,
                step=0.001,
                initial_value=self.sv_offset[2],
            )

        sliders = [ext_x, ext_y, ext_z, off_x, off_y, off_z]

        def _update_sv_box():
            self.sv_extents = [ext_x.value, ext_y.value, ext_z.value]
            self.sv_offset = [off_x.value, off_y.value, off_z.value]
            tf = np.eye(4)
            tf[:3, 3] = self.sv_offset
            visualize_bbox(
                self.vis,
                "sweep_volume_open",
                np.array(self.sv_extents),
                T=tf,
                color=[0, 100, 255],
            )

        for sl in sliders:

            @sl.on_update
            def _on_update(_ev, _fn=_update_sv_box):
                _fn()

        _update_sv_box()

        btn = self.vis.gui.add_button("Confirm Open Sweep Volume", color="green")
        self._gui_handles.append(btn)

        @btn.on_click
        def _on_confirm(_ev):
            self.sv_extents = [ext_x.value, ext_y.value, ext_z.value]
            self.sv_offset = [off_x.value, off_y.value, off_z.value]
            print(
                f"[Step 3] Open sweep volume — extents: {self.sv_extents}, offset: {self.sv_offset}"
            )
            self._next_step()

        self._add_prev_button("open_confirm")

    # ------------------------------------------------------------------
    # Step 4: Annotate half-open sweep volume
    # ------------------------------------------------------------------

    def _step_annotate_half_sv(self):
        self._clear_gui()
        self.vis.scene.reset()
        make_frame(self.vis, "world", h=0.10, radius=0.002)
        self._draw_reference_grippers("close")

        self._add_status("Step 4/6: Annotate Half-Open Sweep Volume")
        self._add_info(
            "Now set the **close** joint configuration, then annotate the "
            "sweep volume at the **half-open** (mid) state.\n\n"
            "First, adjust joints to the **fully-closed** pose, then click "
            "'Set Close Config'. The gripper will be shown at the **half-open** state."
        )

        # Close joint sliders
        self._close_sliders = {}
        with self.vis.gui.add_folder("Joint Configuration (Close)") as folder:
            self._gui_handles.append(folder)
            for jname in self.joint_names:
                joint_obj = next(j for j in self.robot.robot.joints if j.name == jname)
                lo = joint_obj.limit.lower if joint_obj.limit else -3.14
                hi = joint_obj.limit.upper if joint_obj.limit else 3.14
                slider = self.vis.gui.add_slider(
                    jname,
                    min=lo,
                    max=hi,
                    step=0.001,
                    initial_value=self.close_js.get(jname, 0.0),
                )
                self._close_sliders[jname] = slider

                @slider.on_update
                def _on_slider(_ev, _jn=jname, _sl=slider):
                    self.close_js[_jn] = _sl.value
                    self._refresh_gripper(self.close_js)

        self._refresh_gripper(self.close_js)

        set_close_btn = self.vis.gui.add_button(
            "Set Close Config & Show Half-Open", color="blue"
        )
        self._gui_handles.append(set_close_btn)

        @set_close_btn.on_click
        def _on_set_close(_ev):
            for jn, sl in self._close_sliders.items():
                self.close_js[jn] = sl.value
            print(f"[Step 4] Close joint state: {self.close_js}")
            self._show_half_open_annotation()

        self._add_prev_button("annotate_open_sv")

    def _show_half_open_annotation(self):
        # Remove close-config GUI elements but keep the status
        self._clear_gui()
        self.vis.scene.reset()
        make_frame(self.vis, "world", h=0.10, radius=0.002)
        self._draw_reference_grippers("half")

        self._add_status("Step 4/6: Annotate Half-Open Sweep Volume")
        self._add_info(
            "Adjust the bounding box for the **half-open** (mid) configuration."
        )

        half_js = interpolate_joint_states(self.open_js, self.close_js, 0.5)
        self._refresh_gripper(half_js)

        # Show open SV as reference (transparent blue)
        tf_open = np.eye(4)
        tf_open[:3, 3] = self.sv_offset
        visualize_bbox(
            self.vis,
            "sv_open_ref",
            np.array(self.sv_extents),
            T=tf_open,
            color=[0, 100, 255],
        )

        # Half-open SV initial guess: inherit the open SV the user just
        # confirmed in Step 3, but halve extent_x (the closing-axis dimension
        # after base_rotation aligns +X with the closing direction). This
        # keeps the offset identical to the open SV so the user only has to
        # tweak from a known-good starting point.
        self.sv2_extents = [
            self.sv_extents[0] * 0.5,
            self.sv_extents[1],
            self.sv_extents[2],
        ]
        self.sv2_offset = list(self.sv_offset)
        self._refresh_gripper(half_js)
        print(
            f"[Step 4] Initial half SV inherited from open SV (extent_x halved) — "
            f"extents={self.sv2_extents}, offset={self.sv2_offset}"
        )

        with self.vis.gui.add_folder("Sweep Volume V2 (Half) — Extents") as folder:
            self._gui_handles.append(folder)
            ext_x = self.vis.gui.add_slider(
                "extent2_x",
                min=0.001,
                max=0.5,
                step=0.001,
                initial_value=self.sv2_extents[0],
            )
            ext_y = self.vis.gui.add_slider(
                "extent2_y",
                min=0.001,
                max=0.5,
                step=0.001,
                initial_value=self.sv2_extents[1],
            )
            ext_z = self.vis.gui.add_slider(
                "extent2_z",
                min=0.001,
                max=0.5,
                step=0.001,
                initial_value=self.sv2_extents[2],
            )

        with self.vis.gui.add_folder("Sweep Volume V2 (Half) — Offset") as folder:
            self._gui_handles.append(folder)
            off_x = self.vis.gui.add_slider(
                "offset2_x",
                min=-0.3,
                max=0.3,
                step=0.001,
                initial_value=self.sv2_offset[0],
            )
            off_y = self.vis.gui.add_slider(
                "offset2_y",
                min=-0.3,
                max=0.3,
                step=0.001,
                initial_value=self.sv2_offset[1],
            )
            off_z = self.vis.gui.add_slider(
                "offset2_z",
                min=-0.3,
                max=0.3,
                step=0.001,
                initial_value=self.sv2_offset[2],
            )

        sliders = [ext_x, ext_y, ext_z, off_x, off_y, off_z]

        def _update_sv2_box():
            self.sv2_extents = [ext_x.value, ext_y.value, ext_z.value]
            self.sv2_offset = [off_x.value, off_y.value, off_z.value]
            tf = np.eye(4)
            tf[:3, 3] = self.sv2_offset
            visualize_bbox(
                self.vis,
                "sweep_volume_half",
                np.array(self.sv2_extents),
                T=tf,
                color=[255, 165, 0],
            )

        for sl in sliders:

            @sl.on_update
            def _on_update(_ev, _fn=_update_sv2_box):
                _fn()

        _update_sv2_box()

        btn = self.vis.gui.add_button("Confirm Half-Open Sweep Volume", color="green")
        self._gui_handles.append(btn)

        @btn.on_click
        def _on_confirm(_ev):
            self.sv2_extents = [ext_x.value, ext_y.value, ext_z.value]
            self.sv2_offset = [off_x.value, off_y.value, off_z.value]
            print(
                f"[Step 4] Half sweep volume — extents: {self.sv2_extents}, offset: {self.sv2_offset}"
            )
            self._next_step()

        # Going back from the half-SV box editor lands on the close-config setup
        # of Step 4 (lets the user re-pick the close joint state cleanly).
        self._add_prev_button("annotate_half_sv")

    # ------------------------------------------------------------------
    # Step 4: Animate review
    # ------------------------------------------------------------------

    def _step_animate_review(self):
        self._clear_gui()
        self.vis.scene.reset()
        make_frame(self.vis, "world", h=0.10, radius=0.002)

        self._add_status("Step 5/6: Review Animation")
        self._add_info(
            "The gripper will animate from **open** to **close**.\n\n"
            "Both sweep volumes are shown:\n"
            "- Blue = open sweep volume\n"
            "- Orange = half-open sweep volume (v2)"
        )

        # Draw both sweep volumes
        tf_open = np.eye(4)
        tf_open[:3, 3] = self.sv_offset
        visualize_bbox(
            self.vis,
            "sv_open",
            np.array(self.sv_extents),
            T=tf_open,
            color=[0, 100, 255],
        )

        tf_half = np.eye(4)
        tf_half[:3, 3] = self.sv2_offset
        visualize_bbox(
            self.vis,
            "sv_half",
            np.array(self.sv2_extents),
            T=tf_half,
            color=[255, 165, 0],
        )

        self._animating = True

        def _animate():
            traj = get_traj_js(self.open_js, self.close_js, num_steps=30)
            while self._animating:
                for js in traj:
                    if not self._animating:
                        break
                    self._refresh_gripper(js)
                    time.sleep(0.05)
                for js in reversed(traj):
                    if not self._animating:
                        break
                    self._refresh_gripper(js)
                    time.sleep(0.05)

        self._anim_thread = threading.Thread(target=_animate, daemon=True)
        self._anim_thread.start()

        go_back_btn = self.vis.gui.add_button("Go Back (Re-annotate)", color="yellow")
        self._gui_handles.append(go_back_btn)

        @go_back_btn.on_click
        def _on_back(_ev):
            self._animating = False
            # Go back to "annotate_open_sv" (Step 3 in the new 6-step ordering)
            self.step_idx = self.STEPS.index("annotate_open_sv")
            self._run_step()

        btn = self.vis.gui.add_button("Looks Good — Proceed to Save", color="green")
        self._gui_handles.append(btn)

        @btn.on_click
        def _on_confirm(_ev):
            self._animating = False
            self._next_step()

        # Standard previous-step navigation (returns to half-SV annotation)
        self._add_prev_button("annotate_half_sv")

    # ------------------------------------------------------------------
    # Step 6: Save config
    # ------------------------------------------------------------------

    def _step_save(self):
        self._clear_gui()
        self.vis.scene.reset()
        make_frame(self.vis, "world", h=0.10, radius=0.002)
        self._refresh_gripper(self.open_js)

        # Draw final sweep volumes
        tf_open = np.eye(4)
        tf_open[:3, 3] = self.sv_offset
        visualize_bbox(
            self.vis,
            "sv_open",
            np.array(self.sv_extents),
            T=tf_open,
            color=[0, 100, 255],
        )
        tf_half = np.eye(4)
        tf_half[:3, 3] = self.sv2_offset
        visualize_bbox(
            self.vis,
            "sv_half",
            np.array(self.sv2_extents),
            T=tf_half,
            color=[255, 165, 0],
        )

        # Derive fingertip from sweep volume offset
        self.fingertip = list(self.sv_offset)

        # Compute bbox in the canonical (rotated) frame
        bbox_min, bbox_max = compute_gripper_bbox(
            self.robot, self.open_js, base_T=self.base_rotation
        )

        gtype, symmetric = infer_gripper_type(self.robot)

        config = {
            "open": self.open_js,
            "close": self.close_js,
            "fingertip": self.fingertip,
            "sweep_volume": {
                "extents": self.sv_extents,
                "offset": self.sv_offset,
                "extents2": self.sv2_extents,
                "offset2": self.sv2_offset,
            },
            "links": self.link_names,
            "standoff": [0.0, self.sv_extents[2] / 2],
            "symmetric": symmetric,
            "type": gtype,
            "bbox": [bbox_min, bbox_max],
            "base_rotation": self.base_rotation.tolist(),
        }

        self._add_status("Step 6/6: Confirm and Save")

        config_summary = json.dumps(config, indent=2)
        self._add_info(f"**Config preview:**\n```json\n{config_summary}\n```")

        dest_dir = os.path.join(self.output_root, self.gripper_name)
        self._add_info(f"Will save to: `{dest_dir}/`")

        # Gripper type dropdown
        type_dropdown = self.vis.gui.add_dropdown(
            "Gripper Type",
            options=["parallel_2f", "revolute_2f", "revolute_3f"],
            initial_value=gtype,
        )
        self._gui_handles.append(type_dropdown)

        symmetric_cb = self.vis.gui.add_checkbox("Symmetric", initial_value=symmetric)
        self._gui_handles.append(symmetric_cb)

        btn = self.vis.gui.add_button("Confirm and Save Config", color="green")
        self._gui_handles.append(btn)

        @btn.on_click
        def _on_save(_ev):
            config["type"] = type_dropdown.value
            config["symmetric"] = symmetric_cb.value
            self._save_config(config)

        self._add_prev_button("animate_review")

    def _save_config(self, config: Dict):
        dest_dir = os.path.join(self.output_root, self.gripper_name)
        os.makedirs(dest_dir, exist_ok=True)

        # Copy URDF
        urdf_src = self.urdf_path
        urdf_dir = os.path.dirname(urdf_src)

        # Copy the URDF file
        dest_urdf = os.path.join(dest_dir, "gripper.urdf")
        shutil.copy2(urdf_src, dest_urdf)

        # Copy meshes directory if it exists alongside the URDF
        src_meshes = os.path.join(urdf_dir, "meshes")
        if os.path.isdir(src_meshes):
            dest_meshes = os.path.join(dest_dir, "meshes")
            if os.path.exists(dest_meshes):
                shutil.rmtree(dest_meshes)
            shutil.copytree(src_meshes, dest_meshes)

        # Also copy any mesh files referenced in subdirectories
        # Scan URDF for mesh filenames and copy any that exist relative to URDF dir
        import xml.etree.ElementTree as ET

        tree = ET.parse(urdf_src)
        for mesh_elem in tree.iter("mesh"):
            fname = mesh_elem.get("filename", "")
            if fname:
                src_mesh = os.path.join(urdf_dir, fname)
                if os.path.isfile(src_mesh):
                    dest_mesh = os.path.join(dest_dir, fname)
                    os.makedirs(os.path.dirname(dest_mesh), exist_ok=True)
                    if not os.path.exists(dest_mesh):
                        shutil.copy2(src_mesh, dest_mesh)

        # Write config.json
        config_path = os.path.join(dest_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        # Export merged meshes (in canonical / rotated frame)
        try:
            vis_mesh_path = os.path.join(dest_dir, "vis_mesh.obj")
            export_merged_mesh(
                self.robot, self.open_js, vis_mesh_path, base_T=self.base_rotation
            )
            print(f"Exported visual mesh: {vis_mesh_path}")
        except Exception as e:
            print(f"Warning: could not export vis_mesh.obj: {e}")

        print(f"\n{'='*60}")
        print(f"Config saved to: {config_path}")
        print(f"Assets copied to: {dest_dir}")
        print(f"{'='*60}")
        print(f"\nTo visualize this gripper:")
        print(f"  python scripts/vis_gripper_desc.py --gripper {self.gripper_name}")
        print(f"\nTo visualize with sweep volume:")
        print(
            f"  python scripts/vis_gripper_desc.py --gripper {self.gripper_name} --show-sweep-volume"
        )
        print(f"\nNext steps:")
        print(
            f"  Run inference with: python scripts/demo_object_pc.py --gripper_name {self.gripper_name}"
        )

        # Update GUI
        self._clear_gui()
        self._add_status("Done!")
        self._add_info(
            f"Config saved to `{config_path}`.\n\n"
            f"Visualize with:\n```\npython scripts/vis_gripper_desc.py --gripper {self.gripper_name}\n```"
        )

    # ------------------------------------------------------------------
    # Step navigation
    # ------------------------------------------------------------------

    def _next_step(self):
        self.step_idx += 1
        if self.step_idx < len(self.STEPS):
            self._run_step()

    def _run_step(self):
        step = self.current_step
        dispatch = {
            "align_base": self._step_align_base,
            "open_confirm": self._step_open_confirm,
            "annotate_open_sv": self._step_annotate_open_sv,
            "annotate_half_sv": self._step_annotate_half_sv,
            "animate_review": self._step_animate_review,
            "save": self._step_save,
        }
        dispatch[step]()

    def start(self):
        print(f"\nStarting Gripper Wizard for: {self.gripper_name}")
        print(f"Open the Viser GUI at: http://localhost:{self.port}")
        print(f"Follow the steps in the GUI panel.\n")
        self._run_step()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive GUI tool for creating a GraspGenX gripper config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/gripper_config_wizard.py --urdf assets/x_grippers/my_gripper/gripper.urdf

    python scripts/gripper_config_wizard.py --urdf /path/to/robot.urdf --name my_robot_hand --port 8081
        """,
    )
    parser.add_argument(
        "--urdf",
        type=str,
        required=True,
        help="Path to the gripper URDF file",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Gripper name (used as folder name under assets/x_grippers/). "
        "If not provided, will prompt interactively.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for Viser server (default: 8080)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Parent directory under which '<name>/' will be written. "
        "Defaults to the gripper_descriptions assets/x_grippers path "
        "resolved via $GRASPGENX_GRIPPER_CFG_DIR (or the auto-cloned "
        "<repo>/ext/gripper_descriptions/...). Pass an absolute path "
        "to override.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    urdf_path = os.path.abspath(args.urdf)
    if not os.path.isfile(urdf_path):
        print(f"Error: URDF not found: {urdf_path}")
        sys.exit(1)

    # Get gripper name
    if args.name:
        gripper_name = args.name
    else:
        gripper_name = input("Enter gripper name (e.g., my_robot_gripper): ").strip()
        if not gripper_name:
            print("Error: gripper name cannot be empty.")
            sys.exit(1)

    print(f"Loading URDF: {urdf_path}")
    robot = load_urdf(urdf_path)
    joint_names = get_joint_names(robot)
    link_names = get_link_names(robot)
    print(f"  Joints ({len(joint_names)}): {joint_names}")
    print(f"  Links  ({len(link_names)}): {link_names}")

    vis = create_visualizer(port=args.port)
    wizard = ConfigWizard(
        vis,
        robot,
        urdf_path,
        gripper_name,
        args.port,
        output_root=args.output_dir,
    )
    print(f"Output will be written under: {wizard.output_root}/{gripper_name}/")
    wizard.start()

    # Keep alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
