"""Geometry checks for end2end robot YAMLs (no cuRobo / simulation needed).

A GraspGenX grasp pose has +Z as the approach axis and +X as the closing
axis. For each robot whose profile names a pair of finger-pad links, place the
tool frame at ``grasp @ grasp_to_tool_transform`` with ``grasp = I`` and check
that the pads straddle the grasp origin along X at the gripper's fingertip
depth. A wrong ``grasp_to_tool_transform`` (e.g. 90° about Z) moves the pads
onto the Y axis and fails this test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yourdfpy

REPO_ROOT = Path(__file__).resolve().parents[1]
E2E = REPO_ROOT / "end2end"
sys.path.insert(0, str(E2E))

from paths import grippers_dir  # noqa: E402
from robot_profiles import RobotProfile  # noqa: E402
from scene_builder import load_yaml  # noqa: E402

ROBOT_YAMLS = sorted((E2E / "robots").glob("ur10e_*.yaml"))


def _load_profile(robot_yaml: Path) -> RobotProfile:
    return RobotProfile.from_yaml(load_yaml(str(robot_yaml)))


@pytest.mark.parametrize("robot_yaml", ROBOT_YAMLS, ids=lambda p: p.stem)
def test_grasp_frame_aligns_with_finger_pads(robot_yaml):
    profile = _load_profile(robot_yaml)
    pads = profile.gripper_pad_links()
    if pads is None:
        pytest.skip(f"{profile.NAME} has no finger-pad links to check")
    if not Path(profile.urdf_path).is_file():
        pytest.skip(
            f"{Path(profile.urdf_path).name} missing — run "
            "`python end2end/setup_end2end_deps.py` first."
        )

    config_path = grippers_dir() / profile.graspgen.gripper_name / "config.json"
    fingertip_depth = json.loads(config_path.read_text())["fingertip"][2]

    robot = yourdfpy.URDF.load(profile.urdf_path, load_meshes=False)
    for joint_values in (profile.gripper_open, profile.gripper_close):
        robot.update_cfg(dict(joint_values))
        T_world_tool = robot.get_transform(profile.tool_frame)
        # Maps URDF world coordinates into the grasp frame, given that the
        # tool frame sits at grasp_to_tool_transform relative to the grasp.
        T_grasp_world = profile.grasp_to_tool_transform @ np.linalg.inv(T_world_tool)
        left, right = (
            (T_grasp_world @ robot.get_transform(link))[:3, 3] for link in pads
        )

        # Pads close along X, symmetrically about the approach axis.
        assert abs(left[1]) < 0.005 and abs(right[1]) < 0.005, (left, right)
        assert left[0] * right[0] <= 0.0, (left, right)
        assert abs(left[0] + right[0]) < 0.005, (left, right)
        # Pads sit near the fingertip depth along the approach axis.
        for pad in (left, right):
            assert abs(pad[2] - fingertip_depth) < 0.03, (pad, fingertip_depth)

    robot.update_cfg(dict(profile.gripper_open))
    open_left, open_right = (robot.get_transform(link)[:3, 3] for link in pads)
    robot.update_cfg(dict(profile.gripper_close))
    closed_left, closed_right = (robot.get_transform(link)[:3, 3] for link in pads)
    assert np.linalg.norm(open_left - open_right) > np.linalg.norm(
        closed_left - closed_right
    ), "gripper_open should leave a wider gap between the pads than gripper_close"
