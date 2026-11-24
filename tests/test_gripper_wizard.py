"""Tests for gripper type/symmetry inference in scripts/gripper_config_wizard.py."""

import glob
import importlib.util
import json
import os

import pytest
import yourdfpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIZARD_PATH = os.path.join(REPO_ROOT, "scripts", "gripper_config_wizard.py")
PROC_GRIPPERS = os.path.join(REPO_ROOT, "assets", "proc_grippers")

# Grippers in gripper_descriptions whose URDF has only prismatic joints but
# whose config.json is labelled revolute_2f. The inference returns parallel_2f.
KNOWN_LABEL_DISCREPANCIES = {"sawyer_hand", "schunk_wsg50", "tesollo_delto2f"}


def _load_wizard():
    spec = importlib.util.spec_from_file_location("gripper_config_wizard", WIZARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wizard = _load_wizard()


def _urdf_from_string(tmp_path, body):
    path = tmp_path / "gripper.urdf"
    path.write_text(f'<?xml version="1.0"?>\n<robot name="g">{body}</robot>')
    return yourdfpy.URDF.load(str(path), load_meshes=False, build_scene_graph=False)


def _links(*names):
    return "".join(f'<link name="{n}"/>' for n in names)


def _joint(name, jtype, parent, child, mimic=None, axis="1 0 0"):
    mimic_xml = f'<mimic joint="{mimic}" multiplier="1" offset="0"/>' if mimic else ""
    limit = (
        '<limit lower="0" upper="0.8" effort="1" velocity="1"/>'
        if jtype != "fixed"
        else ""
    )
    return (
        f'<joint name="{name}" type="{jtype}"><parent link="{parent}"/>'
        f'<child link="{child}"/><axis xyz="{axis}"/>{limit}{mimic_xml}</joint>'
    )


def test_robotiq_style_linkage_with_mimic_joints_is_revolute_2f(tmp_path):
    # One actuated knuckle drives five mimic joints (outer/inner linkages on
    # both sides), as in the Robotiq 2F-85/2F-140 URDFs.
    body = _links(
        "base", "l_outer", "l_finger", "r_outer", "r_finger", "l_inner", "r_inner"
    )
    body += _joint("finger_joint", "revolute", "base", "l_outer")
    body += _joint("l_finger_joint", "revolute", "l_outer", "l_finger", "finger_joint")
    body += _joint("r_outer_joint", "revolute", "base", "r_outer", "finger_joint")
    body += _joint("r_finger_joint", "revolute", "r_outer", "r_finger", "finger_joint")
    body += _joint("l_inner_joint", "revolute", "base", "l_inner", "finger_joint")
    body += _joint("r_inner_joint", "revolute", "base", "r_inner", "finger_joint")
    robot = _urdf_from_string(tmp_path, body)

    assert wizard.get_joint_names(robot) == ["finger_joint"]
    assert wizard.infer_gripper_type(robot) == ("revolute_2f", True)


def test_prismatic_jaws_with_mimic_are_parallel_2f(tmp_path):
    body = _links("base", "left", "right")
    body += _joint("left_joint", "prismatic", "base", "left")
    body += _joint("right_joint", "prismatic", "base", "right", "left_joint")
    robot = _urdf_from_string(tmp_path, body)

    assert wizard.infer_gripper_type(robot) == ("parallel_2f", True)


def test_single_moving_jaw_is_not_symmetric(tmp_path):
    # A single revolute jaw closing against a fixed one (e.g. Spot).
    body = _links("base", "fixed_jaw", "moving_jaw")
    body += _joint("fixed_jaw_joint", "fixed", "base", "fixed_jaw")
    body += _joint("jaw_joint", "revolute", "base", "moving_jaw")
    robot = _urdf_from_string(tmp_path, body)

    assert wizard.infer_gripper_type(robot) == ("revolute_2f", False)


def test_three_independent_fingers_are_revolute_3f(tmp_path):
    body = _links("base", *[f"f{i}_{s}" for i in range(3) for s in ("prox", "dist")])
    for i in range(3):
        body += _joint(f"f{i}_base", "revolute", "base", f"f{i}_prox")
        body += _joint(f"f{i}_tip", "revolute", f"f{i}_prox", f"f{i}_dist")
    robot = _urdf_from_string(tmp_path, body)

    assert wizard.infer_gripper_type(robot) == ("revolute_3f", False)


def _gripper_dirs(root):
    return [
        d
        for d in sorted(glob.glob(os.path.join(root, "*")))
        if os.path.exists(os.path.join(d, "config.json"))
        and os.path.exists(os.path.join(d, "gripper.urdf"))
    ]


def _check_against_config(gripper_dir):
    with open(os.path.join(gripper_dir, "config.json")) as f:
        cfg = json.load(f)
    robot = yourdfpy.URDF.load(
        os.path.join(gripper_dir, "gripper.urdf"),
        load_meshes=False,
        build_scene_graph=False,
    )
    assert wizard.infer_gripper_type(robot) == (cfg["type"], cfg["symmetric"])


@pytest.mark.parametrize(
    "gripper_dir", _gripper_dirs(PROC_GRIPPERS), ids=os.path.basename
)
def test_matches_procedural_gripper_labels(gripper_dir):
    _check_against_config(gripper_dir)


def _gripper_descriptions_dirs():
    try:
        from graspgenx import get_gripper_descriptions_assets

        root = str(get_gripper_descriptions_assets())
    except Exception:  # noqa: BLE001 - assets are optional for this test
        return []
    return [
        d
        for d in _gripper_dirs(root)
        if os.path.basename(d) not in KNOWN_LABEL_DISCREPANCIES
    ]


_REAL_GRIPPERS = _gripper_descriptions_dirs()


@pytest.mark.skipif(not _REAL_GRIPPERS, reason="gripper_descriptions not available")
@pytest.mark.parametrize("gripper_dir", _REAL_GRIPPERS, ids=os.path.basename)
def test_matches_real_gripper_labels(gripper_dir):
    _check_against_config(gripper_dir)
