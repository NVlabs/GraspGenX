#!/usr/bin/env python3
"""End-to-end grasping demo.

Pipeline:
  1. Build a scene (env + manipulation object + robot) via scene_synthesizer
  2. Visualize it in viser
  3. Run GraspGen inference to predict grasps on the object
  4. Initialize cuRobo MotionPlanner with the env's collision world
  5. Pick a non-colliding start joint config and a random predicted grasp
  6. Plan a collision-free trajectory to the grasp
  7. Animate the trajectory in viser
  8. Export trajectory.json (always) and optionally encode an MP4

Run (from the repo root; with no arguments this is a UR10e + Robotiq 2F-140
pick-and-lift of a HOPE object on a bare tabletop):
  uv run --no-sync python end2end/e2e_grasp_demo.py \
      --robot_config end2end/robots/ur10e_robotiq_2f_140.yaml \
      --env_config   end2end/envs/tabletop_single_nobin.yaml \
      --mesh_file    assets/sample_data/hope_objects/GranolaBars.obj \
      --render-mp4   end2end/runs/demo.mp4
"""

from __future__ import annotations

# Headless rendering: set BEFORE pyrender is imported anywhere downstream.
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYGLET_HEADLESS", "true")

import argparse
import json
import logging
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
import trimesh.transformations as tra
import yaml

# Make sibling modules importable when this file is run directly.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from registry import CollisionObstacle  # noqa: E402
from robot_profiles import RobotProfile  # noqa: E402
from scene_builder import SceneBundle, build_scene, load_yaml  # noqa: E402
from tasks import get_task  # noqa: E402
from trajectory_visualizer import URDFFK, matrix_to_xyz_quat_wxyz  # noqa: E402

logging.basicConfig(format="%(asctime)s [E2E] %(message)s", level=logging.INFO)
log = logging.getLogger("e2e")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="End-to-end grasping demo")
    ap.add_argument(
        "--robot_config",
        type=Path,
        default=_HERE / "robots/ur10e_robotiq_2f_140.yaml",
    )
    ap.add_argument(
        "--env_config", type=Path, default=_HERE / "envs/tabletop_single_nobin.yaml"
    )
    ap.add_argument(
        "--mesh_file",
        type=Path,
        default=_HERE.parent / "assets/sample_data/hope_objects/GranolaBars.obj",
        help="Path to the manipulation-target mesh (.obj/.stl/.ply) for "
        "single-object tasks (default: the HOPE GranolaBars sample). "
        "Ignored for clutter_pick_and_drop, where the env YAML's "
        "object_slots carry mesh paths per object.",
    )
    ap.add_argument("--mesh_scale", type=float, default=1.0)
    ap.add_argument(
        "--num_grasps",
        type=int,
        default=200,
        help="Number of grasps GraspGen should sample per inference call",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Keep top-K grasps by confidence after inference",
    )
    ap.add_argument(
        "--grasp_threshold",
        type=float,
        default=0.7,
        help="Confidence floor for grasps. Matches the default in "
        "scripts/demo_object_pc.py. Use -1 to disable and "
        "rank by topk only.",
    )
    ap.add_argument(
        "--planner",
        choices=["graspmoe", "diffusion", "topdown"],
        default="graspmoe",
        help="GraspGenX planner. graspmoe (default) combines "
        "diffusion + an OBB top-down sweep. diffusion is the "
        "raw diffusion sampler. topdown runs graspmoe but "
        "drops the diffusion branch — equivalent to "
        "`--planner graspmoe --obb_only` but a cleaner UX.",
    )
    ap.add_argument(
        "--moe_obb_density",
        choices=["sparse", "dense", "none"],
        default="dense",
        help="GraspMoE OBB sweep density. sparse=single centroid "
        "top-down pose, dense=positions along OBB's longer "
        "XY axis (top-down), none=skip OBB (diffusion only). "
        "Default 'dense' gives cuRobo more top-down candidates "
        "to pick a reachable grasp from.",
    )
    ap.add_argument(
        "--obb_only",
        action="store_true",
        help="Keep only OBB top-down candidates (drop the diffusion "
        "branch). Use this when you want a 'top-down only' "
        "planner. Requires --planner graspmoe and a non-'none' "
        "--moe_obb_density.",
    )
    ap.add_argument(
        "--force_grasp_idx",
        type=int,
        default=-1,
        help="If >=0, force the planner to use this specific grasp "
        "index (instead of letting cuRobo pick from the "
        "goalset). Indices match the order GraspGen returns "
        "them; use end2end/viz_grasps.py to inspect.",
    )
    ap.add_argument(
        "--singleton_top_conf",
        action="store_true",
        help="Plan to a goalset of EXACTLY ONE pose: the "
        "highest-confidence grasp returned by GraspGenX. "
        "Mutually exclusive with --force_grasp_idx. Used "
        "for debug comparisons against viz_grasps.py "
        "where the visualization picks the top-conf grasp.",
    )
    ap.add_argument(
        "--task",
        default="pick_and_lift",
        help="Task identifier — selects which post-pick action "
        "sequence runs. See end2end/tasks.py. Default "
        "pick_and_lift = original approach+grasp+close+lift. "
        "pick_and_drop_in_bin adds transport+drop+release "
        "(use with an env YAML that defines a `bin` asset).",
    )
    ap.add_argument(
        "--rank_grasps_by_confidence",
        action="store_true",
        help="Instead of passing all top-K grasps to cuRobo as a "
        "single goalset (where cuRobo picks the most "
        "reachable, often a lower-conf one), iterate top-K "
        "in confidence order, attempting plan_grasp on each "
        "as a singleton goalset until one succeeds.",
    )
    ap.add_argument("--num_sample_points", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-viser",
        action="store_true",
        help="Skip the viser server entirely (for headless smoke tests)",
    )
    ap.add_argument("--viser_port", type=int, default=8080)
    ap.add_argument(
        "--export-trajectory",
        type=Path,
        default=None,
        help="Path to write trajectory.json (default: runs/<ts>/trajectory.json)",
    )
    ap.add_argument(
        "--render-mp4",
        type=Path,
        default=None,
        help="If set, encode an MP4 of the trajectory at this path",
    )
    ap.add_argument(
        "--wrist-cam-mp4",
        type=Path,
        default=None,
        help="If set, also encode an MP4 from a wrist-camera POV (camera "
        "follows the gripper through the trajectory).",
    )
    ap.add_argument(
        "--wrist_cam_link",
        type=str,
        default=None,
        help="Link the wrist camera follows (defaults to the cuRobo tool frame).",
    )
    ap.add_argument(
        "--wrist_cam_eye_offset",
        nargs=3,
        type=float,
        default=[-0.07, 0.0, -0.04],
        help="Eye offset in the wrist link's local frame (xyz).",
    )
    ap.add_argument(
        "--wrist_cam_target_offset",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.20],
        help="Look-at target offset in the wrist link's local frame (xyz).",
    )
    ap.add_argument("--mp4_resolution", type=str, default="960x720")
    ap.add_argument("--mp4_fps", type=int, default=30)
    ap.add_argument("--show_grasps_in_mp4", action="store_true", default=True)
    ap.add_argument(
        "--no_show_grasps_in_mp4", action="store_false", dest="show_grasps_in_mp4"
    )
    ap.add_argument(
        "--max_plan_attempts",
        type=int,
        default=40,
        help="Try this many grasps before giving up",
    )
    ap.add_argument(
        "--hold_frames",
        type=int,
        default=60,
        help="Frames to hold pose between phases (pregrasp→close, "
        "close→lift). Default 60 = 1 second at sim_fps=60. "
        "Lets the sim settle and the gripper finish closing "
        "before the lift starts.",
    )
    ap.add_argument(
        "--hold_after_close_frames",
        type=int,
        default=None,
        help="Frames to hold AFTER close before lifting (overrides "
        "--hold_frames for just that segment). Use a larger "
        "value for slow velocity-mode multi-finger hands so "
        "the fingers fully settle on the object before lift, "
        "preventing premature-lift slip. Default: = hold_frames.",
    )
    ap.add_argument(
        "--close_frames",
        type=int,
        default=20,
        help="Number of trajectory frames to ramp the gripper closed at the end",
    )
    ap.add_argument(
        "--gripper_close_value",
        type=float,
        default=0.7,
        help="Target value for the gripper finger joint when closed (rad). "
        "0.7 ≈ fully closed for robotiq_2f_140; ~0.0 for franka.",
    )
    ap.add_argument(
        "--gripper_joint_names",
        nargs="+",
        default=["finger_joint", "panda_finger_joint1", "panda_finger_joint2"],
        help="Joint names that drive the gripper close — first match wins",
    )
    ap.add_argument(
        "--auto_close_to_object",
        action="store_true",
        default=True,
        help="If on (default), stop the gripper at the object surface "
        "instead of fully closing. Computes the object's width along "
        "the gripper closing axis via FK on the finger-pad links.",
    )
    ap.add_argument(
        "--no_auto_close_to_object", action="store_false", dest="auto_close_to_object"
    )
    ap.add_argument(
        "--finger_pad_links",
        nargs="+",
        default=[
            "left_inner_finger_pad",
            "right_inner_finger_pad",
            "panda_leftfinger",
            "panda_rightfinger",
        ],
        help="Two-link names whose midpoint distance is the gripper gap. "
        "First two found in the URDF will be used (so this same default "
        "covers both the robotiq and franka URDFs).",
    )
    ap.add_argument(
        "--gripper_close_padding",
        type=float,
        default=0.005,
        help="Extra clearance (m) added to the object width before computing "
        "the close angle, so fingers stop just shy of the surface.",
    )
    # Camera defaults are taken from `visual.camera` in the env YAML if
    # present, otherwise the values below. CLI flags always override.
    ap.add_argument(
        "--camera_eye",
        nargs=3,
        type=float,
        default=None,
        help="Camera eye in world frame (overrides env YAML)",
    )
    ap.add_argument(
        "--camera_target",
        nargs=3,
        type=float,
        default=None,
        help="Camera target in world frame (defaults to env YAML, then object center)",
    )

    # ----- Playback mode: kinematic (default) or dynamic (Newton sim) -----
    ap.add_argument(
        "--playback_mode",
        choices=["kinematic", "dynamic"],
        default="kinematic",
        help="kinematic = paint each link's mesh at the FK pose per "
        "trajectory waypoint (no physics). dynamic = replay the "
        "trajectory inside Newton with PD control + contacts + "
        "gravity so the gripper actually grasps the object.",
    )
    ap.add_argument(
        "--sim_fps",
        type=int,
        default=60,
        help="[dynamic] frames recorded to trajectory JSON",
    )
    ap.add_argument(
        "--sim_dt",
        type=float,
        default=0.001,
        help="[dynamic] physics step (seconds). Default 0.001 = "
        "1000 Hz. Substeps per frame = round(1/(sim_fps·sim_dt)).",
    )
    ap.add_argument(
        "--arm_kp",
        type=float,
        default=500.0,
        help="[dynamic] joint_target_ke for the 6 arm joints",
    )
    ap.add_argument(
        "--arm_kd",
        type=float,
        default=50.0,
        help="[dynamic] joint_target_kd for the 6 arm joints",
    )
    ap.add_argument(
        "--finger_kp",
        type=float,
        default=2000.0,
        help="[dynamic] joint_target_ke for the master finger_joint. "
        "Higher = harder close (recommended for stable grasps).",
    )
    ap.add_argument(
        "--finger_kd",
        type=float,
        default=200.0,
        help="[dynamic] joint_target_kd for the master finger_joint. "
        "High damping keeps the close from overshooting under "
        "contact, which otherwise tunnels the object out.",
    )
    ap.add_argument(
        "--gravity",
        type=float,
        default=-9.81,
        help="[dynamic] world-frame z-gravity (m/s²)",
    )
    ap.add_argument(
        "--settle_frames",
        type=int,
        default=30,
        help="[dynamic] extra frames to step after the trajectory ends "
        "with the final target held, so contacts can stabilize",
    )
    # Defaults raised from (0.1, 3.0) to (0.2, 10.0) per the diagnosis
    # in dynamic_playback.py:81-86: the lighter, slipperier values
    # caused small/smooth objects to be launched by close-time contact
    # impulse spikes and to slide/rotate in place on the table even
    # after the initial settle. Mirrors NewtonDataGen's newton_grasp_eval.py.
    ap.add_argument(
        "--object_mass",
        type=float,
        default=0.2,
        help="[dynamic] manipulation object mass (kg)",
    )
    ap.add_argument(
        "--object_mu",
        type=float,
        default=10.0,
        help="[dynamic] Coulomb friction on the object mesh",
    )
    ap.add_argument(
        "--finger_mu",
        type=float,
        default=3.0,
        help="[dynamic] Coulomb friction on the gripper finger pads",
    )
    ap.add_argument(
        "--max_retries_per_object",
        type=int,
        default=2,
        help="[clutter_pick_and_drop] How many times each object "
        "may be re-queued after a failed grasp / transport "
        "slip before being marked failed_max_retries. The "
        "HOPE sweep uses 1 to keep wall time bounded.",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# cuRobo robot config loading (with template placeholders)
# ---------------------------------------------------------------------------

PLACEHOLDER_URDF = "ABSOLUTE_PATH_PLACEHOLDER_URDF"
PLACEHOLDER_ASSET_ROOT = "ABSOLUTE_PATH_PLACEHOLDER_ASSET_ROOT"


def resolve_curobo_robot_dict(
    combo_path: Path, combo_cfg: Dict[str, Any]
) -> Dict[str, Any] | str:
    """Return either a dict (if our YAML has placeholders) or a bare filename
    that cuRobo can resolve in its own content tree.
    """
    robot_cfg_ref = combo_cfg["curobo"]["robot_config"]
    p = (
        (combo_path.parent / robot_cfg_ref).resolve()
        if not Path(robot_cfg_ref).is_absolute()
        else Path(robot_cfg_ref)
    )

    if not p.is_file():
        # Treat as a bare filename that cuRobo will resolve in its content path
        # (e.g. "franka.yml"). cuRobo's MotionPlannerCfg.create handles this.
        return str(robot_cfg_ref)

    raw = p.read_text()
    raw = raw.replace(PLACEHOLDER_URDF, combo_cfg["urdf_path"])
    raw = raw.replace(PLACEHOLDER_ASSET_ROOT, combo_cfg["asset_root_path"])
    return yaml.safe_load(raw)


# ---------------------------------------------------------------------------
# Collision world packaging for cuRobo
# ---------------------------------------------------------------------------


def collision_world_to_curobo(
    obstacles: List[CollisionObstacle], robot_base_T: np.ndarray
) -> Dict[str, Any]:
    """Convert our list of CollisionObstacle (poses in WORLD frame) to a
    cuRobo scene_model dict (poses in ROBOT BASE frame).
    """
    T_world_robot_inv = tra.inverse_matrix(robot_base_T)

    def _xform(pose_curobo: List[float]) -> List[float]:
        # pose_curobo = [x, y, z, qw, qx, qy, qz]
        T = np.eye(4)
        T[:3, 3] = pose_curobo[:3]
        T[:3, :3] = tra.quaternion_matrix(pose_curobo[3:7])[:3, :3]
        T_robot = T_world_robot_inv @ T
        q = tra.quaternion_from_matrix(T_robot)  # wxyz
        return [
            float(T_robot[0, 3]),
            float(T_robot[1, 3]),
            float(T_robot[2, 3]),
            float(q[0]),
            float(q[1]),
            float(q[2]),
            float(q[3]),
        ]

    out: Dict[str, Any] = {}
    cuboids: Dict[str, Any] = {}
    meshes: Dict[str, Any] = {}
    for ob in obstacles:
        if ob.type == "cuboid":
            cuboids[ob.name] = {"dims": ob.dims, "pose": _xform(ob.pose)}
        elif ob.type == "mesh":
            meshes[ob.name] = {
                "file_path": ob.mesh_file,
                "pose": _xform(ob.pose),
                "scale": ob.scale,
            }
    if cuboids:
        out["cuboid"] = cuboids
    if meshes:
        out["mesh"] = meshes
    return out


# ---------------------------------------------------------------------------
# GraspGen
# ---------------------------------------------------------------------------


def _resolve_checkpoints_dir(gg_cfg: Dict[str, Any]) -> Path:
    """Resolve the GraspGenX checkpoint dir (containing ``gen/`` + ``dis/``).

    Portable resolution order:
      1. ``graspgen.checkpoints_dir`` from the robot YAML, if given (a literal
         path — supports per-robot custom checkpoints).
      2. Otherwise GraspGenX's managed location via
         ``get_checkpoints_version_dir()`` (honours ``$GRASPGENX_CHECKPOINT_DIR``
         and the auto-download root), so the demo works without hardcoded paths.
    """
    explicit = gg_cfg.get("checkpoints_dir")
    if explicit:
        return Path(explicit).expanduser().resolve()
    from graspgenx import get_checkpoints_version_dir

    return Path(get_checkpoints_version_dir()).resolve()


def run_graspgen(
    bundle: SceneBundle,
    robot_cfg: Dict[str, Any],
    num_sample_points: int,
    num_grasps: int,
    topk: int,
    seed: int,
    grasp_threshold: float = -1.0,
    planner: str = "graspmoe",
    moe_obb_density: str = "sparse",
    obb_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run GraspGenX on the bundle's object mesh. Returns grasps in WORLD
    frame plus per-grasp confidence scores.

    `robot_cfg["graspgen"]` must contain:
        gripper_name    — name resolved by GraspGenX's `resolve_gripper_info`
                          (e.g. "robotiq_2f_140"), looked up under `assets_dir`.
        checkpoints_dir — directory containing `gen/` and `dis/` subdirs (each
                          with `config.yaml` + `epoch_*.pth`).
        assets_dir      — root containing `x_grippers/` and `proc_grippers/`;
                          defaults to <GraspGenX repo>/assets. Pre-shipped
                          robotiq URDFs live in
                          ext/gripper_descriptions/gripper_descriptions/assets.
    """
    from graspgenx.grasp_server import GraspGenXSampler
    from graspgenx.utils.checkpoint_io import load_model_cfg

    gg_cfg = robot_cfg["graspgen"]
    gripper_name = gg_cfg["gripper_name"]
    ckpts_dir = _resolve_checkpoints_dir(gg_cfg)
    gen_dir = ckpts_dir / "gen"
    dis_dir = ckpts_dir / "dis"
    if not gen_dir.is_dir() or not dis_dir.is_dir():
        raise FileNotFoundError(
            f"Expected gen/ and dis/ under {ckpts_dir} (got gen={gen_dir.is_dir()}, "
            f"dis={dis_dir.is_dir()})"
        )
    assets_dir = gg_cfg.get("assets_dir")
    if assets_dir is None:
        # Auto-shipped gripper descriptions (robotiq_2f_140 etc.) live under
        # GraspGenX's `ext/gripper_descriptions/...assets`.
        graspgenx_root = Path(__file__).resolve().parent.parent
        assets_dir = str(
            graspgenx_root / "ext/gripper_descriptions/gripper_descriptions/assets"
        )
    log.info(
        "Loading GraspGenX checkpoints from %s (gripper=%s)", ckpts_dir, gripper_name
    )
    model_cfg = load_model_cfg(
        str(gen_dir), str(dis_dir), gg_cfg.get("gen_pth"), gg_cfg.get("dis_pth")
    )
    sampler = GraspGenXSampler(model_cfg, gripper_name, assets_dir=assets_dir)

    # Sample points in mesh-local frame
    obj_mesh = bundle.object_mesh
    xyz, _ = trimesh.sample.sample_surface(obj_mesh, num_sample_points)
    xyz = np.asarray(xyz, dtype=np.float32)

    # Center the cloud — GraspGenX is trained on pc-mean-centered inputs.
    pc_mean = xyz.mean(axis=0)
    T_center = tra.translation_matrix(-pc_mean)
    xyz_centered = tra.transform_points(xyz, T_center).astype(np.float32)

    # Use the GraspMoE planner (same as scripts/demo_object_pc.py default)
    # — diffusion ∪ OBB top-down sweep, with all candidates scored by the
    # discriminator. Much more reliable for tabletop objects than
    # diffusion-only, especially for boxes/cylinders that GraspGenX
    # otherwise gives very low confidence to. Falls back to the raw
    # diffusion call when planner == "diffusion".
    from graspgenx.samplers import run_planner_on_object

    # 'topdown' is sugar for 'graspmoe + obb_only' — same machinery, just
    # always drops the diffusion branch.
    planner_internal = "graspmoe" if planner == "topdown" else planner
    obb_only_eff = obb_only or (planner == "topdown")
    log.info(
        "Running GraspGenX %s planner (num_grasps=%d, topk=%d, threshold=%.3f, "
        "obb_density=%s%s)",
        planner,
        num_grasps,
        topk,
        grasp_threshold,
        moe_obb_density,
        ", obb_only" if obb_only_eff else "",
    )
    grasps_centered_np, conf_np, _tags, _obb = run_planner_on_object(
        xyz_centered,
        sampler,
        planner=planner_internal,
        grasp_threshold=grasp_threshold,
        num_grasps=num_grasps,
        topk_num_grasps=topk,
        moe_obb_density=moe_obb_density,
    )
    if obb_only_eff and len(_tags) > 0:
        keep = np.array([t == "obb" for t in _tags], dtype=bool)
        n_before = len(grasps_centered_np)
        grasps_centered_np = grasps_centered_np[keep]
        conf_np = conf_np[keep]
        tag = "topdown" if planner == "topdown" else "obb_only"
        log.info(
            "%s: kept %d/%d OBB top-down grasps (dropped diffusion branch)",
            tag,
            int(keep.sum()),
            n_before,
        )
    if len(grasps_centered_np) == 0:
        return np.zeros((0, 4, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    grasps_centered = grasps_centered_np.astype(np.float64)
    grasps_centered[:, 3, 3] = 1.0  # numerical safety (mirrors demo_object_mesh.py)
    conf = conf_np.astype(np.float32)

    # Un-center the grasps back into the object's mesh-local frame, then lift
    # them into world frame via the placed object's transform.
    T_uncenter = tra.inverse_matrix(T_center)
    grasps_local = np.array([T_uncenter @ g for g in grasps_centered])
    grasps_world = np.array([bundle.object_world_T @ g for g in grasps_local])
    return grasps_world.astype(np.float32), conf


# ---------------------------------------------------------------------------
# cuRobo planning
# ---------------------------------------------------------------------------


def init_planner(
    combo_path: Path,
    robot_cfg: Dict[str, Any],
    scene_model: Dict[str, Any],
    max_goalset: int = 128,
):
    """Lazy import + setup of cuRobo MotionPlanner.

    `max_goalset` lets us pass *all* GraspGen grasps as one goalset to
    `plan_grasp`; the planner then picks the most reachable one itself.
    """
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

    robot_arg = resolve_curobo_robot_dict(combo_path, robot_cfg)

    log.info("Initializing cuRobo MotionPlanner (max_goalset=%d)", max_goalset)
    # Use cuRobo's defaults for IK/trajopt seeds, pose tolerances, and the
    # cuda-graph setting. An earlier explicit use_cuda_graph=False (plus
    # tightened seeds/tolerances) silently broke public NVlabs cuRobo — every
    # approach/grasp trajopt failed ("Planning to grasp pose failed"). The
    # defaults plan correctly on both the lab fork and public cuRobo.
    cfg = MotionPlannerCfg.create(
        robot=robot_arg,
        scene_model=scene_model if scene_model else None,
        max_goalset=max_goalset,
    )
    planner = MotionPlanner(cfg)
    log.info("Warming up planner (this can take a minute on first run)")
    planner.warmup(enable_graph=False, num_warmup_iterations=1)
    return planner


def plan_to_grasp(
    planner,
    robot_cfg: Dict[str, Any],
    grasps_world: np.ndarray,
    conf: np.ndarray,
    max_attempts: int,
    seed: int,
    robot_base_T: np.ndarray,
    force_idx: int = -1,
    rank_by_confidence: bool = False,
):
    """Plan a grasp trajectory.

    Default behaviour: pass top-K (by confidence) grasps as one goalset
    and let cuRobo pick the most reachable. Useful when many grasps are
    feasible.

    With ``rank_by_confidence=True``: iterate top-K in confidence order,
    attempting plan_grasp on each as a singleton goalset until one
    succeeds. Use this when the goalset path picks a low-confidence
    grasp just because it's reachable.

    With ``force_idx >= 0``: only try that one specific grasp. Useful for
    debugging or to force a grasp identified via end2end/viz_grasps.py.

    Returns ``(success, result, chosen_idx, pregrasp_traj, lift_traj)``.
    """
    from curobo.types import JointState, Pose

    target_link = robot_cfg["curobo"]["tool_frame"]
    default_q = robot_cfg["curobo"]["default_joint_position"]

    q_start = JointState.from_position(
        torch.tensor([default_q], device="cuda", dtype=torch.float32),
        joint_names=planner.joint_names,
    )

    # Per-grasp tool-frame offset (identity in our default config)
    g2t = robot_cfg.get("grasp_to_tool_transform", {})
    tt = g2t.get("translation", [0, 0, 0])
    qq = g2t.get("quaternion_xyzw", [0, 0, 0, 1])
    T_offset = np.eye(4)
    T_offset[:3, 3] = tt
    if not (
        abs(qq[0]) < 1e-9
        and abs(qq[1]) < 1e-9
        and abs(qq[2]) < 1e-9
        and abs(qq[3] - 1) < 1e-9
    ):
        T_offset[:3, :3] = tra.quaternion_matrix([qq[3], qq[0], qq[1], qq[2]])[:3, :3]

    # cuRobo plans in robot-base frame. Lift each grasp world→robot.
    T_world_robot_inv = tra.inverse_matrix(robot_base_T)

    # Build the list of grasps to try, in confidence-descending order.
    order = np.argsort(-conf)
    if force_idx >= 0:
        if force_idx >= len(grasps_world):
            log.warning(
                "--force_grasp_idx %d out of range (have %d grasps); "
                "falling back to confidence-ranked",
                force_idx,
                len(grasps_world),
            )
            try_idxs_list = [int(i) for i in order[:max_attempts]]
        else:
            try_idxs_list = [int(force_idx)]
            log.info(
                "Forcing grasp idx=%d (conf=%.3f)", force_idx, float(conf[force_idx])
            )
    else:
        try_idxs_list = [int(i) for i in order[:max_attempts]]

    def _grasp_pose_dict(idx_subset):
        """Build the grasp goalset for cuRobo from a list of grasp indices.

        Returns a {link: Pose} dict (lab fork) or a 5D GoalToolPose (public
        cuRobo), whichever the installed planner wants — see curobo_compat.
        """
        positions, quats = [], []
        for idx in idx_subset:
            T_target_robot = T_world_robot_inv @ grasps_world[idx] @ T_offset
            p, q = matrix_to_xyz_quat_wxyz(T_target_robot)
            positions.append(p)
            quats.append(q)
        pos_t = torch.tensor(positions, device="cuda", dtype=torch.float32).unsqueeze(0)
        quat_t = torch.tensor(quats, device="cuda", dtype=torch.float32).unsqueeze(0)
        from curobo_compat import grasp_goals

        return grasp_goals(target_link, pos_t, quat_t)

    def _try(
        grasp_poses,
        approach_offset: float,
        plan_to_grasp: bool,
        plan_to_lift: bool,
        lift_offset: float = -0.20,
    ):
        return planner.plan_grasp(
            grasp_poses,
            q_start,
            grasp_approach_axis="z",
            grasp_approach_offset=approach_offset,
            grasp_approach_in_tool_frame=True,
            grasp_lift_axis="z",
            grasp_lift_offset=lift_offset,
            grasp_lift_in_tool_frame=True,
            plan_approach_to_grasp=plan_to_grasp,
            plan_grasp_to_lift=plan_to_lift,
            disable_collision_links=[target_link],
        )

    # Two modes:
    #   - default: single goalset of all candidates, let cuRobo pick;
    #   - rank-by-confidence (or force_idx): iterate one at a time so
    #     cuRobo plans to a specific grasp (the highest-conf reachable
    #     one) instead of the easiest-to-reach.
    iterate_singletons = rank_by_confidence or (force_idx >= 0)
    if iterate_singletons:
        outer_batches = [[i] for i in try_idxs_list]
        log.info(
            "plan_grasp: iterating %d grasps in confidence order (singleton goalsets)",
            len(outer_batches),
        )
    else:
        outer_batches = [try_idxs_list]
        log.info("plan_grasp goalset: %d candidates", len(try_idxs_list))

    result = None
    last_status = None
    kept_idxs: List[int] = []
    for batch_n, idx_subset in enumerate(outer_batches):
        kept_idxs = idx_subset
        grasp_poses = _grasp_pose_dict(idx_subset)
        # Strategy table. For the "full" strategy (which plans the lift
        # segment too) we sweep BOTH approach offsets AND lift heights
        # because either step can be infeasible:
        #   * `Planning to grasp pose failed` -> the pre-grasp -> grasp
        #     IK/graph search couldn't connect; try a shorter approach
        #     offset (less travel for the wrist).
        #   * Lift step failed (joint limit at the top of the lift arc
        #     for tall objects) -> try a shorter lift.
        #
        # We try all combinations of {approach, lift}, preferring the
        # LARGEST viable approach and the TALLEST viable lift (in that
        # order). Only fall back to no-lift strategies after all
        # combinations exhaust.
        #
        # Sign convention: grasp_lift_axis="z" + tool frame => the
        # lift target is at grasp_pose translated by lift_offset along
        # the tool's +Z. Tool +Z points AWAY from the object (gripper
        # backplate direction), so a NEGATIVE offset lifts the object
        # upward. Higher magnitude = taller lift. Same for approach
        # offset along the approach axis.
        strategies = [
            (-0.15, True, True, -0.20, "full (a=15, lift=20)"),
            (-0.15, True, True, -0.12, "full (a=15, lift=12)"),
            (-0.15, True, True, -0.06, "full (a=15, lift=6)"),
            (-0.10, True, True, -0.20, "full (a=10, lift=20)"),
            (-0.10, True, True, -0.12, "full (a=10, lift=12)"),
            (-0.10, True, True, -0.06, "full (a=10, lift=6)"),
            (-0.07, True, True, -0.20, "full (a=7, lift=20)"),
            (-0.07, True, True, -0.12, "full (a=7, lift=12)"),
            (-0.07, True, True, -0.06, "full (a=7, lift=6)"),
            (-0.10, True, False, -0.20, "approach+grasp"),
            (-0.10, False, False, -0.20, "approach only"),
            (-0.05, False, False, -0.20, "short approach"),
        ]
        for approach_offset, p_grasp, p_lift, lift_offset, label in strategies:
            try:
                result = _try(
                    grasp_poses, approach_offset, p_grasp, p_lift, lift_offset
                )
            except Exception as e:
                log.warning(
                    "plan_grasp raised on '%s' (batch %d): %s", label, batch_n, e
                )
                continue
            last_status = getattr(result, "status", "<no status>")
            success_flag = result.success is not None and bool(result.success.any())
            if iterate_singletons:
                log.info(
                    "  grasp idx=%d (conf=%.3f) '%s' -> status=%s success=%s",
                    idx_subset[0],
                    float(conf[idx_subset[0]]),
                    label,
                    last_status,
                    success_flag,
                )
            else:
                log.info(
                    "plan_grasp '%s' status=%s success=%s",
                    label,
                    last_status,
                    success_flag,
                )
            if success_flag:
                log.info("plan_grasp succeeded with strategy: %s", label)
                break
            if (
                not p_grasp
                and result is not None
                and result.approach_success is not None
                and bool(result.approach_success.any())
            ):
                log.info("plan_grasp accepted approach-only (label: %s)", label)
                break
        else:
            continue  # inner loop exhausted without break → try next grasp
        break  # outer: inner broke (success), stop
    else:
        log.warning("All plan_grasp strategies failed; last status: %s", last_status)
        return False, result, -1, None, None

    if result is None:
        return False, None, -1, None, None
    # We may have "broken" out of the for-else because of approach-only success
    # (where result.success may still be False); use the approach_success flag
    # to determine if we should proceed.
    has_approach = result.approach_success is not None and bool(
        result.approach_success.any()
    )
    if not has_approach:
        return False, result, -1, None, None

    chosen_in_goalset = (
        int(result.goalset_index.view(-1)[0].item())
        if result.goalset_index is not None
        else -1
    )
    target_idx = (
        kept_idxs[chosen_in_goalset] if 0 <= chosen_in_goalset < len(kept_idxs) else -1
    )
    log.info(
        "plan_grasp chose goalset_idx=%d (original grasp #%d, conf=%.3f)",
        chosen_in_goalset,
        target_idx,
        float(conf[target_idx]) if target_idx >= 0 else -1.0,
    )

    # Split the three interpolated trajectories. We keep the lift segment
    # separate from approach+grasp so the caller can insert a "close
    # gripper at the grasp pose" phase between them — physically, you
    # close the fingers BEFORE lifting, not after.
    def _last_idx(x):
        # cuRobo interpolated trajectories are a fixed-length buffer padded
        # with the final pose after the motion ends; *_interpolated_last_tstep
        # marks the last *real* waypoint. Trimming to it drops the static tail
        # (otherwise the arm appears to hang at the goal — very long on public
        # cuRobo, whose interpolation buffer is large).
        if x is None:
            return None
        try:
            return int(x.view(-1)[0].item())
        except Exception:
            try:
                return int(x)
            except Exception:
                return None

    def _traj_to_np(t, last_tstep=None):
        if t is None:
            return None
        pos = t.position.detach().cpu().numpy()
        while pos.ndim > 2:
            pos = pos[0]
        pos = pos.astype(np.float32)
        li = _last_idx(last_tstep)
        if li is not None and 0 <= li < pos.shape[0] - 1:
            pos = pos[: li + 1]
        return pos

    pre_segments = []
    for name, traj, lt in [
        (
            "approach",
            result.approach_interpolated_trajectory,
            getattr(result, "approach_interpolated_last_tstep", None),
        ),
        (
            "grasp",
            result.grasp_interpolated_trajectory,
            getattr(result, "grasp_interpolated_last_tstep", None),
        ),
    ]:
        pos = _traj_to_np(traj, lt)
        if pos is None:
            log.info("  segment %s: none", name)
            continue
        log.info("  segment %s: %d waypoints", name, pos.shape[0])
        pre_segments.append(pos)

    lift_np = _traj_to_np(
        result.lift_interpolated_trajectory,
        getattr(result, "lift_interpolated_last_tstep", None),
    )
    if lift_np is not None:
        log.info("  segment lift: %d waypoints", lift_np.shape[0])

    if not pre_segments:
        return False, result, target_idx, None, None

    # Return the approach + grasp + lift segments concatenated as the
    # "pregrasp" trajectory (kept for backward-compat), and stash the
    # individual segments on ``result`` so tasks that want a finer-grained
    # state machine can pick them apart.
    joint_traj = np.concatenate(pre_segments, axis=0)
    result._segments = {
        "approach": pre_segments[0] if len(pre_segments) >= 1 else None,
        "grasp": pre_segments[1] if len(pre_segments) >= 2 else None,
        "lift": lift_np,
    }
    return True, result, target_idx, joint_traj, lift_np


# ---------------------------------------------------------------------------
# Trajectory export + viser visualization
# ---------------------------------------------------------------------------


def export_trajectory(
    bundle: SceneBundle,
    fk: URDFFK,
    profile: RobotProfile,
    joint_traj: np.ndarray,
    grasps_world: np.ndarray,
    target_idx: int,
    camera_eye: List[float],
    camera_target: List[float],
    output_path: Path,
    fps: int = 30,
):
    """Build the trajectory JSON used by render_trajectory_mp4.py.

    joint_traj: ``(T, n_arm + n_gripper)`` numpy array with column layout
    matching the profile (arm joints first, then each master gripper
    joint).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Static meshes (table, object): write each to a side-car .obj so the
    # renderer doesn't need to know how scene_synthesizer generated them.
    static_dir = output_path.parent / "static_meshes"
    static_dir.mkdir(exist_ok=True)

    static: Dict[str, Dict[str, Any]] = {}
    for name, (mesh, T_world) in bundle.vis_meshes.items():
        rel = f"static_meshes/{name}.obj"
        try:
            mesh.export(static_dir / f"{name}.obj")
        except Exception as e:
            log.warning("Failed to export %s: %s", name, e)
            continue
        static[name] = {
            "mesh_rel": rel,
            "transform": T_world.tolist(),
        }

    # The renderer's base_dir defaults to the JSON file's parent; we place mesh
    # paths so they're either:
    #   * Relative to base_dir for static meshes (under static_meshes/)
    #   * Relative to the URDF asset root for robot link meshes
    # We write `base_dir = output_path.parent` and embed *absolute* link mesh
    # paths so resolution always works.

    actuated = fk.actuated_joint_names()
    n_arm = profile.n_arm

    frames = []
    for t in range(joint_traj.shape[0]):
        cfg_dict: Dict[str, float] = {n: 0.0 for n in actuated}
        for n, v in zip(profile.arm_joint_names, joint_traj[t, :n_arm]):
            cfg_dict[n] = float(v)
        for k, gname in enumerate(profile.gripper_joint_names):
            col = n_arm + k
            if col < joint_traj.shape[1]:
                cfg_dict[gname] = float(joint_traj[t, col])
            else:
                cfg_dict[gname] = profile.open_value(gname)
        # Compute world-frame mesh transforms for every link visual
        link_to_world_mesh = fk.link_poses_with_visual_offset(
            cfg_dict, base_T=bundle.robot_base_T
        )
        parts = []
        for vis, T_world_mesh in link_to_world_mesh:
            # Resolve mesh path to an absolute path for portability.
            mesh_abs = vis.mesh_rel
            try:
                p = Path(vis.mesh_rel)
                if not p.is_absolute():
                    p = (Path(bundle.robot_asset_root) / p).resolve()
                if p.is_file():
                    mesh_abs = str(p)
            except Exception:
                pass
            parts.append(
                {
                    "name": vis.link_name,
                    "mesh_rel": mesh_abs,
                    "transform": T_world_mesh.tolist(),
                }
            )
        frames.append(
            {
                "phase": "plan",
                "joint_position": joint_traj[t].tolist(),
                "parts": parts,
            }
        )

    target_grasp = (
        grasps_world[target_idx].tolist()
        if 0 <= target_idx < len(grasps_world)
        else None
    )
    annotations = {
        "all_grasps": [g.tolist() for g in grasps_world],
        "target_grasp_transform": target_grasp,
    }

    traj = {
        "fps": fps,
        "total_frames": len(frames),
        "base_dir": str(output_path.parent.resolve()),
        "camera": {"eye": camera_eye, "target": camera_target, "up": [0, 0, 1]},
        "static": static,
        "annotations": annotations,
        "frames": frames,
    }
    output_path.write_text(json.dumps(traj))
    log.info("Trajectory JSON: %s (%d frames)", output_path, len(frames))


def viser_visualize(
    bundle: SceneBundle,
    profile: RobotProfile,
    grasps_world: np.ndarray,
    conf: np.ndarray,
    fk: URDFFK,
    joint_traj: Optional[np.ndarray],
    target_idx: int,
    port: int,
):
    """Spin up a viser server and walk through scene → grasps → trajectory.

    Joint mapping uses the profile so the same code drives any robot
    (UR10e+Robotiq, Franka Panda, …) without hardcoded joint orders.
    """
    from graspgenx.utils.viser_utils import (
        create_visualizer,
        visualize_mesh,
        visualize_x_grasp,
        get_color_from_score,
    )

    # GraspGenX renames visualize_grasp → visualize_x_grasp; alias so the rest
    # of the function below can stay symmetric with the original GraspGen path.
    visualize_grasp = visualize_x_grasp

    vis = create_visualizer(port=port)
    log.info("Viser scene running at http://localhost:%d", port)

    # 1. Static scene
    for name, (mesh, T_world) in bundle.vis_meshes.items():
        color = [180, 200, 220] if name == "object" else [200, 180, 140]
        try:
            visualize_mesh(vis, name, mesh, color=color, transform=T_world)
        except Exception as e:
            log.warning("Failed to visualize mesh %s: %s", name, e)

    actuated = fk.actuated_joint_names()
    n_arm = profile.n_arm

    def _cfg_from_traj_row(jp: np.ndarray) -> Dict[str, float]:
        cfg = {n: 0.0 for n in actuated}
        for n, v in zip(profile.arm_joint_names, jp[:n_arm]):
            cfg[n] = float(v)
        for k, gname in enumerate(profile.gripper_joint_names):
            col = n_arm + k
            cfg[gname] = float(jp[col]) if col < len(jp) else profile.open_value(gname)
        return cfg

    # 2. Robot at default arm pose, gripper open
    default_arm = list(profile.default_arm_q)
    grip_open = [profile.open_value(n) for n in profile.gripper_joint_names]
    default_full = np.array(default_arm + grip_open, dtype=float)
    cfg_dict = _cfg_from_traj_row(default_full)
    for i, (lv, T) in enumerate(
        fk.link_poses_with_visual_offset(cfg_dict, base_T=bundle.robot_base_T)
    ):
        try:
            visualize_mesh(
                vis,
                f"robot/{lv.link_name}_{i}",
                lv.mesh,
                color=[130, 145, 165],
                transform=T,
            )
        except Exception:
            pass

    # 3. Grasps (color by confidence). visualize_x_grasp wants an
    # XGripperInfo *object*, not just the name string — otherwise it
    # falls back to a generic 8cm stub.
    if len(grasps_world) > 0:
        from graspgenx.x_grippers import resolve_gripper_info

        gg_cfg = profile.graspgen
        assets_dir = gg_cfg.assets_dir or str(
            Path(__file__).resolve().parent.parent
            / "ext/gripper_descriptions/gripper_descriptions/assets"
        )
        gripper_info = resolve_gripper_info(gg_cfg.gripper_name, assets_dir)
        colors = get_color_from_score(conf, use_255_scale=True)
        for i, g in enumerate(grasps_world):
            try:
                visualize_grasp(
                    vis,
                    f"grasps/g_{i:03d}",
                    g,
                    color=colors[i] if i < len(colors) else [128, 128, 0],
                    gripper_info=gripper_info,
                    linewidth=0.5,
                )
            except Exception:
                pass

    # 4. Animate trajectory
    if joint_traj is not None and len(joint_traj) > 0:
        log.info("Animating trajectory (%d waypoints)…", len(joint_traj))
        try:
            for t in range(joint_traj.shape[0]):
                cfg_dict = _cfg_from_traj_row(joint_traj[t])
                for i, (lv, T_world) in enumerate(
                    fk.link_poses_with_visual_offset(
                        cfg_dict, base_T=bundle.robot_base_T
                    )
                ):
                    visualize_mesh(
                        vis,
                        f"robot/{lv.link_name}_{i}",
                        lv.mesh,
                        color=[130, 145, 165],
                        transform=T_world,
                    )
                time.sleep(0.03)
        except Exception as e:
            log.warning("Trajectory animation interrupted: %s", e)

    log.info("Done; press Ctrl+C to exit (viser still serving)")


def compute_close_value_for_object(
    fk: URDFFK,
    object_mesh: trimesh.Trimesh,
    T_world_object: np.ndarray,
    T_world_grasp: np.ndarray,
    finger_pad_links: List[str],
    gripper_joint: str,
    max_close: float = 1.0,
    padding: float = 0.005,
    n_samples: int = 40,
) -> Tuple[float, float, float]:
    """Find the gripper joint angle that just contains the object plus padding.

    Walks `gripper_joint` from 0 to `max_close`, FK-evaluates the gap between
    `finger_pad_links[0]` and `finger_pad_links[1]`, and returns the largest
    joint value where the gap still ≥ object_width + padding (so the next
    increment would cause penetration). Falls back to `max_close` if the
    object is narrower than the gripper can close to.

    Returns `(close_value, gap_at_close_value, object_width)`.
    """
    actuated = fk.actuated_joint_names()
    cfg = {n: 0.0 for n in actuated}
    if gripper_joint not in actuated:
        raise KeyError(f"{gripper_joint!r} not in actuated joints: {actuated}")

    pad_a, pad_b = finger_pad_links[:2]
    samples = np.linspace(0.0, max_close, n_samples)
    gaps = []
    for v in samples:
        cfg[gripper_joint] = float(v)
        Ts = fk.fk(cfg, base_T=np.eye(4), link_names=[pad_a, pad_b])
        gaps.append(float(np.linalg.norm(Ts[pad_b][:3, 3] - Ts[pad_a][:3, 3])))
    gaps = np.asarray(gaps)

    # Identify the closing axis in the gripper-base frame (the one along which
    # the two pads are separated when the gripper is fully open).
    cfg[gripper_joint] = 0.0
    Ts0 = fk.fk(cfg, base_T=np.eye(4), link_names=[pad_a, pad_b])
    sep_vec = Ts0[pad_b][:3, 3] - Ts0[pad_a][:3, 3]
    closing_axis_idx = int(np.argmax(np.abs(sep_vec)))

    # Object vertices in the grasp frame: world ← object × T_world_grasp⁻¹.
    verts = object_mesh.vertices
    n = len(verts)
    verts_h = np.hstack([verts, np.ones((n, 1))])
    verts_world = (T_world_object @ verts_h.T).T[:, :3]
    T_grasp_world = np.linalg.inv(T_world_grasp)
    verts_grasp = (T_grasp_world @ np.hstack([verts_world, np.ones((n, 1))]).T).T[:, :3]
    obj_width = float(
        verts_grasp[:, closing_axis_idx].max() - verts_grasp[:, closing_axis_idx].min()
    )

    target_gap = obj_width + padding
    # `gaps` is monotonically decreasing in `samples`. Pick the largest sample
    # whose gap still ≥ target_gap. If even sample 0 (fully open) is too narrow,
    # the object is wider than the gripper — return 0 (stay open) as a fallback.
    feasible = np.where(gaps >= target_gap)[0]
    if len(feasible) == 0:
        return 0.0, float(gaps[0]), obj_width
    idx = int(feasible[-1])
    return float(samples[idx]), float(gaps[idx]), obj_width


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load configs
    log.info("Loading robot config: %s", args.robot_config)
    robot_cfg = load_yaml(args.robot_config)
    log.info("Loading env config: %s", args.env_config)
    env_cfg = load_yaml(args.env_config)
    profile: RobotProfile = RobotProfile.from_yaml(robot_cfg)
    log.info(
        "Robot profile: %s — %d arm joints + %d master gripper joints",
        profile.NAME,
        profile.n_arm,
        profile.n_gripper,
    )

    # Run dir
    if args.export_trajectory is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = _HERE / "runs" / ts
        export_path = run_dir / "trajectory.json"
    else:
        export_path = Path(args.export_trajectory).resolve()
        run_dir = export_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Run directory: %s", run_dir)

    # Build scene. Two modes:
    #   * Single-object (default): env YAML has `object_slot:` + --mesh_file
    #     is required. Uses scene_builder.build_scene.
    #   * Multi-object (clutter_pick_and_drop task): env YAML has
    #     `object_slots: [...]` + --mesh_file is ignored. Uses
    #     scene_builder.build_clutter_scene.
    log.info("Building scene…")
    if args.task == "clutter_pick_and_drop":
        from scene_builder import build_clutter_scene  # noqa: E402

        bundle = build_clutter_scene(env_cfg, robot_cfg, seed=args.seed)
        log.info(
            "Clutter scene: %d objects, %d collision obstacles",
            len(bundle.objects),
            len(bundle.collision_world),
        )
    else:
        if args.mesh_file is None:
            log.error("--mesh_file is required for task %r", args.task)
            sys.exit(2)
        bundle = build_scene(env_cfg, robot_cfg, str(args.mesh_file), seed=args.seed)
        log.info(
            "Object placed at %s; %d collision obstacles registered",
            bundle.object_world_T[:3, 3].tolist(),
            len(bundle.collision_world),
        )

    # FK helper (used by viser + JSON export)
    fk = URDFFK(bundle.robot_urdf_path, asset_root=bundle.robot_asset_root)

    # Multi-object clutter dispatch: short-circuit the single-object
    # GraspGenX + cuRobo + simulate path and hand off to clutter_task,
    # which builds its own per-object grasps + per-pick cuRobo planners
    # and drives a DynamicSession through the queue/retry loop.
    if args.task == "clutter_pick_and_drop":
        if args.playback_mode != "dynamic":
            log.error("clutter_pick_and_drop only supports --playback_mode dynamic")
            sys.exit(2)
        # Load the GraspGenX sampler ONCE; clutter_task re-uses it per object.
        from graspgenx.grasp_server import GraspGenXSampler
        from graspgenx.utils.checkpoint_io import load_model_cfg

        gg_cfg = robot_cfg["graspgen"]
        ckpts_dir = _resolve_checkpoints_dir(gg_cfg)
        assets_dir = gg_cfg.get("assets_dir") or str(
            Path(__file__).resolve().parent.parent
            / "ext/gripper_descriptions/gripper_descriptions/assets"
        )
        model_cfg = load_model_cfg(
            str(ckpts_dir / "gen"),
            str(ckpts_dir / "dis"),
            gg_cfg.get("gen_pth"),
            gg_cfg.get("dis_pth"),
        )
        sampler = GraspGenXSampler(
            model_cfg, gg_cfg["gripper_name"], assets_dir=assets_dir
        )
        from clutter_task import run_clutter_task  # noqa: E402

        run_clutter_task(
            bundle=bundle,
            profile=profile,
            robot_cfg=robot_cfg,
            env_cfg=env_cfg,
            combo_path=args.robot_config,
            sampler=sampler,
            args=args,
            out_path=export_path,
        )
        # MP4 rendering happens at the end of main() — fall through to that
        # block (skip the single-object GraspGen + cuRobo + simulate section).
        joint_traj_np = None  # signal "no single-object trajectory to encode"
        _skip_to_mp4 = True
    else:
        _skip_to_mp4 = False

    if _skip_to_mp4:
        # The clutter task already wrote the trajectory JSON via
        # DynamicSession.export. Encode the MP4 directly and exit.
        if args.render_mp4 is not None:
            mp4_path = Path(args.render_mp4).resolve()
            log.info("Encoding clutter MP4 to %s", mp4_path)
            cmd = [
                sys.executable,
                str(_HERE / "render_trajectory_mp4.py"),
                "--trajectory",
                str(export_path),
                "--output",
                str(mp4_path),
                "--resolution",
                args.mp4_resolution,
                "--fps",
                str(args.mp4_fps),
            ]
            if args.show_grasps_in_mp4:
                cmd.append("--show-grasps")
            env = os.environ.copy()
            env["PYOPENGL_PLATFORM"] = "egl"
            env["PYGLET_HEADLESS"] = "true"
            res = subprocess.run(cmd, env=env)
            if res.returncode != 0:
                log.error("Clutter MP4 rendering failed (exit %d)", res.returncode)
        return

    # GraspGen
    grasps_world, conf = run_graspgen(
        bundle,
        robot_cfg,
        num_sample_points=args.num_sample_points,
        num_grasps=args.num_grasps,
        topk=args.topk,
        seed=args.seed,
        grasp_threshold=args.grasp_threshold,
        planner=args.planner,
        moe_obb_density=args.moe_obb_density,
        obb_only=args.obb_only,
    )
    log.info(
        "GraspGen returned %d grasps; conf range [%.3f, %.3f]",
        len(grasps_world),
        float(conf.min()) if len(conf) else 0.0,
        float(conf.max()) if len(conf) else 0.0,
    )

    if len(grasps_world) == 0:
        log.error("GraspGen produced no grasps; aborting before motion planning.")
        sys.exit(2)

    # cuRobo planning
    scene_model = collision_world_to_curobo(bundle.collision_world, bundle.robot_base_T)
    log.info("Collision scene_model: %s", json.dumps(scene_model, default=str)[:400])
    # Goalset must be at least as large as the candidate set we'll send to plan_grasp.
    planner = init_planner(
        args.robot_config,
        robot_cfg,
        scene_model,
        max_goalset=max(args.max_plan_attempts, len(grasps_world), 1),
    )

    force_idx_eff = args.force_grasp_idx
    if args.singleton_top_conf:
        if args.force_grasp_idx >= 0:
            log.error(
                "--singleton_top_conf and --force_grasp_idx are mutually exclusive"
            )
            sys.exit(2)
        if len(conf) == 0:
            log.error("--singleton_top_conf set but GraspGen returned 0 grasps")
            sys.exit(2)
        force_idx_eff = int(np.argmax(conf))
        log.info(
            "--singleton_top_conf: planning to grasp idx=%d (conf=%.3f) as singleton goalset",
            force_idx_eff,
            float(conf[force_idx_eff]),
        )
    success, result, target_idx, pregrasp_traj, lift_traj = plan_to_grasp(
        planner,
        robot_cfg,
        grasps_world,
        conf,
        max_attempts=args.max_plan_attempts,
        seed=args.seed,
        robot_base_T=bundle.robot_base_T,
        force_idx=force_idx_eff,
        rank_by_confidence=args.rank_grasps_by_confidence,
    )
    # The trajectory is built as: approach + grasp + close + lift, so that
    # fingers close at the grasp pose BEFORE the arm lifts. This is what
    # actually happens in a real pick-and-place: you grasp, then lift, not
    # the other way around. In kinematic mode it's purely cosmetic; in
    # dynamic mode it's essential — the object only follows the gripper
    # up if the fingers are already closed around it.
    joint_traj_np = None
    if success and pregrasp_traj is not None:
        # The action sequence (approach + grasp + close + lift, optionally
        # transport + drop + release for pick_and_drop_in_bin, etc.) is
        # owned by the Task profile. The task returns the full
        # arm-plus-gripper joint trajectory.
        task = get_task(args.task)
        log.info("Task: %s", task.NAME)
        task_result = task.plan_actions(
            planner=planner,
            bundle=bundle,
            profile=profile,
            grasps_world=grasps_world,
            conf=conf,
            target_idx=target_idx,
            pregrasp_traj=pregrasp_traj,
            lift_traj=lift_traj,
            env_cfg=env_cfg,
            close_frames=args.close_frames,
            hold_frames=args.hold_frames,
            hold_after_close_frames=args.hold_after_close_frames,
            playback_mode=args.playback_mode,
            result=result,
        )
        joint_traj_np = task_result.joint_traj
        log.info(
            "Trajectory: %d waypoints, %d cols (arm=%d, gripper=%d). " "Segments: %s",
            joint_traj_np.shape[0],
            joint_traj_np.shape[1],
            profile.n_arm,
            profile.n_gripper,
            ", ".join(f"{n}={k}" for n, k in task_result.segments),
        )
    else:
        log.warning("Skipping trajectory export — plan_grasp failed.")

    # Resolve camera defaults: CLI > env YAML > sensible default.
    env_camera = (env_cfg.get("visual") or {}).get("camera", {})
    cam_eye = (
        args.camera_eye
        if args.camera_eye is not None
        else env_camera.get("eye", [1.2, -1.0, 1.1])
    )
    cam_target = args.camera_target
    if cam_target is None:
        cam_target = env_camera.get("target", "object")
    if cam_target == "object":
        cam_target = bundle.object_world_T[:3, 3].tolist()

    # Always export trajectory JSON if we have one. In dynamic mode, the
    # JSON written here is the SIMULATED trajectory (object moves, joint
    # angles reflect what Newton's PD controller actually achieved under
    # contact). In kinematic mode it's the planned joint trajectory with
    # the object pinned at its initial pose.
    if joint_traj_np is not None and args.playback_mode == "dynamic":
        from dynamic_playback import simulate_and_export  # noqa: E402

        log.info("Playback mode: dynamic — replaying trajectory in Newton")
        simulate_and_export(
            bundle=bundle,
            profile=profile,
            joint_traj=joint_traj_np,
            out_path=export_path,
            grasps_world=grasps_world,
            target_idx=target_idx,
            camera_eye=list(cam_eye),
            camera_target=list(cam_target),
            sim_fps=args.sim_fps,
            sim_dt=args.sim_dt,
            arm_kp=args.arm_kp,
            arm_kd=args.arm_kd,
            finger_kp=args.finger_kp,
            finger_kd=args.finger_kd,
            gravity=args.gravity,
            settle_frames=args.settle_frames,
            object_mass=args.object_mass,
            object_mu=args.object_mu,
            finger_mu=args.finger_mu,
        )
    elif joint_traj_np is not None:
        export_trajectory(
            bundle=bundle,
            fk=fk,
            profile=profile,
            joint_traj=joint_traj_np,
            grasps_world=grasps_world,
            target_idx=target_idx,
            camera_eye=list(cam_eye),
            camera_target=list(cam_target),
            output_path=export_path,
            fps=args.mp4_fps,
        )

    # MP4 (third-person view)
    if args.render_mp4 is not None and joint_traj_np is not None:
        mp4_path = Path(args.render_mp4).resolve()
        log.info("Encoding MP4 to %s", mp4_path)
        cmd = [
            sys.executable,
            str(_HERE / "render_trajectory_mp4.py"),
            "--trajectory",
            str(export_path),
            "--output",
            str(mp4_path),
            "--resolution",
            args.mp4_resolution,
            "--fps",
            str(args.mp4_fps),
        ]
        if args.show_grasps_in_mp4:
            cmd.append("--show-grasps")
        env = os.environ.copy()
        env["PYOPENGL_PLATFORM"] = "egl"
        env["PYGLET_HEADLESS"] = "true"
        res = subprocess.run(cmd, env=env)
        if res.returncode != 0:
            log.error("MP4 rendering failed (exit %d)", res.returncode)

    # Wrist-cam MP4: re-render with per-frame camera attached to the gripper.
    if args.wrist_cam_mp4 is not None and joint_traj_np is not None:
        wrist_link = args.wrist_cam_link or profile.tool_frame
        eye_off = np.array(args.wrist_cam_eye_offset, dtype=float)
        tgt_off = np.array(args.wrist_cam_target_offset, dtype=float)

        # Recompute per-waypoint world transforms for the wrist link via FK.
        # Read joint positions from each frame's own `joint_position` slot —
        # in dynamic mode that's the SIMULATED state, and the frame count
        # may exceed `len(joint_traj_np)` because of settle frames.
        actuated = fk.actuated_joint_names()
        n_arm = profile.n_arm
        wrist_traj = json.loads(export_path.read_text())
        for t, frame in enumerate(wrist_traj["frames"]):
            jp = frame.get("joint_position")
            if jp is None:
                jp = list(joint_traj_np[min(t, len(joint_traj_np) - 1)])
            cfg_dict = {n: 0.0 for n in actuated}
            for n, v in zip(profile.arm_joint_names, jp[:n_arm]):
                cfg_dict[n] = float(v)
            for k, gname in enumerate(profile.gripper_joint_names):
                col = n_arm + k
                if col < len(jp):
                    cfg_dict[gname] = float(jp[col])
            T_world_link = fk.fk(cfg_dict, base_T=bundle.robot_base_T)[wrist_link]
            eye_world = (T_world_link @ np.append(eye_off, 1.0))[:3]
            target_world = (T_world_link @ np.append(tgt_off, 1.0))[:3]
            # Up = wrist link's +X axis (the closing direction of robotiq fingers),
            # so the camera image is upright relative to the gripper.
            up_world = T_world_link[:3, 0]
            frame["camera"] = {
                "eye": eye_world.tolist(),
                "target": target_world.tolist(),
                "up": up_world.tolist(),
            }

        wrist_traj_path = export_path.with_name(export_path.stem + "_wristcam.json")
        wrist_traj_path.write_text(json.dumps(wrist_traj))
        wrist_mp4 = Path(args.wrist_cam_mp4).resolve()
        log.info("Encoding wrist-cam MP4 to %s", wrist_mp4)

        # Skip rendering arm links that would clip into / occlude the camera.
        # Keep only the gripper base + fingers visible by skipping the arm.
        skip = [
            "base_link_inertia",
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
        ]
        cmd = [
            sys.executable,
            str(_HERE / "render_trajectory_mp4.py"),
            "--trajectory",
            str(wrist_traj_path),
            "--output",
            str(wrist_mp4),
            "--resolution",
            args.mp4_resolution,
            "--fps",
            str(args.mp4_fps),
        ]
        for s in skip:
            cmd += ["--skip-link", s]
        env = os.environ.copy()
        env["PYOPENGL_PLATFORM"] = "egl"
        env["PYGLET_HEADLESS"] = "true"
        res = subprocess.run(cmd, env=env)
        if res.returncode != 0:
            log.error("Wrist-cam MP4 rendering failed (exit %d)", res.returncode)

    # Live viser visualization
    if not args.no_viser:
        viser_visualize(
            bundle,
            profile,
            grasps_world,
            conf,
            fk,
            joint_traj=joint_traj_np,
            target_idx=target_idx,
            port=args.viser_port,
        )
        # Keep alive so user can inspect
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("Bye.")


if __name__ == "__main__":
    main()
