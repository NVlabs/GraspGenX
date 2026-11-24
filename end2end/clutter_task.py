"""Clutter pick-and-drop driver — Task D.

This module orchestrates a multi-object pick-and-drop sequence:

    For each object in the FIFO queue:
      1. Run GraspGenX inference on the object's mesh.
      2. Plan an approach + grasp + close + lift trajectory via cuRobo,
         with the OTHER objects (not yet dropped) registered as cuboid
         obstacles in the planner's world model.
      3. Drive the dynamic sim through the pick segments.
      4. Read the object's world pose at the end of the lift segment;
         if it has risen by ≥ LIFT_SUCCESS_DZ from its initial z, the
         grasp succeeded.
      5. On success: plan_pose to above the bin, drive the sim through
         the transport + open + hold segments, drop the object. Then
         return the arm to home.
         On failure: re-queue the object, return the arm to home, move
         on. After all objects have been tried, retry the failed ones
         up to a per-object retry cap.

The big design lever: a single :class:`DynamicSession` (built once)
holds the Newton model + state. Each call to ``session.drive_segments``
appends to a single growing frames buffer. After the queue is empty,
we call ``session.export`` to write one big trajectory JSON the
renderer consumes.

cuRobo's MotionPlanner is rebuilt per pick so the obstacle set reflects
which objects are still on the table. This adds 10–20 s per pick on
first run, but warmup is cached on subsequent rebuilds so it's manageable
(<5 s typical). The cost is acceptable for the demo's clarity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
import trimesh.transformations as tra

from dynamic_playback import DynamicSession
from registry import CollisionObstacle
from robot_profiles import RobotProfile
from scene_builder import SceneBundle, SceneObject
from tasks import (
    PickAndDropInBinTask,
    _hold,
    _open_close_vals,
    _ramp,
    _resample_traj,
    _slice_arm,
    _stack_arm_and_gripper,
)

log = logging.getLogger("clutter")


# Lift-success threshold: object must have risen by at least this many
# meters from its initial table-top z to count as "in the gripper".
LIFT_SUCCESS_DZ = 0.05

# Default max retries per object across the whole sequence. Once an
# object has been re-queued this many times after a failed grasp /
# transport slip, it's marked failed_max_retries. Overridable at runtime
# via the CLI ``--max_retries_per_object`` flag (lives on args).
MAX_RETRIES_PER_OBJECT_DEFAULT = 2

# Frames the lift segment is resampled to (kept slow so held objects
# don't slip out — same justification as PickAndLiftTask.LIFT_FRAMES).
LIFT_FRAMES = 240
# Frames the move-to-above-bin segment runs over (slow swing).
MOVE_TO_BIN_FRAMES = 360
# Frames for the return-to-home segment after each pick.
RETURN_HOME_FRAMES = 180
# Frames to hold above the bin before opening fingers.
HOLD_ABOVE_BIN_FRAMES = 30
# Frames the gripper takes to open (drop the object).
OPEN_FRAMES = 30
# Frames to wait after opening so the object falls into the bin.
HOLD_AFTER_DROP_FRAMES = 60
# Drop height above the bin pose (m). With bin pose z=0.49 and bin
# walls 0.10 m tall, this puts the drop pose at world z = 0.49 + 0.30
# = 0.79, which is ~20 cm above the bin's top rim — plenty for the
# gripper fingers to clear the rim while keeping the pose well within
# the franka's reach (the previous 0.40 value put the pre-drop pose
# at z=0.94 which was at the EDGE of the franka's workspace and tipped
# out of reach as soon as we added xy jitter to the drop pose).
DROP_HEIGHT_ABOVE_BIN = 0.30
# Extra height above the drop pose for the PRE_DROP waypoint. The
# gripper first swings to (bin_xy, drop_z + PRE_DROP_EXTRA_Z) before
# descending vertically to the drop pose. Keeps the held object well
# above the bin's rim during the lateral move, then approaches the
# release pose with a clean purely-vertical descent.
PRE_DROP_EXTRA_Z = 0.05
# Frames for the pre-drop vertical descent. Short so the descent feels
# crisp but slow enough that physics doesn't shake the held object.
PRE_DROP_DESCEND_FRAMES = 60
# After a release, allow this much xy margin outside the bin's footprint
# when classifying "object landed in bin". Objects can rest against the
# bin's inner wall; we don't want to call those a slip.
BIN_LAND_XY_MARGIN = 0.02


@dataclass
class ClutterObjectState:
    """Per-object bookkeeping across the clutter run."""

    idx: int  # index into bundle.objects
    obj: SceneObject  # the SceneObject reference
    initial_z: float  # world z of the centroid at sim start
    retries: int = 0  # how many times we've attempted this object
    outcome: str = "pending"  # pending / in_bin / failed_max_retries


def _object_aabb_world(
    mesh: trimesh.Trimesh, T_world: np.ndarray
) -> Tuple[List[float], List[float]]:
    """Return ``(world_center, world_half_extents)`` for an axis-aligned
    cuboid that bounds the mesh in the world frame after applying
    ``T_world`` (a 4×4 transform).

    Used to add other on-table objects as cuRobo cuboid obstacles when
    planning a pick on the *target* object. AABB is a loose bound (gets
    larger when the mesh is rotated), but that's fine — we want
    conservative clearance, not tight reachability.
    """
    verts = mesh.vertices
    verts_h = np.hstack([verts, np.ones((len(verts), 1))])
    verts_world = (T_world @ verts_h.T).T[:, :3]
    lo = verts_world.min(axis=0)
    hi = verts_world.max(axis=0)
    center = ((lo + hi) / 2.0).tolist()
    half = ((hi - lo) / 2.0).tolist()
    return center, half


def _bin_translation(env_cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    """World-frame xyz translation of the bin asset, or (0.55, -0.20, 0.50)
    if no bin found (matches the franka envs' nominal layout)."""
    for a in env_cfg.get("assets", []):
        if a.get("id") == "bin":
            t = a.get("pose", {}).get("translation", [0.55, -0.20, 0.50])
            return (float(t[0]), float(t[1]), float(t[2]))
    return (0.55, -0.20, 0.50)


def _bin_footprint_world(
    env_cfg: Dict[str, Any],
) -> Optional[Tuple[float, float, float, float]]:
    """Return ``(x_lo, x_hi, y_lo, y_hi)`` of the bin's outer xy
    footprint in world coords, or None if no bin in env_cfg.

    Used to classify "object landed in the bin" vs "slipped en route" by
    checking the object's xy after the drop + settle.
    """
    for a in env_cfg.get("assets", []):
        if a.get("id") != "bin":
            continue
        pose = a.get("pose", {})
        t = pose.get("translation", [0.0, 0.0, 0.0])
        params = a.get("params", {}) or {}
        # ProceduralBin's width is along x, depth along y. Both default
        # to 0.30 in our envs; thickness is the wall (does NOT extend
        # the footprint).
        w = float(params.get("width", 0.30))
        d = float(params.get("depth", 0.30))
        # Add a small margin: objects can rest against the bin's inner
        # wall, slightly outside the strict "center of mass inside" check.
        return (
            float(t[0]) - w / 2.0 - BIN_LAND_XY_MARGIN,
            float(t[0]) + w / 2.0 + BIN_LAND_XY_MARGIN,
            float(t[1]) - d / 2.0 - BIN_LAND_XY_MARGIN,
            float(t[1]) + d / 2.0 + BIN_LAND_XY_MARGIN,
        )
    return None


def _object_in_bin(
    T_obj: np.ndarray, footprint: Optional[Tuple[float, float, float, float]]
) -> bool:
    """Is the object's centroid inside the bin's xy footprint?

    A "yes" means the object was successfully transported and released
    over the bin; a "no" usually means it slipped during transport and
    landed on the table somewhere short of the bin.
    """
    if footprint is None:
        return True  # no bin to check against; trust the lift-success path
    x, y = float(T_obj[0, 3]), float(T_obj[1, 3])
    x_lo, x_hi, y_lo, y_hi = footprint
    return x_lo <= x <= x_hi and y_lo <= y <= y_hi


def _aabb_obstacle(
    name: str, center: List[float], half: List[float]
) -> CollisionObstacle:
    """Build a cuRobo cuboid CollisionObstacle from a world-frame AABB."""
    dims = [2 * half[0], 2 * half[1], 2 * half[2]]
    # cuRobo pose convention: [x, y, z, qw, qx, qy, qz]
    pose = [center[0], center[1], center[2], 1.0, 0.0, 0.0, 0.0]
    return CollisionObstacle(name=name, type="cuboid", dims=dims, pose=pose)


# How far from the robot base a grasp pose is considered "in reach".
# Matches franka's effective workspace radius with a small safety margin.
GRASP_REACH_MAX = 0.85
GRASP_REACH_MIN = 0.20
# Margin added to neighbouring-object AABBs when checking whether a
# grasp pose intersects them. Matches the obstacle-inflation we use
# during cuRobo planning.
GRASP_AABB_MARGIN = 0.02


def _collision_free_grasp_indices(
    grasps_world: np.ndarray,
    target_idx: int,
    states_per_obj: list,
    session,
    robot_base_T: np.ndarray,
    gripper_mesh: trimesh.Trimesh,
    static_meshes: dict,
    include_target: bool = True,
) -> np.ndarray:
    """Return the indices of grasps that are (a) within the robot's reach
    AND (b) the gripper's vis_mesh at the canonical grasp pose does NOT
    intersect table + bin + every other on-table (non-in-bin) object, and
    (when ``include_target``) the target object itself.

    Uses trimesh.collision.CollisionManager (fcl-backed) — same logic
    as end2end/visualize_scene_grasps.py:classify_grasps so the viser
    preview and the sweep's picker / cuRobo goalset stay in sync.

    ``include_target=True`` (fully-observed mesh mode) rejects grasps whose
    gripper geometry penetrates the target. These indices are used both to
    ORDER candidates (by count) AND to FILTER the goalset handed to cuRobo,
    so cuRobo only ever plans to geometrically collision-free grasps.

    Args:
        gripper_mesh: trimesh.Trimesh from the gripper's vis_mesh.obj
            (canonical frame).
        static_meshes: {name: (mesh, T_world)} for table, bin, ...
            (manipulation objects are added per-call using session poses).
    """
    if len(grasps_world) == 0:
        return np.zeros(0, dtype=int)
    base_xyz = robot_base_T[:3, 3]

    # Build a fresh CollisionManager containing static scene meshes
    # (table, bin) plus every non-target, non-in-bin object at its
    # CURRENT session pose (objects move during the clutter loop), and
    # the target itself when include_target is set.
    mgr = trimesh.collision.CollisionManager()
    for name, (mesh, T_world) in static_meshes.items():
        mgr.add_object(name, mesh, transform=np.asarray(T_world, dtype=np.float64))
    for other in states_per_obj:
        if other.idx == target_idx and not include_target:
            continue
        if other.idx != target_idx and other.outcome == "in_bin":
            continue
        T_now = session.current_object_pose(other.idx)
        mgr.add_object(
            other.obj.asset_id,
            other.obj.mesh,
            transform=np.asarray(T_now, dtype=np.float64),
        )

    cf_idx = []
    for i, T in enumerate(grasps_world):
        pos = T[:3, 3]
        d = float(np.linalg.norm(pos - base_xyz))
        if d > GRASP_REACH_MAX or d < GRASP_REACH_MIN:
            continue
        if not mgr.in_collision_single(
            gripper_mesh, transform=np.asarray(T, dtype=np.float64)
        ):
            cf_idx.append(i)
    return np.asarray(cf_idx, dtype=int)


def run_clutter_task(
    *,
    bundle: SceneBundle,
    profile: RobotProfile,
    robot_cfg: Dict[str, Any],
    env_cfg: Dict[str, Any],
    combo_path: Path,
    sampler,  # GraspGenXSampler, already loaded
    args,  # parsed CLI args
    out_path: Path,
) -> Path:
    """Drive the franka_panda_clutter_pick_and_drop task end-to-end.

    Returns the path the trajectory JSON was written to.
    """
    n_arm = profile.n_arm
    n_grip = profile.n_gripper
    open_vals, close_vals = _open_close_vals(profile)

    home_q = np.array(profile.default_arm_q, dtype=np.float32)

    MAX_RETRIES_PER_OBJECT = int(
        getattr(args, "max_retries_per_object", MAX_RETRIES_PER_OBJECT_DEFAULT)
    )

    log.info(
        "Clutter task: %d objects in the queue (max_retries=%d)",
        len(bundle.objects),
        MAX_RETRIES_PER_OBJECT,
    )
    states_per_obj: List[ClutterObjectState] = []
    for i, obj in enumerate(bundle.objects):
        states_per_obj.append(
            ClutterObjectState(
                idx=i,
                obj=obj,
                initial_z=float(obj.world_T[2, 3]),
            )
        )

    # Load the gripper vis_mesh ONCE for the smart-picker's fcl
    # collision check (mirrors visualize_scene_grasps.py's check so
    # the viser counts and the picker scores stay consistent).
    gg_cfg = robot_cfg["graspgen"]
    gripper_name = gg_cfg["gripper_name"]
    if gg_cfg.get("assets_dir"):
        x_grippers = Path(gg_cfg["assets_dir"]) / "x_grippers"
    else:
        # Same resolution as the GraspGenX sampler: $GRASPGENX_GRIPPER_CFG_DIR,
        # else the auto-cloned ext/gripper_descriptions.
        from graspgenx._setup_dependencies import get_gripper_descriptions_assets

        x_grippers = get_gripper_descriptions_assets()
    vis_mesh_path = x_grippers / gripper_name / "vis_mesh.obj"
    if not vis_mesh_path.exists():
        raise FileNotFoundError(
            f"Smart-picker CF check needs gripper vis_mesh.obj at {vis_mesh_path}"
        )
    gripper_mesh = trimesh.load(str(vis_mesh_path), force="mesh")
    if len(gripper_mesh.vertices) == 0:
        # An un-pulled git-LFS pointer loads as an empty mesh and crashes fcl.
        raise RuntimeError(
            f"{vis_mesh_path} has no vertices (git-LFS pointer?). Run "
            f"`git lfs pull` in {x_grippers.parents[2]}."
        )
    # Static meshes (table, bin, ...) — every entry in bundle.vis_meshes
    # whose name is NOT one of the manipulation objects (those move).
    object_ids = {o.asset_id for o in bundle.objects}
    static_meshes = {
        name: (mesh, T)
        for name, (mesh, T) in bundle.vis_meshes.items()
        if name not in object_ids
    }
    log.info(
        "Smart-picker fcl: gripper_mesh=%d verts, %d static obstacles",
        len(gripper_mesh.vertices),
        len(static_meshes),
    )

    # Build the dynamic session ONCE (expensive: parses URDF, runs
    # CoACD, initializes MuJoCo solver). The session holds Newton state
    # across all picks.
    initial_row = np.concatenate(
        [
            home_q,
            np.array(
                [profile.open_value(n) for n in profile.gripper_joint_names],
                dtype=np.float32,
            ),
        ]
    )
    session = DynamicSession(
        bundle=bundle,
        profile=profile,
        sim_fps=args.sim_fps,
        sim_dt=args.sim_dt,
        arm_kp=args.arm_kp,
        arm_kd=args.arm_kd,
        finger_kp=args.finger_kp,
        finger_kd=args.finger_kd,
        gravity=args.gravity,
        object_mass=args.object_mass,
        object_mu=args.object_mu,
        finger_mu=args.finger_mu,
        initial_joint_q=initial_row,
    )

    # Step the sim a few frames at the start so the objects settle on
    # the table before we attempt the first pick. Otherwise the first
    # grasp may close around an object that's still falling.
    log.info("Settling objects on the table (%d frames)...", args.settle_frames * 2)
    empty_traj = np.zeros((0, n_arm + n_grip), dtype=np.float32)
    session.drive_segments(empty_traj, settle_frames=int(args.settle_frames * 2))

    # Refresh each object's initial_z to the post-settle pose (objects
    # may have shifted slightly when contact resolved with the table).
    for s in states_per_obj:
        T = session.current_object_pose(s.idx)
        s.initial_z = float(T[2, 3])
        log.info("  %s settled at world z=%.3f", s.obj.asset_id, s.initial_z)

    # Build the queue: try objects in declaration order initially.
    queue: List[ClutterObjectState] = list(states_per_obj)
    grasp_annotations: List[Dict[str, Any]] = []

    iteration = 0
    # Hard ceiling on total iterations so a fully-failing object set
    # can't trap us in an infinite re-queue cycle (which CAN happen
    # because re-queuing is "object back to end of queue", not "object
    # removed once retries exhausted" — see the per-object cap below).
    MAX_ITERATIONS = max(16, 6 * len(states_per_obj))

    while queue:
        iteration += 1
        if iteration > MAX_ITERATIONS:
            log.warning(
                "Reached iteration cap (%d); abandoning %d remaining object(s): %s",
                MAX_ITERATIONS,
                len(queue),
                ", ".join(s.obj.asset_id for s in queue),
            )
            for s in queue:
                if s.outcome == "pending":
                    s.outcome = "failed_max_retries"
            queue.clear()
            break

        # ---- Smart picking order ----
        # Run GraspGenX inference on ALL pending objects (skip already
        # in_bin / over the retry cap). Score each by how many of its
        # grasps are (a) within the franka's reach and (b) don't fall
        # inside another on-table object's inflated AABB. Pick the
        # object with the most "collision-free" grasps. If every
        # remaining object has zero collision-free grasps, fall back to
        # a random pick among the candidates so the loop makes progress.
        candidates = [c for c in queue if c.retries < MAX_RETRIES_PER_OBJECT]
        if not candidates:
            # All remaining are over the cap — mark them failed.
            for c in queue:
                if c.outcome == "pending":
                    c.outcome = "failed_max_retries"
            queue.clear()
            break

        per_candidate_grasps: Dict[int, Tuple[np.ndarray, np.ndarray, int]] = {}
        for c in candidates:
            T_now = session.current_object_pose(c.idx)
            try:
                g_world, g_conf = _graspgen_for_object(
                    sampler=sampler,
                    mesh=c.obj.mesh,
                    T_obj_world=T_now,
                    num_sample_points=args.num_sample_points,
                    num_grasps=args.num_grasps,
                    topk=args.topk,
                    grasp_threshold=args.grasp_threshold,
                    planner_name=args.planner,
                    moe_obb_density=args.moe_obb_density,
                )
            except Exception as e:
                log.warning("GraspGenX failed on %s: %s", c.obj.asset_id, e)
                g_world = np.zeros((0, 4, 4), dtype=np.float32)
                g_conf = np.zeros((0,), dtype=np.float32)
            cf_idx = _collision_free_grasp_indices(
                g_world,
                target_idx=c.idx,
                states_per_obj=states_per_obj,
                session=session,
                robot_base_T=bundle.robot_base_T,
                gripper_mesh=gripper_mesh,
                static_meshes=static_meshes,
                include_target=True,
            )
            per_candidate_grasps[c.idx] = (g_world, g_conf, cf_idx)
            log.info(
                "  candidate %s: %d grasps total, %d collision-free",
                c.obj.asset_id,
                len(g_world),
                len(cf_idx),
            )

        # Order candidates by collision-free count (descending). The
        # first one with cf > 0 is our pick; if all are 0, pick a random
        # candidate (the fallback the user asked for).
        scored = sorted(
            candidates,
            key=lambda c: len(per_candidate_grasps[c.idx][2]),
            reverse=True,
        )
        best = scored[0]
        if len(per_candidate_grasps[best.idx][2]) == 0:
            import random

            best = random.choice(candidates)
            log.warning(
                "All candidates have 0 collision-free grasps; "
                "falling back to RANDOM pick: %s",
                best.obj.asset_id,
            )
        s = best
        queue.remove(s)

        log.info("=" * 60)
        log.info(
            "Iteration %d: target=%s (retry %d/%d, " "%d collision-free grasps)",
            iteration,
            s.obj.asset_id,
            s.retries,
            MAX_RETRIES_PER_OBJECT,
            len(per_candidate_grasps[s.idx][2]),
        )

        # Build per-pick obstacle set for cuRobo.
        per_pick_obstacles: List[CollisionObstacle] = list(bundle.collision_world)
        for other in states_per_obj:
            if other.idx == s.idx:
                continue
            if other.outcome == "in_bin":
                continue
            T_now = session.current_object_pose(other.idx)
            center, half = _object_aabb_world(other.obj.mesh, T_now)
            half_safe = [h + 0.01 for h in half]
            per_pick_obstacles.append(
                _aabb_obstacle(f"obj_{other.idx}_aabb", center, half_safe)
            )

        grasps_world, conf, cf_idx = per_candidate_grasps[s.idx]

        if len(grasps_world) == 0:
            s.retries += 1
            log.warning(
                "  no grasps — skipping %s and trying again later", s.obj.asset_id
            )
            queue.append(s)
            continue

        # Feed cuRobo ONLY the fcl-collision-free grasps (gripper mesh
        # clear of table/bin/neighbours/target) so it never plans to a
        # geometrically-colliding grasp. If none are collision-free (the
        # all-zero fallback that selected this object at random), fall back
        # to the full set so cuRobo's own sphere check still gets a chance.
        if len(cf_idx) > 0:
            grasps_world = grasps_world[cf_idx]
            conf = conf[cf_idx]
            log.info(
                "  feeding cuRobo %d collision-free grasps (of %d)",
                len(cf_idx),
                len(per_candidate_grasps[s.idx][0]),
            )
        else:
            log.warning(
                "  no collision-free grasps for %s — feeding cuRobo "
                "the full set as fallback",
                s.obj.asset_id,
            )

        # Plan + drive the pick segments. Starts from the session's
        # current arm config (held at home from the previous pick).
        ok = _plan_and_drive_pick(
            session=session,
            bundle=bundle,
            profile=profile,
            robot_cfg=robot_cfg,
            combo_path=combo_path,
            obstacles=per_pick_obstacles,
            grasps_world=grasps_world,
            conf=conf,
            args=args,
            phase_prefix=f"obj{s.idx}_",
            target_object=s.obj,
        )
        if not ok or session._nan_seen:
            s.retries += 1
            log.warning("  pick segments failed/NaN — skipping %s", s.obj.asset_id)
            if not session._nan_seen:
                queue.append(s)
                # Release whatever the fingers may have partially grabbed
                # (close_fingers may have been mid-segment when the
                # trajectory failed), then return home so the next iter
                # starts clean.
                _release_in_place(
                    session=session,
                    profile=profile,
                    phase_prefix=f"obj{s.idx}_release_after_pick_fail_",
                )
                _plan_and_drive_return_home(
                    session=session,
                    profile=profile,
                    robot_cfg=robot_cfg,
                    combo_path=combo_path,
                    obstacles=per_pick_obstacles,
                    home_q=home_q,
                    args=args,
                    phase_prefix=f"obj{s.idx}_return_after_pick_fail_",
                )
            else:
                # NaN means physics state is corrupt; can't recover.
                break
            continue

        # Grasp success check: is the object higher than its initial z?
        T_lift_end = session.current_object_pose(s.idx)
        dz = float(T_lift_end[2, 3]) - s.initial_z
        log.info(
            "  lift-end object z=%.3f (initial=%.3f, delta=%.3f)",
            float(T_lift_end[2, 3]),
            s.initial_z,
            dz,
        )
        if dz < LIFT_SUCCESS_DZ:
            log.warning("  grasp FAILED (dz=%.3f < %.2f)", dz, LIFT_SUCCESS_DZ)
            s.retries += 1
            # Plan back to home and continue.
            _plan_and_drive_return_home(
                session=session,
                profile=profile,
                robot_cfg=robot_cfg,
                combo_path=combo_path,
                obstacles=per_pick_obstacles,
                home_q=home_q,
                args=args,
                phase_prefix=f"obj{s.idx}_return_after_fail_",
            )
            queue.append(s)
            continue
        log.info("  grasp SUCCEEDED")

        # Plan transport + drop above bin. After release, the object
        # falls into the bin; we mark it as in_bin so it's removed from
        # future obstacle sets.
        ok_drop = _plan_and_drive_drop(
            session=session,
            bundle=bundle,
            profile=profile,
            robot_cfg=robot_cfg,
            combo_path=combo_path,
            obstacles=per_pick_obstacles,
            env_cfg=env_cfg,
            args=args,
            phase_prefix=f"obj{s.idx}_",
        )
        if not ok_drop:
            log.warning("  transport plan failed; re-queueing %s", s.obj.asset_id)
            s.retries += 1
            queue.append(s)
            # 1. Release whatever's in the gripper at the current pose
            #    (the object falls to wherever the gripper is now).
            # 2. Plan return-to-home + open fingers so the next
            #    iteration starts from a clean arm config.
            _release_in_place(
                session=session,
                profile=profile,
                phase_prefix=f"obj{s.idx}_release_after_transport_fail_",
            )
            _plan_and_drive_return_home(
                session=session,
                profile=profile,
                robot_cfg=robot_cfg,
                combo_path=combo_path,
                obstacles=per_pick_obstacles,
                home_q=home_q,
                args=args,
                phase_prefix=f"obj{s.idx}_return_after_transport_fail_",
            )
            continue

        # Post-drop verification: the lift-success check above told us
        # the object was *in the gripper* at lift-end, but the object can
        # still slip out during the move_to_above_bin swing (especially
        # the bowl, which has only ~1 cm of rim wall for the gripper to
        # close around). Check that the object's xy ends up inside the
        # bin's footprint after the drop + settle; if it's outside, it
        # slipped onto the table somewhere along the transport and we
        # need to re-queue.
        bin_footprint = _bin_footprint_world(env_cfg)
        T_after_drop = session.current_object_pose(s.idx)
        if not _object_in_bin(T_after_drop, bin_footprint):
            log.warning(
                "  %s slipped during transport — final xy=(%.3f, %.3f, %.3f) "
                "is outside bin footprint; re-queueing",
                s.obj.asset_id,
                float(T_after_drop[0, 3]),
                float(T_after_drop[1, 3]),
                float(T_after_drop[2, 3]),
            )
            s.retries += 1
            # Reset initial_z for the next attempt to the object's
            # current resting z (it may have settled to a different
            # height on the table).
            s.initial_z = float(T_after_drop[2, 3])
            queue.append(s)
            # Still return to home so the next iteration starts clean.
            _plan_and_drive_return_home(
                session=session,
                profile=profile,
                robot_cfg=robot_cfg,
                combo_path=combo_path,
                obstacles=per_pick_obstacles,
                home_q=home_q,
                args=args,
                phase_prefix=f"obj{s.idx}_return_after_slip_",
            )
            continue

        s.outcome = "in_bin"
        log.info(
            "  %s dropped in bin (final xy=(%.3f, %.3f), z=%.3f)",
            s.obj.asset_id,
            float(T_after_drop[0, 3]),
            float(T_after_drop[1, 3]),
            float(T_after_drop[2, 3]),
        )

        # Return to home so the next pick has predictable start state.
        _plan_and_drive_return_home(
            session=session,
            profile=profile,
            robot_cfg=robot_cfg,
            combo_path=combo_path,
            obstacles=per_pick_obstacles,
            home_q=home_q,
            args=args,
            phase_prefix=f"obj{s.idx}_return_home_",
        )

    # Final settle so the trailing MP4 frames show the final scene
    # state (all objects in bin / on table).
    log.info("Final settle: %d frames", int(args.settle_frames * 2))
    session.drive_segments(
        np.zeros((0, n_arm + n_grip), dtype=np.float32),
        settle_frames=int(args.settle_frames * 2),
    )

    # Final sweep: scan every object's actual resting pose and reconcile
    # ``outcome`` with reality. An object can be marked ``in_bin`` by the
    # per-iteration drop check but later get bumped out (e.g. by a
    # subsequent pick swinging close to the bin) — this sweep catches
    # those and either re-queues for one more attempt (if retries left)
    # or marks them as failed.
    bin_footprint = _bin_footprint_world(env_cfg)
    if bin_footprint is not None:
        leftovers: List[ClutterObjectState] = []
        for s in states_per_obj:
            if s.outcome != "in_bin":
                continue
            T_final = session.current_object_pose(s.idx)
            if not _object_in_bin(T_final, bin_footprint):
                log.warning(
                    "Final sweep: %s was marked in_bin but ended up at "
                    "xy=(%.3f, %.3f, %.3f) — reverting to pending",
                    s.obj.asset_id,
                    float(T_final[0, 3]),
                    float(T_final[1, 3]),
                    float(T_final[2, 3]),
                )
                s.outcome = "pending"
                if s.retries < MAX_RETRIES_PER_OBJECT:
                    leftovers.append(s)
                else:
                    s.outcome = "failed_max_retries"

        # If anything still needs a retry, run one more pass of the
        # queue loop on just the leftovers. Bounded to a single sweep so
        # we don't loop forever on hopelessly-fragile objects.
        if leftovers:
            log.info(
                "Final sweep: %d object(s) need a retry pass: %s",
                len(leftovers),
                ", ".join(s.obj.asset_id for s in leftovers),
            )
            queue.extend(leftovers)
            while queue:
                s = queue.pop(0)
                log.info(
                    "Retry pass: target=%s (retry %d/%d)",
                    s.obj.asset_id,
                    s.retries,
                    MAX_RETRIES_PER_OBJECT,
                )
                if s.retries >= MAX_RETRIES_PER_OBJECT:
                    s.outcome = "failed_max_retries"
                    continue
                # Build obstacles, re-run pick+drop. Reuse the same
                # helpers as the main loop.
                per_pick_obstacles = list(bundle.collision_world)
                for other in states_per_obj:
                    if other.idx == s.idx:
                        continue
                    if other.outcome == "in_bin":
                        continue
                    T_now = session.current_object_pose(other.idx)
                    center, half = _object_aabb_world(other.obj.mesh, T_now)
                    half_safe = [h + 0.01 for h in half]
                    per_pick_obstacles.append(
                        _aabb_obstacle(f"obj_{other.idx}_aabb", center, half_safe)
                    )
                T_obj_now = session.current_object_pose(s.idx)
                try:
                    grasps_world, conf = _graspgen_for_object(
                        sampler=sampler,
                        mesh=s.obj.mesh,
                        T_obj_world=T_obj_now,
                        num_sample_points=args.num_sample_points,
                        num_grasps=args.num_grasps,
                        topk=args.topk,
                        grasp_threshold=args.grasp_threshold,
                        planner_name=args.planner,
                        moe_obb_density=args.moe_obb_density,
                    )
                except Exception as e:
                    log.warning("  GraspGenX failed: %s", e)
                    s.retries += 1
                    continue
                if len(grasps_world) == 0:
                    s.retries += 1
                    continue
                # Same CF filter as the main loop: feed cuRobo only the
                # collision-free grasps (fall back to all if none).
                cf_idx = _collision_free_grasp_indices(
                    grasps_world,
                    target_idx=s.idx,
                    states_per_obj=states_per_obj,
                    session=session,
                    robot_base_T=bundle.robot_base_T,
                    gripper_mesh=gripper_mesh,
                    static_meshes=static_meshes,
                    include_target=True,
                )
                if len(cf_idx) > 0:
                    grasps_world = grasps_world[cf_idx]
                    conf = conf[cf_idx]
                ok = _plan_and_drive_pick(
                    session=session,
                    bundle=bundle,
                    profile=profile,
                    robot_cfg=robot_cfg,
                    combo_path=combo_path,
                    obstacles=per_pick_obstacles,
                    grasps_world=grasps_world,
                    conf=conf,
                    args=args,
                    phase_prefix=f"obj{s.idx}_retry_",
                    target_object=s.obj,
                )
                if not ok or session._nan_seen:
                    s.retries += 1
                    if session._nan_seen:
                        break
                    continue
                T_lift_end = session.current_object_pose(s.idx)
                dz = float(T_lift_end[2, 3]) - s.initial_z
                if dz < LIFT_SUCCESS_DZ:
                    s.retries += 1
                    continue
                ok_drop = _plan_and_drive_drop(
                    session=session,
                    bundle=bundle,
                    profile=profile,
                    robot_cfg=robot_cfg,
                    combo_path=combo_path,
                    obstacles=per_pick_obstacles,
                    env_cfg=env_cfg,
                    args=args,
                    phase_prefix=f"obj{s.idx}_retry_",
                )
                if not ok_drop:
                    s.retries += 1
                    continue
                T_after_drop = session.current_object_pose(s.idx)
                if _object_in_bin(T_after_drop, bin_footprint):
                    s.outcome = "in_bin"
                    log.info("  retry succeeded: %s in bin", s.obj.asset_id)
                else:
                    s.retries += 1
                    log.warning("  retry slipped: %s still outside bin", s.obj.asset_id)
                _plan_and_drive_return_home(
                    session=session,
                    profile=profile,
                    robot_cfg=robot_cfg,
                    combo_path=combo_path,
                    obstacles=per_pick_obstacles,
                    home_q=home_q,
                    args=args,
                    phase_prefix=f"obj{s.idx}_retry_return_",
                )
            # Settle once more so the MP4 ends on a stable state.
            session.drive_segments(
                np.zeros((0, n_arm + n_grip), dtype=np.float32),
                settle_frames=int(args.settle_frames * 2),
            )

    # Build annotations summarizing what happened.
    summary = {
        "queue_order": [s.obj.asset_id for s in states_per_obj],
        "outcomes": {s.obj.asset_id: s.outcome for s in states_per_obj},
        "retries": {s.obj.asset_id: s.retries for s in states_per_obj},
        "per_object_grasps": grasp_annotations,
    }
    n_done = sum(1 for s in states_per_obj if s.outcome == "in_bin")
    log.info(
        "Clutter task complete: %d/%d objects dropped in bin",
        n_done,
        len(states_per_obj),
    )
    for s in states_per_obj:
        log.info("  %s: %s (retries=%d)", s.obj.asset_id, s.outcome, s.retries)

    # Per-session arm tracking error (target vs measured joint_q).
    session.log_tracking_summary(label="clutter_run")

    # Camera defaults from env YAML.
    env_camera = (env_cfg.get("visual") or {}).get("camera", {})
    cam_eye = (
        args.camera_eye
        if args.camera_eye is not None
        else env_camera.get("eye", [1.3, 0.0, 1.1])
    )
    cam_target = args.camera_target
    if cam_target is None:
        cam_target = env_camera.get("target", [0.5, 0.0, 0.55])
    if cam_target == "object":
        cam_target = bundle.objects[0].world_T[:3, 3].tolist()

    session.export(
        out_path,
        camera_eye=list(cam_eye),
        camera_target=list(cam_target),
        annotations=summary,
    )
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _graspgen_for_object(
    sampler,
    mesh: trimesh.Trimesh,
    T_obj_world: np.ndarray,
    num_sample_points: int,
    num_grasps: int,
    topk: int,
    grasp_threshold: float,
    planner_name: str,
    moe_obb_density: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run GraspGenX on a single object's mesh and return grasps in WORLD frame."""
    from graspgenx.samplers import run_planner_on_object

    xyz, _ = trimesh.sample.sample_surface(mesh, num_sample_points)
    xyz = np.asarray(xyz, dtype=np.float32)
    pc_mean = xyz.mean(axis=0)
    T_center = tra.translation_matrix(-pc_mean)
    xyz_centered = tra.transform_points(xyz, T_center).astype(np.float32)

    # 'topdown' is sugar for 'graspmoe + obb_only' (same as e2e_grasp_demo's
    # run_graspgen): run graspmoe, then keep only the OBB (top-down) grasps,
    # dropping the diffusion branch. The sampler itself only knows
    # graspmoe/diffusion, so we translate here.
    planner_internal = "graspmoe" if planner_name == "topdown" else planner_name
    obb_only_eff = planner_name == "topdown"
    # When obb_only (top-down): request ALL grasps (topk=-1) so the OBB
    # top-down grasps aren't dropped by the topk cut (the diffusion branch
    # tends to outscore OBB, so a combined topk keeps 0 OBB). We filter to
    # OBB first, THEN take the top-`topk` by confidence.
    sampler_topk = -1 if obb_only_eff else topk
    grasps_centered_np, conf_np, _tags, _obb = run_planner_on_object(
        xyz_centered,
        sampler,
        planner=planner_internal,
        grasp_threshold=grasp_threshold,
        num_grasps=num_grasps,
        topk_num_grasps=sampler_topk,
        moe_obb_density=moe_obb_density,
    )
    if obb_only_eff and len(_tags) > 0:
        keep = np.array([t == "obb" for t in _tags], dtype=bool)
        grasps_centered_np = grasps_centered_np[keep]
        conf_np = conf_np[keep]
        if topk and topk > 0 and len(grasps_centered_np) > topk:
            order = np.argsort(-conf_np)[:topk]
            grasps_centered_np = grasps_centered_np[order]
            conf_np = conf_np[order]
    if len(grasps_centered_np) == 0:
        return np.zeros((0, 4, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    grasps_centered = grasps_centered_np.astype(np.float64)
    grasps_centered[:, 3, 3] = 1.0
    conf = conf_np.astype(np.float32)
    T_uncenter = tra.inverse_matrix(T_center)
    grasps_local = np.array([T_uncenter @ g for g in grasps_centered])
    grasps_world = np.array([T_obj_world @ g for g in grasps_local])
    return grasps_world.astype(np.float32), conf


def _init_planner_for_pick(
    combo_path: Path,
    robot_cfg: Dict[str, Any],
    obstacles: List[CollisionObstacle],
    robot_base_T: np.ndarray,
    max_goalset: int,
):
    """Build a cuRobo MotionPlanner with the given obstacle set."""
    # Import here to avoid pulling cuRobo into the import graph at module load.
    from e2e_grasp_demo import collision_world_to_curobo, init_planner

    scene_model = collision_world_to_curobo(obstacles, robot_base_T)
    return init_planner(combo_path, robot_cfg, scene_model, max_goalset=max_goalset)


def _plan_and_drive_pick(
    *,
    session: DynamicSession,
    bundle: SceneBundle,
    profile: RobotProfile,
    robot_cfg: Dict[str, Any],
    combo_path: Path,
    obstacles: List[CollisionObstacle],
    grasps_world: np.ndarray,
    conf: np.ndarray,
    args,
    phase_prefix: str,
    target_object: SceneObject,
) -> bool:
    """Build the pick segments (approach + grasp + close + lift) and drive
    the session through them. Returns True on success (segments built +
    driven), False on plan or NaN failure.
    """
    from e2e_grasp_demo import plan_to_grasp

    planner = _init_planner_for_pick(
        combo_path,
        robot_cfg,
        obstacles,
        bundle.robot_base_T,
        max_goalset=max(args.max_plan_attempts, len(grasps_world), 1),
    )

    # Start from the current arm config (held at home from last pick).
    # We need to pass it via robot_cfg.curobo.default_joint_position to
    # plan_to_grasp, which uses default_q for the start. Set it transiently.
    prev_default = list(robot_cfg["curobo"]["default_joint_position"])
    robot_cfg["curobo"]["default_joint_position"] = session.current_robot_q().tolist()
    try:
        success, result, target_idx, pregrasp_traj, lift_traj = plan_to_grasp(
            planner,
            robot_cfg,
            grasps_world,
            conf,
            max_attempts=args.max_plan_attempts,
            seed=args.seed,
            robot_base_T=bundle.robot_base_T,
            force_idx=-1,
            rank_by_confidence=False,
        )
    finally:
        robot_cfg["curobo"]["default_joint_position"] = prev_default

    if not success or pregrasp_traj is None:
        log.warning("plan_to_grasp failed for %s", target_object.asset_id)
        return False

    # Reject plans that don't include a valid lift trajectory. cuRobo's
    # plan_grasp falls back to "approach + grasp only" (no lift) when it
    # can't find a feasible retraction from the grasp pose. If we drive
    # such a plan we'd close the fingers around the object and then
    # immediately open them again — never lifting it. The clutter task
    # also can't satisfy its grasp-success check (object z must rise),
    # so this would burn a retry. Better to skip the grasp entirely now
    # so the retry budget goes to a target with a feasible lift.
    if lift_traj is None or lift_traj.shape[0] == 0:
        log.warning(
            "plan_to_grasp returned no lift segment for %s — "
            "skipping this grasp attempt (gripper would never lift)",
            target_object.asset_id,
        )
        return False

    # Build pick + lift segments by reusing PickAndLiftTask's helper.
    pick_task = PickAndDropInBinTask()
    chunks, segments = pick_task._build_pick_and_lift(
        profile,
        pregrasp_traj,
        lift_traj,
        args.close_frames,
        args.hold_frames,
        result,
    )
    if not chunks:
        return False
    joint_traj = np.concatenate(chunks, axis=0).astype(np.float32)

    # Prefix segment names so the JSON's phase labels indicate which
    # object's pick this is.
    prefixed_segments = [(phase_prefix + name, k) for name, k in segments]

    log.info(
        "  driving %d pick frames (segments: %s)",
        joint_traj.shape[0],
        ", ".join(f"{n}={k}" for n, k in prefixed_segments),
    )
    return session.drive_segments(joint_traj, prefixed_segments)


def _plan_and_drive_drop(
    *,
    session: DynamicSession,
    bundle: SceneBundle,
    profile: RobotProfile,
    robot_cfg: Dict[str, Any],
    combo_path: Path,
    obstacles: List[CollisionObstacle],
    env_cfg: Dict[str, Any],
    args,
    phase_prefix: str,
) -> bool:
    """Plan + drive: move_to_above_bin → hold → open_fingers → hold."""
    n_arm = profile.n_arm
    n_grip = profile.n_gripper
    open_vals, close_vals = _open_close_vals(profile)

    # Locate the bin in env_cfg.
    pick_task = PickAndDropInBinTask()
    bin_T_world = pick_task._bin_world_pose(env_cfg)
    if bin_T_world is None:
        log.warning("clutter drop: no 'bin' asset in env; skipping drop")
        return False

    # Compute the gripper orientation at the current sim state via URDF
    # FK — we want to keep this orientation through the drop.
    try:
        import yourdfpy

        fk = yourdfpy.URDF.load(
            profile.urdf_path, build_collision_scene_graph=False, load_meshes=False
        )
        current_q = session.current_robot_q()
        cfg = {n: float(v) for n, v in zip(profile.arm_joint_names, current_q)}
        for gn in profile.gripper_joint_names:
            cfg[gn] = float(profile.close_value(gn))
        fk.update_cfg(cfg)
        T_curr_link = fk.get_transform(frame_to=profile.tool_frame)
        T_curr_world = bundle.robot_base_T @ T_curr_link
        R_drop = T_curr_world[:3, :3]
    except Exception as e:
        log.warning("clutter drop: FK failed (%s); using identity rot", e)
        R_drop = np.eye(3)

    # Randomize the xy of the drop pose so successive drops don't pile
    # on top of each other. Capped at ±DROP_XY_JITTER_MAX (5 cm) to keep
    # the resulting pose well within the franka's reach — wider jitter
    # (±0.15 m) was demonstrated to push the pre-drop pose out of reach.
    # z stays at DROP_HEIGHT_ABOVE_BIN. Reproducible per-prefix seed so
    # the same object drops at the same xy across reruns.
    DROP_XY_JITTER_MAX = 0.05
    seed = abs(hash(phase_prefix)) & 0xFFFF
    rng = np.random.default_rng(seed)
    dx = float(rng.uniform(-DROP_XY_JITTER_MAX, DROP_XY_JITTER_MAX))
    dy = float(rng.uniform(-DROP_XY_JITTER_MAX, DROP_XY_JITTER_MAX))
    log.info("  drop xy jitter for %s: dx=%+.3f dy=%+.3f", phase_prefix, dx, dy)

    drop_T = np.eye(4)
    drop_T[:3, :3] = R_drop
    drop_T[:3, 3] = bin_T_world[:3, 3] + np.array([dx, dy, DROP_HEIGHT_ABOVE_BIN])
    # Pre-drop waypoint: 5 cm higher than the drop pose. The gripper
    # swings the long lateral arc to this higher pose first (so the held
    # object stays well above the bin's rim during the swing), then
    # descends purely vertically to the actual drop pose. Two benefits:
    #  * Lateral swing has more clearance over the bin walls.
    #  * The descent is a clean straight z-line, so cuRobo's interpolated
    #    trajectory doesn't accidentally clip the bin during the final
    #    approach to the drop pose.
    pre_drop_T = drop_T.copy()
    pre_drop_T[2, 3] += PRE_DROP_EXTRA_Z

    # Build a fresh planner for the drop motion. Skip target's AABB
    # since we're holding it; everything else stays an obstacle.
    planner = _init_planner_for_pick(
        combo_path,
        robot_cfg,
        obstacles,
        bundle.robot_base_T,
        max_goalset=1,
    )
    # Plan the lateral swing to the pre-drop pose.
    move_arm = pick_task._plan_to_world_pose(
        planner,
        profile,
        session.current_robot_q(),
        pre_drop_T,
        bundle.robot_base_T,
        target_frames=MOVE_TO_BIN_FRAMES,
    )
    if move_arm is None or move_arm.shape[0] == 0:
        # FALLBACK: plan_pose returned None for the single pre-drop
        # target. Retry against a small grid of alternative pre-drop
        # poses around the bin centre — cuRobo's plan_pose is flaky
        # when the held-object collision spheres make the single goal's
        # IK marginal, so giving it 8 alternatives drops the failure
        # rate dramatically without adding singletons-mode complexity.
        log.warning(
            "clutter drop: plan_pose to pre-drop pose failed; "
            "trying fallback grid of alternative pre-drop poses"
        )
        bin_t = _bin_translation(env_cfg)
        fallback_offsets = [
            (0.00, 0.00, 0.00),
            (0.05, 0.00, 0.00),
            (-0.05, 0.00, 0.00),
            (0.00, 0.05, 0.00),
            (0.00, -0.05, 0.00),
            (0.00, 0.00, 0.05),  # higher pre-drop
            (0.05, 0.05, 0.05),
            (-0.05, -0.05, 0.05),
        ]
        move_arm = None
        for k, (dx, dy, dz) in enumerate(fallback_offsets):
            alt = pre_drop_T.copy()
            alt[0, 3] = bin_t[0] + dx
            alt[1, 3] = bin_t[1] + dy
            alt[2, 3] = pre_drop_T[2, 3] + dz
            cand = pick_task._plan_to_world_pose(
                planner,
                profile,
                session.current_robot_q(),
                alt,
                bundle.robot_base_T,
                target_frames=MOVE_TO_BIN_FRAMES,
            )
            if cand is not None and cand.shape[0] > 0:
                log.info(
                    "  fallback pre-drop #%d succeeded "
                    "(offset=%+0.2f,%+0.2f,%+0.2f)",
                    k,
                    dx,
                    dy,
                    dz,
                )
                move_arm = cand
                # Recompute drop_T (vertical descent target) to be
                # directly below this successful pre-drop.
                drop_T = alt.copy()
                drop_T[2, 3] -= PRE_DROP_EXTRA_Z
                break
        if move_arm is None:
            log.warning(
                "clutter drop: all %d fallback pre-drop poses also failed",
                len(fallback_offsets),
            )
            return False
    # Plan the vertical descent from pre-drop to drop pose. Re-using the
    # same planner (same obstacle set, same scene) but with a new start
    # config = the end of move_arm.
    descend_arm = pick_task._plan_to_world_pose(
        planner,
        profile,
        move_arm[-1],
        drop_T,
        bundle.robot_base_T,
        target_frames=PRE_DROP_DESCEND_FRAMES,
    )
    if descend_arm is None or descend_arm.shape[0] == 0:
        log.warning(
            "clutter drop: plan_pose for pre-drop descent failed; "
            "falling back to direct release at pre-drop pose"
        )
        descend_arm = np.tile(move_arm[-1], (1, 1)).astype(np.float32)

    chunks: List[np.ndarray] = []
    segments: List[Tuple[str, int]] = []

    # Segment 1: swing to pre-drop (5 cm above drop pose).
    move_full = _stack_arm_and_gripper(move_arm, close_vals, n_grip)
    chunks.append(move_full)
    segments.append((phase_prefix + "move_to_pre_drop", move_full.shape[0]))
    if HOLD_ABOVE_BIN_FRAMES > 0:
        chunks.append(_hold(move_full[-1], HOLD_ABOVE_BIN_FRAMES))
        segments.append((phase_prefix + "hold_at_pre_drop", HOLD_ABOVE_BIN_FRAMES))

    # Segment 2: vertical descent to the drop pose.
    descend_full = _stack_arm_and_gripper(descend_arm, close_vals, n_grip)
    chunks.append(descend_full)
    segments.append((phase_prefix + "descend_to_drop_pose", descend_full.shape[0]))
    if HOLD_ABOVE_BIN_FRAMES > 0:
        chunks.append(_hold(descend_full[-1], HOLD_ABOVE_BIN_FRAMES))
        segments.append((phase_prefix + "hold_at_drop_pose", HOLD_ABOVE_BIN_FRAMES))

    # Open fingers (release) — hold at the DROP pose (post-descent), not
    # the pre-drop pose.
    if n_grip > 0:
        arm_hold = np.tile(descend_arm[-1], (OPEN_FRAMES, 1))
        ramp_open = _ramp(close_vals, open_vals, OPEN_FRAMES)
        release_full = np.concatenate(
            [arm_hold.astype(np.float32), ramp_open],
            axis=1,
        )
        chunks.append(release_full)
        segments.append((phase_prefix + "open_fingers_to_drop", OPEN_FRAMES))
        if HOLD_AFTER_DROP_FRAMES > 0:
            chunks.append(_hold(release_full[-1], HOLD_AFTER_DROP_FRAMES))
            segments.append((phase_prefix + "hold_after_drop", HOLD_AFTER_DROP_FRAMES))

    joint_traj = np.concatenate(chunks, axis=0).astype(np.float32)
    log.info(
        "  driving %d drop frames (segments: %s)",
        joint_traj.shape[0],
        ", ".join(f"{n}={k}" for n, k in segments),
    )
    return session.drive_segments(joint_traj, segments)


def _release_in_place(
    *,
    session: DynamicSession,
    profile: RobotProfile,
    phase_prefix: str,
    settle_frames: int = 60,
):
    """Open the gripper at the current arm config, hold, let the held
    object fall to the table/floor. Called BEFORE return_home in the
    failure branches so the object lands deterministically near where
    the gripper was, instead of being dragged along the return path.
    """
    # session.current_robot_q() returns just the arm joints (n_arm,);
    # use current_gripper_q() for the actual gripper-master values.
    current_arm = session.current_robot_q()
    current_grip_dict = session.current_gripper_q()
    current_grip = np.array(
        [current_grip_dict[n] for n in profile.gripper_joint_names],
        dtype=np.float32,
    )
    open_grip = np.array(
        [profile.open_value(n) for n in profile.gripper_joint_names],
        dtype=np.float32,
    )
    # Ramp gripper from current -> open over OPEN_FRAMES.
    rows = []
    for k in range(OPEN_FRAMES):
        alpha = (k + 1) / OPEN_FRAMES
        grip = (1 - alpha) * current_grip + alpha * open_grip
        rows.append(np.concatenate([current_arm, grip]))
    open_row = np.concatenate([current_arm, open_grip])
    rows.extend([open_row] * settle_frames)
    traj = np.asarray(rows, dtype=np.float32)
    log.info(
        "  release-in-place: %d frames (open=%d, settle=%d)",
        traj.shape[0],
        OPEN_FRAMES,
        settle_frames,
    )
    session.drive_segments(
        traj,
        [
            (phase_prefix + "release_open_fingers", OPEN_FRAMES),
            (phase_prefix + "release_settle", settle_frames),
        ],
    )


def _plan_and_drive_return_home(
    *,
    session: DynamicSession,
    profile: RobotProfile,
    robot_cfg: Dict[str, Any],
    combo_path: Path,
    obstacles: List[CollisionObstacle],
    home_q: np.ndarray,
    args,
    phase_prefix: str,
):
    """Plan a free-space joint-space return to ``home_q`` and drive it.

    Uses cuRobo's plan_pose with the home-q's tool pose computed via
    URDF FK. (Avoiding plan_cspace because the planner's API surface
    has shifted between cuRobo versions; plan_pose is stable.)
    """
    import yourdfpy

    n_arm = profile.n_arm
    n_grip = profile.n_gripper
    open_vals, _ = _open_close_vals(profile)

    fk = yourdfpy.URDF.load(
        profile.urdf_path, build_collision_scene_graph=False, load_meshes=False
    )
    cfg = {n: float(v) for n, v in zip(profile.arm_joint_names, home_q)}
    for gn in profile.gripper_joint_names:
        cfg[gn] = float(profile.open_value(gn))
    fk.update_cfg(cfg)
    T_home_link = fk.get_transform(frame_to=profile.tool_frame)
    bundle_robot_base_T = robot_cfg.get("_robot_base_T")
    # Fallback: read from session.bundle (cleaner).
    if bundle_robot_base_T is None:
        bundle_robot_base_T = session.bundle.robot_base_T
    T_home_world = bundle_robot_base_T @ T_home_link

    planner = _init_planner_for_pick(
        combo_path,
        robot_cfg,
        obstacles,
        bundle_robot_base_T,
        max_goalset=1,
    )
    pick_task = PickAndDropInBinTask()
    home_arm = pick_task._plan_to_world_pose(
        planner,
        profile,
        session.current_robot_q(),
        T_home_world,
        bundle_robot_base_T,
        target_frames=RETURN_HOME_FRAMES,
    )
    if home_arm is None or home_arm.shape[0] == 0:
        log.warning("return_home plan failed; just holding current pose")
        # Synthesize a do-nothing segment so the MP4 doesn't jump.
        current = session.current_robot_q()
        gripper_open = np.array(open_vals, dtype=np.float32)
        row = np.concatenate([current, gripper_open])
        empty = np.tile(row, (HOLD_AFTER_DROP_FRAMES, 1)).astype(np.float32)
        session.drive_segments(
            empty, [(phase_prefix + "hold_in_place", empty.shape[0])]
        )
        return

    home_full = _stack_arm_and_gripper(home_arm, open_vals, n_grip)
    log.info("  driving %d return-home frames", home_full.shape[0])
    session.drive_segments(
        home_full, [(phase_prefix + "return_home", home_full.shape[0])]
    )
