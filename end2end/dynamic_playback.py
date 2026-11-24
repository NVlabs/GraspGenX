"""Newton-backed dynamic playback for the end2end grasping demo.

Given a cuRobo-planned joint trajectory plus the SceneBundle returned by
:mod:`scene_builder`, this module:

  1. Builds a Newton :class:`newton.Model` containing the same robot URDF,
     the environment's static collision meshes (table / bin walls / …),
     and the manipulation target as a free rigid body sitting at its
     scene-builder pose.
  2. Steps a :class:`newton.solvers.SolverMuJoCo` solver under PD position
     control, feeding the planned joint trajectory into ``joint_target_q``
     waypoint-by-waypoint. The Robotiq ``finger_joint`` close (already
     appended to the trajectory by the demo) drives the URDF's mimic
     constraints, so all six finger DOFs close together.
  3. Records the *simulated* per-link world poses and the *simulated* object
     pose into a trajectory JSON with the same schema as the kinematic
     exporter — the static ``object`` slot is omitted and the object
     appears as a per-frame entry under ``frames[t].parts`` instead.

The render pipeline (``render_trajectory_mp4.py``) consumes the JSON
unchanged.

Design notes:

- We deliberately use the same URDF that cuRobo planned against
  (``robot_cfg['urdf_path']``) so the FK chain is consistent across modes.
- ``yourdfpy``-backed :class:`URDFFK` from :mod:`trajectory_visualizer`
  is reused for per-frame visual mesh world transforms — that way both
  kinematic and dynamic exporters paint the same meshes at the same
  visual offsets.
- ``parse_visuals_as_colliders=True`` is set on ``add_urdf`` so the
  Robotiq fingers and the UR10e arm have collision shapes; otherwise the
  fingers would pass through the object.
- Static environment meshes are added with ``body=-1`` (world-frame) using
  ``builder.add_shape_mesh``, with their ``vis_meshes`` trimeshes converted
  to :class:`newton.Mesh` objects.
- The manipulation target gets its own free body so gravity + contact act on it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import trimesh
import trimesh.transformations as tra
import warp as wp
import yourdfpy

# newton 1.0.0 reads ``wp.config.verbose`` at import time, which warp-lang 1.19
# no longer defines. cuRobo sets it as a side effect of its own init, so the
# import only worked when cuRobo was loaded first; set it here so the import
# doesn't depend on import order.
if not hasattr(wp.config, "verbose"):
    wp.config.verbose = False

import newton  # noqa: E402

from robot_profiles import RobotProfile
from scene_builder import SceneBundle
from trajectory_visualizer import URDFFK

log = logging.getLogger("dyn")


# Defaults matched to UR10e example (ke=500/kd=50). Finger gains tuned
# so the gripper closes firm enough to hold the object through the
# lift, but not so aggressive that contact tunnels the box out the
# side: kp=2000 with relatively high kd=200 (heavy damping prevents
# the close from overshooting under contact).
ARM_KP_DEFAULT = 2000.0  # 4x baseline (was 500); stiffer arm tracking
ARM_KD_DEFAULT = 100.0  # 2x baseline (was 50)
FINGER_KP_DEFAULT = 2000.0
FINGER_KD_DEFAULT = 200.0

# Mimic joints (the 5 follower revolute joints in the Robotiq URDF) need
# *some* drive gain to actually track the constrained position under
# finger-pad contact. Keep these gentle so the PD doesn't fight Newton's
# mimic constraint when the per-mimic target equals multiplier·finger_q
# (which is what `_set_joint_targets` writes each frame). Borrowed from
# `newton_grasp_eval.py`'s mimic follower block.
MIMIC_KP = 50.0
MIMIC_KD = 10.0

# Object mass + friction. Values copied from NewtonDataGen's
# `newton_grasp_eval.py` (OBJECT_MASS=0.2, OBJECT_FRICTION=10.0).
# The 0.1 / 3.0 defaults that lived here previously were too light and
# too low-friction relative to the validated grasp-eval pipeline,
# making small/smooth objects (e.g. our 4 cm battery cylinder) prone to
# being launched by close-time contact impulse spikes.
DEFAULT_OBJECT_MASS = 0.2
DEFAULT_OBJECT_MU = 10.0
DEFAULT_FINGER_MU = 3.0

# Solver settings matched to NewtonDataGen's `newton_grasp_eval.py`
# (newton/scripts) — 100 Newton iterations per substep is what they use
# for tight contact resolution during grasp evaluation. With only 10
# iterations the constraint solver doesn't fully converge and grasps
# slip during the lift segment.
SOLVER_ITERATIONS = 100
SOLVER_LS_ITERATIONS = 50
SOLVER_IMPRATIO = 1000.0
# Collide-every-Nth-substep: NewtonDataGen uses 4 (so at their 500 Hz
# they collide at 125 Hz). At our 1000 Hz this gives 250 Hz collision,
# matching the relative ratio.
COLLIDE_SUBSTEPS = 4
CONTACT_MAX = 524288
# MuJoCo's njmax/nconmax must be large enough for all simultaneous
# contacts. With CoACD-decomposed gripper meshes (~300 convex parts)
# AND a CoACDed object near them, the broad phase can report >140k
# overlapping pairs. 256k seats comfortably above what we've seen;
# overflowing this buffer silently drops contacts (including critical
# table-vs-object pairs that hold the object up against gravity).
MJC_NCONMAX = 262144


@dataclass
class _RobotLayout:
    """Bookkeeping for how Newton laid out the URDF after add_urdf.

    Indices reference ``builder.joint_label`` order. ``q_start`` values
    are per-DOF offsets into ``model.joint_q`` / ``control.joint_target_q``;
    ``qd_start`` values are per-DOF offsets into ``model.joint_qd`` /
    ``control.joint_target_qd`` (same as q_start for single-DOF joints,
    which is everything we have).
    """

    # Per-DOF offsets for arm joints, in the order cuRobo emits them.
    arm_q_starts: List[int]
    # Per master gripper joint: ``(name, q_offset, open_value, close_value)``.
    # The demo ramps each from open to close over the close phase; physics
    # naturally limits how far the close actually goes when there's contact.
    gripper_drives: List[Tuple[str, int, float, float]]
    # Per mimic follower joint: ``(q_offset, master_q_offset, multiplier,
    # offset)``. The PD target each frame is
    # ``multiplier · joint_q[master_q_offset] + offset`` so the drive
    # aligns with Newton's mimic constraint instead of fighting it.
    mimic_drives: List[Tuple[int, int, float, float]]
    # Newton body indices of every manipulation-target free body. For
    # single-object scenes this list has one entry (which legacy code
    # accesses via :attr:`object_body_idx`). Clutter scenes have one
    # entry per object.
    object_body_idxs: List[int] = field(default_factory=list)
    # Per-object free-joint coordinate offsets into ``model.joint_q``.
    # Each object occupies seven values: xyz followed by quaternion xyzw.
    object_q_starts: List[int] = field(default_factory=list)
    # Velocity-mode close: for each gripper joint, a (qd_offset,
    # closing_velocity) tuple. Empty for position-mode profiles.
    velocity_close_drives: List[Tuple[int, float]] = field(default_factory=list)

    @property
    def object_body_idx(self) -> int:
        """Single-object compatibility shim — returns objects[0]."""
        return self.object_body_idxs[0] if self.object_body_idxs else -1


def _matrix_to_wp_transform(T: np.ndarray) -> wp.transform:
    """4x4 → wp.transform (xyz + xyzw quat)."""
    quat_wxyz = tra.quaternion_from_matrix(T)  # [w, x, y, z]
    return wp.transform(
        wp.vec3(float(T[0, 3]), float(T[1, 3]), float(T[2, 3])),
        wp.quat(
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
            float(quat_wxyz[0]),  # warp quat is xyzw
        ),
    )


def _wp_transform_to_matrix(t_xyzw: np.ndarray) -> np.ndarray:
    """[x, y, z, qx, qy, qz, qw] → 4x4."""
    T = np.eye(4)
    q_wxyz = np.array([t_xyzw[6], t_xyzw[3], t_xyzw[4], t_xyzw[5]], dtype=float)
    T[:3, :3] = tra.quaternion_matrix(q_wxyz)[:3, :3]
    T[:3, 3] = t_xyzw[:3]
    return T


def _trimesh_to_newton_mesh(mesh: trimesh.Trimesh) -> newton.Mesh:
    """Convert a trimesh.Trimesh into a newton.Mesh (no SDF — geometric contacts)."""
    verts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32).reshape(-1)
    return newton.Mesh(verts, faces, compute_inertia=True)


def _bare(name: str) -> str:
    return name.split("/")[-1] if "/" in name else name


# ---------------------------------------------------------------------------
# Per-shape MuJoCo contact overrides — ported verbatim from
# NewtonDataGen/scripts/newton_grasp_eval.py (`_load_mjc_geom_overrides`,
# `_resolve_mjc_geom_overrides`, `_apply_mjc_geom_overrides_to_template`,
# `_apply_mjc_geom_overrides_to_model`). Driven by a JSON config:
#
#   {"patterns": [
#       {"link_regex": ".*",           "condim": 3, "priority": 0, "mu": 0.6},
#       {"link_regex": "(^|/)(thumb_distal|index_intermediate|...)$",
#        "condim": 6, "priority": 1, "mu": 3.0,
#        "mu_torsional": 7.5e-3, "mu_rolling": 7.5e-4,
#        "ke": 40000.0, "kd": 400.0}
#   ]}
#
# Apply order is "last-match wins" via dict.update.
# ---------------------------------------------------------------------------
def _load_mjc_geom_overrides(path: str | Path | None):
    """Load mjc-geom-overrides JSON. Returns list of {regex, spec} or None."""
    if not path:
        return None
    import json as _json
    import re as _re

    with open(path, "r", encoding="utf-8") as f:
        cfg = _json.load(f)
    out = []
    for pat in cfg.get("patterns", []):
        out.append(
            {
                "regex": _re.compile(pat["link_regex"]),
                "spec": {k: v for k, v in pat.items() if k != "link_regex"},
            }
        )
    return out


def _resolve_mjc_geom_overrides(builder, shape_indices, patterns):
    """Per-shape merged override dict, last-match-wins. Returns {shape_idx: spec}."""
    if not patterns:
        return {}
    out = {}
    for si in shape_indices:
        bi = builder.shape_body[si]
        body_lbl = builder.body_label[bi] if 0 <= bi < len(builder.body_label) else ""
        merged = {}
        for entry in patterns:
            if entry["regex"].search(str(body_lbl)):
                merged.update(entry["spec"])
        if merged:
            out[si] = merged
    return out


def _apply_mjc_geom_overrides_to_template(builder, overrides):
    """Apply per-shape material params (mu/ke/kd/mu_torsional/mu_rolling/...)."""
    for si, spec in overrides.items():
        if "mu" in spec:
            builder.shape_material_mu[si] = float(spec["mu"])
        if "ke" in spec:
            builder.shape_material_ke[si] = float(spec["ke"])
        if "kd" in spec:
            builder.shape_material_kd[si] = float(spec["kd"])
        if "mu_torsional" in spec and hasattr(builder, "shape_material_mu_torsional"):
            builder.shape_material_mu_torsional[si] = float(spec["mu_torsional"])
        if "mu_rolling" in spec and hasattr(builder, "shape_material_mu_rolling"):
            builder.shape_material_mu_rolling[si] = float(spec["mu_rolling"])
        if "collision_group" in spec:
            builder.shape_collision_group[si] = int(spec["collision_group"])


def _apply_mjc_geom_overrides_to_model(model, overrides):
    """Set model.mujoco.condim and geom_priority post-finalize. These are NOT
    in the SHAPE_PROPERTIES runtime-update list, so this MUST run before
    SolverMuJoCo(model, ...) is constructed."""
    if not overrides:
        return
    if not hasattr(model, "mujoco"):
        log.warning(
            "[mjc-overrides] model has no .mujoco namespace; "
            "did you forget SolverMuJoCo.register_custom_attributes?"
        )
        return
    has_condim = hasattr(model.mujoco, "condim") and any(
        "condim" in s for s in overrides.values()
    )
    has_priority = hasattr(model.mujoco, "geom_priority") and any(
        "priority" in s for s in overrides.values()
    )
    if has_condim:
        condim = model.mujoco.condim.numpy()
    if has_priority:
        priority = model.mujoco.geom_priority.numpy()
    for si, spec in overrides.items():
        if has_condim and si < len(condim) and "condim" in spec:
            condim[si] = int(spec["condim"])
        if has_priority and si < len(priority) and "priority" in spec:
            priority[si] = int(spec["priority"])
    if has_condim:
        model.mujoco.condim.assign(condim)
    if has_priority:
        model.mujoco.geom_priority.assign(priority)


def _read_mimic_info(urdf_path: str) -> Dict[str, Tuple[str, float, float]]:
    """Return ``{follower_joint_name: (master_joint_name, multiplier, offset)}``.

    Kept for backward compat; :class:`RobotProfile` now exposes the same
    via :meth:`RobotProfile.mimic_constraints`.
    """
    u = yourdfpy.URDF.load(
        str(urdf_path),
        build_collision_scene_graph=False,
        load_meshes=False,
    )
    out: Dict[str, Tuple[str, float, float]] = {}
    for j in u.robot.joints:
        if j.mimic is None:
            continue
        out[j.name] = (j.mimic.joint, float(j.mimic.multiplier), float(j.mimic.offset))
    return out


_PHYSICS_OVERRIDE_KEYS = (
    "set_gap_margin",
    "gap",
    "margin",
    "set_contact_stiffness",
    "contact_ke",
    "contact_kd",
    "contact_kf",
    "set_env_mu",
    "object_armature",
    "force_object_mass",
    "deterministic_inertia",
)


def _physics_overrides_from_profile(profile: RobotProfile) -> dict:
    """Pull physics-override keys out of the robot YAML's `dynamic` block
    (profile.dynamic_overrides). Empty for robots that don't set any (e.g.
    Franka) — so they keep _build_scene's defaults (Newton-default gap/margin).
    """
    return {
        k: profile.dynamic_overrides[k]
        for k in _PHYSICS_OVERRIDE_KEYS
        if k in profile.dynamic_overrides
    }


def _build_scene(
    bundle: SceneBundle,
    profile: RobotProfile,
    gravity: float,
    object_mass: float,
    object_mu: float,
    finger_mu: float,
    physics_overrides: dict | None = None,
) -> Tuple[newton.Model, _RobotLayout, List[str]]:
    """Construct a Newton :class:`Model` for dynamic playback.

    Returns the finalized model, a :class:`_RobotLayout` describing where
    things live, and the list of URDF-prefixed link labels Newton assigned
    to each body (so we can locate fingers for friction overrides downstream).

    ``physics_overrides`` (default None → current production behaviour) is a
    debug hook for the settle-stability sweep. Recognised keys:
      set_contact_stiffness (bool, def True) — if False, don't set ke/kd/kf
        on any ShapeConfig (use Newton's defaults).
      contact_ke / contact_kd / contact_kf (floats) — values when set.
      set_gap_margin (bool, def True) — if False, don't set gap/margin.
      gap / margin (floats) — values when set.
      set_env_mu (bool, def True) — if False, don't set mu on robot/env
        default shapes (i.e. don't apply finger_mu to the table).
      force_object_mass (bool, def False) — if True, force body_mass to
        ``object_mass`` AFTER shapes (mirrors NewtonDataGen).
      deterministic_inertia (bool, def False) — compute object inertia +
        com from the visual mesh (deterministic) instead of CoACD shapes.
      object_armature (float, def 0.002) — object free-joint armature.
    """
    po = physics_overrides or {}
    set_ke = po.get("set_contact_stiffness", True)
    ov_ke = po.get("contact_ke", 5.0e4)
    ov_kd = po.get("contact_kd", 5.0e2)
    ov_kf = po.get("contact_kf", 1.0e3)
    # FIX (settle-stability sweep, 2026-05-29): do NOT set gap/margin by
    # default — let Newton pick its defaults. The old hardcoded
    # gap=0.001/margin=0.003 (a 1 mm contact band) caused resting-contact
    # jitter that made flat/upright objects shake and tip over (e.g.
    # GranolaBars_upright fell at ~17 s). The settle sweep showed dropping
    # this override takes GranolaBars from mean_angvel 0.815 deg/frame
    # (falls) to 0.003 (rock-steady). Pass set_gap_margin=True to restore.
    set_gm = po.get("set_gap_margin", False)
    ov_gap = po.get("gap", 0.001)
    ov_margin = po.get("margin", 0.003)
    set_env_mu = po.get("set_env_mu", True)
    obj_armature = float(po.get("object_armature", 0.002))

    def _shape_cfg(mu_val):
        """Build a ShapeConfig honouring the physics_overrides toggles."""
        kw = dict(is_hydroelastic=False)
        if set_gm:
            kw.update(gap=ov_gap, margin=ov_margin)
        if set_ke:
            kw.update(ke=ov_ke, kd=ov_kd, kf=ov_kf)
        if mu_val is not None:
            kw.update(mu=float(mu_val))
        return newton.ModelBuilder.ShapeConfig(**kw)

    # gravity is a scalar magnitude along the up axis (Z by default); a
    # negative value points "down". Newton expects float, not vec3.
    builder = newton.ModelBuilder(gravity=float(gravity))
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    # Default shape cfg with PROPER contact gains. Without ke/kd/kf the
    # SolverMuJoCo contact response is too soft and objects sink straight
    # through the table. Values copied from
    # newton/examples/ik/example_ik_cube_stacking.py which uses Franka
    # + table + cubes — same physics shape as our setup.
    # Contact-geometry settings match NewtonDataGen's `build_grasp_builder`
    # shape_cfg (gap=0.001, margin=0.003). Contact-stiffness (ke/kd/kf) is
    # kept explicit at ke=5e4/kd=5e2 because Newton's defaults (ke=2500,
    # kd=100) are too soft for a tabletop scene — an object resting on
    # the table under gravity sinks ~2 mm into the table before the
    # spring force balances its weight. NewtonDataGen doesn't have this
    # problem because their grasp eval has no table (floating gripper).
    # The per-shape JSON override for the inspire fingertip pads still
    # sets ke=40000, kd=400 (close to these values).
    builder.default_shape_cfg = _shape_cfg(finger_mu if set_env_mu else None)

    # ---- 1. Robot URDF -----------------------------------------------------
    # parse_visuals_as_colliders=False uses the URDF's <collision> tags
    # (simple primitives / low-poly meshes) instead of the full <visual>
    # meshes. The Franka collision/link*.obj files are ~200 verts each;
    # the visual meshes are 10x larger and would CoACD into hundreds of
    # parts each, overflowing the contact buffer (>250k contacts vs
    # MuJoCo's nconmax limit). The Newton ik_cube_stacking example also
    # uses False — see newton/examples/ik/example_ik_cube_stacking.py:358.
    robot_base_xform = _matrix_to_wp_transform(profile.robot_base_T)
    n_shapes_before = builder.shape_count
    builder.add_urdf(
        profile.urdf_path,
        xform=robot_base_xform,
        enable_self_collisions=False,
        parse_visuals_as_colliders=False,
        collapse_fixed_joints=profile.collapse_fixed_joints,
    )
    urdf_shape_end = builder.shape_count  # last URDF shape index + 1
    # Per-shape MuJoCo geom overrides (condim, priority, mu, mu_torsional,
    # mu_rolling, ke, kd) — driven by a JSON file declared in the robot
    # profile (e.g. inspire_hand_mjc_geom_overrides.json). The template-
    # side params (mu / ke / kd / mu_torsional / mu_rolling) get applied
    # here; condim + priority are deferred until after model.finalize().
    mjc_override_path = profile.dynamic_overrides.get("mjc_geom_overrides_json")
    pending_mjc_overrides: Dict[int, dict] = {}
    if mjc_override_path:
        # Resolve relative paths against the robot YAML's directory if needed.
        pat_path = Path(mjc_override_path).expanduser()
        if not pat_path.is_absolute():
            pat_path = (Path(__file__).resolve().parent / pat_path).resolve()
        patterns = _load_mjc_geom_overrides(str(pat_path))
        if patterns:
            pending_mjc_overrides = _resolve_mjc_geom_overrides(
                builder,
                range(urdf_shape_end),
                patterns,
            )
            _apply_mjc_geom_overrides_to_template(builder, pending_mjc_overrides)
            log.info(
                "[mjc-overrides] loaded %s, applied to %d URDF shapes",
                pat_path.name,
                len(pending_mjc_overrides),
            )
            # First few entries for sanity in the log.
            for cnt, (si, spec) in enumerate(pending_mjc_overrides.items()):
                if cnt >= 6:
                    log.info("  ... (%d more)", len(pending_mjc_overrides) - 6)
                    break
                bi = builder.shape_body[si]
                lbl = (
                    builder.body_label[bi] if 0 <= bi < len(builder.body_label) else "?"
                )
                log.info("  shape[%d] body=%s -> %s", si, lbl, spec)
    # SolverMuJoCo's update_geom_properties_kernel fails on raw triangle
    # meshes, so every mesh shape must be replaced by something convex.
    # Strategy:
    #  - Gripper bodies (per `profile.gripper_body_keywords`) get CoACD —
    #    a single convex hull would collapse the inward-facing pad
    #    concavity and the object would slip right out.
    #  - Arm links are smooth, mostly-convex; one convex hull per link is
    #    fine and *much* cheaper than CoACD (a smooth arm link can
    #    otherwise decompose into 80+ parts).
    # Split mesh shapes into three buckets:
    #   - "convex" — arm links, plus gripper bodies that don't need CoACD
    #     (panda_hand etc.). Single convex hull each.
    #   - "coacd"  — gripper bodies that need preserved concavity (fingertips).
    convex_mesh_indices: List[int] = []
    coacd_mesh_indices: List[int] = []
    keywords = profile.coacd_link_keywords  # empty = all gripper bodies get CoACD
    for si in range(n_shapes_before, builder.shape_count):
        if int(builder.shape_type[si]) != int(newton.GeoType.MESH):
            continue
        if builder.shape_source[si] is None:
            continue
        bi = builder.shape_body[si]
        lbl = _bare(str(builder.body_label[bi])) if bi >= 0 else ""
        is_gripper = any(k in lbl for k in profile.gripper_body_keywords)
        is_collider = bool(
            builder.shape_flags[si] & newton.ShapeFlags.COLLIDE_SHAPES
        )
        if is_collider and is_gripper and profile.use_coacd_for_gripper:
            if not keywords or any(k in lbl for k in keywords):
                coacd_mesh_indices.append(si)
            else:
                convex_mesh_indices.append(si)
        else:
            convex_mesh_indices.append(si)

    coacd_threshold = float(profile.dynamic_overrides.get("coacd_threshold", 0.05))
    if convex_mesh_indices:
        builder.approximate_meshes(
            method="convex_hull",
            shape_indices=convex_mesh_indices,
            keep_visual_shapes=False,
        )
        log.info(
            "Convex-hull meshes (arm + non-finger gripper): %d shapes",
            len(convex_mesh_indices),
        )
    if coacd_mesh_indices:
        try:
            n_before = builder.shape_count
            builder.approximate_meshes(
                method="coacd",
                shape_indices=coacd_mesh_indices,
                keep_visual_shapes=False,
                threshold=coacd_threshold,
            )
            log.info(
                "CoACD meshes (finger pads): threshold=%.3f, %d -> %d shapes",
                coacd_threshold,
                len(coacd_mesh_indices),
                builder.shape_count - (n_before - len(coacd_mesh_indices)),
            )
        except Exception as e:
            log.warning("CoACD failed (%s); falling back to convex_hull", e)
            builder.approximate_meshes(
                method="convex_hull",
                shape_indices=coacd_mesh_indices,
                keep_visual_shapes=False,
            )

    # ---- 2. Map URDF joints to Newton indices via the profile -------------
    # Profiles list arm joint names in cuRobo's planning order, plus the
    # set of master gripper joints we actively drive. Mimic followers
    # come from the URDF (or a profile override).
    mimic_info = profile.mimic_constraints()
    bare_to_idx: Dict[str, int] = {
        _bare(str(lbl)): j for j, lbl in enumerate(builder.joint_label)
    }
    joint_q_start = builder.joint_q_start  # list of ints during build phase

    missing_arm = [n for n in profile.arm_joint_names if n not in bare_to_idx]
    if missing_arm:
        raise RuntimeError(
            f"Profile {profile.NAME}: arm joints not found in URDF: " f"{missing_arm}"
        )
    arm_q_starts = [joint_q_start[bare_to_idx[n]] for n in profile.arm_joint_names]

    missing_grip = [n for n in profile.gripper_joint_names if n not in bare_to_idx]
    if missing_grip:
        raise RuntimeError(
            f"Profile {profile.NAME}: gripper joints not found in URDF: "
            f"{missing_grip}"
        )
    gripper_drives: List[Tuple[str, int, float, float]] = []
    for gname in profile.gripper_joint_names:
        q_off = joint_q_start[bare_to_idx[gname]]
        gripper_drives.append(
            (
                gname,
                q_off,
                profile.open_value(gname),
                profile.close_value(gname),
            )
        )

    # Per-mimic: q_offset, master q_offset, multiplier, offset.
    # Two things happen here:
    #   1. For PD: store (q_offset, master_q_offset, mult, off) so we can
    #      keep the follower's PD target aligned with the master each frame.
    #   2. For physics: if the URDF didn't declare this mimic but the
    #      profile injected it (e.g. the Franka Panda fingers, which the
    #      cuRobo URDF leaves uncoupled), add an actual Newton mimic
    #      constraint via `builder.add_constraint_mimic`. Without this
    #      the two fingers move independently — one stalls on object
    #      contact while the other closes past empty space.
    mimic_drives: List[Tuple[int, int, float, float]] = []
    # Detect which mimics the URDF already provided so we don't add
    # duplicate constraints.
    urdf_mimics = set()
    try:
        import yourdfpy as _ydf

        _u = _ydf.URDF.load(
            profile.urdf_path, build_collision_scene_graph=False, load_meshes=False
        )
        for j in _u.robot.joints:
            if j.mimic is not None:
                urdf_mimics.add(j.name)
    except Exception:
        pass
    n_added_constraints = 0
    for follower, (master, mult, off) in mimic_info.items():
        if follower not in bare_to_idx or master not in bare_to_idx:
            continue
        mimic_drives.append(
            (
                joint_q_start[bare_to_idx[follower]],
                joint_q_start[bare_to_idx[master]],
                float(mult),
                float(off),
            )
        )
        if follower not in urdf_mimics:
            try:
                builder.add_constraint_mimic(
                    joint0=bare_to_idx[follower],
                    joint1=bare_to_idx[master],
                    coef0=float(off),
                    coef1=float(mult),
                )
                n_added_constraints += 1
            except Exception as e:
                log.warning(
                    "add_constraint_mimic for %s→%s failed: %s", follower, master, e
                )
    log.info(
        "Profile %s: %d arm joints, %d gripper joints, %d mimic followers "
        "(%d Newton mimic constraints added by profile)",
        profile.NAME,
        len(arm_q_starts),
        len(gripper_drives),
        len(mimic_drives),
        n_added_constraints,
    )

    # ---- 3. Drive gains (PD on all driven DOFs) ---------------------------
    arm_kp = profile.dynamic_overrides.get("arm_kp", ARM_KP_DEFAULT)
    arm_kd = profile.dynamic_overrides.get("arm_kd", ARM_KD_DEFAULT)
    finger_kp = profile.dynamic_overrides.get("finger_kp", FINGER_KP_DEFAULT)
    finger_kd = profile.dynamic_overrides.get("finger_kd", FINGER_KD_DEFAULT)

    # Arm: position control always.
    for q_off in arm_q_starts:
        builder.joint_target_ke[q_off] = float(arm_kp)
        builder.joint_target_kd[q_off] = float(arm_kd)
        builder.joint_target_mode[q_off] = int(newton.JointTargetMode.POSITION)

    # Gripper: position or velocity, per the profile.
    velocity_close_drives: List[Tuple[int, float]] = []
    if profile.gripper_control_mode == "velocity":
        # NewtonDataGen-style parallel jaw: ke=0, high kd, velocity mode.
        # Closing happens by setting joint_target_qd to a constant
        # closing rate during the close phase; contact equilibrium stops
        # the motion. Match `newton_grasp_eval.py`'s `FINGER_KD=800`.
        velocity_kd = float(profile.dynamic_overrides.get("finger_velocity_kd", 800.0))
        for name, q_off, _, _ in gripper_drives:
            builder.joint_target_ke[q_off] = 0.0
            builder.joint_target_kd[q_off] = velocity_kd
            builder.joint_target_mode[q_off] = int(newton.JointTargetMode.VELOCITY)
            close_vel = float(profile.gripper_close_velocity.get(name, -0.1))
            velocity_close_drives.append((q_off, close_vel))
        log.info(
            "Gripper drive: VELOCITY mode (kd=%.0f), closing velocities=%s",
            velocity_kd,
            [
                (n, profile.gripper_close_velocity.get(n))
                for n, _, _, _ in gripper_drives
            ],
        )
    else:
        for _, q_off, _, _ in gripper_drives:
            builder.joint_target_ke[q_off] = float(finger_kp)
            builder.joint_target_kd[q_off] = float(finger_kd)
            builder.joint_target_mode[q_off] = int(newton.JointTargetMode.POSITION)
        log.info(
            "Gripper drive: POSITION mode (ke=%.0f, kd=%.0f)", finger_kp, finger_kd
        )

    # Mimic followers: position mode tracking master joint position.
    for q_off, _, _, _ in mimic_drives:
        builder.joint_target_ke[q_off] = MIMIC_KP
        builder.joint_target_kd[q_off] = MIMIC_KD
        builder.joint_target_mode[q_off] = int(newton.JointTargetMode.POSITION)

    # Optional gripper armature (Newton's PD is unstable on low-inertia
    # prismatic joints without it).
    if profile.gripper_armature > 0:
        for _, q_off, _, _ in gripper_drives:
            builder.joint_armature[q_off] = float(profile.gripper_armature)
        log.info("Gripper armature set to %.3f", profile.gripper_armature)

    # Override URDF effort limit on the finger joints (Newton parses the
    # URDF's `effort=20` for the panda fingers, which caps the PD force
    # so the fingers can't close against gravity / contact). The
    # panda_hydro example uses 20 N too but its gains are well-tuned;
    # bumping for our setup gives the gripper authority to close.
    finger_effort = float(profile.dynamic_overrides.get("finger_effort_limit", 200.0))
    for _, q_off, _, _ in gripper_drives:
        builder.joint_effort_limit[q_off] = finger_effort

    # ---- 4. Higher friction on gripper-pad shapes for stable grasps -------
    # Skip this blanket loop when per-shape JSON overrides are in effect —
    # the JSON config already sets mu (typically 0.6 on default shapes and
    # 3.0 on fingertip pads) and we don't want to clobber it.
    if not pending_mjc_overrides:
        for si in range(builder.shape_count):
            bi = builder.shape_body[si]
            if bi < 0 or bi >= len(builder.body_label):
                continue
            lbl = _bare(str(builder.body_label[bi]))
            if any(k in lbl for k in profile.gripper_body_keywords):
                builder.shape_material_mu[si] = float(finger_mu)

    # ---- 4. Static environment collision shapes (table, bin walls, …) -----
    # Use the cuRobo-style cuboid obstacles from `bundle.collision_world`
    # rather than the full meshes. World-attached mesh shapes are not
    # well-supported by SolverMuJoCo without hydroelastic SDFs; cuboids
    # ("primitives") work natively. Pose is in cuRobo's
    # [x, y, z, qw, qx, qy, qz] convention.
    env_shape_cfg = _shape_cfg(finger_mu if set_env_mu else None)
    # BIN COLLISION MODE (env hook): the procedural bin's visual mesh is a
    # hollow open container, but `collision: cuboid_from_extents` registers
    # it as a SOLID cuboid up to the rim, so dropped objects land on top of
    # the block (~10 cm above the real floor). Modes:
    #   primitives  (DEFAULT) — build the bin from floor + 4 wall boxes
    #                 (hollow tray): objects drop INSIDE and rest on the
    #                 real floor (~0.505), stable axis-aligned boxes. This
    #                 replaced the old solid-cuboid default, which made
    #                 objects float on top of the block at the rim (~0.59).
    #   coacd       — decompose the bin's VISUAL mesh via CoACD. Hollow but
    #                 UNSTABLE on thin-walled bins (spiky convex pieces
    #                 launch objects out) — kept for comparison only.
    #   solid       — legacy: one solid AABB cuboid (object lands on top).
    import os as _os

    bin_mode = _os.environ.get("GRASPGENX_BIN_COLLISION", "primitives").lower()
    n_static = 0
    bin_ob = None
    for ob in bundle.collision_world:
        if ob.type != "cuboid" or ob.dims is None or ob.pose is None:
            continue
        if "bin" in ob.name.lower() and bin_mode != "solid":
            bin_ob = ob  # handle below (skip the solid cuboid)
            continue
        x, y, z, qw, qx, qy, qz = ob.pose
        xform = wp.transform(
            wp.vec3(float(x), float(y), float(z)),
            wp.quat(float(qx), float(qy), float(qz), float(qw)),
        )
        hx, hy, hz = (float(d) / 2.0 for d in ob.dims)
        builder.add_shape_box(
            body=-1,
            xform=xform,
            hx=hx,
            hy=hy,
            hz=hz,
            cfg=env_shape_cfg,
            label=f"env_{ob.name}",
        )
        log.info(
            "  static[%s]: center=(%.3f, %.3f, %.3f) half_extents=(%.3f, %.3f, %.3f) "
            "z_top=%.3f",
            ob.name,
            x,
            y,
            z,
            hx,
            hy,
            hz,
            z + hz,
        )
        n_static += 1
    log.info("Added %d static env collision boxes", n_static)

    # ---- 4b. Hollow bin (coacd / primitives) ------------------------------
    if bin_mode == "coacd":
        binmesh, binT = bundle.vis_meshes.get("bin", (None, None))
        if binmesh is None:
            log.warning(
                "bin_mode=coacd but no 'bin' in vis_meshes; "
                "falling back to solid cuboid was skipped — bin has "
                "NO collision now."
            )
        else:
            bin_n_mesh = _trimesh_to_newton_mesh(binmesh)
            bsi = builder.add_shape_mesh(
                body=-1,
                xform=_matrix_to_wp_transform(binT),
                mesh=bin_n_mesh,
                cfg=env_shape_cfg,
                label="env_bin_coacd",
            )
            try:
                builder.approximate_meshes(
                    method="coacd",
                    shape_indices=[bsi],
                    keep_visual_shapes=False,
                    threshold=float(
                        profile.dynamic_overrides.get("coacd_threshold", 0.05)
                    ),
                )
                log.info(
                    "BIN collision built via CoACD on the visual mesh "
                    "(hollow): %d verts",
                    len(binmesh.vertices),
                )
            except Exception as e:  # noqa: BLE001
                log.warning("bin CoACD failed (%s); convex_hull fallback", e)
                builder.approximate_meshes(
                    method="convex_hull", shape_indices=[bsi], keep_visual_shapes=False
                )
    elif bin_mode == "primitives" and bin_ob is not None:
        cx, cy, cz, _qw, _qx, _qy, _qz = bin_ob.pose
        hx, hy, hz = (float(d) / 2.0 for d in bin_ob.dims)
        t = 0.0075  # half wall/floor thickness (1.5 cm walls)
        # Floor slab at the bottom + 4 perimeter walls = open tray.
        boxes = [
            ("floor", (cx, cy, cz - hz + t), (hx, hy, t)),
            ("wall_px", (cx + hx - t, cy, cz), (t, hy, hz)),
            ("wall_nx", (cx - hx + t, cy, cz), (t, hy, hz)),
            ("wall_py", (cx, cy + hy - t, cz), (hx, t, hz)),
            ("wall_ny", (cx, cy - hy + t, cz), (hx, t, hz)),
        ]
        for nm, (bx, by, bz), (bhx, bhy, bhz) in boxes:
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(
                    wp.vec3(float(bx), float(by), float(bz)),
                    wp.quat(0.0, 0.0, 0.0, 1.0),
                ),
                hx=bhx,
                hy=bhy,
                hz=bhz,
                cfg=env_shape_cfg,
                label=f"env_bin_{nm}",
            )
        log.info(
            "BIN collision built from 5 primitives (floor + 4 walls); "
            "floor_top_z=%.3f (was solid z_top=%.3f)",
            cz - hz + 2 * t,
            cz + hz,
        )

    log.info("  object_world_T = %s (initial)", bundle.object_world_T[:3, 3].tolist())
    # Most legacy scenes rely on this safety plane. Exact collision-world
    # reproductions can disable it (for example collision_test.yml has a
    # tabletop below z=0, where an extra z=0 plane would intersect objects).
    if bool(profile.dynamic_overrides.get("add_ground_plane", True)):
        try:
            builder.add_ground_plane()
        except Exception as e:
            log.warning("add_ground_plane failed: %s", e)

    # ---- 5. Manipulation target(s) as free rigid bodies -------------------
    # Multi-object scenes have N>1 entries in `bundle.objects`; single-object
    # scenes have exactly one. Each gets its own free body + CoACD-decomposed
    # mesh shape so finger / object contacts work the same as in the
    # single-object path.
    object_body_idxs: List[int] = []
    object_q_starts: List[int] = []
    import os as _os

    object_shape_cfg = _shape_cfg(object_mu)
    for obj in bundle.objects:
        obj_xform = _matrix_to_wp_transform(obj.world_T)
        body_idx = builder.add_body(
            xform=obj_xform,
            mass=float(object_mass),
            label=obj.asset_id,
        )
        object_body_idxs.append(body_idx)
        # Set armature on the object's free joint (matches
        # NewtonDataGen newton_grasp_eval.py:671-683 OBJECT_ARMATURE=0.002).
        # Without this, the constraint solver has no regularization on
        # the object's 6 free DOFs and edge contacts on the table can
        # cause persistent oscillation — object never fully settles,
        # drifts and rotates slowly under gravity even with high mu.
        obj_joint_idx = builder.joint_count - 1
        if obj_joint_idx >= 0:
            object_q_starts.append(int(builder.joint_q_start[obj_joint_idx]))
            obj_qd_start = builder.joint_qd_start[obj_joint_idx]
            for q_off in range(obj_qd_start, len(builder.joint_armature)):
                builder.joint_armature[q_off] = obj_armature
        obj_n_mesh = _trimesh_to_newton_mesh(obj.mesh)
        shape_idx = builder.add_shape_mesh(
            body=body_idx,
            xform=wp.transform_identity(),
            mesh=obj_n_mesh,
            cfg=object_shape_cfg,
            label=obj.asset_id,
        )
        n_shapes_before_obj = builder.shape_count
        # Diagnostic env-var hook: GRASPGENX_OBJECT_DECOMP=convex_hull to
        # skip CoACD on the object (useful for already-convex meshes like
        # the battery cylinder, where CoACD artifacts confuse the solver).
        _obj_decomp = _os.environ.get("GRASPGENX_OBJECT_DECOMP", "coacd")
        try:
            if _obj_decomp == "convex_hull":
                builder.approximate_meshes(
                    method="convex_hull",
                    shape_indices=[shape_idx],
                    keep_visual_shapes=False,
                )
                log.info(
                    "%s collision built as single convex hull (env override)",
                    obj.asset_id,
                )
            else:
                builder.approximate_meshes(
                    method="coacd",
                    shape_indices=[shape_idx],
                    keep_visual_shapes=False,
                    threshold=coacd_threshold,
                )
                log.info(
                    "%s mesh decomposed via CoACD (threshold=%.3f); " "%d -> %d shapes",
                    obj.asset_id,
                    coacd_threshold,
                    1,
                    builder.shape_count - n_shapes_before_obj + 1,
                )
        except Exception as e:
            log.warning(
                "%s decomposition failed (%s); falling back to convex_hull",
                obj.asset_id,
                e,
            )
            builder.approximate_meshes(
                method="convex_hull",
                shape_indices=[shape_idx],
                keep_visual_shapes=False,
            )
        # Optional: override the shape-accumulated (density=1000, CoACD-
        # derived, non-deterministic) mass/inertia with deterministic values.
        # Done AFTER approximate_meshes so the shape decomposition doesn't
        # re-clobber it. ``force_object_mass`` mirrors NewtonDataGen
        # (body_mass = object_mass); ``deterministic_inertia`` computes a
        # smooth inertia + com from the VISUAL mesh (independent of CoACD).
        if po.get("deterministic_inertia", False):
            m_target = float(object_mass)
            try:
                vm = obj.mesh.copy()
                vol = float(vm.volume) if vm.is_volume else float(vm.convex_hull.volume)
                if vol <= 1e-9:
                    raise ValueError("non-positive volume")
                vm.density = m_target / vol
                I = np.asarray(vm.moment_inertia, dtype=np.float64)
                com = np.asarray(vm.center_mass, dtype=np.float64)
                if not np.all(np.isfinite(I)):
                    raise ValueError("non-finite inertia")
            except Exception as e:  # noqa: BLE001
                hull = obj.mesh.convex_hull
                hull.density = m_target / max(float(hull.volume), 1e-9)
                I = np.asarray(hull.moment_inertia, dtype=np.float64)
                com = np.asarray(hull.center_mass, dtype=np.float64)
                log.warning(
                    "%s deterministic inertia from convex hull (%s)", obj.asset_id, e
                )
            builder.body_mass[body_idx] = m_target
            builder.body_inv_mass[body_idx] = 1.0 / m_target
            I_inv = np.linalg.inv(I)
            builder.body_inertia[body_idx] = wp.mat33(*I.reshape(-1).tolist())
            builder.body_inv_inertia[body_idx] = wp.mat33(
                *I_inv.reshape(-1).tolist()
            )
            builder.body_com[body_idx] = wp.vec3(*com.tolist())
            log.info(
                "%s deterministic mass=%.3f kg, inertia diag=%s",
                obj.asset_id,
                m_target,
                np.diag(I).round(6).tolist(),
            )
        elif po.get("force_object_mass", False):
            m_target = float(object_mass)
            builder.body_mass[body_idx] = m_target
            builder.body_inv_mass[body_idx] = 1.0 / m_target
            log.info(
                "%s forced mass=%.3f kg (inertia left shape-derived)",
                obj.asset_id,
                m_target,
            )

        log.info(
            "Object %s collision built from %d-vertex mesh; bounds=%s",
            obj.asset_id,
            len(obj.mesh.vertices),
            obj.mesh.bounds.tolist(),
        )

    # ---- 6. Initialize joint_q to match the planned start config ----------
    # (Filled in by simulate_and_export from joint_traj[0])

    model = builder.finalize()
    # MuJoCo-side overrides (condim, priority) — MUST run after finalize() but
    # BEFORE SolverMuJoCo construction, since SolverMuJoCo reads these into
    # its internal MJData at init.
    if pending_mjc_overrides:
        _apply_mjc_geom_overrides_to_model(model, pending_mjc_overrides)
        log.info(
            "[mjc-overrides] applied condim/priority to model (%d shapes)",
            len(pending_mjc_overrides),
        )

    # NOTE: condim is left at MuJoCo's default (3) — we no longer bump it
    # to 4. The earlier condim=4 override (torsional friction) was a
    # band-aid for the object-spin regression; per the §18.2 debug plan we
    # leave it at the default and address the spin at its root (object
    # mass/inertia determinism + support friction) instead.
    body_labels = [str(lbl) for lbl in builder.body_label]
    layout = _RobotLayout(
        arm_q_starts=arm_q_starts,
        gripper_drives=gripper_drives,
        mimic_drives=mimic_drives,
        object_body_idxs=object_body_idxs,
        object_q_starts=object_q_starts,
        velocity_close_drives=velocity_close_drives,
    )
    return model, layout, body_labels


def _set_joint_targets(
    targets: np.ndarray,
    layout: _RobotLayout,
    arm_q: np.ndarray,
    gripper_q: Dict[str, float],
) -> np.ndarray:
    """Build the per-DOF ``joint_target_q`` array for one waypoint.

    Args:
        arm_q: array of length ``len(layout.arm_q_starts)`` with arm joint
            targets, in profile order.
        gripper_q: ``{gripper_joint_name: target}`` — every joint in
            ``layout.gripper_drives`` must appear.

    Mimic followers get ``multiplier · joint_q[master] + offset`` so the
    PD drive stays aligned with Newton's mimic constraint instead of
    fighting it back toward 0.
    """
    out = targets.copy()
    for off, v in zip(layout.arm_q_starts, arm_q):
        out[off] = float(v)
    for name, q_off, _, _ in layout.gripper_drives:
        out[q_off] = float(gripper_q[name])
    for q_off, master_q_off, mult, ofs in layout.mimic_drives:
        out[q_off] = float(mult) * float(out[master_q_off]) + float(ofs)
    return out


def simulate_and_export(
    bundle: SceneBundle,
    profile: RobotProfile,
    joint_traj: np.ndarray,
    out_path: Path,
    grasps_world: np.ndarray | None = None,
    target_idx: int = -1,
    camera_eye: List[float] | None = None,
    camera_target: List[float] | None = None,
    sim_fps: int = 60,
    sim_dt: float = 0.001,
    arm_kp: float = ARM_KP_DEFAULT,
    arm_kd: float = ARM_KD_DEFAULT,
    finger_kp: float = FINGER_KP_DEFAULT,
    finger_kd: float = FINGER_KD_DEFAULT,
    gravity: float = -9.81,
    settle_frames: int = 30,
    object_mass: float = DEFAULT_OBJECT_MASS,
    object_mu: float = DEFAULT_OBJECT_MU,
    finger_mu: float = DEFAULT_FINGER_MU,
) -> Path:
    """Replay ``joint_traj`` in Newton and write the simulated trajectory JSON.

    Args:
        bundle: scene_builder.SceneBundle produced by build_scene().
        robot_cfg: parsed robot/gripper combo YAML.
        joint_traj: ``(T, n)`` numpy array of joint positions per waypoint.
            ``n`` is either 6 (arm only) or ≥7 (arm + finger_joint + …).
            Column 6 is the master finger_joint angle for Robotiq close.
        out_path: file to write the trajectory JSON to.
        grasps_world: optional ``(K, 4, 4)`` annotation array, pass-through.
        target_idx: index into ``grasps_world`` for the chosen target.
        camera_eye, camera_target: optional camera defaults for the JSON.
        sim_fps: target trajectory frame rate (also written to JSON).
        sim_dt: physics step in seconds. The number of solver substeps
            per recorded frame is computed as ``round(1/(sim_fps*sim_dt))``.
            Default 0.001 s (1000 Hz) — gives the contact solver plenty
            of headroom for tight finger-pad/object contacts.
        arm_kp, arm_kd, finger_kp, finger_kd: PD gains. The robot YAML
            may override via ``robot_cfg['dynamic']`` (CLI flag → YAML →
            this argument default).
        gravity: world-frame z-gravity in m/s².
        settle_frames: extra frames to step after the trajectory ends with
            the *final* target held, so contacts can stabilize.
        object_mass: kg.
        object_mu, finger_mu: Coulomb friction coefficients.

    Returns:
        The path the JSON was written to.
    """
    # CLI gains > YAML overrides > module defaults. Inject into the
    # profile's `dynamic_overrides` dict so _build_scene picks them up.
    profile.dynamic_overrides.setdefault("arm_kp", arm_kp)
    profile.dynamic_overrides.setdefault("arm_kd", arm_kd)
    profile.dynamic_overrides.setdefault("finger_kp", finger_kp)
    profile.dynamic_overrides.setdefault("finger_kd", finger_kd)

    # Derive substeps so sim_dt is honoured. Round to nearest int >= 1.
    sim_substeps = max(1, int(round(1.0 / (float(sim_fps) * float(sim_dt)))))
    effective_dt = 1.0 / (float(sim_fps) * float(sim_substeps))
    log.info(
        "Dynamic playback (%s): %d planned waypoints, sim_fps=%d, "
        "sim_dt=%.4f (%d substeps -> effective dt=%.4f)",
        profile.NAME,
        joint_traj.shape[0],
        sim_fps,
        sim_dt,
        sim_substeps,
        effective_dt,
    )

    # 1. Build the Newton scene. Physics overrides (gap/margin/contact
    # stiffness/inertia) are read from the robot YAML's `dynamic` block, so
    # they're per-robot — a gripper combo can set gap/margin while Franka
    # (no such keys) keeps Newton's defaults.
    physics_overrides = _physics_overrides_from_profile(profile)
    model, layout, body_labels = _build_scene(
        bundle=bundle,
        profile=profile,
        gravity=gravity,
        object_mass=object_mass,
        object_mu=object_mu,
        finger_mu=finger_mu,
        physics_overrides=physics_overrides,
    )
    log.info(
        "Newton scene: %d bodies, %d joints, %d shapes; object_body=%d",
        model.body_count,
        model.joint_count,
        model.shape_count,
        layout.object_body_idx,
    )

    # 2. Seed joint_q from joint_traj[0].
    #
    # Trajectory column layout (from e2e_grasp_demo.py): the first
    # ``profile.n_arm`` columns are the arm joints in profile order;
    # the remaining columns are the master gripper joints in
    # ``profile.gripper_joint_names`` order. A trajectory shorter than
    # ``n_arm + n_gripper`` falls back to ``open_value`` for missing
    # gripper columns.
    n_arm = profile.n_arm
    n_traj_dof = joint_traj.shape[1]
    joint_q_init = model.joint_q.numpy().copy()
    for off, v in zip(layout.arm_q_starts, joint_traj[0, :n_arm]):
        joint_q_init[off] = float(v)
    gripper_q0: Dict[str, float] = {}
    for k, (name, q_off, open_v, _close_v) in enumerate(layout.gripper_drives):
        col = n_arm + k
        v = float(joint_traj[0, col]) if col < n_traj_dof else open_v
        joint_q_init[q_off] = v
        gripper_q0[name] = v
    # Mimics: start consistent with their masters so the constraint
    # solver doesn't do a violent correction in the first step.
    for q_off, master_q_off, mult, ofs in layout.mimic_drives:
        joint_q_init[q_off] = mult * joint_q_init[master_q_off] + ofs
    model.joint_q.assign(joint_q_init)

    # 3. Initial state via FK so body_q reflects joint_q.
    states = [model.state(), model.state()]
    newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])

    # 4. Solver + collision pipeline.
    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        solver="newton",
        integrator="implicitfast",
        cone="elliptic",
        iterations=SOLVER_ITERATIONS,
        ls_iterations=SOLVER_LS_ITERATIONS,
        impratio=SOLVER_IMPRATIO,
        njmax=MJC_NCONMAX,
        nconmax=MJC_NCONMAX,
    )
    collision_pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=True,
        broad_phase="explicit",
    )
    contacts = newton.Contacts(
        rigid_contact_max=CONTACT_MAX,
        soft_contact_max=0,
        device=model.device,
    )
    control = model.control()

    step_dt = effective_dt

    # Cache: yourdfpy FK for visual mesh world transforms (matches kinematic mode).
    fk = URDFFK(bundle.robot_urdf_path, asset_root=bundle.robot_asset_root)
    actuated = fk.actuated_joint_names()

    # Map Newton joint index → bare URDF joint name → URDFFK actuated index.
    # We use this to build a {joint_name: value} dict from state.joint_q each
    # frame. yourdfpy resolves mimic joints internally so we only need master
    # joints in the dict.
    actuated_set = set(actuated)
    joint_q_start_np = model.joint_q_start.numpy()
    # Pre-compute the bare-name → q_offset mapping for actuated joints only.
    actuated_q_offset: Dict[str, int] = {}
    for j in range(model.joint_count):
        bare = _bare(str(model.joint_label[j]))
        if bare in actuated_set:
            actuated_q_offset[bare] = int(joint_q_start_np[j])

    # Per-frame recorder.
    static_dir = out_path.parent / "static_meshes"
    static_dir.mkdir(parents=True, exist_ok=True)
    static: Dict[str, Dict[str, Any]] = {}
    object_asset_ids = {obj.asset_id for obj in bundle.objects}
    for name, (mesh, T_world) in bundle.vis_meshes.items():
        if name in object_asset_ids:
            # Object meshes move in dynamic mode — written into the
            # ``objects`` list + frames[t].object_poses instead.
            continue
        try:
            mesh.export(static_dir / f"{name}.obj")
        except Exception as e:
            log.warning("Failed to export static mesh %r: %s", name, e)
            continue
        static[name] = {
            "mesh_rel": f"static_meshes/{name}.obj",
            "transform": T_world.tolist(),
        }
    # Write each object mesh under its own subdirectory so textured
    # meshes' .mtl + .jpeg sidecars don't clobber each other (trimesh
    # writes them as the generic `material.mtl` / `material_0.jpeg`).
    objects_meta: List[Dict[str, Any]] = []
    for obj in bundle.objects:
        sub = static_dir / obj.asset_id
        sub.mkdir(parents=True, exist_ok=True)
        rel = f"static_meshes/{obj.asset_id}/{obj.asset_id}.obj"
        try:
            obj.mesh.export(sub / f"{obj.asset_id}.obj")
        except Exception as e:
            log.warning("Failed to export object mesh %s: %s", obj.asset_id, e)
            rel = obj.mesh_path  # absolute fallback
        objects_meta.append(
            {
                "id": obj.asset_id,
                "label": obj.label or obj.asset_id,
                "mesh_rel": rel,
            }
        )

    # 5. Step loop.
    frames: List[Dict[str, Any]] = []

    def _record_frame(phase: str):
        body_q_np = states[0].body_q.numpy()
        joint_q_np = states[0].joint_q.numpy()

        # Build URDF cfg dict from the *simulated* joint angles. For the
        # finger_joint, the achieved value may be < commanded because of
        # contact — that's exactly the physics we want to visualize.
        cfg_dict: Dict[str, float] = {n: 0.0 for n in actuated}
        for bare, off in actuated_q_offset.items():
            cfg_dict[bare] = float(joint_q_np[off])

        link_to_world_mesh = fk.link_poses_with_visual_offset(
            cfg_dict,
            base_T=bundle.robot_base_T,
        )
        parts: List[Dict[str, Any]] = []
        for vis, T_world_mesh in link_to_world_mesh:
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

        # Per-object world transforms.
        object_poses: Dict[str, List[List[float]]] = {}
        for obj, bidx in zip(bundle.objects, layout.object_body_idxs):
            T = _wp_transform_to_matrix(body_q_np[bidx])
            object_poses[obj.asset_id] = T.tolist()
        # Backward-compat: also emit the first object as a ``parts`` entry
        # named "object" so the legacy single-object renderer code path
        # still finds something to draw.
        if bundle.objects:
            obj0 = bundle.objects[0]
            parts.append(
                {
                    "name": "object",
                    "mesh_rel": objects_meta[0]["mesh_rel"],
                    "transform": object_poses[obj0.asset_id],
                }
            )

        # Joint position vector mirrors the trajectory column layout:
        # arm joints in profile order, then each master gripper joint.
        joint_pos_out = [float(joint_q_np[off]) for off in layout.arm_q_starts]
        for _name, q_off, _o, _c in layout.gripper_drives:
            joint_pos_out.append(float(joint_q_np[q_off]))
        frames.append(
            {
                "phase": phase,
                "joint_position": joint_pos_out,
                "parts": parts,
                "object_poses": object_poses,
            }
        )

    nan_seen = False
    total_traj = joint_traj.shape[0]
    total_with_settle = total_traj + max(0, int(settle_frames))
    log.info(
        "Stepping %d trajectory frames + %d settle frames", total_traj, settle_frames
    )

    # Per-frame arm tracking-error accumulator: |target_q - measured_q|
    # per arm joint. We log peak + RMS at the end so we can correlate
    # gain values (arm_kp/kd) with how well the PD controller is
    # actually following the cuRobo trajectory under load.
    arm_q_offsets_np = np.asarray(layout.arm_q_starts, dtype=np.int64)
    n_arm_q = arm_q_offsets_np.shape[0]
    track_err_peak = np.zeros(n_arm_q, dtype=np.float64)
    track_err_sq_sum = np.zeros(n_arm_q, dtype=np.float64)
    track_err_n = 0

    for t in range(total_with_settle):
        if nan_seen:
            break
        # Pick the target waypoint: clamp to the last waypoint during settle.
        wp_idx = min(t, total_traj - 1)
        arm_q = joint_traj[wp_idx, :n_arm]
        gripper_q: Dict[str, float] = {}
        for k, (name, _q_off, open_v, _close_v) in enumerate(layout.gripper_drives):
            col = n_arm + k
            gripper_q[name] = (
                float(joint_traj[wp_idx, col]) if col < n_traj_dof else open_v
            )
        targets_np = control.joint_target_q.numpy()
        targets_np = _set_joint_targets(targets_np, layout, arm_q, gripper_q)
        control.joint_target_q.assign(targets_np)

        # Velocity-mode close (parallel-jaw style): instead of ramping
        # position, set joint_target_qd directly. The trajectory tells
        # us *which* phase we're in (open / closing / closed) by
        # comparing the current gripper target to the open value.
        if layout.velocity_close_drives:
            vel_targets = control.joint_target_qd.numpy()
            for k, (q_off, close_vel) in enumerate(layout.velocity_close_drives):
                name, _, open_v, close_v = layout.gripper_drives[k]
                target = gripper_q[name]
                # Closing if we're not at the open value (with a tiny
                # tolerance). For pre-grasp frames the trajectory holds
                # target == open_v so target_vel = 0 (no closing pressure).
                is_open = abs(target - open_v) < 1e-6
                vel_targets[q_off] = 0.0 if is_open else close_vel
            control.joint_target_qd.assign(vel_targets)

        for s in range(sim_substeps):
            states[0].clear_forces()
            # Match NewtonDataGen's pattern: collide every Nth substep
            # (every step is wasteful; the contact set doesn't change
            # that fast). At sim_dt=0.001 + COLLIDE_SUBSTEPS=4, contacts
            # update at 250 Hz which is plenty for finger-pad/object
            # contacts under gravity.
            if s % COLLIDE_SUBSTEPS == 0:
                collision_pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], control, contacts, step_dt)
            states[0], states[1] = states[1], states[0]

        # NaN guard — bail rather than write garbage frames.
        body_q_np = states[0].body_q.numpy()
        if np.any(np.isnan(body_q_np)):
            log.warning(
                "NaN detected in body_q at frame %d (phase=%s); halting sim",
                t,
                "traj" if t < total_traj else "settle",
            )
            nan_seen = True
            break

        phase = "plan" if t < total_traj else "settle"
        _record_frame(phase)

        # --- Tracking-error accumulator (per arm joint) ---
        joint_q_now = states[0].joint_q.numpy()
        tgt_q_now = control.joint_target_q.numpy()
        arm_measured = joint_q_now[arm_q_offsets_np]
        arm_target = tgt_q_now[arm_q_offsets_np]
        arm_err = np.abs(arm_target - arm_measured)
        track_err_peak = np.maximum(track_err_peak, arm_err)
        track_err_sq_sum += arm_err * arm_err
        track_err_n += 1

        if (t + 1) % max(1, total_with_settle // 10) == 0:
            grip_q_now = [
                joint_q_now[q_off] for _, q_off, _, _ in layout.gripper_drives
            ]
            grip_target = [tgt_q_now[q_off] for _, q_off, _, _ in layout.gripper_drives]
            running_rms = float(np.sqrt(np.mean(arm_err * arm_err)))
            log.info(
                "  ... step %d/%d (phase=%s) grip_q=%s target=%s "
                "arm_track_err(rad): rms=%.4f peak_so_far=%.4f",
                t + 1,
                total_with_settle,
                phase,
                [f"{v:.4f}" for v in grip_q_now],
                [f"{v:.4f}" for v in grip_target],
                running_rms,
                float(track_err_peak.max()),
            )

    log.info(
        "Dynamic sim done: %d frames recorded%s",
        len(frames),
        " (NaN detected)" if nan_seen else "",
    )

    # Final tracking-error summary (per arm joint + overall).
    if track_err_n > 0:
        rms_per_joint = np.sqrt(track_err_sq_sum / track_err_n)
        log.info("arm_tracking_err over %d frames (rad):", track_err_n)
        log.info("  per-joint peak: %s", [f"{v:.4f}" for v in track_err_peak.tolist()])
        log.info("  per-joint rms : %s", [f"{v:.4f}" for v in rms_per_joint.tolist()])
        log.info(
            "  overall: peak=%.4f rad (%.2f deg) rms=%.4f rad (%.2f deg)",
            float(track_err_peak.max()),
            float(np.rad2deg(track_err_peak.max())),
            float(rms_per_joint.max()),
            float(np.rad2deg(rms_per_joint.max())),
        )

    # 6. Write JSON.
    target_grasp = (
        grasps_world[target_idx].tolist()
        if grasps_world is not None and 0 <= target_idx < len(grasps_world)
        else None
    )
    annotations = {
        "all_grasps": [
            g.tolist() for g in (grasps_world if grasps_world is not None else [])
        ],
        "target_grasp_transform": target_grasp,
    }
    if camera_eye is None:
        camera_eye = [1.2, -1.0, 1.1]
    if camera_target is None:
        camera_target = bundle.object_world_T[:3, 3].tolist()

    traj_out = {
        "fps": int(sim_fps),
        "total_frames": len(frames),
        "base_dir": str(out_path.parent.resolve()),
        "camera": {
            "eye": list(camera_eye),
            "target": list(camera_target),
            "up": [0, 0, 1],
        },
        "static": static,
        "objects": objects_meta,
        "annotations": annotations,
        "frames": frames,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(traj_out))
    log.info("Dynamic trajectory JSON: %s (%d frames)", out_path, len(frames))
    return out_path


# ---------------------------------------------------------------------------
# DynamicSession — resumable sim driver for multi-pick tasks
# ---------------------------------------------------------------------------


class DynamicSession:
    """A resumable Newton playback session built around the existing
    single-pass :func:`simulate_and_export` pipeline.

    The clutter task drives the sim incrementally — one pick's segments
    at a time — so it can read object pose between picks and decide
    whether the grasp succeeded. To support that we need to:

      1. Build the model + states + solver + contacts + control ONCE
         (the expensive part — robot URDF parse, CoACD, MuJoCo solver
         setup all happen here).
      2. Hand back a ``drive_segments(joint_traj, phase_labels)`` method
         the caller can call N times, each time appending to a single
         growing frames list.
      3. Expose accessors for current robot joint config and per-object
         world pose, so the caller can plan the next move from where
         the sim actually is.
      4. Write the final JSON via :meth:`export`.

    The single-pass :func:`simulate_and_export` is kept as a thin
    wrapper around this class (so existing single-object code paths are
    unchanged).
    """

    def __init__(
        self,
        bundle: SceneBundle,
        profile: RobotProfile,
        *,
        sim_fps: int = 60,
        sim_dt: float = 0.001,
        arm_kp: float = ARM_KP_DEFAULT,
        arm_kd: float = ARM_KD_DEFAULT,
        finger_kp: float = FINGER_KP_DEFAULT,
        finger_kd: float = FINGER_KD_DEFAULT,
        gravity: float = -9.81,
        object_mass: float = DEFAULT_OBJECT_MASS,
        object_mu: float = DEFAULT_OBJECT_MU,
        finger_mu: float = DEFAULT_FINGER_MU,
        initial_joint_q: np.ndarray | None = None,
        physics_overrides: dict | None = None,
    ):
        # Inject CLI overrides into the profile so _build_scene picks them up.
        profile.dynamic_overrides.setdefault("arm_kp", arm_kp)
        profile.dynamic_overrides.setdefault("arm_kd", arm_kd)
        profile.dynamic_overrides.setdefault("finger_kp", finger_kp)
        profile.dynamic_overrides.setdefault("finger_kd", finger_kd)

        self.bundle = bundle
        self.profile = profile
        self.sim_fps = int(sim_fps)
        self.sim_dt = float(sim_dt)
        self.sim_substeps = max(1, int(round(1.0 / (self.sim_fps * self.sim_dt))))
        self.effective_dt = 1.0 / (self.sim_fps * self.sim_substeps)
        self.n_arm = profile.n_arm

        log.info(
            "DynamicSession %s: %d objects, sim_fps=%d, sim_dt=%.4f "
            "(%d substeps -> dt=%.4f)",
            profile.NAME,
            len(bundle.objects),
            self.sim_fps,
            self.sim_dt,
            self.sim_substeps,
            self.effective_dt,
        )

        # Default physics overrides from the robot YAML's `dynamic` block
        # (per-robot gap/margin/etc.); explicit arg still wins if given.
        if physics_overrides is None:
            physics_overrides = _physics_overrides_from_profile(profile)
        self.model, self.layout, self.body_labels = _build_scene(
            bundle=bundle,
            profile=profile,
            gravity=gravity,
            object_mass=object_mass,
            object_mu=object_mu,
            finger_mu=finger_mu,
            physics_overrides=physics_overrides,
        )
        log.info(
            "Newton model: %d bodies, %d joints, %d shapes; object_bodies=%s",
            self.model.body_count,
            self.model.joint_count,
            self.model.shape_count,
            self.layout.object_body_idxs,
        )

        # Initial joint_q. If the caller provides initial_joint_q (a
        # full trajectory row of arm + master gripper joints), use it;
        # otherwise hold the URDF default with the gripper open.
        joint_q_init = self.model.joint_q.numpy().copy()
        if initial_joint_q is not None:
            n_traj_dof = initial_joint_q.shape[0]
            for off, v in zip(self.layout.arm_q_starts, initial_joint_q[: self.n_arm]):
                joint_q_init[off] = float(v)
            for k, (name, q_off, open_v, _close_v) in enumerate(
                self.layout.gripper_drives
            ):
                col = self.n_arm + k
                v = float(initial_joint_q[col]) if col < n_traj_dof else open_v
                joint_q_init[q_off] = v
        else:
            # Hold at URDF default + gripper open.
            for _, q_off, open_v, _ in self.layout.gripper_drives:
                joint_q_init[q_off] = open_v
        for q_off, master_q_off, mult, ofs in self.layout.mimic_drives:
            joint_q_init[q_off] = mult * joint_q_init[master_q_off] + ofs
        self.model.joint_q.assign(joint_q_init)

        self.states = [self.model.state(), self.model.state()]
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.states[0]
        )
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            iterations=SOLVER_ITERATIONS,
            ls_iterations=SOLVER_LS_ITERATIONS,
            impratio=SOLVER_IMPRATIO,
            njmax=MJC_NCONMAX,
            nconmax=MJC_NCONMAX,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            broad_phase="explicit",
        )
        self.contacts = newton.Contacts(
            rigid_contact_max=CONTACT_MAX,
            soft_contact_max=0,
            device=self.model.device,
        )
        self.control = self.model.control()
        # Initialize the PD target to the START config (home arm + open
        # gripper) so the initial settle HOLDS the home pose. Without this,
        # control.joint_target_q defaults to ~zero and the PD drags the
        # arm off home during settle, which then shows up as a visible
        # "jump" when the first planned segment begins from the (correct)
        # home pose. Mirrors the per-frame target set in drive_segments.
        if initial_joint_q is not None:
            init_arm = np.asarray(initial_joint_q[: self.n_arm], dtype=np.float64)
        else:
            init_arm = np.asarray(
                [float(joint_q_init[off]) for off in self.layout.arm_q_starts],
                dtype=np.float64,
            )
        init_grip = {
            name: open_v
            for name, _q_off, open_v, _close_v in self.layout.gripper_drives
        }
        init_tgt = self.control.joint_target_q.numpy()
        init_tgt = _set_joint_targets(init_tgt, self.layout, init_arm, init_grip)
        self.control.joint_target_q.assign(init_tgt)
        self.step_dt = self.effective_dt

        # The model defaults now contain the caller's requested home arm,
        # open gripper, and the initial free-object poses. Keep explicit
        # copies so repeated trials can reset without rebuilding Newton or
        # carrying MuJoCo warm-start state across episodes.
        self._reset_joint_q = self.model.joint_q.numpy().copy()
        self._reset_joint_qd = self.model.joint_qd.numpy().copy()
        self._reset_arm_q = init_arm.copy()
        self._reset_gripper_q = init_grip.copy()

        # URDF FK for per-frame visual mesh transforms.
        self.fk = URDFFK(bundle.robot_urdf_path, asset_root=bundle.robot_asset_root)
        self.actuated = self.fk.actuated_joint_names()
        actuated_set = set(self.actuated)
        joint_q_start_np = self.model.joint_q_start.numpy()
        self.actuated_q_offset: Dict[str, int] = {}
        for j in range(self.model.joint_count):
            bare = _bare(str(self.model.joint_label[j]))
            if bare in actuated_set:
                self.actuated_q_offset[bare] = int(joint_q_start_np[j])

        # Frames buffer + objects metadata for export. We don't know the
        # output path yet; the caller passes it to ``export``.
        self.frames: List[Dict[str, Any]] = []
        self.objects_meta: List[Dict[str, Any]] = []
        # Mesh exports happen lazily on first export() call.
        self._exported = False
        self._nan_seen = False
        # Arm tracking-error accumulator: |target_q - measured_q| per
        # arm joint, accumulated across all drive_segments() calls in
        # this session. Logged on demand via log_tracking_summary().
        self._arm_q_offsets = np.asarray(self.layout.arm_q_starts, dtype=np.int64)
        self._track_err_peak = np.zeros(self._arm_q_offsets.shape[0], dtype=np.float64)
        self._track_err_sq_sum = np.zeros(
            self._arm_q_offsets.shape[0], dtype=np.float64
        )
        self._track_err_n = 0
        self._sim_frame_count = 0

    # -- accessors ---------------------------------------------------------
    @property
    def total_frames(self) -> int:
        return len(self.frames)

    def current_robot_q(self) -> np.ndarray:
        """Return the current arm joint config (length ``n_arm``)."""
        joint_q_np = self.states[0].joint_q.numpy()
        return np.array(
            [float(joint_q_np[off]) for off in self.layout.arm_q_starts],
            dtype=np.float32,
        )

    def current_gripper_q(self) -> Dict[str, float]:
        joint_q_np = self.states[0].joint_q.numpy()
        return {
            name: float(joint_q_np[q_off])
            for name, q_off, _, _ in self.layout.gripper_drives
        }

    def current_object_pose(self, obj_idx: int) -> np.ndarray:
        """4x4 world transform of object index `obj_idx`."""
        body_q_np = self.states[0].body_q.numpy()
        return _wp_transform_to_matrix(body_q_np[self.layout.object_body_idxs[obj_idx]])

    def reset(
        self,
        *,
        clear_frames: bool = True,
        object_world_poses: List[np.ndarray] | None = None,
    ) -> None:
        """Restore the robot and all free objects to their initial state.

        This also clears MuJoCo's persistent solver buffers, controller
        targets, NaN state, and tracking statistics. The expensive model,
        collision decomposition, and solver construction are retained.

        Args:
            clear_frames: Clear accumulated export frames.
            object_world_poses: Optional replacement 4x4 world poses, one per
                free object. These update both Newton's reset coordinates and
                the scene-bundle metadata without rebuilding the model.
        """
        reset_joint_q = self._reset_joint_q
        if object_world_poses is not None:
            if len(object_world_poses) != len(self.layout.object_q_starts):
                raise ValueError(
                    "object_world_poses must contain one pose per free object: "
                    f"got {len(object_world_poses)}, expected "
                    f"{len(self.layout.object_q_starts)}"
                )
            reset_joint_q = self._reset_joint_q.copy()
            for index, (q_start, world_pose) in enumerate(
                zip(self.layout.object_q_starts, object_world_poses)
            ):
                pose = np.asarray(world_pose, dtype=np.float64)
                if pose.shape != (4, 4) or not np.isfinite(pose).all():
                    raise ValueError(
                        f"Object pose {index} must be a finite 4x4 transform"
                    )
                quat_wxyz = tra.quaternion_from_matrix(pose)
                reset_joint_q[q_start : q_start + 7] = [
                    float(pose[0, 3]),
                    float(pose[1, 3]),
                    float(pose[2, 3]),
                    float(quat_wxyz[1]),
                    float(quat_wxyz[2]),
                    float(quat_wxyz[3]),
                    float(quat_wxyz[0]),
                ]
                self.bundle.objects[index].world_T = pose.copy()
            if object_world_poses:
                self.bundle.object_world_T = np.asarray(
                    object_world_poses[0], dtype=np.float64
                ).copy()

        for state in self.states:
            state.joint_q.assign(reset_joint_q)
            state.joint_qd.assign(self._reset_joint_qd)
            state.clear_forces()
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

        # flags=0 preserves the explicit state assigned above while clearing
        # qacc warm-start, applied forces, activations, and controller buffers.
        self.solver.reset(self.states[0], flags=0)

        targets = self.control.joint_target_q.numpy()
        targets = _set_joint_targets(
            targets,
            self.layout,
            self._reset_arm_q,
            self._reset_gripper_q,
        )
        self.control.joint_target_q.assign(targets)
        self.control.joint_target_qd.zero_()

        if clear_frames:
            self.frames.clear()
        self._nan_seen = False
        self._track_err_peak.fill(0.0)
        self._track_err_sq_sum.fill(0.0)
        self._track_err_n = 0
        self._sim_frame_count = 0

    # -- driving ----------------------------------------------------------
    def drive_segments(
        self,
        joint_traj: np.ndarray,
        phase_labels: List[Tuple[str, int]] | None = None,
        settle_frames: int = 0,
        *,
        record_frames: bool = True,
        frame_callback: Callable[["DynamicSession", str], None] | None = None,
    ) -> bool:
        """Step the sim through ``joint_traj`` (T, n_arm+n_gripper).

        Optionally appends ``settle_frames`` extra frames after the
        trajectory holding the final target. ``record_frames=False`` avoids
        accumulating export data for long-running validation loops. A
        ``frame_callback`` can update a live visualizer after every simulated
        frame. Returns False if a NaN was detected (caller should stop driving
        further segments).

        ``phase_labels`` is a list of ``(name, n_frames)`` mirroring
        :class:`TaskResult.segments`; if provided, each recorded frame's
        ``"phase"`` field is the matching segment name, otherwise
        ``"plan"`` / ``"settle"``.
        """
        if self._nan_seen:
            return False
        if joint_traj.shape[0] == 0 and settle_frames == 0:
            return True

        n_traj_dof = joint_traj.shape[1]
        total_traj = joint_traj.shape[0]
        total_with_settle = total_traj + max(0, int(settle_frames))

        # Pre-compute the per-frame phase name from phase_labels.
        if phase_labels is not None:
            phase_per_frame: List[str] = []
            for name, k in phase_labels:
                phase_per_frame.extend([name] * int(k))
            # Pad or truncate to total_traj.
            if len(phase_per_frame) < total_traj:
                phase_per_frame.extend(["plan"] * (total_traj - len(phase_per_frame)))
            else:
                phase_per_frame = phase_per_frame[:total_traj]
        else:
            phase_per_frame = ["plan"] * total_traj
        phase_per_frame = phase_per_frame + ["settle"] * max(0, int(settle_frames))

        for t in range(total_with_settle):
            if self._nan_seen:
                return False
            wp_idx = min(t, total_traj - 1) if total_traj > 0 else 0
            if total_traj > 0:
                arm_q = joint_traj[wp_idx, : self.n_arm]
                gripper_q: Dict[str, float] = {}
                for k, (name, _q_off, open_v, _close_v) in enumerate(
                    self.layout.gripper_drives
                ):
                    col = self.n_arm + k
                    gripper_q[name] = (
                        float(joint_traj[wp_idx, col]) if col < n_traj_dof else open_v
                    )
                targets_np = self.control.joint_target_q.numpy()
                targets_np = _set_joint_targets(
                    targets_np, self.layout, arm_q, gripper_q
                )
                self.control.joint_target_q.assign(targets_np)

                if self.layout.velocity_close_drives:
                    vel_targets = self.control.joint_target_qd.numpy()
                    for k, (q_off, close_vel) in enumerate(
                        self.layout.velocity_close_drives
                    ):
                        name, _, open_v, _close_v = self.layout.gripper_drives[k]
                        target = gripper_q[name]
                        is_open = abs(target - open_v) < 1e-6
                        vel_targets[q_off] = 0.0 if is_open else close_vel
                    self.control.joint_target_qd.assign(vel_targets)

            for s in range(self.sim_substeps):
                self.states[0].clear_forces()
                if s % COLLIDE_SUBSTEPS == 0:
                    self.collision_pipeline.collide(self.states[0], self.contacts)
                self.solver.step(
                    self.states[0],
                    self.states[1],
                    self.control,
                    self.contacts,
                    self.step_dt,
                )
                self.states[0], self.states[1] = self.states[1], self.states[0]

            body_q_np = self.states[0].body_q.numpy()
            if np.any(np.isnan(body_q_np)):
                log.warning(
                    "NaN detected at session frame %d; halting drive",
                    self._sim_frame_count,
                )
                self._nan_seen = True
                return False

            # Per-frame arm tracking-error accumulator (target vs measured).
            joint_q_np = self.states[0].joint_q.numpy()
            tgt_q_np = self.control.joint_target_q.numpy()
            arm_err = np.abs(
                tgt_q_np[self._arm_q_offsets] - joint_q_np[self._arm_q_offsets]
            )
            self._track_err_peak = np.maximum(self._track_err_peak, arm_err)
            self._track_err_sq_sum += arm_err * arm_err
            self._track_err_n += 1

            phase = phase_per_frame[t]
            if record_frames:
                self._record_frame(phase)
            self._sim_frame_count += 1
            if frame_callback is not None:
                frame_callback(self, phase)

        return True

    def log_tracking_summary(self, label: str = "session"):
        """Log per-joint peak + RMS arm tracking error since session
        start (or last reset). Call once at the end of a clutter run."""
        if self._track_err_n <= 0:
            return
        rms = np.sqrt(self._track_err_sq_sum / self._track_err_n)
        log.info("arm_tracking_err [%s] over %d frames:", label, self._track_err_n)
        log.info(
            "  per-joint peak (rad): %s",
            [f"{v:.4f}" for v in self._track_err_peak.tolist()],
        )
        log.info("  per-joint rms  (rad): %s", [f"{v:.4f}" for v in rms.tolist()])
        log.info(
            "  overall: peak=%.4f rad (%.2f deg) rms=%.4f rad (%.2f deg)",
            float(self._track_err_peak.max()),
            float(np.rad2deg(self._track_err_peak.max())),
            float(rms.max()),
            float(np.rad2deg(rms.max())),
        )

    # -- recording --------------------------------------------------------
    def _record_frame(self, phase: str):
        body_q_np = self.states[0].body_q.numpy()
        joint_q_np = self.states[0].joint_q.numpy()
        cfg_dict: Dict[str, float] = {n: 0.0 for n in self.actuated}
        for bare, off in self.actuated_q_offset.items():
            cfg_dict[bare] = float(joint_q_np[off])
        link_to_world_mesh = self.fk.link_poses_with_visual_offset(
            cfg_dict,
            base_T=self.bundle.robot_base_T,
        )
        parts: List[Dict[str, Any]] = []
        for vis, T_world_mesh in link_to_world_mesh:
            mesh_abs = vis.mesh_rel
            try:
                p = Path(vis.mesh_rel)
                if not p.is_absolute():
                    p = (Path(self.bundle.robot_asset_root) / p).resolve()
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
        object_poses: Dict[str, List[List[float]]] = {}
        for obj, bidx in zip(self.bundle.objects, self.layout.object_body_idxs):
            object_poses[obj.asset_id] = _wp_transform_to_matrix(
                body_q_np[bidx]
            ).tolist()
        joint_pos_out = [float(joint_q_np[off]) for off in self.layout.arm_q_starts]
        for _name, q_off, _o, _c in self.layout.gripper_drives:
            joint_pos_out.append(float(joint_q_np[q_off]))
        self.frames.append(
            {
                "phase": phase,
                "joint_position": joint_pos_out,
                "parts": parts,
                "object_poses": object_poses,
            }
        )

    # -- export ----------------------------------------------------------
    def export(
        self,
        out_path: Path,
        *,
        camera_eye: List[float] | None = None,
        camera_target: List[float] | None = None,
        annotations: Dict[str, Any] | None = None,
    ) -> Path:
        out_path = Path(out_path)
        static_dir = out_path.parent / "static_meshes"
        static_dir.mkdir(parents=True, exist_ok=True)
        static: Dict[str, Dict[str, Any]] = {}
        object_asset_ids = {obj.asset_id for obj in self.bundle.objects}
        for name, (mesh, T_world) in self.bundle.vis_meshes.items():
            if name in object_asset_ids:
                continue
            try:
                mesh.export(static_dir / f"{name}.obj")
            except Exception as e:
                log.warning("Failed to export static mesh %r: %s", name, e)
                continue
            static[name] = {
                "mesh_rel": f"static_meshes/{name}.obj",
                "transform": T_world.tolist(),
            }
        objects_meta: List[Dict[str, Any]] = []
        for obj in self.bundle.objects:
            sub = static_dir / obj.asset_id
            sub.mkdir(parents=True, exist_ok=True)
            rel = f"static_meshes/{obj.asset_id}/{obj.asset_id}.obj"
            try:
                obj.mesh.export(sub / f"{obj.asset_id}.obj")
            except Exception as e:
                log.warning("Failed to export object mesh %s: %s", obj.asset_id, e)
                rel = obj.mesh_path
            objects_meta.append(
                {
                    "id": obj.asset_id,
                    "label": obj.label or obj.asset_id,
                    "mesh_rel": rel,
                }
            )
        if camera_eye is None:
            camera_eye = [1.2, -1.0, 1.1]
        if camera_target is None:
            camera_target = self.bundle.objects[0].world_T[:3, 3].tolist()
        traj_out = {
            "fps": int(self.sim_fps),
            "total_frames": len(self.frames),
            "base_dir": str(out_path.parent.resolve()),
            "camera": {
                "eye": list(camera_eye),
                "target": list(camera_target),
                "up": [0, 0, 1],
            },
            "static": static,
            "objects": objects_meta,
            "annotations": annotations or {},
            "frames": self.frames,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(traj_out))
        log.info("DynamicSession exported %d frames -> %s", len(self.frames), out_path)
        return out_path
