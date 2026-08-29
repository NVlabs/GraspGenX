"""Robot/gripper profile abstraction for the end2end demo.

The kinematic and dynamic pipelines used to be UR10e+Robotiq-only — joint
names, finger close logic, and mimic constraints were all hardcoded.
This module factors that out into a :class:`RobotProfile` base + one
concrete subclass per (robot, gripper) combination, so adding a new
robot is a single new class + YAML field rather than scattered edits in
``dynamic_playback.py`` / ``e2e_grasp_demo.py``.

A profile owns:

  * **Arm joint names** in cuRobo's planning order — used to map cuRobo's
    flat (T, n_arm) trajectory into URDF-named joint configs and to set
    Newton PD targets on the correct DOFs.
  * **Gripper joint names** the demo actively drives. May be one
    (Robotiq: ``finger_joint`` with mimic followers) or several (Franka:
    independent ``panda_finger_joint1``/``panda_finger_joint2``).
  * **Open / close values per gripper joint** — used to seed the close
    ramp at the grasp pose. Close is the target the demo ramps to in
    dynamic mode; the physics naturally limits the actual close on
    contact.
  * **cuRobo robot config reference** + **tool frame** + **default
    arm-only q**, for backwards compat with the existing YAML loader.
  * **GraspGenX gripper_name + checkpoints_dir** for inference lookups.

Path fields (``urdf_path``, ``asset_root_path``, ``robot_base_T``,
``grasp_to_tool_transform``) are populated *per instance* from the
loaded YAML so a single profile can be reused across deployments that
keep the URDF in different places.

Mimic constraints are parsed from the URDF via yourdfpy by default; if
a robot needs custom mimic handling (e.g. an inferred multiplier) a
subclass can override :meth:`RobotProfile.mimic_constraints`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import trimesh.transformations as tra

log = logging.getLogger("robot_profile")


@dataclass
class GraspGenInfo:
    """GraspGenX inference parameters tied to a particular gripper."""

    gripper_name: str
    checkpoints_dir: str
    assets_dir: str | None = None


@dataclass
class RobotProfile:
    """Per-robot configuration consumed by the end2end demo.

    Subclasses set the class-attribute defaults (joint names, open/close
    values, tool frame, etc.) and inherit ``__init__`` to absorb the
    per-instance paths read from the YAML. To register a new robot:

    1. Subclass :class:`RobotProfile`.
    2. Set ``NAME`` and override the class-level defaults.
    3. Add the class to :data:`ROBOT_PROFILES`.
    """

    # -- Per-instance fields (populated from YAML) -------------------------
    urdf_path: str
    asset_root_path: str
    robot_base_T: np.ndarray
    grasp_to_tool_transform: np.ndarray
    graspgen: GraspGenInfo

    # PD / friction overrides (rarely needed — most robots use defaults).
    # Values are usually floats (gains, friction coefficients) but may also
    # be strings (e.g. a path to a JSON mjc_geom_overrides file).
    dynamic_overrides: Dict[str, Any] = field(default_factory=dict)

    # -- Class-level defaults (override in subclasses) ---------------------
    NAME: str = "base"

    # Joint ordering matters: cuRobo emits trajectories in this order, so
    # subclasses must list arm joints in the exact same order as the
    # ``cspace.joint_names`` field of the matching cuRobo robot YAML.
    arm_joint_names: List[str] = field(default_factory=list)
    tool_frame: str = ""
    default_arm_q: List[float] = field(default_factory=list)
    curobo_robot_config: str = ""

    # Master gripper joints (the ones the demo actively drives). Mimic
    # followers are handled separately via :meth:`mimic_constraints`.
    gripper_joint_names: List[str] = field(default_factory=list)
    gripper_open: Dict[str, float] = field(default_factory=dict)
    gripper_close: Dict[str, float] = field(default_factory=dict)

    # Heuristic for which URDF link labels are part of the gripper (used
    # to decide which mesh shapes get the "gripper" treatment vs the
    # "arm" treatment in dynamic mode). Default catches typical naming
    # conventions; override if a gripper uses unusual link names.
    gripper_body_keywords: Tuple[str, ...] = (
        "finger",
        "knuckle",
        "gripper",
        "hand",
        "thumb",
        "robotiq",
        "arg2f",
    )

    # If True, the gripper's mesh shapes get CoACD (multi-piece convex
    # decomposition) — needed for any gripper whose inner pads have
    # meaningful concavity (Robotiq, Panda, etc.). False is only a
    # speed/stability fallback. This concerns *collision representation*
    # — independent of how the joint is actuated.
    use_coacd_for_gripper: bool = True

    # Fixed-link collapsing is normally desirable, but some profiles select
    # collision treatment by the names of fixed child links.  Those profiles
    # can preserve the link bodies so the selector remains unambiguous.
    collapse_fixed_joints: bool = True

    # Joint armature for the gripper master joints. Newton's PD is
    # unstable on low-inertia prismatic joints without armature; the
    # Newton Panda examples use 0.15–0.5 here.
    gripper_armature: float = 0.0

    # Per-profile filter for which gripper bodies actually get CoACD
    # (vs convex_hull). Empty tuple means *all* gripper bodies get
    # CoACD — right for the Robotiq where every finger link has
    # meaningful concavity. Restrict for grippers like the Franka where
    # only some links (the fingertips) need the multi-piece treatment.
    coacd_link_keywords: Tuple[str, ...] = ()

    # Actuation mode for the master gripper joints. ``"position"`` is
    # right for revolute Robotiq-style grippers (the close target is a
    # fixed angle the URDF respects). ``"velocity"`` is what
    # NewtonDataGen uses for parallel-jaw 2F grippers and what we use
    # for the Franka Panda: set ``joint_target_vel`` to a constant
    # closing rate, leave ke=0 and kd high, and let the velocity command
    # squeeze on contact until equilibrium. Independent of the
    # trajectory column layout — the trajectory still stores positions
    # for compatibility with the kinematic renderer; dynamic_playback
    # converts them per-mode.
    gripper_control_mode: str = "position"  # "position" | "velocity"
    # Closing velocity per master joint, signed. Only used when
    # ``gripper_control_mode == "velocity"``.
    gripper_close_velocity: Dict[str, float] = field(default_factory=dict)

    # ---------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, robot_cfg: Dict[str, Any]) -> "RobotProfile":
        """Instantiate from a parsed robot YAML.

        Looks up ``robot_cfg['profile']`` against :data:`ROBOT_PROFILES`
        (falling back to ``robot_cfg['name']`` for backward compatibility
        with YAMLs that pre-date the profile field) and populates the
        instance fields from YAML keys.
        """
        key = robot_cfg.get("profile") or robot_cfg.get("name") or ""
        if key not in ROBOT_PROFILES:
            raise KeyError(
                f"Unknown robot profile {key!r}. Known: {sorted(ROBOT_PROFILES)}"
            )
        profile_cls = ROBOT_PROFILES[key]

        base_pose = robot_cfg.get("robot_base_pose", {})
        base_T = _xyzw_to_matrix(
            base_pose.get("translation", [0.0, 0.0, 0.0]),
            base_pose.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]),
        )

        g2t = robot_cfg.get("grasp_to_tool_transform", {})
        grasp_to_tool_T = _xyzw_to_matrix(
            g2t.get("translation", [0.0, 0.0, 0.0]),
            g2t.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]),
        )

        gg = robot_cfg.get("graspgen", {})
        graspgen = GraspGenInfo(
            gripper_name=gg.get("gripper_name", ""),
            checkpoints_dir=gg.get("checkpoints_dir", ""),
            assets_dir=gg.get("assets_dir"),
        )

        inst = profile_cls(
            urdf_path=robot_cfg["urdf_path"],
            asset_root_path=robot_cfg.get(
                "asset_root_path", str(Path(robot_cfg["urdf_path"]).parent)
            ),
            robot_base_T=base_T,
            grasp_to_tool_transform=grasp_to_tool_T,
            graspgen=graspgen,
            dynamic_overrides=robot_cfg.get("dynamic", {}),
        )
        # Optional YAML overrides of the gripper/arm specifics. Backward
        # compatible (franka / inspire YAMLs omit these and keep their class
        # defaults); used by the generic `ur10e_gripper` profile so a single
        # class can serve many UR10e + <gripper> combos straight from YAML.
        for key in (
            "arm_joint_names",
            "tool_frame",
            "default_arm_q",
            "gripper_joint_names",
            "gripper_open",
            "gripper_close",
            "gripper_control_mode",
            "gripper_close_velocity",
            "gripper_armature",
            "collapse_fixed_joints",
            "curobo_robot_config",
        ):
            if key in robot_cfg:
                setattr(inst, key, robot_cfg[key])
        if "coacd_link_keywords" in robot_cfg:
            inst.coacd_link_keywords = tuple(robot_cfg["coacd_link_keywords"])
        if "gripper_body_keywords" in robot_cfg:
            inst.gripper_body_keywords = tuple(robot_cfg["gripper_body_keywords"])
        return inst

    # ---------------------------------------------------------------------

    @property
    def n_arm(self) -> int:
        return len(self.arm_joint_names)

    @property
    def n_gripper(self) -> int:
        return len(self.gripper_joint_names)

    def close_value(self, joint_name: str) -> float:
        """Closing target for a single gripper joint."""
        if joint_name in self.gripper_close:
            return float(self.gripper_close[joint_name])
        raise KeyError(f"{joint_name!r} has no close value in {self.NAME}")

    def open_value(self, joint_name: str) -> float:
        return float(self.gripper_open.get(joint_name, 0.0))

    def mimic_constraints(self) -> Dict[str, Tuple[str, float, float]]:
        """Return ``{follower_name: (master_name, multiplier, offset)}``.

        Parses the URDF via yourdfpy. Subclass override if you need to
        inject constraints the URDF lacks.
        """
        import yourdfpy

        u = yourdfpy.URDF.load(
            self.urdf_path,
            build_collision_scene_graph=False,
            load_meshes=False,
        )
        out: Dict[str, Tuple[str, float, float]] = {}
        for j in u.robot.joints:
            if j.mimic is None:
                continue
            out[j.name] = (
                j.mimic.joint,
                float(j.mimic.multiplier),
                float(j.mimic.offset),
            )
        return out

    def gripper_pad_links(self) -> Tuple[str, str] | None:
        """Two URDF link names whose midpoint distance equals the gripper gap.

        Used by the object-aware close in kinematic mode. Returning
        ``None`` disables object-aware close for this robot.
        """
        return None


# ---------------------------------------------------------------------------
# Concrete profiles
# ---------------------------------------------------------------------------


@dataclass
class UR10eRobotiq2F140Profile(RobotProfile):
    """UR10e arm + Robotiq 2F-140 gripper (one master finger_joint with
    5 mimic followers in the URDF).
    """

    NAME: str = "ur10e_robotiq_2f_140"
    arm_joint_names: List[str] = field(
        default_factory=lambda: [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
    )
    tool_frame: str = "robotiq_arg2f_base_link"
    default_arm_q: List[float] = field(
        default_factory=lambda: [
            0.0,
            -1.5,
            1.5,
            -1.5708,
            -1.5708,
            0.0,
        ]
    )
    curobo_robot_config: str = "ur10e_robotiq_2f_140.yml"

    gripper_joint_names: List[str] = field(default_factory=lambda: ["finger_joint"])
    # 0 = fully open, 0.7 ≈ fully closed for the 2F-140.
    gripper_open: Dict[str, float] = field(
        default_factory=lambda: {"finger_joint": 0.0}
    )
    gripper_close: Dict[str, float] = field(
        default_factory=lambda: {"finger_joint": 0.7}
    )
    # Revolute master joint with mimic followers — position control
    # works cleanly because the close angle is well-defined.
    gripper_control_mode: str = "position"

    def gripper_pad_links(self) -> Tuple[str, str] | None:
        return ("left_inner_finger_pad", "right_inner_finger_pad")


@dataclass
class FrankaPandaProfile(RobotProfile):
    """Franka Emika Panda — 7 arm joints + 2 independent prismatic
    fingers (no mimic; both fingers driven symmetrically by the demo).
    """

    NAME: str = "franka_panda"
    arm_joint_names: List[str] = field(
        default_factory=lambda: [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
    )
    tool_frame: str = "panda_hand"
    default_arm_q: List[float] = field(
        default_factory=lambda: [
            0.0,
            -1.3,
            0.0,
            -2.5,
            0.0,
            1.5,
            0.8,
        ]
    )
    curobo_robot_config: str = "franka.yml"

    # Single master joint; panda_finger_joint2 follows via a Newton
    # mimic constraint injected in ``mimic_constraints()`` below. This
    # mirrors how the Robotiq URDF works (one driven joint, followers
    # constrained to it) and prevents the two fingers from closing
    # independently — previously one would stop on the object while the
    # other closed past empty space (`(0.028, 0.000)` pattern).
    gripper_joint_names: List[str] = field(
        default_factory=lambda: [
            "panda_finger_joint1",
        ]
    )
    # Panda fingers are prismatic; 0.04 m fully open, 0.0 fully closed.
    gripper_open: Dict[str, float] = field(
        default_factory=lambda: {
            "panda_finger_joint1": 0.04,
        }
    )
    gripper_close: Dict[str, float] = field(
        default_factory=lambda: {
            "panda_finger_joint1": 0.0,
        }
    )
    # Position control with the gains from `newton/examples/.../
    # example_robot_panda_hydro.py`: ke=650, kd=100, effort_limit=20,
    # armature=0.5. Velocity control was asymmetric in practice (one
    # finger closed, the other stuck near open) because the URDF's
    # `damping=10` plus a kd=800 stalls the finger that doesn't have
    # immediate contact.
    gripper_control_mode: str = "position"
    gripper_armature: float = 0.5
    # Only the fingertips need CoACD (their inward-facing surfaces have
    # concavity that matters for object contact). The panda_hand body is
    # large and mostly convex — CoACDing it produces hundreds of parts
    # that overwhelm the contact pipeline. Restrict CoACD to bodies
    # whose bare URDF link name matches one of these keywords; everything
    # else in the gripper falls back to convex_hull.
    coacd_link_keywords: Tuple[str, ...] = ("finger",)

    def gripper_pad_links(self) -> Tuple[str, str] | None:
        return ("panda_leftfinger", "panda_rightfinger")

    def mimic_constraints(self) -> Dict[str, Tuple[str, float, float]]:
        """Add the missing mechanical coupling between the two panda fingers.

        cuRobo's shipped franka_panda.urdf declares ``panda_finger_joint1``
        and ``panda_finger_joint2`` as independent prismatic joints, with
        opposite axes (``axis="0 1 0"`` vs ``"0 -1 0"``). On the real
        gripper they're mechanically coupled so the two fingers always
        move symmetrically. Without that constraint our PD drives each
        finger independently and the unblocked finger over-closes while
        the blocked one stalls on contact — visible as ``(0.028, 0.000)``
        in the dynamic sweep summaries.

        We synthesize the mimic here: ``panda_finger_joint2`` follows
        ``panda_finger_joint1`` with multiplier=+1 (same joint_q value;
        the URDF's opposite axes already make the fingers move in
        opposite world directions).
        """
        out = super().mimic_constraints()
        out["panda_finger_joint2"] = ("panda_finger_joint1", 1.0, 0.0)
        return out


@dataclass
class YamFlexibleProfile(RobotProfile):
    """YAM six-axis arm with its coupled flexible parallel-jaw gripper."""

    NAME: str = "yam_flexible"
    arm_joint_names: List[str] = field(
        default_factory=lambda: [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
    )
    tool_frame: str = "franka_gripper"
    default_arm_q: List[float] = field(
        default_factory=lambda: [0.0, 0.3, 1.0, 0.0, 0.0, 0.0]
    )
    curobo_robot_config: str = "yam_single.yml"

    # joint8 follows joint7 through the mimic declaration in the URDF.
    gripper_joint_names: List[str] = field(default_factory=lambda: ["joint7"])
    gripper_open: Dict[str, float] = field(
        default_factory=lambda: {"joint7": -0.0475}
    )
    gripper_close: Dict[str, float] = field(default_factory=lambda: {"joint7": 0.0})
    gripper_control_mode: str = "velocity"
    gripper_close_velocity: Dict[str, float] = field(
        default_factory=lambda: {"joint7": 0.06}
    )
    gripper_armature: float = 0.15
    # Keep one concave collision mesh per finger component in the URDF.
    # Newton decomposes only these four meshes into convex MuJoCo geoms when
    # building the dynamic model; the shared left/right sources are cached by
    # ModelBuilder, so each unique mesh is decomposed only once.
    use_coacd_for_gripper: bool = True
    # Keep the fixed finger_set/finger_flex body labels: Newton's URDF importer
    # currently emits generic shape labels, so link names are the reliable way
    # to distinguish these envelopes from the detailed tip-base meshes.
    collapse_fixed_joints: bool = False
    gripper_body_keywords: Tuple[str, ...] = (
        "gripper",
        "finger",
        "tip_left",
        "tip_right",
    )
    # Exclude the gripper mount and tip bodies: their collision meshes are
    # already adequately represented by a single hull.  The finger-set and
    # flexible-pad meshes contain the inward recess that must survive.
    coacd_link_keywords: Tuple[str, ...] = ("finger_set", "finger_flex")

    def gripper_pad_links(self) -> Tuple[str, str] | None:
        return ("left_finger_flex", "right_finger_flex")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class UR10eInspireHandProfile(RobotProfile):
    """UR10e arm + Inspire 5-finger hand (Task G).

    Six master finger joints (thumb yaw, thumb pitch, 4 finger proximal)
    are explicitly position-controlled to the OPEN / CLOSE values from
    the Inspire ``config.json``. Five mimic followers (thumb_intermediate,
    thumb_distal, and the four ``<finger>_intermediate_joint``) are
    parsed from the URDF and propagated by Newton's built-in mimic
    constraint — same pattern NewtonDataGen's ``newton_grasp_eval.py``
    uses for this gripper.
    """

    NAME: str = "ur10e_inspire_hand"
    arm_joint_names: List[str] = field(
        default_factory=lambda: [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
    )
    tool_frame: str = "tool0"
    default_arm_q: List[float] = field(
        default_factory=lambda: [
            0.0,
            -2.2,
            1.9,
            -1.383,
            -1.57,
            0.0,
        ]
    )
    curobo_robot_config: str = "ur10e_inspire_hand.yml"

    # Six master joints — values from
    # gripper_descriptions/x_grippers/inspire_hand/config.json.
    gripper_joint_names: List[str] = field(
        default_factory=lambda: [
            "thumb_proximal_yaw_joint",
            "thumb_proximal_pitch_joint",
            "index_proximal_joint",
            "middle_proximal_joint",
            "ring_proximal_joint",
            "pinky_proximal_joint",
        ]
    )
    gripper_open: Dict[str, float] = field(
        default_factory=lambda: {
            "thumb_proximal_yaw_joint": 1.308,
            "thumb_proximal_pitch_joint": 0.0,
            "index_proximal_joint": 0.0,
            "middle_proximal_joint": 0.0,
            "ring_proximal_joint": 0.0,
            "pinky_proximal_joint": 0.0,
        }
    )
    gripper_close: Dict[str, float] = field(
        default_factory=lambda: {
            "thumb_proximal_yaw_joint": 1.308,  # locked (same as open)
            "thumb_proximal_pitch_joint": 0.6,
            "index_proximal_joint": 1.47,
            "middle_proximal_joint": 1.47,
            "ring_proximal_joint": 1.47,
            "pinky_proximal_joint": 1.47,
        }
    )
    gripper_control_mode: str = "position"
    # Only the fingertips need CoACD; the palm + proximal/intermediate
    # phalanges are mostly convex per their URDF collision tags (boxes
    # / cylinders), so a single convex hull each is fine. The thumb
    # tip / distal and the *_intermediate links have the concavity that
    # matters for object contact.
    coacd_link_keywords: Tuple[str, ...] = ("intermediate", "distal")

    def gripper_pad_links(self) -> Tuple[str, str] | None:
        # Inspire is a 5-finger hand — there isn't a 2-link "pad" pair
        # the way Robotiq / Panda fingers have. Return None so the
        # demo's auto-close-to-object gap heuristic falls back to the
        # full close_value.
        return None


@dataclass
class UR10eGripperProfile(RobotProfile):
    """Generic UR10e + <any gripper> profile.

    Carries the UR10e arm defaults; every gripper-specific field
    (gripper_joint_names / gripper_open / gripper_close / tool_frame /
    coacd_link_keywords / gripper_control_mode / ...) is supplied by the
    robot YAML and applied in :meth:`RobotProfile.from_yaml`. This lets one
    class serve many UR10e + <gripper> combos (surge_hand, unitree_g1,
    arx_x5, ...) without a bespoke subclass each.
    """

    NAME: str = "ur10e_gripper"
    arm_joint_names: List[str] = field(
        default_factory=lambda: [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
    )
    tool_frame: str = "tool0"
    default_arm_q: List[float] = field(
        default_factory=lambda: [
            0.0,
            -2.2,
            1.9,
            -1.383,
            -1.57,
            0.0,
        ]
    )
    gripper_control_mode: str = "position"

    def gripper_pad_links(self) -> Tuple[str, str] | None:
        return None


ROBOT_PROFILES: Dict[str, type] = {
    UR10eRobotiq2F140Profile.NAME: UR10eRobotiq2F140Profile,
    FrankaPandaProfile.NAME: FrankaPandaProfile,
    YamFlexibleProfile.NAME: YamFlexibleProfile,
    UR10eInspireHandProfile.NAME: UR10eInspireHandProfile,
    UR10eGripperProfile.NAME: UR10eGripperProfile,
}


def list_profiles() -> List[str]:
    return sorted(ROBOT_PROFILES.keys())


def _xyzw_to_matrix(translation: List[float], q_xyzw: List[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = translation
    if q_xyzw is None:
        return T
    x, y, z, w = q_xyzw
    if abs(x) < 1e-9 and abs(y) < 1e-9 and abs(z) < 1e-9 and abs(w - 1.0) < 1e-9:
        return T
    T[:3, :3] = tra.quaternion_matrix([w, x, y, z])[:3, :3]
    return T
