#!/usr/bin/env python3
"""Build a combined UR10e + <gripper> cuRobo robot (URDF + YAML + spheres).

Generalizes build_ur10e_inspire_urdf.py + generate_inspire_spheres.py +
the hand-written ur10e_inspire_hand.yml into a single parametrized builder.
For a gripper onboarded in gripper_descriptions/x_grippers/<gripper>/ it:

  1. Merges UR10e's URDF + the gripper's URDF onto `tool0` (drops the
     gripper's standalone `world` link; the base link = child of the
     `world_joint` is fixed-jointed to tool0 with a configurable mount).
  2. Auto-fits cuRobo collision spheres (VOXEL) for every gripper link.
  3. Assembles the cuRobo robot YAML: UR10e arm block (spheres / buffers /
     arm self-collision-ignore, copied verbatim from ur10e_inspire_hand.yml)
     + the gripper links/spheres, a cspace (6 arm + gripper master joints),
     lock_joints holding the gripper open during planning, and an
     auto-generated self_collision_ignore matrix (every gripper link ignores
     every other gripper link + all arm links — the fix that stops cuRobo's
     IK from silently failing on a self-colliding default hand pose).

Outputs (under end2end/curobo_assets/):
  ur10e_<gripper>.urdf
  ur10e_<gripper>.yml

The fixed-joint mount (rpy/xyz) is a FIRST GUESS; eyeball + refine it in
visualize_robot_curobo_spheres.py.

Run::
  uv run python end2end/build_ur10e_gripper.py --gripper surge_hand
  uv run python end2end/build_ur10e_gripper.py --gripper robotiq_2f_140 \
      --mount_rpy 0 0 0
"""

from __future__ import annotations

import argparse
import logging
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(
    format="%(asctime)s [BUILD_GRIPPER] %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
CUROBO_ASSETS = _HERE / "curobo_assets"
# Resolve the UR10e arm URDF and the gripper_descriptions root portably (no
# absolute paths): cuRobo's content/assets from the installed package, and
# the x_grippers dir via GraspGenX (see end2end/paths.py). Works wherever the
# repo is cloned.
from paths import curobo_assets_dir, grippers_dir  # noqa: E402

UR10E_URDF = curobo_assets_dir() / "robot/ur_description/ur10e.urdf"
GRIPPERS_ROOT = grippers_dir()
# UR10e arm block (collision spheres + self-collision buffers for the 7 arm
# links) — committed alongside this script so the build is self-contained
# (no dependency on a previously-built gripper config). A locally-built
# curobo_assets/ur10e_inspire_hand.yml takes precedence if present.
_COMMITTED_ARM_TEMPLATE = _HERE / "ur10e_arm_template.yml"
_LEGACY_ARM_TEMPLATE = CUROBO_ASSETS / "ur10e_inspire_hand.yml"
ARM_TEMPLATE_YML = (
    _LEGACY_ARM_TEMPLATE if _LEGACY_ARM_TEMPLATE.is_file() else _COMMITTED_ARM_TEMPLATE
)

ARM_LINKS = [
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "tool0",
]
ARM_MESH_LINKS = ARM_LINKS[:-1]  # tool0 has no mesh
ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
ARM_DEFAULT_Q = [0.0, -2.2, 1.9, -1.383, -1.57, 0.0]

# First-guess mount (mirrors the inspire mount; refine per gripper in viser).
DEFAULT_MOUNT_RPY = (-math.pi / 2, 0.0, -math.pi / 2)
DEFAULT_MOUNT_XYZ = (0.0, 0.0, 0.0)


def _absolutize_mesh_paths(elem: ET.Element, urdf_dir: Path) -> None:
    for mesh in elem.iter("mesh"):
        fn = mesh.attrib.get("filename")
        if fn and not Path(fn).is_absolute():
            mesh.set("filename", str((urdf_dir / fn).resolve()))


def _snap_joint_axes(root: ET.Element, tol: float = 0.95) -> int:
    """Snap near-principal revolute/continuous joint axes to the exact
    principal axis (±X/±Y/±Z).

    cuRobo's kinematics loader only maps revolute joints whose axis is a
    principal axis to its X/Y/Z_ROT enum; a slightly-tilted axis (e.g.
    surge's Index_MCP at (0, 0.9956, 0.0933)) is left as the raw string
    'revolute' and crashes at ``joint_type.value``. Snapping the dominant
    component to ±1 (when |component| >= tol) fixes it. Joints here are
    locked during planning, so a few-degree axis change is negligible.
    """
    n = 0
    for j in root.findall("joint"):
        if j.get("type") not in ("revolute", "continuous"):
            continue
        ax = j.find("axis")
        if ax is None:
            continue
        v = [float(x) for x in ax.get("xyz", "0 0 0").split()]
        a = [abs(x) for x in v]
        k = a.index(max(a))
        if a[k] >= tol and (a[k] < 0.999999 or sum(1 for x in a if x > 1e-6) > 1):
            snapped = [0.0, 0.0, 0.0]
            snapped[k] = 1.0 if v[k] >= 0 else -1.0
            ax.set("xyz", " ".join(f"{x:.6f}" for x in snapped))
            n += 1
    return n


def _find_base_link(gripper_root: ET.Element) -> str:
    """The gripper base = child of the fixed joint whose parent is 'world'."""
    for j in gripper_root.findall("joint"):
        parent = j.find("parent")
        if parent is not None and parent.get("link") == "world":
            return j.find("child").get("link")
    raise RuntimeError("No joint with parent 'world' found in gripper URDF")


def merge_urdf(
    gripper_urdf: Path, out_urdf: Path, robot_name: str, mount_rpy, mount_xyz
) -> str:
    ur_tree = ET.parse(str(UR10E_URDF))
    ur_root = ur_tree.getroot()
    g_tree = ET.parse(str(gripper_urdf))
    g_root = g_tree.getroot()
    base_link = _find_base_link(g_root)

    _absolutize_mesh_paths(ur_root, UR10E_URDF.parent)
    _absolutize_mesh_paths(g_root, gripper_urdf.parent)
    ur_root.set("name", robot_name)

    bridge = ET.Element("joint", {"name": f"tool0_to_{base_link}", "type": "fixed"})
    ET.SubElement(bridge, "parent", {"link": "tool0"})
    ET.SubElement(bridge, "child", {"link": base_link})
    ET.SubElement(
        bridge,
        "origin",
        {
            "rpy": " ".join(f"{v:.9f}" for v in mount_rpy),
            "xyz": " ".join(f"{v:.9f}" for v in mount_xyz),
        },
    )
    ur_root.append(bridge)

    n_l = n_j = 0
    for child in list(g_root):
        nm = child.attrib.get("name")
        if child.tag == "link":
            if nm == "world":
                continue
            ur_root.append(child)
            n_l += 1
        elif child.tag == "joint":
            if nm == "world_joint" or (
                child.find("parent") is not None
                and child.find("parent").get("link") == "world"
            ):
                continue
            ur_root.append(child)
            n_j += 1
        else:
            ur_root.append(child)
    n_snap = _snap_joint_axes(ur_root)
    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(ur_tree, "  ")
    ur_tree.write(str(out_urdf), encoding="utf-8", xml_declaration=True)
    log.info(
        "Merged URDF: base=%s, +%d links +%d joints (%d axes snapped to "
        "principal) -> %s",
        base_link,
        n_l,
        n_j,
        n_snap,
        out_urdf.name,
    )
    return base_link


def fit_link_spheres(mesh, density=1.0):
    try:
        from curobo._src.geom.sphere_fit import SphereFitType, fit_spheres_to_mesh
    except ImportError:  # public NVlabs cuRobo moved these into submodules
        from curobo._src.geom.sphere_fit.types import SphereFitType
        from curobo._src.geom.sphere_fit.fit_spheres import fit_spheres_to_mesh
    # Omit the sphere-count kwarg (default None) — the lab fork names it
    # `n_spheres`, public cuRobo `num_spheres`; relying on the default keeps
    # this version-agnostic.
    r = fit_spheres_to_mesh(mesh, sphere_density=density, fit_type=SphereFitType.VOXEL)
    c = r.centers.detach().cpu().numpy()
    rad = r.radii.detach().cpu().numpy()
    return [
        {"center": [float(p[0]), float(p[1]), float(p[2])], "radius": float(x)}
        for p, x in zip(c, rad)
        if float(x) > 0
    ]


def _link_collision_mesh(link):
    import trimesh

    pieces = []
    for c in link.collisions:
        g = c.geometry
        origin = np.asarray(c.origin, float) if c.origin is not None else np.eye(4)
        piece = None
        if getattr(g, "box", None) is not None:
            piece = trimesh.creation.box(extents=list(g.box.size))
        elif getattr(g, "cylinder", None) is not None:
            piece = trimesh.creation.cylinder(
                radius=float(g.cylinder.radius),
                height=float(g.cylinder.length),
                sections=24,
            )
        elif getattr(g, "sphere", None) is not None:
            piece = trimesh.creation.icosphere(radius=float(g.sphere.radius))
        elif getattr(g, "mesh", None) is not None:
            try:
                piece = trimesh.load(g.mesh.filename, force="mesh")
                if not isinstance(piece, trimesh.Trimesh):
                    piece = trimesh.util.concatenate(
                        [
                            m
                            for m in piece.geometry.values()
                            if isinstance(m, trimesh.Trimesh)
                        ]
                    )
                if g.mesh.scale is not None:
                    piece.apply_scale(list(g.mesh.scale))
            except Exception as e:  # noqa: BLE001
                log.warning("    mesh load failed %s: %s", g.mesh.filename, e)
                continue
        if piece is None:
            continue
        if len(piece.vertices) == 0:
            raise RuntimeError(
                f"Collision mesh for link {link.name!r} has no vertices"
                + (f" ({g.mesh.filename})" if getattr(g, "mesh", None) else "")
                + ". If gripper_descriptions was cloned without its git-LFS "
                "objects the mesh is a pointer file; run `git lfs pull` in "
                "the gripper_descriptions checkout."
            )
        piece.apply_transform(origin)
        pieces.append(piece)
    import trimesh as _t

    return _t.util.concatenate(pieces) if pieces else None


def build(gripper: str, mount_rpy, mount_xyz, tag: str = ""):
    gdir = GRIPPERS_ROOT / gripper
    gripper_urdf = gdir / "gripper.urdf"
    config = yaml.safe_load((gdir / "config.json").read_text())
    masters = list(config["open"].keys())
    open_vals = config["open"]

    stem = f"ur10e_{gripper}{('_' + tag) if tag else ''}"
    out_urdf = CUROBO_ASSETS / f"{stem}.urdf"
    out_yml = CUROBO_ASSETS / f"{stem}.yml"
    base_link = merge_urdf(gripper_urdf, out_urdf, stem, mount_rpy, mount_xyz)

    # Fit spheres for every gripper link that has collision geometry.
    import yourdfpy

    u = yourdfpy.URDF.load(
        str(out_urdf), build_collision_scene_graph=True, load_meshes=True
    )
    gripper_links, gripper_spheres = [], {}
    for ln, link in u.link_map.items():
        if ln in ARM_LINKS or ln == "base_link" or ln.startswith("base"):
            continue
        if ln in ("shoulder_link",) or not getattr(link, "collisions", None):
            continue
        m = _link_collision_mesh(link)
        if m is None:
            continue
        sph = fit_link_spheres(m)
        if sph:
            gripper_links.append(ln)
            gripper_spheres[ln] = sph
    log.info(
        "Fitted spheres for %d gripper links (%d total)",
        len(gripper_links),
        sum(len(v) for v in gripper_spheres.values()),
    )

    # Arm block from the known-good template.
    tmpl = yaml.safe_load(ARM_TEMPLATE_YML.read_text())["robot_cfg"]["kinematics"]
    arm_spheres = {k: tmpl["collision_spheres"][k] for k in ARM_LINKS}
    # The template's wrist_3_link spheres (r=0.05, 0.035) sit at the gripper
    # mount and visually balloon over the hand base. Shrink to one small
    # sphere — the gripper base-link spheres already cover this region.
    arm_spheres["wrist_3_link"] = [{"center": [0.0, 0.0, 0.02], "radius": 0.02}]
    arm_buffer = {k: tmpl["self_collision_buffer"].get(k, 0) for k in ARM_LINKS}
    arm_ignore = {  # arm-arm adjacency ignores (from ur10e)
        "upper_arm_link": ["forearm_link", "shoulder_link"],
        "wrist_1_link": ["wrist_2_link", "wrist_3_link"],
        "wrist_2_link": ["wrist_3_link", "tool0"],
        "wrist_3_link": ["tool0", base_link],
    }

    # Auto self-collision-ignore: every gripper link ignores all other gripper
    # links + all arm links + tool0 (hand joints are locked during planning, so
    # intra-hand collisions are never a planning constraint).
    arm_set = ARM_LINKS
    sci = dict(arm_ignore)
    for gl in gripper_links:
        sci[gl] = sorted(set(gripper_links) - {gl} | set(arm_set))

    yml = {
        "robot_cfg": {
            "kinematics": {
                "urdf_path": str(out_urdf),
                "asset_root_path": str(CUROBO_ASSETS),
                "base_link": "base_link",
                "collision_sphere_buffer": 0.0,
                "use_global_cumul": True,
                # NOTE: no "use_usd_kinematics" key — public NVlabs cuRobo's
                # KinematicsLoaderCfg rejects it; the lab fork defaults to False anyway.
                "collision_link_names": ARM_LINKS + gripper_links,
                "collision_spheres": {**arm_spheres, **gripper_spheres},
                "cspace": {
                    "joint_names": ARM_JOINTS + masters,
                    "cspace_distance_weight": [1.0] * (len(ARM_JOINTS) + len(masters)),
                    "null_space_weight": [1.0] * (len(ARM_JOINTS) + len(masters)),
                    "max_acceleration": 12.0,
                    "max_jerk": 500.0,
                    "position_limit_clip": 0.1,
                    "default_joint_position": ARM_DEFAULT_Q
                    + [float(open_vals[m]) for m in masters],
                },
                "tool_frames": ["tool0"],
                "lock_joints": {m: float(open_vals[m]) for m in masters},
                "mesh_link_names": ARM_MESH_LINKS + [base_link],
                "self_collision_buffer": {
                    **arm_buffer,
                    **{gl: 0 for gl in gripper_links},
                },
                "self_collision_ignore": sci,
            }
        }
    }
    out_yml.write_text(yaml.safe_dump(yml, sort_keys=False, default_flow_style=None))
    log.info(
        "Wrote %s  (cspace: 6 arm + %d gripper masters, locked)",
        out_yml.name,
        len(masters),
    )
    return out_yml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gripper", required=True)
    ap.add_argument("--mount_rpy", type=float, nargs=3, default=list(DEFAULT_MOUNT_RPY))
    ap.add_argument("--mount_xyz", type=float, nargs=3, default=list(DEFAULT_MOUNT_XYZ))
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    build(args.gripper, tuple(args.mount_rpy), tuple(args.mount_xyz), tag=args.tag)


if __name__ == "__main__":
    main()
