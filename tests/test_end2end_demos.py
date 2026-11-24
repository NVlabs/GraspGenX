"""End-to-end pick-and-place demo tests.

Runs each of the four featured ``end2end/`` demos through the real pipeline
(GraspGenX → cuRobo → Newton) and checks the *outcome* from the exported
``trajectory.json`` — no rendering. The four demos:

  1. franka_single   Franka Panda picks one HOPE object → bin.
  2. franka_clutter3 Franka Panda clears a 3-object HOPE scene → bin.
  3. arx_lift        UR10e + arx_x5 grasps + lifts one object (no bin).
  4. robotiq_lift    UR10e + Robotiq 2F-140 grasps + lifts one object (no bin).

These are slow GPU integration tests (model load + sim per demo). Run with::

    uv run pytest tests/test_end2end_demos.py -m end2end -v -s

The UR10e demos need the merged robot URDFs, generated once into the
gitignored ``end2end/curobo_assets/`` by ``setup_end2end_deps.py``; if they're
missing the UR10e cases skip with a hint instead of failing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
E2E = REPO_ROOT / "end2end"
ROBOTS = E2E / "robots"
ENVS = E2E / "envs"
HOPE = REPO_ROOT / "assets/sample_data/hope_objects"

pytestmark = [pytest.mark.integration, pytest.mark.end2end]


# --- demo specs -----------------------------------------------------------
# Each spec: how to invoke the demo + how to judge the trajectory outcome.
DEMOS = {
    "franka_single": dict(
        robot="franka_panda.yaml",
        env="single_bin_demo.yaml",
        task="clutter_pick_and_drop",
        mesh=None,
        outcome="drop", min_in_bin=1, n_objects=1,
    ),
    "franka_clutter3": dict(
        robot="franka_panda.yaml",
        env="franka_clutter3_demo.yaml",
        task="clutter_pick_and_drop",
        mesh=None,
        outcome="drop", min_in_bin=2, n_objects=3,
    ),
    "arx_lift": dict(
        robot="ur10e_arx_x5.yaml",
        env="tabletop_single_nobin.yaml",
        task="pick_and_lift",
        mesh=str(HOPE / "GranolaBars.obj"),
        extra_args=["--hold_after_close_frames", "150"],
        outcome="lift", min_in_bin=0, n_objects=1,
    ),
    "robotiq_lift": dict(
        robot="ur10e_robotiq_2f_140.yaml",
        env="tabletop_single_nobin.yaml",
        task="pick_and_lift",
        mesh=str(HOPE / "GranolaBars.obj"),
        extra_args=["--hold_after_close_frames", "150"],
        outcome="lift", min_in_bin=0, n_objects=1,
    ),
}

# Bin footprint half-extent (+ margin) used to decide "landed in the bin".
_BIN_HALF = 0.22
# Minimum rise (m) of an object's peak height over its start to count as lifted.
_LIFT_RISE = 0.05


def _curobo_available() -> bool:
    try:
        import curobo  # noqa: F401
        import warp as wp

        # Same workaround as end2end/dynamic_playback.py: newton reads
        # wp.config.verbose, which newer warp-lang no longer defines; without
        # it this check failed and every case here was silently skipped.
        if not hasattr(wp.config, "verbose"):
            wp.config.verbose = False
        import newton  # noqa: F401
        return True
    except Exception:
        return False


def _object_outcomes(traj_path: Path) -> dict:
    """Per-object {lifted, in_bin, start_z, peak_z} from a trajectory JSON."""
    d = json.loads(Path(traj_path).read_text())
    frames = d["frames"]
    bin_xy = None
    binmeta = (d.get("static") or {}).get("bin")
    if binmeta is not None:
        T = np.asarray(binmeta["transform"], dtype=float)
        bin_xy = (T[0, 3], T[1, 3])

    out: dict = {}
    for obj in (d.get("objects") or []):
        oid = obj["id"]
        zs, start_z, final_P = [], None, None
        for f in frames:
            op = (f.get("object_poses") or {}).get(oid)
            if op is None:
                continue
            P = np.asarray(op, dtype=float)
            if start_z is None:
                start_z = float(P[2, 3])
            zs.append(float(P[2, 3]))
            final_P = P
        if not zs:
            out[oid] = dict(seen=False)
            continue
        in_bin = False
        if bin_xy is not None and final_P is not None:
            dx = abs(final_P[0, 3] - bin_xy[0])
            dy = abs(final_P[1, 3] - bin_xy[1])
            in_bin = dx < _BIN_HALF and dy < _BIN_HALF
        out[oid] = dict(
            seen=True,
            lifted=(max(zs) - start_z) > _LIFT_RISE,
            in_bin=in_bin,
            start_z=start_z,
            peak_z=max(zs),
        )
    return out


def _run_demo(spec: dict, traj_path: Path) -> None:
    cmd = [
        sys.executable, "end2end/e2e_grasp_demo.py",
        "--robot_config", str(ROBOTS / spec["robot"]),
        "--env_config", str(ENVS / spec["env"]),
        "--task", spec["task"],
        "--playback_mode", "dynamic", "--no-viser",
        "--num_grasps", "200", "--topk", "80",
        "--grasp_threshold", "0.7", "--planner", "graspmoe",
        "--seed", "0",
        "--export-trajectory", str(traj_path),
    ]
    if spec["mesh"] is not None:
        cmd += ["--mesh_file", spec["mesh"]]
    cmd += spec.get("extra_args", [])
    env = dict(os.environ, PYOPENGL_PLATFORM="egl", PYGLET_HEADLESS="true")
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=1800,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.splitlines()[-40:])
        raise AssertionError(
            f"demo exited {proc.returncode}\n--- stderr tail ---\n{tail}"
        )
    assert traj_path.is_file(), "demo produced no trajectory.json"


@pytest.mark.skipif(not _curobo_available(),
                    reason="cuRobo/Newton not installed (need `.[end2end]`)")
@pytest.mark.parametrize("name", list(DEMOS))
def test_end2end_demo(name, tmp_path):
    spec = DEMOS[name]

    # UR10e demos need the generated merged URDF + cuRobo YAML (gitignored).
    # Skip with a hint on a fresh checkout where they haven't been built yet.
    if spec["robot"].startswith("ur10e"):
        stem = spec["robot"].replace(".yaml", "")
        merged = [E2E / "curobo_assets" / f"{stem}{ext}" for ext in (".urdf", ".yml")]
        if not all(p.is_file() for p in merged):
            pytest.skip(
                f"{stem} robot assets missing — run "
                "`python end2end/setup_end2end_deps.py` first."
            )

    traj = tmp_path / "trajectory.json"
    _run_demo(spec, traj)
    outcomes = _object_outcomes(traj)

    seen = [o for o in outcomes.values() if o.get("seen")]
    assert len(seen) == spec["n_objects"], (
        f"expected {spec['n_objects']} objects in trajectory, "
        f"got {len(seen)}: {outcomes}"
    )

    if spec["outcome"] == "lift":
        assert any(o["lifted"] for o in seen), (
            f"no object was lifted off the table: {outcomes}"
        )
    else:  # drop
        n_in_bin = sum(1 for o in seen if o["in_bin"])
        assert n_in_bin >= spec["min_in_bin"], (
            f"{n_in_bin} object(s) in bin, expected >= {spec['min_in_bin']}: "
            f"{outcomes}"
        )
