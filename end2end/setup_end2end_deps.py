#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bootstrap the cuRobo robot ASSETS for the end2end examples.

cuRobo's Python package is installed via uv (a pinned git source — see
``[tool.uv.sources]`` in ``pyproject.toml``), but its *built wheel* ships only
a minimal subset of ``content/assets`` — no meshes and no ``ur_description``.
The end2end demos need the full robot URDFs + meshes (franka visual/collision
meshes, the UR10e arm for the merged grippers), so this script clones the
cuRobo *source* — at the same pinned, validated commit, with its git-LFS mesh
assets pulled — into ``ext/curobo``.

``end2end/paths.py`` then resolves ``${CUROBO_ASSETS}`` to this full checkout
(falling back to the installed package only if absent). This mirrors how
``gripper_descriptions`` / ``graspgenx_checkpoints`` are cloned into ``ext/``.

It also re-installs cuRobo editable from that checkout, and builds the merged
UR10e + arx_x5 and UR10e + Robotiq 2F-140 URDFs (the gitignored
``end2end/curobo_assets/`` artifacts that the UR10e demos need).

Idempotent. Run once after ``uv sync --extra end2end``:

    python end2end/setup_end2end_deps.py

Only used by the end2end examples; the base install is unaffected.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = REPO_ROOT / "ext"

# cuRobo source repo + the validated revision the end2end demos were built on:
# public NVlabs cuRobo, pinned by the v0.8.0 commit SHA (must match the
# nvidia-curobo rev in pyproject.toml). Fetched directly by SHA from GitHub.
# Overridable via env for forks / offline mirrors.
CUROBO_URL = os.environ.get(
    "GRASPGENX_CUROBO_URL",
    "https://github.com/NVlabs/curobo.git",
)
CUROBO_REF = os.environ.get(
    "GRASPGENX_CUROBO_REF", "057a96ffb1088531535f9915154f9d0dabd62428"
)
CUROBO_DIR = EXT_DIR / "curobo"

# Sentinel that proves the LFS meshes + ur_description are actually present
# (not just URDF text or LFS pointers).
_ASSET_SENTINEL = (
    "curobo/content/assets/robot/ur_description/meshes/ur10e/collision/base.stl"
)


def _assets_ok(d: Path) -> bool:
    f = d / _ASSET_SENTINEL
    # A pulled LFS .stl is a few KB+ of binary; an un-pulled pointer is ~130 B.
    return f.is_file() and f.stat().st_size > 2048


def _ensure_curobo_assets() -> None:
    """Clone the cuRobo source (with LFS meshes) into ext/curobo if missing."""
    if _assets_ok(CUROBO_DIR):
        print(f"[setup-e2e] cuRobo assets already present at {CUROBO_DIR}")
        return
    if shutil.which("git") is None:
        sys.exit("[setup-e2e] git not found; install git (+ git-lfs) first.")
    EXT_DIR.mkdir(parents=True, exist_ok=True)

    if not (CUROBO_DIR / ".git").exists():
        print(f"[setup-e2e] fetching cuRobo @ {CUROBO_REF} -> {CUROBO_DIR}")
        # Fetch the exact commit by SHA (works for a SHA, tag, or branch) — a
        # plain `clone --branch` can't target an arbitrary commit or a
        # remote-absent tag.
        subprocess.run(["git", "init", "-q", str(CUROBO_DIR)], check=True)
        subprocess.run(
            ["git", "-C", str(CUROBO_DIR), "remote", "add", "origin", CUROBO_URL],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(CUROBO_DIR), "fetch", "--depth", "1",
             "origin", CUROBO_REF],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(CUROBO_DIR), "checkout", "-q", "FETCH_HEAD"], check=True
        )

    print("[setup-e2e] git lfs pull (mesh assets — may take a minute)")
    subprocess.run(["git", "-C", str(CUROBO_DIR), "lfs", "pull"], check=True)

    if not _assets_ok(CUROBO_DIR):
        sys.exit(
            "[setup-e2e] cuRobo mesh assets still missing after clone + lfs "
            "pull — check git-lfs is installed and you have repo access."
        )
    print(f"[setup-e2e] cuRobo assets ready at {CUROBO_DIR}")


# Validated tool0 mounts (rpy) for the merged UR10e + <gripper> robots. Each
# must match the grasp_to_tool_transform in end2end/robots/ur10e_<gripper>.yaml.
UR10E_GRIPPER_MOUNTS = {
    "arx_x5": ["-1.5708", "-1.5708", "0"],
    "robotiq_2f_140": ["0", "0", "0"],
}


def _build_ur10e_gripper(gripper: str, mount_rpy: list[str]) -> None:
    """Generate the merged UR10e + <gripper> URDF/YAML for the UR10e demos.

    They're gitignored build artifacts under end2end/curobo_assets/, so build
    them once here. Non-fatal: the Franka demos don't need them.
    """
    assets = REPO_ROOT / "end2end" / "curobo_assets"
    outputs = [assets / f"ur10e_{gripper}.urdf", assets / f"ur10e_{gripper}.yml"]
    # Check both files: the URDF is written before sphere fitting, so a failed
    # build can leave a URDF without its cuRobo YAML.
    if all(p.is_file() for p in outputs):
        print(f"[setup-e2e] UR10e+{gripper} already built ({outputs[0].name})")
        return
    print(f"[setup-e2e] building merged UR10e + {gripper} URDF")
    cmd = [
        "uv", "run", "--no-sync", "python",
        str(REPO_ROOT / "end2end" / "build_ur10e_gripper.py"),
        "--gripper", gripper, "--mount_rpy", *mount_rpy,
    ]
    # Run in the project venv (which now has the editable cuRobo) via uv.
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if res.returncode == 0 and all(p.is_file() for p in outputs):
        print(f"[setup-e2e] UR10e+{gripper} ready ({outputs[0]})")
    else:
        print(
            f"[setup-e2e] WARNING: UR10e+{gripper} build failed — the Franka "
            f"demos still work. Rebuild later with: {' '.join(cmd)}"
        )


def main() -> None:
    _ensure_curobo_assets()

    # cuRobo's built wheel ships an incomplete content/ tree (no robot configs
    # like franka.yml, no meshes), and cuRobo resolves those from its OWN
    # installed package — so re-install it editable from this full source
    # checkout. --no-deps: its dependencies are already satisfied by the
    # `uv sync --extra end2end` install. NOTE: a later `uv sync` will revert
    # this to the (incomplete) wheel — run end2end commands with
    # `uv run --no-sync ...` (or this script again) so the editable install
    # sticks.
    print("[setup-e2e] re-installing cuRobo editable from source (--no-deps)")
    subprocess.run(
        ["uv", "pip", "install", "-e", str(CUROBO_DIR), "--no-deps"], check=True
    )

    # Build the merged UR10e + <gripper> robots the UR10e demos need.
    for gripper, mount_rpy in UR10E_GRIPPER_MOUNTS.items():
        _build_ur10e_gripper(gripper, mount_rpy)

    print("[setup-e2e] done. Run end2end commands with `uv run --no-sync ...`.")


if __name__ == "__main__":
    main()
