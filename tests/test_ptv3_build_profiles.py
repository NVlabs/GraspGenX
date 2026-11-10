# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys

import pytest

pytest.importorskip("tensorrt")

from cpp_runtime.tools import build_engines
from graspgenx.fast_inference._profiles import PROFILE_FILENAME, read_stage_capacities


@pytest.mark.parametrize("failure", [None, "build", "branch_shape"])
def test_plan_build_publishes_only_a_completed_consistent_profile(
    tmp_path, monkeypatch, failure
):
    onnx = tmp_path / "onnx"
    plans = tmp_path / "engines"
    (onnx / "ptv3").mkdir(parents=True)
    plans.mkdir()
    capacities = (3500, 2048, 1024, 512, 512)
    for branch in ("generator", "discriminator"):
        (onnx / "ptv3" / f"ptv3_{branch}.onnx").touch()
    for name in ("denoiser_core_d100.onnx", "scorer_core_c150.onnx"):
        (onnx / name).touch()
    (plans / PROFILE_FILENAME).write_text(json.dumps({"stage_capacities": capacities}))

    def build(path, *args):
        if "ptv3_" not in path.name:
            return {"input_shapes": {}}
        if failure == "build" and "discriminator" in path.name:
            raise RuntimeError("interrupted build")
        shapes = {
            f"stage{i}_neighbor_indices_k3": [capacity, 27]
            for i, capacity in enumerate(capacities)
        }
        if failure == "branch_shape" and "discriminator" in path.name:
            shapes["stage2_neighbor_indices_k3"][0] = 512
        return {"input_shapes": shapes}

    monkeypatch.setattr(build_engines, "build_engine", build)
    monkeypatch.setattr(build_engines.ctypes, "CDLL", lambda *a, **kw: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_engines",
            "--onnx-dir",
            str(onnx),
            "--output-dir",
            str(plans),
            "--plugin-library",
            str(tmp_path / "plugin.so"),
        ],
    )
    if failure is None:
        build_engines.main()
        assert read_stage_capacities(plans) == capacities
    else:
        with pytest.raises((RuntimeError, ValueError)):
            build_engines.main()
        with pytest.raises(ValueError, match="incomplete"):
            read_stage_capacities(plans)
