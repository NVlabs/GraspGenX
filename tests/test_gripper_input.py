"""Tests for gripper point-cloud conditioning inputs.

``sample_gripper_pointclouds`` feeds both the inference sampler
(``GraspGenXSampler.load_gripper_input``) and the training dataset.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from graspgenx.x_grippers import sample_gripper_pointclouds

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _gripper(open_ptc, close_ptc):
    return SimpleNamespace(
        depth=0.1,
        sweep_volume=np.zeros(6, dtype=np.float32),
        sweep_volume_mid=np.zeros(6, dtype=np.float32),
        gripper_type="parallel_2f",
        open_pointcloud=open_ptc,
        close_pointcloud=close_ptc,
        vol_tsdf={
            "open_tsdf": np.zeros((2, 2, 2), dtype=np.float32),
            "close_tsdf": np.zeros((2, 2, 2), dtype=np.float32),
        },
        pointnet_vae={
            "open": np.zeros(4, dtype=np.float32),
            "half": np.zeros(4, dtype=np.float32),
            "close": np.zeros(4, dtype=np.float32),
        },
    )


def test_open_and_close_points_come_from_their_own_states():
    gripper = _gripper(
        np.ones((200, 3), dtype=np.float64), -np.ones((50, 3), dtype=np.float64)
    )

    open_pts, close_pts = sample_gripper_pointclouds(gripper, 64)

    assert open_pts.shape == close_pts.shape == (64, 3)
    assert open_pts.dtype == close_pts.dtype == np.float32
    assert np.all(open_pts == 1.0)
    assert np.all(close_pts == -1.0)


def test_sampling_uses_each_cloud_size():
    # The open cloud is larger than the close cloud, so indices drawn from the
    # open cloud must not be used to index the close cloud.
    open_ptc = np.arange(300 * 3, dtype=np.float32).reshape(300, 3)
    close_ptc = np.arange(10 * 3, dtype=np.float32).reshape(10, 3)

    open_pts, close_pts = sample_gripper_pointclouds(_gripper(open_ptc, close_ptc), 500)

    assert {tuple(r) for r in open_pts} <= {tuple(r) for r in open_ptc}
    assert {tuple(r) for r in close_pts} <= {tuple(r) for r in close_ptc}
    # 500 draws from 300 rows cover far more than the 10 close rows.
    assert len({tuple(r) for r in open_pts}) > 10


@requires_cuda
def test_sampler_load_gripper_input_uses_open_and_close_clouds():
    from graspgenx.grasp_server import GraspGenXSampler

    # Bypass __init__: load_gripper_input only needs the gripper info and cfg.
    sampler = GraspGenXSampler.__new__(GraspGenXSampler)
    sampler.cfg = SimpleNamespace(data=SimpleNamespace(num_points=64))
    sampler.gripper = _gripper(
        np.ones((200, 3), dtype=np.float32), -np.ones((50, 3), dtype=np.float32)
    )

    outputs = sampler.load_gripper_input({})

    assert outputs["gripper_open_ptc"].shape == (64, 3)
    assert outputs["gripper_open_ptc"].is_cuda
    assert torch.all(outputs["gripper_open_ptc"] == 1.0)
    assert torch.all(outputs["gripper_close_ptc"] == -1.0)
