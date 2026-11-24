import numpy as np
import pytest
import torch

from graspgenx.dataset.dataset import ObjectPickDataset


class _Grasps:
    def __init__(self, x):
        self.positive_grasps = np.stack([np.eye(4) for _ in x])
        self.positive_grasps[:, 0, 3] = x


def _dataset(grasps_per_scene):
    # Build the object without __init__, as if it was preloaded.
    ds = ObjectPickDataset.__new__(ObjectPickDataset)
    ds.scenes = [f"obj{i}" for i in range(len(grasps_per_scene))]
    ds.num_points = 4
    ds.cache = {}
    for key, x in zip(ds.scenes, grasps_per_scene):
        outputs = {
            "mesh_mode": None,
            "load_contact_batch": False,
            "points": torch.zeros(16, 3),
            "T_move_to_pc_mean": np.eye(4),
        }
        ds.cache[key] = (_Grasps(x), [outputs])
    return ds


def test_calculate_dataset_kappa():
    # Extents per object are (0.09, 0, 0) and (0.03, 0, 0), so the mean is 0.02.
    ds = _dataset([np.arange(10) * 0.01, np.arange(4) * 0.01])
    assert ds.calculate_dataset_kappa() == pytest.approx(50.0)


def test_calculate_dataset_kappa_keeps_cache():
    ds = _dataset([np.arange(10) * 0.01])
    ds.calculate_dataset_kappa()
    assert ds.cache["obj0"][1][0]["points"].shape == (16, 3)
