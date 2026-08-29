import numpy as np
import pytest
import torch

from graspgenx._constants import MIN_OBJECT_POINT_COUNT
from graspgenx.utils.point_cloud import (
    filter_and_resample_point_cloud,
    knn_points,
    point_cloud_outlier_removal,
)


def test_knn_points_basic():
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )

    K = 2
    dists, idxs = knn_points(points, K=K, norm=2)

    assert dists.shape == (5, K)
    assert idxs.shape == (5, K)

    # Distances should be sorted ascending
    assert torch.all(dists[:, 0] <= dists[:, 1])

    # Self-indices should not appear
    for i in range(5):
        assert i not in idxs[i]


def test_knn_points_identical_points():
    points = torch.ones((5, 3))
    K = 2

    dists, idxs = knn_points(points, K=K, norm=2)

    # All distances should be 0 since points are identical
    assert torch.allclose(dists, torch.zeros(5, K))

    # Indices should be different from self
    for i in range(5):
        assert i not in idxs[i]


def test_knn_points_single_neighbor():
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    K = 1
    dists, idxs = knn_points(points, K=K, norm=2)

    assert dists.shape == (3, 1)
    assert idxs.shape == (3, 1)

    # Nearest neighbor of point 0 (origin) should be point 1 (at distance 1)
    assert idxs[0, 0] == 1
    assert torch.isclose(dists[0, 0], torch.tensor(1.0))


def test_knn_points_l1_norm():
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    K = 1
    dists, idxs = knn_points(points, K=K, norm=1)

    assert dists.shape == (3, 1)
    # L1 distance from origin to (1,1,0) is 2, to (2,0,0) is 2 — both equal
    # L1 distance from (1,1,0) to origin is 2, to (2,0,0) is 2 — both equal
    assert dists[0, 0] >= 0


def test_knn_points_large_K():
    N = 10
    points = torch.randn(N, 3)
    K = N - 1  # all other points
    dists, idxs = knn_points(points, K=K, norm=2)

    assert dists.shape == (N, K)
    assert idxs.shape == (N, K)

    # Each row should contain all indices except self
    for i in range(N):
        assert i not in idxs[i]
        assert len(set(idxs[i].tolist())) == K


def test_point_cloud_outlier_removal():
    # Dense cluster near origin + outliers far away
    cluster = torch.randn(100, 3) * 0.01
    outliers = torch.tensor(
        [
            [10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0],
        ]
    )
    points = torch.cat([cluster, outliers], dim=0)

    filtered_pc, removed_pc = point_cloud_outlier_removal(points, threshold=0.5)

    assert filtered_pc.shape[0] < points.shape[0]
    assert removed_pc.shape[0] > 0
    assert filtered_pc.shape[0] + removed_pc.shape[0] == points.shape[0]


def test_point_cloud_outlier_removal_no_outliers():
    points = torch.randn(100, 3) * 0.001  # very tight cluster

    filtered_pc, removed_pc = point_cloud_outlier_removal(points, threshold=0.5)

    # Most (or all) points should be kept
    assert filtered_pc.shape[0] >= points.shape[0] * 0.9


def test_point_cloud_outlier_removal_identical_points():
    identical_point = torch.tensor([1.0, 2.0, 3.0])
    points = identical_point.repeat(50, 1)

    filtered_pc, removed_pc = point_cloud_outlier_removal(points, threshold=0.5)

    # All points should be kept since they are identical (zero distance)
    assert filtered_pc.shape[0] == points.shape[0]
    assert removed_pc.shape[0] == 0
    assert torch.allclose(filtered_pc, points)


def test_point_cloud_outlier_removal_returns_correct_shapes():
    points = torch.randn(200, 3)
    filtered_pc, removed_pc = point_cloud_outlier_removal(points)

    assert filtered_pc.ndim == 2
    assert filtered_pc.shape[1] == 3
    assert removed_pc.ndim == 2
    assert removed_pc.shape[1] == 3


def test_filter_and_resample_point_cloud_returns_fixed_count():
    points = torch.zeros(MIN_OBJECT_POINT_COUNT, 3)

    output = filter_and_resample_point_cloud(points, num_points=128)

    assert output.shape == (128, 3)
    assert torch.equal(output, torch.zeros(128, 3))


def test_filter_and_resample_point_cloud_enforces_minimum_count():
    with pytest.raises(ValueError, match=f"at least {MIN_OBJECT_POINT_COUNT}"):
        filter_and_resample_point_cloud(
            torch.zeros(MIN_OBJECT_POINT_COUNT - 1, 3),
            num_points=128,
        )
