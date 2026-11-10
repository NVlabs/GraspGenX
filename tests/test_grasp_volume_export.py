"""Capture-volume geometry used by the offline native bundle exporter."""
import numpy as np
import pytest

from cpp_runtime.tools.export_gripper_embeddings import grasp_volume_from_config, validate_grasp_volume


def test_capture_volume_includes_standoff_and_retains_gripper_frame():
    config = dict(sweep_volume=dict(extents=[.08, .02, .04], offset=[.01, -.03, .15]),
                  standoff=[-.005, .025])
    bounds = grasp_volume_from_config(config)
    np.testing.assert_allclose(bounds, [[-.03, -.04, .125], [.05, -.02, .195]], atol=1e-8)
    assert bounds.dtype == np.float32 and bounds.shape == (2, 3)


@pytest.mark.parametrize("bounds", [np.zeros((2,3)), [[1,0,0],[0,1,1]],
    [[0,0,0],[1,np.nan,1]], [[0,0,-np.inf],[1,1,1]]])
def test_invalid_capture_bounds_are_rejected(bounds):
    with pytest.raises(ValueError, match="finite minimum"):
        validate_grasp_volume(bounds)


@pytest.mark.parametrize("extents", [[0,1,1],[-1,1,1],[np.nan,1,1]])
def test_invalid_config_extents_are_rejected(extents):
    with pytest.raises(ValueError, match="extents"):
        grasp_volume_from_config(dict(sweep_volume=dict(extents=extents, offset=[0,0,0]), standoff=[0,0]))
