#!/usr/bin/env python3

# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Test inference installation by checking if GraspGenX can run end-to-end inference
with random weights (no checkpoints needed). Validates that:

1. All dependencies are correctly installed (torch, spconv, etc.)
2. Models can be initialized with random weights
3. Inference pipeline runs correctly end-to-end with gripper conditioning
4. Expected number of grasps (100) are generated

Unlike GraspGen (per-gripper models), GraspGenX conditions the model on a
gripper representation. This test uses the simplest conditioning mode
(z_offset) to validate the cross-embodiment architecture.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

from graspgenx.models.grasp_gen import GraspGen
from graspgenx.batching import collate


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def random_point_cloud():
    """Generate a random point cloud (2000 points centered at origin)."""
    torch.manual_seed(42)
    points = torch.randn(2000, 3, dtype=torch.float32) * 0.05
    points -= points.mean(dim=0)
    return points


def _make_generator_cfg(backbone: str) -> DictConfig:
    """Build a minimal GraspGenX generator config for the given backbone."""
    return DictConfig({
        "num_embed_dim": 256,
        "num_object_dim": 512,
        "num_gripper_dim": 512,
        "diffusion_embed_dim": 512,
        "image_size": 256,
        "num_diffusion_iters": 10,
        "num_diffusion_iters_eval": 10,
        "object_backbone": backbone,
        "gripper_backbone": "z_offset",
        "compositional_schedular": False,
        "loss_pointmatching": True,
        "loss_l1_pos": False,
        "loss_l1_rot": False,
        "grasp_repr": "r3_6d",
        "kappa": -1.0,
        "clip_sample": True,
        "beta_schedule": "squaredcos_cap_v2",
        "attention": "cat",
        "pose_repr": "mlp",
        "num_grasps_per_object": 100,
        "checkpoint_object_encoder_pretrained": None,
        "ptv3": DictConfig({"grid_size": 0.02}),
    })


def _make_discriminator_cfg(backbone: str) -> DictConfig:
    """Build a minimal GraspGenX discriminator config for the given backbone."""
    return DictConfig({
        "num_object_dim": 512,
        "num_gripper_dim": 512,
        "num_embed_dim": 512,
        "object_backbone": backbone,
        "gripper_backbone": "z_offset",
        "grasp_repr": "r3_6d",
        "topk_ratio": 0.40,
        "checkpoint_object_encoder_pretrained": None,
        "kappa": 3.30,
        "pose_repr": "mlp",
        "ptv3": DictConfig({"grid_size": 0.01}),
    })


def _prepare_batch(point_cloud: torch.Tensor, device: torch.device) -> dict:
    """Prepare a collated batch dict with gripper conditioning (z_offset)."""
    pc = point_cloud.to(device)
    pc_center = pc.mean(dim=0)
    pc_centered = pc - pc_center[None]
    pc_color = torch.zeros_like(pc)

    data = {
        "task": "pick",
        "inputs": torch.cat([pc_centered, pc_color[:, :3]], dim=-1).float(),
        "points": pc_centered,
        "z_offset": torch.tensor([0.1], dtype=torch.float32).to(device),
    }
    return collate([data])


def _run_inference(model, data_batch, num_grasps: int = 100):
    """Run model inference and return the predicted grasps tensor."""
    model.grasp_generator.num_grasps_per_object = num_grasps
    with torch.inference_mode():
        outputs, _, _ = model.infer(data_batch)

    assert "grasps_pred" in outputs, "Missing 'grasps_pred' key in model outputs"
    grasps = outputs["grasps_pred"][0]  # first (only) batch element
    return grasps


# ─── Tests ──────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backbone", ["ptv3vanilla"])
def test_inference_100_grasps(backbone, random_point_cloud):
    """
    End-to-end inference test: create a model with random weights, feed in
    a random point cloud with z_offset gripper conditioning, and verify we
    get back exactly 100 grasps that are valid 4x4 homogeneous matrices.
    """
    device = torch.device("cuda")

    gen_cfg = _make_generator_cfg(backbone)
    disc_cfg = _make_discriminator_cfg(backbone)
    model = GraspGen.from_config(gen_cfg, disc_cfg).to(device).eval()

    data_batch = _prepare_batch(random_point_cloud, device)
    grasps = _run_inference(model, data_batch, num_grasps=100)

    assert len(grasps) == 100, (
        f"[{backbone}] Expected 100 grasps, got {len(grasps)}"
    )

    assert grasps.shape == torch.Size([100, 4, 4]), (
        f"[{backbone}] Expected shape [100, 4, 4], got {list(grasps.shape)}"
    )

    bottom_rows = grasps[:, 3, :]
    expected_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    assert torch.allclose(bottom_rows, expected_bottom.expand_as(bottom_rows), atol=1e-5), (
        f"[{backbone}] Grasp matrices have invalid bottom row"
    )

    print(f"[{backbone}] Generated {len(grasps)} valid 4x4 grasps")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backbone", ["ptv3vanilla"])
def test_model_components(backbone):
    """Verify generator and discriminator are properly initialised."""
    gen_cfg = _make_generator_cfg(backbone)
    disc_cfg = _make_discriminator_cfg(backbone)
    model = GraspGen.from_config(gen_cfg, disc_cfg)

    assert hasattr(model, "grasp_generator"), "Missing grasp_generator"
    assert hasattr(model, "grasp_discriminator"), "Missing grasp_discriminator"
    assert model.grasp_generator.object_backbone == backbone
    assert model.grasp_generator.gripper_backbone == "z_offset"
    assert model.grasp_generator.num_grasps_per_object == 100

    print(f"[{backbone}] Model components OK")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
