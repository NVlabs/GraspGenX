# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The discriminator's object embedding is grasp-independent, so injecting a
precomputed embedding must reproduce the from-scratch scores exactly. This is
what lets GraspMoE encode each object once and reuse it for OBB-candidate
scoring instead of re-encoding per object.
"""

import pytest
import torch

from graspgenx.batching import collate
from graspgenx.models.discriminator import GraspGenDiscriminator

import sys

sys.path.insert(0, "tests")
from test_inference_installation import _make_discriminator_cfg  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _disc_batch(n_points=2000, n_grasps=64):
    torch.manual_seed(0)
    pc = torch.randn(n_points, 3) * 0.05
    pc -= pc.mean(0)
    grasps = torch.eye(4).repeat(n_grasps, 1, 1)
    grasps[:, :3, 3] = torch.randn(n_grasps, 3) * 0.05
    data = {
        "task": "pick",
        "points": pc.to(DEVICE),
        "grasps": grasps.to(DEVICE),
        "z_offset": torch.tensor([0.1], dtype=torch.float32).to(DEVICE),
    }
    batch = collate([data])
    batch["grasp_key"] = "grasps"
    return batch


def test_injected_object_embedding_reproduces_scores():
    """Forward with an injected embedding == forward that encodes from scratch."""
    disc = (
        GraspGenDiscriminator.from_config(_make_discriminator_cfg("ptv3vanilla"))
        .to(DEVICE)
        .eval()
    )

    batch = _disc_batch()
    with torch.inference_mode():
        out1, _, _ = disc.infer(batch)
        assert "object_embedding" in out1, "discriminator must expose object_embedding"
        emb = out1["object_embedding"]  # [num_objects, num_object_dim]

        batch2 = _disc_batch()
        batch2["object_embedding"] = emb  # inject -> should skip the encoder
        out2, _, _ = disc.infer(batch2)

    # Bit-identical: same encoder output, same downstream MLP.
    assert torch.allclose(out1["logits"], out2["logits"], atol=1e-5)
    assert torch.allclose(
        out1["grasp_confidence"], out2["grasp_confidence"], atol=1e-5
    )
