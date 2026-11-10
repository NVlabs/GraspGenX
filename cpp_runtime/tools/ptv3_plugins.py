# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Symbolic V3 reductions used only during complete-encoder ONNX export."""

import torch


def segment_max(features, order, index_ptr, valid_count, output_capacity):
    return torch.onnx.ops.symbolic(
        "trt.plugins::GraspGenXSegmentMax",
        (features, order, index_ptr, valid_count),
        {"plugin_version": "1", "plugin_namespace": ""},
        dtype=features.dtype,
        shape=(output_capacity, features.shape[1]),
        version=1,
    )


def valid_mean(features, valid_count):
    return torch.onnx.ops.symbolic(
        "trt.plugins::GraspGenXValidMean",
        (features, valid_count),
        {"plugin_version": "1", "plugin_namespace": ""},
        dtype=features.dtype,
        shape=(features.shape[1],),
        version=1,
    )
