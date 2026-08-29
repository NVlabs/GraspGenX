# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable discriminator label identifiers."""

MAPPING_ID2NAME = {
    0: "pos_true",
    1: "neg_true",
    2: "neg_hncolliding",
    3: "neg_freespace",
    4: "neg_hnretract",
    5: "pos_true_onpolicy",
    6: "neg_true_onpolicy",
}
MAPPING_NAME2ID = {name: label_id for label_id, name in MAPPING_ID2NAME.items()}

__all__ = ["MAPPING_ID2NAME", "MAPPING_NAME2ID"]
