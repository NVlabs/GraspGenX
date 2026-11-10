#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import os

import torch
import torch.nn as nn
from omegaconf import DictConfig

from graspgenx.models.discriminator import GraspGenDiscriminator
from graspgenx.models.generator import GraspGenGenerator
from graspgenx.utils.logging_config import get_logger

logger = get_logger(__name__)


def find_the_last_ckpt(ckpt_dir):
    ckpt_dir = "/".join(ckpt_dir.split("/")[:-1])

    ckpt_list = [
        ckpt for ckpt in glob.glob(ckpt_dir + "/*.pth") if ckpt.find("last") < 0
    ]
    highest_ckpt_idx = sorted(
        [
            int(os.path.basename(ckpt_file).split("epoch_")[1].split(".pth")[0])
            for ckpt_file in ckpt_list
        ]
    )[-1]
    ckpt_file = os.path.join(ckpt_dir, f"epoch_{str(highest_ckpt_idx)}.pth")

    ckpt = torch.load(ckpt_file, map_location="cpu")
    return ckpt


class GraspGen(nn.Module):
    """Combined model that uses both diffusion-based generation and discriminative evaluation.

    This class combines a GraspGenGenerator generator with a GraspGenDiscriminator to both
    generate and evaluate grasps in a single pipeline.

    Args:
        grasp_generator_cfg (DictConfig): Configuration for the grasp generator
        grasp_discriminator_cfg (DictConfig): Configuration for the grasp discriminator
    """

    def __init__(
        self, grasp_generator_cfg: DictConfig, grasp_discriminator_cfg: DictConfig
    ):
        super(GraspGen, self).__init__()
        self.grasp_generator = GraspGenGenerator.from_config(grasp_generator_cfg)
        self.grasp_discriminator = GraspGenDiscriminator.from_config(
            grasp_discriminator_cfg
        )

    def forward(self, data):
        """Generate diffusion grasps and score them with the eager model."""
        outputs, _, stats = self.grasp_generator.infer(data, return_metrics=True)
        data.update(outputs)
        data["grasp_key"] = "grasps_pred"
        outputs, _, _ = self.grasp_discriminator.infer(data)
        return outputs, {}, stats

    def infer(self, data, return_metrics=False):
        """Inference method for generating and evaluating grasps.

        Args:
            data: Input data dictionary containing point clouds
            return_metrics (bool): Whether to compute evaluation metrics

        Returns:
            tuple: (outputs, losses, stats) containing generated and scored grasps with metrics
        """
        return self.forward(data)

    @classmethod
    def from_config(
        cls, grasp_generator_cfg: DictConfig, grasp_discriminator_cfg: DictConfig
    ):
        """Creates a GraspGen instance from configuration objects.

        Args:
            grasp_generator_cfg (DictConfig): Configuration for the grasp generator
            grasp_discriminator_cfg (DictConfig): Configuration for the grasp discriminator

        Returns:
            GraspGen: Instantiated model
        """
        return GraspGen(grasp_generator_cfg, grasp_discriminator_cfg)

    def load_state_dict(
        self, grasp_generator_ckpt_filepath: str, grasp_discriminator_ckpt_filepath: str
    ):
        """Loads pretrained weights for both generator and discriminator.

        Args:
            grasp_generator_ckpt_filepath (str): Path to generator checkpoint
            grasp_discriminator_ckpt_filepath (str): Path to discriminator checkpoint
        """
        logger.info(
            f"Loading generator checkpoint from {grasp_generator_ckpt_filepath}"
        )
        try:
            ckpt = torch.load(grasp_generator_ckpt_filepath, map_location="cpu")
        except:
            ckpt = find_the_last_ckpt(grasp_generator_ckpt_filepath)

        self.grasp_generator.load_state_dict(ckpt["model"])

        logger.info(
            f"Loading discriminator checkpoint from {grasp_discriminator_ckpt_filepath}"
        )
        try:
            ckpt = torch.load(grasp_discriminator_ckpt_filepath, map_location="cpu")
        except:
            ckpt = find_the_last_ckpt(grasp_discriminator_ckpt_filepath)

        self.grasp_discriminator.load_state_dict(ckpt["model"])
