# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import time
import numpy as np
import omegaconf
import torch

from graspgenx.dataset.dataset import collate
from graspgenx.models.grasp_gen import GraspGen
from graspgenx.utils.point_cloud import knn_points, point_cloud_outlier_removal
from graspgenx.x_grippers import (
    XGripperInfo,
    make_sweep_volume_gripper_info,
    resolve_gripper_info,
    sample_gripper_pointclouds,
)
from graspgenx.utils.logging_config import get_logger

logger = get_logger(__name__)

# gripper_backbone values that can be conditioned from raw sweep-volume
# parameters alone (no gripper point clouds / TSDF / VAE assets required).
SWEEP_VOLUME_ONLY_BACKBONES = {
    "sweep_volume_v2",
    "sweep_volume",
    "gripper_type+sweep_volume_v2",
    "gripper_type",
    "z_offset",
}


def load_grasp_gen_model(cfg: omegaconf.DictConfig):
    """Load the (gripper-independent) GraspGen model from cfg checkpoints.

    The returned model can back any number of samplers — the gripper
    conditioning travels with each request's data batch, not the weights.
    """
    model = GraspGen.from_config(cfg.diffusion, cfg.discriminator)
    if not os.path.exists(cfg.eval.gen_checkpoint):
        raise FileNotFoundError(f"Checkpoint {cfg.eval.gen_checkpoint} does not exist")
    if not os.path.exists(cfg.eval.dis_checkpoint):
        raise FileNotFoundError(f"Checkpoint {cfg.eval.dis_checkpoint} does not exist")
    model.load_state_dict(cfg.eval.gen_checkpoint, cfg.eval.dis_checkpoint)
    model.eval()
    return model.cuda().eval()


class GraspGenXSampler:
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        gripper_name: str = None,
        assets_dir: str = None,
        use_tensorrt: bool = False,
        tensorrt_precision: str = "fp32",
        gripper_info: XGripperInfo = None,
        model=None,
    ):
        """
        Args:
            cfg: Hydra config object
            gripper_name: Name of the gripper (must exist in assets). Mutually
                          exclusive with gripper_info.
            assets_dir: Root directory containing x_grippers/ and proc_grippers/
                        subdirectories. Defaults to /code/assets (docker path).
            use_tensorrt: If True, compile the diffusion denoiser to TensorRT
                          (opt-in; requires the 'tensorrt' extra). Falls back to
                          eager PyTorch if unavailable or compilation fails.
            tensorrt_precision: 'fp32' (default, exact parity) or 'fp16'.
            gripper_info: Pre-built XGripperInfo (e.g. from raw sweep-volume
                          params) used instead of a name-based asset lookup.
            model: Already-loaded GraspGen model to reuse. The model is
                   gripper-independent, so one instance can back many samplers
                   (each request carries its own gripper conditioning).
        """

        self.cfg = cfg

        if gripper_info is not None:
            self.gripper = gripper_info
            self.gripper_name = gripper_info.gripper_name
        else:
            if gripper_name is None:
                raise ValueError(
                    "GraspGenXSampler needs either gripper_name or gripper_info."
                )
            if assets_dir is None:
                assets_dir = "/code/assets"
            self.gripper = resolve_gripper_info(gripper_name, assets_dir)
            self.gripper_name = gripper_name

        if model is None:
            model = load_grasp_gen_model(cfg)

        self.model = model

        if use_tensorrt:
            from graspgenx.models.tensorrt_utils import accelerate_sampler
            from graspgenx.samplers.graspmoe import set_gpu_obb

            accelerated = accelerate_sampler(self, precision=tensorrt_precision)
            if accelerated:
                logger.info("Diffusion denoiser accelerated with TensorRT.")
            else:
                logger.warning(
                    "TensorRT acceleration not applied; using eager PyTorch."
                )
            # Run the GraspMoE OBB outlier-removal on GPU only when TensorRT is
            # requested; otherwise it stays on the CPU (scipy).
            set_gpu_obb(True)

    @classmethod
    def from_sweep_volume(
        cls,
        cfg: omegaconf.DictConfig,
        params,
        model=None,
        use_tensorrt: bool = False,
        tensorrt_precision: str = "fp32",
    ) -> "GraspGenXSampler":
        """Build a sampler from raw sweep-volume parameters — no gripper assets.

        Args:
            cfg: Hydra config object. Both cfg.diffusion.gripper_backbone and
                 cfg.discriminator.gripper_backbone must be conditioned on
                 sweep-volume-derivable representations (the released
                 checkpoint uses 'sweep_volume_v2' for both).
            params: A graspgenx.serving.types.SweepVolumeParams, an equivalent
                    dict, or a flat (12,) array
                    [extents_open, offset_open, extents_mid, offset_mid].
            model: Optional already-loaded GraspGen model to reuse.
        """
        from graspgenx.serving.types import SweepVolumeParams

        params = SweepVolumeParams.coerce(params)

        for side_name, side in (
            ("diffusion", cfg.diffusion),
            ("discriminator", cfg.discriminator),
        ):
            backbone = str(side.gripper_backbone)
            if backbone not in SWEEP_VOLUME_ONLY_BACKBONES:
                raise ValueError(
                    f"cfg.{side_name}.gripper_backbone={backbone!r} needs real "
                    f"gripper assets and cannot be conditioned from sweep-volume "
                    f"params alone. Supported backbones: "
                    f"{sorted(SWEEP_VOLUME_ONLY_BACKBONES)}. Use a name-based "
                    f"sampler instead."
                )

        gripper_info = make_sweep_volume_gripper_info(
            extents_open=params.extents_open,
            offset_open=params.offset_open,
            extents_mid=params.extents_mid,
            offset_mid=params.offset_mid,
            gripper_type=params.gripper_type,
            fingertip_depth=params.resolved_fingertip_depth,
        )
        return cls(
            cfg,
            gripper_info=gripper_info,
            model=model,
            use_tensorrt=use_tensorrt,
            tensorrt_precision=tensorrt_precision,
        )

    @staticmethod
    def run_inference(
        object_pc: np.ndarray | torch.Tensor,
        grasp_sampler: "GraspGenXSampler",
        grasp_threshold: float = -1.0,
        num_grasps: int = 200,
        topk_num_grasps: int = -1,
        min_grasps: int = 40,
        max_tries: int = 6,
        remove_outliers: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run grasp generation inference on a point cloud.

        Args:
            object_pc: Point cloud to generate grasps for
            grasp_sampler: Initialized GraspGenXSampler instance
            grasp_threshold: Threshold for valid grasps. If -1.0, then the top topk_grasps grasps will be ranked and returned
            num_grasps: Number of grasps to generate
            topk_grasps: Maximum number of grasps to return
            min_grasps: Minimum number of grasps required. If fewer grasps are found, inference will be retried
            max_tries: Maximum number of inference attempts to make before returning results

        Returns:
            grasps: Generated grasp poses
            grasp_conf: Confidence scores for the grasps
        """
        if type(object_pc) == np.ndarray:
            object_pc = torch.from_numpy(object_pc).cuda().float()

        if grasp_threshold == -1.0 and topk_num_grasps == -1:
            topk_num_grasps = 100

        all_grasps = []
        all_conf = []
        num_tries = 0

        while sum(len(g) for g in all_grasps) < min_grasps and num_tries < max_tries:
            num_tries += 1
            t0 = time.time()
            output = grasp_sampler.sample(
                object_pc,
                threshold=grasp_threshold,
                num_grasps=num_grasps,
                remove_outliers=remove_outliers,
            )
            grasp_conf = output[1]
            grasps = output[0]

            # Sort and prune grasps within this iteration
            if topk_num_grasps != -1 and len(grasps) > 0:
                grasp_conf, grasps = zip(
                    *sorted(zip(grasp_conf, grasps), key=lambda x: x[0], reverse=True)
                )
                grasps = torch.stack(grasps)
                grasp_conf = torch.stack(grasp_conf)
                grasps = grasps[:topk_num_grasps]
                grasp_conf = grasp_conf[:topk_num_grasps]

            all_grasps.append(grasps)
            all_conf.append(grasp_conf)

            logger.info(
                f"Found {len(grasps)} grasps in iteration {len(all_grasps)}, total grasps: {sum(len(g) for g in all_grasps)}"
            )
            t1 = time.time()
            logger.info(f"Time taken for inference: {t1 - t0} seconds")

        # Drop empty per-iteration results (plain lists, not tensors) so the
        # concatenation below is well-defined even when some tries found nothing.
        all_conf = [c for g, c in zip(all_grasps, all_conf) if len(g) > 0]
        all_grasps = [g for g in all_grasps if len(g) > 0]
        if len(all_grasps) == 0:
            return torch.tensor([]), torch.tensor([])

        # Concatenate all grasps and confidences
        grasps = torch.cat(all_grasps, dim=0)
        grasp_conf = torch.cat(all_conf, dim=0)
        grasps[:, 3, 3] = 1  # TODO: Fix this in grasp_gen.py later.

        return grasps, grasp_conf

    @staticmethod
    def run_inference_batch(
        object_pcs: list,
        grasp_sampler: "GraspGenXSampler",
        grasp_threshold: float = -1.0,
        num_grasps: int = 200,
        topk_num_grasps: int = -1,
        remove_outliers: bool = True,
        return_object_embedding: bool = False,
    ) -> list:
        """Batched run_inference: one diffusion + discriminator forward pass
        over N object PCs. Equivalent to ``[run_inference(pc, ...) for pc in
        object_pcs]`` but folds the reverse-diffusion loop into a single
        batched call.

        Each input PC is resampled (with replacement) to
        ``grasp_sampler.cfg.data.num_points`` so that ``collate`` can stack
        them — this matches the training distribution. Per-object centers
        are tracked and added back to the predicted grasps.

        Args:
            object_pcs: list of N point clouds, each (Mi, 3) np.ndarray or
                        torch.Tensor in the same world frame.
            grasp_threshold: per-object discriminator threshold; -1.0 keeps
                             everything then prunes by topk.
            num_grasps: diffusion samples per object (= the model's
                        ``num_grasps_per_object``).
            topk_num_grasps: cap per object after thresholding; -1 means no
                             top-k cap unless ``grasp_threshold == -1.0``
                             (in which case defaults to 100, matching
                             ``run_inference``).
            remove_outliers: run outlier removal per object before
                             resampling.

        Returns:
            List of (grasps, grasp_conf) tuples, one per input PC, in
            input order. Shapes: grasps (Ki, 4, 4), grasp_conf (Ki,) on the
            sampler's device.
        """
        if len(object_pcs) == 0:
            return []
        if grasp_threshold == -1.0 and topk_num_grasps == -1:
            topk_num_grasps = 100

        device = next(grasp_sampler.model.parameters()).device
        target_n = int(grasp_sampler.cfg.data.num_points)

        centers: list = []
        data_items: list = []
        for pc in object_pcs:
            if isinstance(pc, np.ndarray):
                pc_t = torch.from_numpy(pc).float()
            else:
                pc_t = pc.float()
            if remove_outliers:
                pc_f, _ = point_cloud_outlier_removal(pc_t)
                if len(pc_f) >= 10:
                    pc_t = pc_f
                else:
                    logger.warning(
                        "Outlier removal left %d/%d points; using the "
                        "unfiltered point cloud.",
                        len(pc_f),
                        len(pc_t),
                    )
            if len(pc_t) == 0:
                raise ValueError(
                    "run_inference_batch received an empty point cloud."
                )
            # Resample with replacement to the training-time budget so all
            # batch items have the same shape (collate uses torch.stack).
            n = pc_t.shape[0]
            if n != target_n:
                idx = torch.randint(0, n, (target_n,))
                pc_t = pc_t[idx]
            pc_t = pc_t.to(device)
            center = pc_t.mean(dim=0)
            centers.append(center)
            centered = pc_t - center[None]
            color = torch.zeros_like(centered)

            data = {}
            data["task"] = "pick"
            data["inputs"] = torch.cat([centered, color[:, :3]], dim=-1).float()
            data["points"] = centered
            data = grasp_sampler.load_gripper_input(data)
            data_items.append(data)

        data_batch = collate(data_items)
        data_batch["grasp_key"] = "grasps_pred"

        # Set the diffusion sampler to produce num_grasps per object; the
        # model's batched forward already returns (N, num_grasps, 4, 4).
        grasp_sampler.model.grasp_generator.num_grasps_per_object = num_grasps

        with torch.inference_mode():
            model_outputs, _, _ = grasp_sampler.model.infer(data_batch)

        grasps_pred = model_outputs["grasps_pred"]  # (N, K, 4, 4)
        grasp_conf = model_outputs["grasp_confidence"][..., 0]  # (N, K)

        outputs: list = []
        for i, center in enumerate(centers):
            g_i = grasps_pred[i].to(device)
            c_i = grasp_conf[i].to(device)
            if grasp_threshold > 0.0:
                keep = c_i >= grasp_threshold
                g_i = g_i[keep]
                c_i = c_i[keep]
            if topk_num_grasps != -1 and len(g_i) > topk_num_grasps:
                order = torch.argsort(c_i, descending=True)[:topk_num_grasps]
                g_i = g_i[order]
                c_i = c_i[order]
            # Restore world frame.
            if len(g_i) > 0:
                g_i = g_i.clone()
                g_i[:, :3, 3] = g_i[:, :3, 3] + center.to(g_i.device)
                g_i[:, 3, 3] = 1.0
            outputs.append((g_i, c_i))

        if return_object_embedding:
            # Per-object discriminator embedding [N, num_object_dim] (or None if
            # the discriminator did not expose it), for reuse without re-encoding.
            return outputs, model_outputs.get("object_embedding", None)
        return outputs

    @torch.inference_mode()
    def sample(
        self,
        obj_pcd: np.ndarray,
        threshold: float = -1.0,
        num_grasps: int = 200,
        remove_outliers: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obj_pcd: np.array of shape (N, 3)
            obj_pts_color (Optional): np.array of shape (N, 4)

        Returns:
            grasps: torch.tensor of shape (M, 6)
            grasp_conf: torch.tensor of shape (M,)
            grasp_contacts: torch.tensor of shape (M, 3)
        """

        if remove_outliers:
            obj_pcd_filtered, _ = point_cloud_outlier_removal(obj_pcd)
            if len(obj_pcd_filtered) >= 10:
                obj_pcd = obj_pcd_filtered
            else:
                logger.warning(
                    "Outlier removal left %d/%d points; using the unfiltered "
                    "point cloud.",
                    len(obj_pcd_filtered),
                    len(obj_pcd),
                )
        if len(obj_pcd) == 0:
            return [], [], []

        obj_pcd_center = obj_pcd.mean(axis=0)
        obj_pts_color = torch.zeros_like(obj_pcd)
        obj_mean_points = obj_pcd - obj_pcd_center[None]

        data = {}
        data["task"] = "pick"
        data["inputs"] = torch.cat(
            [obj_mean_points, obj_pts_color[:, :3].squeeze(1)], dim=-1
        ).float()
        data["points"] = obj_mean_points
        data = self.load_gripper_input(data)

        data_batch = collate([data])
        grasp_key = "grasps"
        with torch.inference_mode():
            grasp_key = "grasps_pred"
            self.model.grasp_generator.num_grasps_per_object = num_grasps
            model_outputs, _, _ = self.model.infer(data_batch)

        if len(model_outputs[grasp_key][0]) == 0:
            return [], [], []

        grasps = model_outputs[grasp_key][0]

        grasp_conf = model_outputs["grasp_confidence"][0][:, 0]
        logger.info(
            f"Confidences min: {grasp_conf.min():.5f}, max: {grasp_conf.max():.5f}"
        )
        mask_best_grasps = grasp_conf >= threshold
        logger.info(
            f"Thresholding grasps @ {threshold}. Only {mask_best_grasps.sum()}/{mask_best_grasps.shape[0]} grasps remaining"
        )

        grasps = grasps[mask_best_grasps]
        grasp_conf = grasp_conf[mask_best_grasps]

        grasps[:, :3, 3] += obj_pcd_center
        return grasps, grasp_conf, None

    @torch.inference_mode()
    def load_gripper_input(self, outputs):

        outputs["z_offset"] = torch.tensor(
            (self.gripper.depth,), dtype=torch.float32
        ).cuda()
        outputs["sweep_volume"] = torch.from_numpy(
            self.gripper.sweep_volume.astype(np.float32)
        ).cuda()
        outputs["gripper_type"] = self.gripper.gripper_type

        sweep_volume_open = self.gripper.sweep_volume.astype(np.float32)
        sweep_volume_mid = self.gripper.sweep_volume_mid.astype(np.float32)
        outputs["sweep_volume_open_and_mid"] = torch.from_numpy(
            np.concatenate([sweep_volume_open, sweep_volume_mid], axis=0)
        ).cuda()

        gripper_open_ptc, gripper_close_ptc = sample_gripper_pointclouds(
            self.gripper, self.cfg.data.num_points
        )
        outputs["gripper_open_ptc"] = torch.from_numpy(gripper_open_ptc).cuda()
        outputs["gripper_close_ptc"] = torch.from_numpy(gripper_close_ptc).cuda()

        vol_tsdf = np.stack(
            [self.gripper.vol_tsdf[f"open_tsdf"], self.gripper.vol_tsdf[f"close_tsdf"]],
            axis=0,
        )
        outputs["gripper_vol_tsdf"] = torch.from_numpy(
            vol_tsdf.astype(np.float32)
        ).cuda()

        outputs["gripper_pointnet_repr"] = torch.from_numpy(
            np.concatenate(
                [
                    self.gripper.pointnet_vae["open"],
                    self.gripper.pointnet_vae["half"],
                    self.gripper.pointnet_vae["close"],
                ],
                axis=0,
            ).astype(np.float32)
        ).cuda()

        return outputs

    def get_gripper_info(self):
        return self.gripper
