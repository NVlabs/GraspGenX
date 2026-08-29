# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time

import numpy as np
import omegaconf
import torch

from graspgenx.batching import collate
from graspgenx.inference.preprocessing import (
    MIN_OBJECT_POINT_COUNT,
    filter_and_resample_point_cloud,
)
from graspgenx.models.grasp_gen import GraspGen
from graspgenx.utils.logging_config import get_logger
from graspgenx.utils.point_cloud import resample_point_cloud_fixed
from graspgenx.x_grippers import (
    XGripperInfo,
    make_sweep_volume_gripper_info,
    resolve_gripper_info,
)

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
        gripper_name: str | None = None,
        assets_dir: str | None = None,
        use_fast_inference: bool = False,
        diffusion_grasps: int = 100,
        obb_grasps: int = 50,
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
            use_fast_inference: If True, delegate the complete optimized pipeline
                                to the native C++ runtime.
            diffusion_grasps: Fixed diffusion candidate count for fast
                              inference.
            obb_grasps: Fixed analytic candidate count for optimized inference.
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
        self.diffusion_grasps = int(diffusion_grasps)
        self.obb_grasps = int(obb_grasps)
        self._gripper_input_cache = {}
        self.using_fast_inference = False
        self.native_runtime = getattr(model, "native_cpp_runtime", None)
        self._native_gripper = None

        if use_fast_inference and self.native_runtime is None:
            from graspgenx.fast_inference import install_fast_runtime

            accelerated = install_fast_runtime(
                self,
                diffusion_grasps=self.diffusion_grasps,
                obb_grasps=self.obb_grasps,
            )
            if accelerated:
                logger.info("Inference delegated to the native C++ runtime.")
            else:
                logger.warning("Fast inference not enabled; using eager PyTorch.")
            self.native_runtime = getattr(model, "native_cpp_runtime", None)
        self.using_fast_inference = self.native_runtime is not None
        if self.using_fast_inference:
            if (
                self.diffusion_grasps != self.native_runtime.diffusion_grasps
                or self.obb_grasps != self.native_runtime.obb_grasps
            ):
                raise ValueError(
                    "sampler candidate counts do not match the shared native runtime"
                )
            self._native_gripper = self._prepare_native_gripper()

    def _prepare_native_gripper(self):
        from graspgenx.fast_inference.cpp_runtime import encode_gripper_embeddings

        descriptor = np.concatenate(
            (self.gripper.sweep_volume, self.gripper.sweep_volume_mid), axis=0
        )
        return encode_gripper_embeddings(
            self.model, descriptor, float(self.gripper.depth)
        )

    def _native_result(self, object_pc, num_grasps: int):
        if int(num_grasps) != self.native_runtime.diffusion_grasps:
            raise ValueError(
                "native C++ inference uses the construction-time diffusion count "
                f"{self.native_runtime.diffusion_grasps}; received {num_grasps}"
            )
        if isinstance(object_pc, np.ndarray):
            points = torch.from_numpy(np.asarray(object_pc, dtype=np.float32))
        else:
            points = object_pc
        result = self.native_runtime.infer(points, self._native_gripper)
        if bool(result.ptv3_topology_overflow.any().item()):
            raise RuntimeError(
                "native PTV3 topology capacity overflow; inference result is invalid"
            )
        return result

    def _sample_native(self, object_pc, threshold: float, num_grasps: int):
        result = self._native_result(object_pc, num_grasps)
        branch = result.source_indices < self.native_runtime.diffusion_grasps
        grasps = result.grasps[branch]
        confidence = result.confidence[branch]
        if threshold > 0.0:
            keep = confidence >= threshold
            grasps = grasps[keep]
            confidence = confidence[keep]
        return grasps, confidence

    @classmethod
    def from_sweep_volume(
        cls,
        cfg: omegaconf.DictConfig,
        params,
        model=None,
        use_fast_inference: bool = False,
        diffusion_grasps: int = 100,
        obb_grasps: int = 50,
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
            use_fast_inference=use_fast_inference,
            diffusion_grasps=diffusion_grasps,
            obb_grasps=obb_grasps,
        )

    @staticmethod
    def run_inference(
        object_pc: np.ndarray | torch.Tensor,
        grasp_sampler: "GraspGenXSampler",
        grasp_threshold: float = -1.0,
        num_grasps: int | None = None,
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
        if num_grasps is None:
            num_grasps = grasp_sampler.diffusion_grasps
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
                keep_count = min(topk_num_grasps, len(grasps))
                grasp_conf, order = torch.topk(
                    grasp_conf, keep_count, largest=True, sorted=True
                )
                grasps = grasps[order]

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
        num_grasps: int | None = None,
        topk_num_grasps: int = -1,
        remove_outliers: bool = True,
        return_object_embedding: bool = False,
    ) -> list:
        """Run one batched eager-PyTorch inference call over N object PCs.

        The native C++ runtime is intentionally fixed to one object and rejects
        ``N > 1``; call this method once per object there.

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
        if grasp_sampler.using_fast_inference and len(object_pcs) > 1:
            raise ValueError(
                "The native C++ runtime supports exactly one "
                "object per call; loop over objects for batch size > 1."
            )
        if num_grasps is None:
            num_grasps = grasp_sampler.diffusion_grasps
        if grasp_threshold == -1.0 and topk_num_grasps == -1:
            topk_num_grasps = 100

        if grasp_sampler.using_fast_inference:
            if return_object_embedding:
                raise ValueError(
                    "the native C++ fast path returns final ranked grasps, not "
                    "framework object embeddings"
                )
            grasps, confidence = grasp_sampler._sample_native(
                object_pcs[0], grasp_threshold, int(num_grasps)
            )
            if topk_num_grasps != -1 and len(grasps) > topk_num_grasps:
                confidence, order = torch.topk(
                    confidence, topk_num_grasps, largest=True, sorted=True
                )
                grasps = grasps[order]
            return [(grasps, confidence)]

        device = next(grasp_sampler.model.parameters()).device
        target_n = int(grasp_sampler.cfg.data.num_points)

        centers: list = []
        data_items: list = []
        for pc in object_pcs:
            if isinstance(pc, np.ndarray):
                pc_t = torch.from_numpy(pc).float()
            else:
                pc_t = pc.float()
            if len(pc_t) < MIN_OBJECT_POINT_COUNT:
                raise ValueError(
                    "run_inference_batch requires at least "
                    f"{MIN_OBJECT_POINT_COUNT} object points; received {len(pc_t)}"
                )
            pc_t = pc_t.to(device)
            if remove_outliers:
                # Scene segmentations may contain tens of thousands of pixels.
                # PTV3 consumes exactly target_n points, so deterministically
                # reduce oversized inputs before the exact O(N^2) 20-neighbor
                # filter instead of spending work on points that are discarded.
                if len(pc_t) > target_n:
                    pc_t = resample_point_cloud_fixed(pc_t, target_n)
                pc_t = filter_and_resample_point_cloud(pc_t, num_points=target_n)
            else:
                pc_t = resample_point_cloud_fixed(pc_t, target_n)
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

        # Set the diffusion sampler to produce num_grasps per object.
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
                c_i, order = torch.topk(c_i, topk_num_grasps, largest=True, sorted=True)
                g_i = g_i[order]
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
        obj_pcd: np.ndarray | torch.Tensor,
        threshold: float = -1.0,
        num_grasps: int | None = None,
        remove_outliers: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obj_pcd: np.array of shape (N, 3)

        Returns:
            grasps: torch.tensor of shape (M, 6)
            grasp_conf: torch.tensor of shape (M,)
            grasp_contacts: torch.tensor of shape (M, 3)
        """

        if num_grasps is None:
            num_grasps = self.diffusion_grasps
        if self.using_fast_inference:
            grasps, grasp_conf = self._sample_native(
                obj_pcd, threshold, int(num_grasps)
            )
            return grasps, grasp_conf, None
        if isinstance(obj_pcd, np.ndarray):
            obj_pcd = torch.from_numpy(obj_pcd).cuda().float()
        if len(obj_pcd) < MIN_OBJECT_POINT_COUNT:
            raise ValueError(
                f"sample requires at least {MIN_OBJECT_POINT_COUNT} object "
                f"points; received {len(obj_pcd)}"
            )
        target_n = int(self.cfg.data.num_points)
        if remove_outliers:
            if len(obj_pcd) > target_n:
                obj_pcd = resample_point_cloud_fixed(obj_pcd, target_n)
            obj_pcd = filter_and_resample_point_cloud(obj_pcd, num_points=target_n)
        else:
            obj_pcd = resample_point_cloud_fixed(obj_pcd, target_n)

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
    def sample_moe(
        self,
        obj_pcd: np.ndarray | torch.Tensor,
        threshold: float = -1.0,
        num_grasps: int | None = None,
        remove_outliers: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run the native C++ diffusion + OBB pipeline once."""
        if num_grasps is None:
            num_grasps = self.diffusion_grasps
        if not self.using_fast_inference:
            raise RuntimeError("sample_moe requires the native C++ runtime")
        result = self._native_result(obj_pcd, int(num_grasps))
        diffusion_branch = result.source_indices < self.native_runtime.diffusion_grasps
        grasps_diff = result.grasps[diffusion_branch]
        scores_diff = result.confidence[diffusion_branch]
        grasps_obb = result.grasps[~diffusion_branch]
        scores_obb = result.confidence[~diffusion_branch]
        if threshold > 0.0:
            keep_diff = scores_diff >= threshold
            grasps_diff = grasps_diff[keep_diff]
            scores_diff = scores_diff[keep_diff]
            keep_obb = scores_obb >= threshold
            grasps_obb = grasps_obb[keep_obb]
            scores_obb = scores_obb[keep_obb]

        bounds = result.obb_bounds
        center_local = 0.5 * (bounds[:3] + bounds[3:6])
        cosine, sine = bounds[6], bounds[7]
        obb_parameters = torch.empty((8,), device=bounds.device, dtype=bounds.dtype)
        obb_parameters[0] = (
            cosine * center_local[0] - sine * center_local[1] + result.object_center[0]
        )
        obb_parameters[1] = (
            sine * center_local[0] + cosine * center_local[1] + result.object_center[1]
        )
        obb_parameters[2] = center_local[2] + result.object_center[2]
        obb_parameters[3:6] = 0.5 * (bounds[3:6] - bounds[:3])
        obb_parameters[6:] = bounds[6:]
        return grasps_diff, scores_diff, grasps_obb, scores_obb, obb_parameters

    @torch.inference_mode()
    def load_gripper_input(self, outputs):
        device = next(self.model.parameters()).device
        cache_key = (device.type, device.index)
        cached = self._gripper_input_cache.get(cache_key)
        if cached is None:
            backbones = {
                self.model.grasp_generator.gripper_backbone,
                self.model.grasp_discriminator.gripper_backbone,
            }
            cached = {}
            if "z_offset" in backbones:
                cached["z_offset"] = torch.tensor(
                    (self.gripper.depth,), dtype=torch.float32, device=device
                )
            if "sweep_volume" in backbones:
                cached["sweep_volume"] = torch.as_tensor(
                    self.gripper.sweep_volume,
                    dtype=torch.float32,
                    device=device,
                )
            if backbones.intersection(
                {"sweep_volume_v2", "gripper_type+sweep_volume_v2"}
            ):
                sweep = np.concatenate(
                    [self.gripper.sweep_volume, self.gripper.sweep_volume_mid],
                    axis=0,
                )
                cached["sweep_volume_open_and_mid"] = torch.as_tensor(
                    sweep, dtype=torch.float32, device=device
                )
            if backbones.intersection({"gripper_type", "gripper_type+sweep_volume_v2"}):
                cached["gripper_type"] = self.gripper.gripper_type
            if "pointcloud" in backbones:
                target_n = int(self.cfg.data.num_points)
                for output_key, point_cloud in (
                    ("gripper_open_ptc", self.gripper.open_pointcloud),
                    ("gripper_close_ptc", self.gripper.close_pointcloud),
                ):
                    point_cloud = torch.as_tensor(
                        point_cloud, dtype=torch.float32, device=device
                    )
                    cached[output_key] = resample_point_cloud_fixed(
                        point_cloud, target_n
                    )
            if "volume_tsdf" in backbones:
                vol_tsdf = np.stack(
                    [
                        self.gripper.vol_tsdf["open_tsdf"],
                        self.gripper.vol_tsdf["close_tsdf"],
                    ],
                    axis=0,
                )
                cached["gripper_vol_tsdf"] = torch.as_tensor(
                    vol_tsdf, dtype=torch.float32, device=device
                )
            if "pointnet_repr" in backbones:
                pointnet_repr = np.concatenate(
                    [
                        self.gripper.pointnet_vae["open"],
                        self.gripper.pointnet_vae["half"],
                        self.gripper.pointnet_vae["close"],
                    ],
                    axis=0,
                )
                cached["gripper_pointnet_repr"] = torch.as_tensor(
                    pointnet_repr, dtype=torch.float32, device=device
                )
            self._gripper_input_cache[cache_key] = cached
        outputs.update(cached)
        return outputs

    def get_gripper_info(self):
        return self.gripper
