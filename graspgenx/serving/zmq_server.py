# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ZMQ REQ/REP server that wraps :class:`GraspGenXSampler` for remote inference.

Wire protocol (msgpack with ``msgpack_numpy.patch()`` so numpy arrays travel
natively):

* ``{"action": "health"}`` → ``{"status": "ok"}``
* ``{"action": "metadata"}`` →
  ``{"default_gripper": str|None, "loaded_grippers": [str], "model": {...}}``
* ``{"action": "infer", "point_cloud": (N,3) float32,
       "gripper_name": str (optional, falls back to default),
       "num_grasps": int = 100,
       "grasp_threshold": float = -1.0,
       "topk_num_grasps": int = 100}`` →
  ``{"grasps": (K,4,4) float32, "confidences": (K,) float32,
     "gripper_name": str, "timing": {"infer_ms": float}}``

Sweep-volume-params actions (no gripper assets needed on the server; the
gripper is specified per request as raw sweep-volume parameters — see
:class:`graspgenx.serving.types.SweepVolumeParams`). All three accept
``planner`` ("graspmoe" default | "diffusion"), ``num_grasps`` and the
``moe_*`` knobs of :func:`graspgenx.samplers.run_planner_on_object`. They
return **all** generated grasps with their scores — ``grasp_threshold`` /
``topk_num_grasps`` are rejected; thresholding is the client's job:

* ``{"action": "infer_object", "point_cloud": (N,3) float32,
       "sweep_volume_params": dict | (12,) float32, ...}`` →
  ``{"grasps": (K,4,4), "confidences": (K,), "branch_tags": [str],
     "timing": {...}}`` — grasps in the frame of the input point cloud.
* ``{"action": "infer_scene_depth", "depth": (H,W) float32 meters,
       "intrinsics": (3,3), "instance_mask": (H,W) int (0 = ignore),
       "sweep_volume_params": ..., "min_object_points": int = 100, ...}`` →
  per-instance results in the **camera frame** (see below).
* ``{"action": "infer_scene_pc", "point_cloud": (N,3) | (H,W,3),
       "instance_mask": int array with N elements, "sweep_volume_params": ...,
       "min_object_points": int = 100, ...}`` →
  per-instance results in the frame of the input point cloud.

Scene responses use parallel lists (msgpack's default ``strict_map_key``
rejects integer map keys):
``{"instance_ids": (M,) int32, "grasps": [(Ki,4,4)], "confidences": [(Ki,)],
   "branch_tags": [[str]], "skipped_instance_ids": (S,) int32,
   "timing": {"infer_ms": float}}``.

Any unhandled error is returned as ``{"error": str(exc)}`` — the client raises.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import msgpack
import msgpack_numpy
import numpy as np
import zmq

from graspgenx.grasp_server import GraspGenXSampler, load_grasp_gen_model
from graspgenx.samplers import run_planner_on_batch, run_planner_on_object
from graspgenx.serving.types import SweepVolumeParams
from graspgenx.utils.checkpoint_io import load_model_cfg
from graspgenx.utils.logging_config import get_logger

msgpack_numpy.patch()

logger = get_logger(__name__)

ACTIONS = (
    "health",
    "metadata",
    "infer",
    "infer_object",
    "infer_scene_depth",
    "infer_scene_pc",
)

# Wire-level planner knobs and their defaults, mirroring
# graspgenx.samplers.run_planner_on_object / run_planner_on_batch.
# NOTE: grasp_threshold / topk_num_grasps are deliberately absent — the new
# actions return ALL generated grasps with their scores; thresholding and
# top-k selection are the client's job.
PLANNER_KNOB_DEFAULTS = {
    "num_grasps": 100,
    "moe_num_yaws": None,
    "moe_z_offsets_cm": (-8.0, -6.0, -4.0, -2.0, 0.0),
    "moe_outlier_threshold": 0.014,
    "moe_outlier_k": 20,
    "moe_obb_mode": "pca",
    "moe_skip_obb_rule": "never",
    "moe_obb_density": "sparse",
    "moe_obb_position_spacing_cm": 1.0,
}


class GraspGenXZMQServer:
    """ZMQ REQ/REP wrapper around :class:`GraspGenXSampler`.

    Samplers are loaded lazily on the first ``infer`` request for a given
    gripper and cached in-process — so the first request per gripper pays the
    model-init cost, every subsequent request is hot.

    Args:
        config_path: Path to the checkpoint root containing ``gen/`` and
            ``dis/`` subdirectories (each with ``config.yaml`` + ``epoch_*.pth``).
            For backward compatibility a path that ends in ``config.yaml``
            inside a ``gen/`` directory is accepted — its grandparent is used
            as the root.
        assets_dir: Root directory containing ``x_grippers/`` and
            ``proc_grippers/`` subdirectories. Passed through to each sampler.
        host: ZMQ bind address (default ``0.0.0.0``).
        port: ZMQ bind port (default ``5556``).
        default_gripper: If set, pre-load this gripper at startup and use it
            when clients omit ``gripper_name``.
    """

    def __init__(
        self,
        config_path: str,
        assets_dir: str,
        host: str = "0.0.0.0",
        port: int = 5556,
        default_gripper: str | None = None,
        use_fast_inference: bool = False,
    ) -> None:
        self.assets_dir = str(Path(assets_dir).expanduser().resolve())
        self.host = host
        self.port = port
        self.default_gripper = default_gripper
        self.use_fast_inference = use_fast_inference
        self._fast_inference_enabled = False

        self.cfg = self._load_cfg(config_path)
        self._samplers: dict[str, GraspGenXSampler] = {}
        self._samplers_lock = threading.Lock()
        # The GraspGen model is gripper-independent; load it eagerly at
        # startup and share it across every sampler (name-based and
        # params-based alike) — the first request pays no model-init cost.
        logger.info("Loading GraspGen model weights ...")
        self._shared_model = load_grasp_gen_model(self.cfg)
        logger.info("Model loaded.")

        if use_fast_inference:
            from types import SimpleNamespace

            from graspgenx.fast_inference import install_fast_runtime

            # The accelerator mutates sampler.model in place; the shim lets
            # us accelerate the shared model once, before any sampler exists.
            self._fast_inference_enabled = install_fast_runtime(
                SimpleNamespace(model=self._shared_model)
            )
            if self._fast_inference_enabled:
                logger.info("Native C++ fast inference enabled.")
            else:
                logger.warning("Fast inference not enabled; using eager PyTorch.")

        if default_gripper:
            logger.info("Pre-loading default gripper: %s", default_gripper)
            self._get_sampler(default_gripper)

    @staticmethod
    def _load_cfg(config_path: str):
        """Resolve a user-supplied path to a merged gen+dis config."""
        p = Path(config_path).expanduser().resolve()
        if p.is_file() and p.name == "config.yaml" and p.parent.name == "gen":
            root = p.parent.parent
        elif p.is_dir():
            root = p
        else:
            raise FileNotFoundError(
                f"--config must be a checkpoint root containing gen/ and dis/ "
                f"subdirs (or path to gen/config.yaml). Got: {config_path}"
            )
        gen_dir = root / "gen"
        dis_dir = root / "dis"
        if not gen_dir.is_dir() or not dis_dir.is_dir():
            raise FileNotFoundError(
                f"Checkpoint root {root} must contain gen/ and dis/ subdirs."
            )
        return load_model_cfg(str(gen_dir), str(dis_dir))

    def _get_sampler(self, gripper_name: str) -> GraspGenXSampler:
        with self._samplers_lock:
            sampler = self._samplers.get(gripper_name)
            if sampler is None:
                logger.info("Loading sampler for gripper: %s", gripper_name)
                sampler = GraspGenXSampler(
                    self.cfg,
                    gripper_name,
                    assets_dir=self.assets_dir,
                    model=self._shared_model,
                )
                sampler.using_fast_inference = self._fast_inference_enabled
                self._samplers[gripper_name] = sampler
            return sampler

    def _get_sampler_from_params(self, params: SweepVolumeParams) -> GraspGenXSampler:
        key = "sweep:" + params.cache_key()
        with self._samplers_lock:
            sampler = self._samplers.get(key)
            if sampler is None:
                logger.info("Building sampler from sweep-volume params (%s)", key)
                sampler = GraspGenXSampler.from_sweep_volume(
                    self.cfg, params, model=self._shared_model
                )
                sampler.using_fast_inference = self._fast_inference_enabled
                self._samplers[key] = sampler
            return sampler

    @staticmethod
    def _parse_sweep_params(request: dict) -> SweepVolumeParams:
        if "gripper_name" in request:
            raise ValueError(
                f"Action {request.get('action')!r} takes sweep-volume params "
                f"only — 'gripper_name' is not accepted. Pass "
                f"'sweep_volume_params' (use the legacy 'infer' action for "
                f"name-based lookup)."
            )
        params = request.get("sweep_volume_params")
        if params is None:
            raise ValueError("Request is missing 'sweep_volume_params'.")
        return SweepVolumeParams.coerce(params)

    @staticmethod
    def _parse_planner_kwargs(request: dict) -> dict:
        planner = str(request.get("planner", "graspmoe"))
        if planner not in ("diffusion", "graspmoe"):
            raise ValueError(
                f"Unknown planner {planner!r}; expected 'diffusion' or 'graspmoe'."
            )
        rejected = {"grasp_threshold", "topk_num_grasps"} & set(request.keys())
        if rejected:
            raise ValueError(
                f"{sorted(rejected)} are not accepted by action "
                f"{request.get('action')!r} — the server returns ALL generated "
                f"grasps with their scores; apply thresholding/top-k client-side."
            )
        kwargs = {"planner": planner}
        for key, default in PLANNER_KNOB_DEFAULTS.items():
            value = request.get(key, default)
            if value is None:
                pass
            elif key == "moe_z_offsets_cm":
                value = tuple(float(x) for x in value)
            elif isinstance(default, int):
                value = int(value)
            elif isinstance(default, float):
                value = float(value)
            else:
                value = str(value)
            kwargs[key] = value
        # Return everything: no score threshold, and a top-k that never prunes
        # (num_grasps for the diffusion planner; -1 = keep the whole
        # diffusion ∪ OBB union for graspmoe).
        kwargs["grasp_threshold"] = -1.0
        kwargs["topk_num_grasps"] = (
            -1 if planner == "graspmoe" else kwargs["num_grasps"]
        )
        return kwargs

    def _handle_metadata(self) -> dict:
        diff = self.cfg.diffusion
        dis = self.cfg.discriminator
        return {
            "default_gripper": self.default_gripper,
            "loaded_grippers": sorted(self._samplers.keys()),
            "model": {
                "generator_backbone": str(diff.object_backbone),
                "discriminator_backbone": str(dis.object_backbone),
                "grasp_repr": str(diff.grasp_repr),
                "num_diffusion_iters_eval": int(diff.num_diffusion_iters_eval),
            },
            "assets_dir": self.assets_dir,
            "actions": list(ACTIONS),
            "precision": {
                "fast_inference": bool(self._fast_inference_enabled),
                "backend": (
                    "cpp_runtime" if self._fast_inference_enabled else "pytorch"
                ),
                "ptv3": "fp16" if self._fast_inference_enabled else "fp32",
                "diffusion": "fp16" if self._fast_inference_enabled else "fp32",
                "scorer": "fp32",
            },
        }

    @staticmethod
    def _validate_object_pc(point_cloud) -> np.ndarray:
        if point_cloud is None:
            raise ValueError("Request is missing 'point_cloud'.")
        # msgpack-numpy decodes arrays over an immutable byte buffer. Make the
        # ownership explicit before torch.from_numpy consumes it downstream.
        pc = np.array(point_cloud, dtype=np.float32, order="C", copy=True)
        if pc.ndim != 2 or pc.shape[1] != 3 or len(pc) == 0:
            raise ValueError(f"point_cloud must be non-empty (N, 3); got {pc.shape}")
        return pc

    def _handle_infer_object(self, request: dict) -> dict:
        """Mode 1: sweep-volume params + segmented object PC → grasps in the
        frame of the input point cloud."""
        params = self._parse_sweep_params(request)
        pc = self._validate_object_pc(request.get("point_cloud"))
        knobs = self._parse_planner_kwargs(request)
        sampler = self._get_sampler_from_params(params)

        t0 = time.monotonic()
        grasps, confidences, branch_tags, _ = run_planner_on_object(
            pc, sampler, **knobs
        )
        infer_ms = (time.monotonic() - t0) * 1000.0

        return {
            "grasps": np.asarray(grasps, dtype=np.float32).reshape(-1, 4, 4),
            "confidences": np.asarray(confidences, dtype=np.float32).reshape(-1),
            "branch_tags": [str(t) for t in branch_tags],
            "timing": {"infer_ms": float(infer_ms)},
        }

    def _handle_infer_scene_depth(self, request: dict) -> dict:
        """Mode 2: depth image + intrinsics + instance mask → per-instance
        grasps in the camera frame."""
        # Deferred import: scene_loaders pulls in PIL, which pure serving
        # deployments may not have installed.
        from graspgenx.utils.scene_loaders import depth_to_camera_xyz

        depth = request.get("depth")
        if depth is None:
            raise ValueError("Request is missing 'depth'.")
        depth = np.asarray(depth)
        if depth.ndim != 2:
            raise ValueError(f"depth must be (H, W); got shape {depth.shape}")
        if not np.issubdtype(depth.dtype, np.floating):
            raise ValueError(
                f"depth must be float meters; got dtype {depth.dtype} "
                f"(convert millimeter uint16 depth to float32 meters first)"
            )
        depth = depth.astype(np.float32)

        intrinsics = request.get("intrinsics")
        if intrinsics is None:
            raise ValueError("Request is missing 'intrinsics'.")
        K = np.asarray(intrinsics, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"intrinsics must be (3, 3); got shape {K.shape}")

        mask = self._validate_instance_mask(
            request.get("instance_mask"), expected_shape=depth.shape
        )

        xyz = depth_to_camera_xyz(depth, K).reshape(-1, 3)
        valid = ((depth > 0) & np.isfinite(depth)).reshape(-1)
        return self._infer_instances(xyz, mask.reshape(-1), valid, request)

    def _handle_infer_scene_pc(self, request: dict) -> dict:
        """Mode 3: scene point cloud + instance mask → per-instance grasps in
        the frame of the input point cloud."""
        point_cloud = request.get("point_cloud")
        if point_cloud is None:
            raise ValueError("Request is missing 'point_cloud'.")
        # msgpack-numpy decodes arrays over an immutable byte buffer. Make the
        # ownership explicit before torch.from_numpy consumes it downstream.
        pc = np.array(point_cloud, dtype=np.float32, order="C", copy=True)
        if pc.ndim == 3 and pc.shape[-1] == 3:
            pc = pc.reshape(-1, 3)
        elif pc.ndim != 2 or pc.shape[-1] != 3:
            raise ValueError(
                f"point_cloud must be (N, 3) or (H, W, 3); got shape {pc.shape}"
            )

        mask = self._validate_instance_mask(
            request.get("instance_mask"), expected_size=len(pc)
        )

        valid = np.isfinite(pc).all(axis=1)
        return self._infer_instances(pc, mask.reshape(-1), valid, request)

    @staticmethod
    def _validate_instance_mask(
        mask, expected_shape: tuple | None = None, expected_size: int | None = None
    ) -> np.ndarray:
        if mask is None:
            raise ValueError("Request is missing 'instance_mask'.")
        mask = np.asarray(mask)
        if not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(
                f"instance_mask must be an integer array (0 = background); "
                f"got dtype {mask.dtype}"
            )
        if expected_shape is not None and mask.shape != expected_shape:
            raise ValueError(
                f"instance_mask shape {mask.shape} does not match depth "
                f"shape {expected_shape}"
            )
        if expected_size is not None and mask.size != expected_size:
            raise ValueError(
                f"instance_mask has {mask.size} elements but the point cloud "
                f"has {expected_size} points"
            )
        return mask

    def _infer_instances(
        self,
        xyz_flat: np.ndarray,
        labels_flat: np.ndarray,
        valid_flat: np.ndarray,
        request: dict,
    ) -> dict:
        """Shared per-instance batching for the scene actions.

        Splits the flat point set by instance id (> 0), skips instances with
        fewer than min_object_points valid points, runs one batched planner
        call, and packs parallel-list results (msgpack rejects int map keys).
        """
        params = self._parse_sweep_params(request)
        knobs = self._parse_planner_kwargs(request)
        min_object_points = int(request.get("min_object_points", 100))
        sampler = self._get_sampler_from_params(params)

        candidate_ids = np.unique(labels_flat[valid_flat & (labels_flat > 0)])
        instance_pcs: list[np.ndarray] = []
        instance_ids: list[int] = []
        skipped_ids: list[int] = []
        for inst_id in candidate_ids.tolist():
            pts = xyz_flat[valid_flat & (labels_flat == inst_id)]
            if len(pts) < min_object_points:
                skipped_ids.append(int(inst_id))
                continue
            instance_ids.append(int(inst_id))
            instance_pcs.append(np.ascontiguousarray(pts, dtype=np.float32))

        t0 = time.monotonic()
        batch_results = run_planner_on_batch(instance_pcs, sampler, **knobs)
        infer_ms = (time.monotonic() - t0) * 1000.0

        out_ids: list[int] = []
        out_grasps: list[np.ndarray] = []
        out_conf: list[np.ndarray] = []
        out_tags: list[list[str]] = []
        for inst_id, (grasps, conf, tags, _) in zip(instance_ids, batch_results):
            if len(grasps) == 0:
                skipped_ids.append(inst_id)
                continue
            out_ids.append(inst_id)
            out_grasps.append(np.asarray(grasps, dtype=np.float32).reshape(-1, 4, 4))
            out_conf.append(np.asarray(conf, dtype=np.float32).reshape(-1))
            out_tags.append([str(t) for t in tags])

        return {
            "instance_ids": np.asarray(out_ids, dtype=np.int32),
            "grasps": out_grasps,
            "confidences": out_conf,
            "branch_tags": out_tags,
            "skipped_instance_ids": np.asarray(sorted(skipped_ids), dtype=np.int32),
            "timing": {"infer_ms": float(infer_ms)},
        }

    def _handle_infer(self, request: dict) -> dict:
        gripper_name = request.get("gripper_name") or self.default_gripper
        if not gripper_name:
            raise ValueError(
                "Request omitted 'gripper_name' and no default_gripper is set."
            )
        point_cloud = request.get("point_cloud")
        if point_cloud is None:
            raise ValueError("Request is missing 'point_cloud'.")
        # msgpack-numpy decodes arrays over an immutable byte buffer. Make the
        # ownership explicit before torch.from_numpy consumes it downstream.
        pc = np.array(point_cloud, dtype=np.float32, order="C", copy=True)
        if pc.ndim != 2 or pc.shape[1] != 3:
            raise ValueError(f"point_cloud must be (N, 3); got {pc.shape}")

        num_grasps = int(request.get("num_grasps", 100))
        grasp_threshold = float(request.get("grasp_threshold", -1.0))
        topk_num_grasps = int(request.get("topk_num_grasps", 100))

        sampler = self._get_sampler(gripper_name)

        t0 = time.monotonic()
        grasps, confidences = GraspGenXSampler.run_inference(
            pc,
            sampler,
            grasp_threshold=grasp_threshold,
            num_grasps=num_grasps,
            topk_num_grasps=topk_num_grasps,
        )
        infer_ms = (time.monotonic() - t0) * 1000.0

        grasps_np = (
            grasps.detach().cpu().numpy().astype(np.float32)
            if len(grasps)
            else np.zeros((0, 4, 4), dtype=np.float32)
        )
        conf_np = (
            confidences.detach().cpu().numpy().astype(np.float32)
            if len(confidences)
            else np.zeros((0,), dtype=np.float32)
        )

        return {
            "grasps": grasps_np,
            "confidences": conf_np,
            "gripper_name": gripper_name,
            "timing": {"infer_ms": float(infer_ms)},
        }

    def _dispatch(self, request: dict) -> dict:
        action = request.get("action")
        if action == "health":
            return {"status": "ok"}
        if action == "metadata":
            return self._handle_metadata()
        if action == "infer":
            return self._handle_infer(request)
        if action == "infer_object":
            return self._handle_infer_object(request)
        if action == "infer_scene_depth":
            return self._handle_infer_scene_depth(request)
        if action == "infer_scene_pc":
            return self._handle_infer_scene_pc(request)
        raise ValueError(f"Unknown action: {action!r}")

    def serve_forever(self) -> None:
        """Bind the REP socket and serve requests until interrupted."""
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REP)
        addr = f"tcp://{self.host}:{self.port}"
        sock.bind(addr)
        logger.info("GraspGenX ZMQ server listening on %s", addr)
        try:
            while True:
                raw = sock.recv()
                try:
                    request = msgpack.unpackb(raw, raw=False)
                    response = self._dispatch(request)
                except Exception as exc:
                    logger.exception("Request failed")
                    response = {"error": f"{type(exc).__name__}: {exc}"}
                sock.send(msgpack.packb(response, use_bin_type=True))
        except KeyboardInterrupt:
            logger.info("Shutting down ZMQ server (KeyboardInterrupt).")
        finally:
            sock.close(linger=0)
