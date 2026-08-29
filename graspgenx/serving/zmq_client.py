# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin ZMQ REQ client for :class:`GraspGenXZMQServer`.

This module deliberately has no dependency on torch, the model weights, or any
gripper asset — it's a pure msgpack/ZMQ wire-protocol shim. See
:mod:`graspgenx.serving.zmq_server` for the protocol.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import msgpack
import msgpack_numpy
import numpy as np
import zmq

from graspgenx.serving.types import SweepVolumeParams
from graspgenx.utils.logging_config import get_logger

msgpack_numpy.patch()

logger = get_logger(__name__)

# Planner knobs forwarded verbatim to the server (validated there against
# graspgenx.samplers.run_planner_on_object defaults).
_PLANNER_KNOB_KEYS = frozenset(
    {
        "moe_num_yaws",
        "moe_z_offsets_cm",
        "moe_outlier_threshold",
        "moe_outlier_k",
        "moe_obb_mode",
        "moe_skip_obb_rule",
        "moe_obb_density",
        "moe_obb_position_spacing_cm",
    }
)

SweepVolumeParamsLike = Union[SweepVolumeParams, dict, np.ndarray, list, tuple]


def _planner_payload(planner: str, planner_kwargs: dict) -> dict:
    unknown = set(planner_kwargs) - _PLANNER_KNOB_KEYS
    if unknown:
        raise TypeError(
            f"Unknown planner kwargs: {sorted(unknown)}. "
            f"Allowed: {sorted(_PLANNER_KNOB_KEYS)}"
        )
    payload = {"planner": str(planner)}
    payload.update(planner_kwargs)
    return payload


def _apply_threshold_topk(
    grasps: np.ndarray,
    confidences: np.ndarray,
    tags: list,
    grasp_threshold: float,
    topk_num_grasps: int,
):
    """Client-side selection: the server returns ALL generated grasps; the
    score threshold and top-k cap are applied here."""
    if grasp_threshold > 0.0:
        keep = confidences >= grasp_threshold
        grasps, confidences = grasps[keep], confidences[keep]
        tags = [t for t, k in zip(tags, keep) if k]
    if topk_num_grasps is not None and topk_num_grasps > 0:
        order = np.argsort(-confidences)[:topk_num_grasps]
        grasps, confidences = grasps[order], confidences[order]
        tags = [tags[i] for i in order] if tags else tags
    return grasps, confidences, tags


class GraspGenXClient:
    """ZMQ REQ client that round-trips msgpack payloads to a GraspGenX server.

    Usage::

        with GraspGenXClient(host="localhost", port=5556) as client:
            print(client.server_metadata)
            grasps, confidences = client.infer(
                point_cloud=xyz, gripper_name="franka_panda",
            )

    Args:
        host: Server hostname (default ``localhost``).
        port: Server port (default ``5556``).
        timeout_ms: Per-request send/recv timeout in milliseconds. ``None``
            disables timeouts (the request blocks until the server replies).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5556,
        timeout_ms: Optional[int] = 60_000,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._ctx: Optional[zmq.Context] = None
        self._sock: Optional[zmq.Socket] = None
        self._metadata_cache: Optional[dict] = None

    @property
    def address(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    def __enter__(self) -> "GraspGenXClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:
        if self._sock is not None:
            return
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        if self.timeout_ms is not None:
            self._sock.setsockopt(zmq.RCVTIMEO, int(self.timeout_ms))
            self._sock.setsockopt(zmq.SNDTIMEO, int(self.timeout_ms))
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self.address)
        logger.info("Connected to GraspGenX server at %s", self.address)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None

    def _request(self, payload: dict) -> dict:
        if self._sock is None:
            self.connect()
        assert self._sock is not None
        try:
            self._sock.send(msgpack.packb(payload, use_bin_type=True))
            raw = self._sock.recv()
        except zmq.error.Again as exc:
            # Socket is in a bad state after a timeout — reset so callers can retry.
            self.close()
            raise TimeoutError(
                f"GraspGenX server at {self.address} did not respond within {self.timeout_ms} ms"
            ) from exc
        response = msgpack.unpackb(raw, raw=False)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GraspGenX server error: {response['error']}")
        return response

    @property
    def server_metadata(self) -> dict:
        """Cached metadata response. Re-fetched once per client lifetime."""
        if self._metadata_cache is None:
            self._metadata_cache = self._request({"action": "metadata"})
        return self._metadata_cache

    def health(self) -> dict:
        return self._request({"action": "health"})

    def infer(
        self,
        point_cloud: np.ndarray,
        gripper_name: Optional[str] = None,
        num_grasps: int = 100,
        grasp_threshold: float = -1.0,
        topk_num_grasps: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Send a point cloud + gripper to the server, return ranked grasps.

        Returns:
            (grasps, confidences) — grasps is (K, 4, 4) float32, confidences
            is (K,) float32. Both arrays may be empty if the model produced
            no above-threshold grasps.
        """
        pc = np.asarray(point_cloud, dtype=np.float32)
        if pc.ndim != 2 or pc.shape[1] != 3:
            raise ValueError(f"point_cloud must be (N, 3); got {pc.shape}")

        payload = {
            "action": "infer",
            "point_cloud": pc,
            "num_grasps": int(num_grasps),
            "grasp_threshold": float(grasp_threshold),
            "topk_num_grasps": int(topk_num_grasps),
        }
        if gripper_name is not None:
            payload["gripper_name"] = gripper_name

        response = self._request(payload)
        grasps = np.asarray(response["grasps"], dtype=np.float32)
        confidences = np.asarray(response["confidences"], dtype=np.float32)
        return grasps, confidences

    def infer_object(
        self,
        point_cloud: np.ndarray,
        sweep_volume_params: SweepVolumeParamsLike,
        planner: str = "graspmoe",
        num_grasps: int = 100,
        grasp_threshold: float = -1.0,
        topk_num_grasps: int = 100,
        return_branch_tags: bool = False,
        **planner_kwargs,
    ):
        """Mode 1: segmented object PC + sweep-volume params → ranked grasps.

        The server returns every generated grasp; ``grasp_threshold`` and
        ``topk_num_grasps`` are applied locally (client-side).

        Returns:
            (grasps (K,4,4) float32, confidences (K,) float32) in the frame of
            the input point cloud, sorted by confidence when top-k is applied.
            With ``return_branch_tags=True`` a third element is appended: a
            list of "diff" | "obb" per grasp.
        """
        pc = np.asarray(point_cloud, dtype=np.float32)
        if pc.ndim != 2 or pc.shape[1] != 3 or len(pc) == 0:
            raise ValueError(f"point_cloud must be non-empty (N, 3); got {pc.shape}")

        payload = {
            "action": "infer_object",
            "point_cloud": pc,
            "sweep_volume_params": SweepVolumeParams.coerce(
                sweep_volume_params
            ).to_dict(),
            "num_grasps": int(num_grasps),
        }
        payload.update(_planner_payload(planner, planner_kwargs))

        response = self._request(payload)
        grasps, confidences, tags = _apply_threshold_topk(
            np.asarray(response["grasps"], dtype=np.float32),
            np.asarray(response["confidences"], dtype=np.float32),
            list(response.get("branch_tags", [])),
            float(grasp_threshold),
            int(topk_num_grasps),
        )
        if return_branch_tags:
            return grasps, confidences, tags
        return grasps, confidences

    def infer_scene_depth(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        instance_mask: np.ndarray,
        sweep_volume_params: SweepVolumeParamsLike,
        planner: str = "graspmoe",
        min_object_points: int = 100,
        num_grasps: int = 100,
        grasp_threshold: float = -1.0,
        topk_num_grasps: int = 100,
        return_branch_tags: bool = False,
        **planner_kwargs,
    ) -> Dict[int, tuple]:
        """Mode 2: depth image + intrinsics + instance mask → per-instance grasps.

        Args:
            depth: (H, W) float32 depth in **meters** (<= 0 / NaN = invalid).
            intrinsics: (3, 3) pinhole camera matrix.
            instance_mask: (H, W) integer instance ids; 0 = background/ignore.

        Returns:
            {instance_id: (grasps (Ki,4,4), confidences (Ki,))} in the
            **camera frame**. The server returns every generated grasp;
            ``grasp_threshold``/``topk_num_grasps`` are applied locally per
            instance. Instances with too few points, no grasps, or nothing
            above threshold are absent. With ``return_branch_tags=True`` each
            value gains a third element: the per-grasp branch tags.
        """
        depth = np.asarray(depth)
        if depth.ndim != 2:
            raise ValueError(f"depth must be (H, W); got shape {depth.shape}")
        if not np.issubdtype(depth.dtype, np.floating):
            raise ValueError(
                f"depth must be float meters; got dtype {depth.dtype}"
            )
        K = np.asarray(intrinsics, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"intrinsics must be (3, 3); got shape {K.shape}")
        mask = np.asarray(instance_mask)
        if mask.shape != depth.shape:
            raise ValueError(
                f"instance_mask shape {mask.shape} does not match depth "
                f"shape {depth.shape}"
            )

        payload = {
            "action": "infer_scene_depth",
            "depth": depth.astype(np.float32),
            "intrinsics": K,
            "instance_mask": mask,
            "sweep_volume_params": SweepVolumeParams.coerce(
                sweep_volume_params
            ).to_dict(),
            "min_object_points": int(min_object_points),
            "num_grasps": int(num_grasps),
        }
        payload.update(_planner_payload(planner, planner_kwargs))
        return self._unpack_scene_response(
            self._request(payload),
            return_branch_tags,
            float(grasp_threshold),
            int(topk_num_grasps),
        )

    def infer_scene_pc(
        self,
        point_cloud: np.ndarray,
        instance_mask: np.ndarray,
        sweep_volume_params: SweepVolumeParamsLike,
        planner: str = "graspmoe",
        min_object_points: int = 100,
        num_grasps: int = 100,
        grasp_threshold: float = -1.0,
        topk_num_grasps: int = 100,
        return_branch_tags: bool = False,
        **planner_kwargs,
    ) -> Dict[int, tuple]:
        """Mode 3: scene point cloud + instance mask → per-instance grasps.

        Args:
            point_cloud: (N, 3) or organized (H, W, 3) float32 points in
                meters; non-finite points are ignored.
            instance_mask: integer array with N elements (any shape
                reshapeable to the point count); 0 = background/ignore.

        Returns:
            {instance_id: (grasps (Ki,4,4), confidences (Ki,))} in the same
            frame as the input point cloud. Thresholding/top-k are applied
            locally, as in :meth:`infer_scene_depth`.
        """
        pc = np.asarray(point_cloud, dtype=np.float32)
        if pc.ndim == 3 and pc.shape[-1] == 3:
            n_points = pc.shape[0] * pc.shape[1]
        elif pc.ndim == 2 and pc.shape[-1] == 3:
            n_points = pc.shape[0]
        else:
            raise ValueError(
                f"point_cloud must be (N, 3) or (H, W, 3); got shape {pc.shape}"
            )
        mask = np.asarray(instance_mask)
        if mask.size != n_points:
            raise ValueError(
                f"instance_mask has {mask.size} elements but the point cloud "
                f"has {n_points} points"
            )

        payload = {
            "action": "infer_scene_pc",
            "point_cloud": pc,
            "instance_mask": mask,
            "sweep_volume_params": SweepVolumeParams.coerce(
                sweep_volume_params
            ).to_dict(),
            "min_object_points": int(min_object_points),
            "num_grasps": int(num_grasps),
        }
        payload.update(_planner_payload(planner, planner_kwargs))
        return self._unpack_scene_response(
            self._request(payload),
            return_branch_tags,
            float(grasp_threshold),
            int(topk_num_grasps),
        )

    @staticmethod
    def _unpack_scene_response(
        response: dict,
        return_branch_tags: bool,
        grasp_threshold: float = -1.0,
        topk_num_grasps: int = -1,
    ) -> Dict[int, tuple]:
        """Reassemble the wire-level parallel lists into an id-keyed dict and
        apply the client-side threshold/top-k per instance."""
        ids = np.asarray(response["instance_ids"]).reshape(-1)
        grasps_list = response["grasps"]
        conf_list = response["confidences"]
        tags_list = response.get("branch_tags", [[]] * len(grasps_list))
        results: Dict[int, tuple] = {}
        for i, inst_id in enumerate(ids.tolist()):
            grasps, conf, tags = _apply_threshold_topk(
                np.asarray(grasps_list[i], dtype=np.float32).reshape(-1, 4, 4),
                np.asarray(conf_list[i], dtype=np.float32).reshape(-1),
                list(tags_list[i]),
                grasp_threshold,
                topk_num_grasps,
            )
            if len(grasps) == 0:
                continue  # everything below threshold for this instance
            if return_branch_tags:
                results[int(inst_id)] = (grasps, conf, tags)
            else:
                results[int(inst_id)] = (grasps, conf)
        return results
