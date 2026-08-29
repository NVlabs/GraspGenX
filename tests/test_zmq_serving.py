# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the ZMQ serving layer.

Covers the dispatch routing and the msgpack/ZMQ wire round-trip without
loading any model weights. GPU-based inference is exercised by the integration
test at the bottom (skipped unless ``GRASPGENX_RUN_SERVE_INTEGRATION=1``).

The whole file is skipped when the ``serve`` extras aren't installed
(``pip install graspgenx[serve]`` brings in pyzmq + msgpack + msgpack-numpy).
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

pytest.importorskip("zmq", reason="install graspgenx[serve] to exercise serving tests")
pytest.importorskip(
    "msgpack", reason="install graspgenx[serve] to exercise serving tests"
)
pytest.importorskip(
    "msgpack_numpy", reason="install graspgenx[serve] to exercise serving tests"
)


def test_serving_module_imports():
    from graspgenx.serving import GraspGenXClient, GraspGenXZMQServer

    assert GraspGenXClient is not None
    assert GraspGenXZMQServer is not None


def _stub_server():
    """Construct a GraspGenXZMQServer with __init__ bypassed.

    This skips :func:`load_model_cfg` so the dispatch logic can be exercised
    without checkpoints or CUDA. Only attributes actually read by ``_dispatch``
    / ``_handle_metadata`` are populated.
    """
    from graspgenx.serving.zmq_server import GraspGenXZMQServer

    srv = GraspGenXZMQServer.__new__(GraspGenXZMQServer)
    srv._samplers = {}
    srv._samplers_lock = threading.Lock()
    srv.default_gripper = None
    srv.assets_dir = "/tmp/assets-stub"
    return srv


def test_dispatch_health_returns_ok():
    srv = _stub_server()
    assert srv._dispatch({"action": "health"}) == {"status": "ok"}


def test_dispatch_unknown_action_raises():
    srv = _stub_server()
    with pytest.raises(ValueError, match="Unknown action"):
        srv._dispatch({"action": "bogus"})


def test_dispatch_infer_without_gripper_or_default_raises():
    srv = _stub_server()
    pc = np.zeros((10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="gripper_name"):
        srv._dispatch({"action": "infer", "point_cloud": pc})


def test_shared_native_runtime_is_propagated_to_new_sampler(monkeypatch):
    import graspgenx.serving.zmq_server as server_module

    srv = _stub_server()
    srv._fast_inference_enabled = True
    srv.cfg = object()
    srv._shared_model = object()

    class FakeSampler:
        def __init__(self, *_args, **_kwargs):
            self.using_fast_inference = False

    monkeypatch.setattr(server_module, "GraspGenXSampler", FakeSampler)

    sampler = srv._get_sampler("test_gripper")

    assert isinstance(sampler, FakeSampler)
    assert sampler.using_fast_inference


def test_round_trip_health_over_zmq():
    """End-to-end: bind a REP socket, dispatch in a worker thread, hit it with the client."""
    import msgpack
    import msgpack_numpy
    import zmq

    msgpack_numpy.patch()

    from graspgenx.serving.zmq_client import GraspGenXClient

    srv = _stub_server()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    port = sock.bind_to_random_port("tcp://127.0.0.1")
    stop = threading.Event()

    def _serve():
        while not stop.is_set():
            try:
                raw = sock.recv(flags=zmq.NOBLOCK)
            except zmq.error.Again:
                time.sleep(0.01)
                continue
            try:
                req = msgpack.unpackb(raw, raw=False)
                resp = srv._dispatch(req)
            except Exception as exc:  # noqa: BLE001
                resp = {"error": f"{type(exc).__name__}: {exc}"}
            sock.send(msgpack.packb(resp, use_bin_type=True))

    worker = threading.Thread(target=_serve, daemon=True)
    worker.start()
    try:
        with GraspGenXClient(host="127.0.0.1", port=port, timeout_ms=5_000) as client:
            assert client.health() == {"status": "ok"}
    finally:
        stop.set()
        worker.join(timeout=2.0)
        sock.close(linger=0)


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("GRASPGENX_RUN_SERVE_INTEGRATION") != "1",
    reason="Requires a live GPU + checkpoints + assets; set GRASPGENX_RUN_SERVE_INTEGRATION=1 to run.",
)
def test_end_to_end_infer_round_trip():
    """Full inference round-trip with real model weights. Off by default."""
    import msgpack
    import msgpack_numpy
    import zmq

    msgpack_numpy.patch()

    from graspgenx import get_checkpoints_version_dir
    from graspgenx.serving.zmq_client import GraspGenXClient
    from graspgenx.serving.zmq_server import GraspGenXZMQServer

    ckpt_root = str(get_checkpoints_version_dir())
    assets_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
    )

    srv = GraspGenXZMQServer(
        config_path=ckpt_root,
        assets_dir=assets_dir,
        default_gripper="franka_panda",
    )

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    port = sock.bind_to_random_port("tcp://127.0.0.1")
    stop = threading.Event()

    def _serve():
        while not stop.is_set():
            try:
                raw = sock.recv(flags=zmq.NOBLOCK)
            except zmq.error.Again:
                time.sleep(0.01)
                continue
            try:
                req = msgpack.unpackb(raw, raw=False)
                resp = srv._dispatch(req)
            except Exception as exc:  # noqa: BLE001
                resp = {"error": f"{type(exc).__name__}: {exc}"}
            sock.send(msgpack.packb(resp, use_bin_type=True))

    worker = threading.Thread(target=_serve, daemon=True)
    worker.start()
    try:
        rng = np.random.default_rng(0)
        pc = rng.normal(scale=0.05, size=(2000, 3)).astype(np.float32)
        with GraspGenXClient(host="127.0.0.1", port=port, timeout_ms=120_000) as client:
            meta = client.server_metadata
            assert meta["default_gripper"] == "franka_panda"
            grasps, confidences = client.infer(pc, num_grasps=64, topk_num_grasps=8)
            assert grasps.ndim == 3 and grasps.shape[-2:] == (4, 4)
            assert confidences.ndim == 1 and confidences.shape[0] == grasps.shape[0]
    finally:
        stop.set()
        worker.join(timeout=2.0)
        sock.close(linger=0)
