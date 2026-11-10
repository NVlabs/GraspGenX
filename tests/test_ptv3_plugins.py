# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in V3 serialization, reduction parity and CUDA Graph tests."""

import ctypes
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("GRASPGENX_PLUGIN_LIBRARY"),
    reason="requires built native plugin and CUDA",
)


@pytest.mark.parametrize(
    "rows,capacity", [(100, 73), (3500, 2048), (1024, 512), (512, 512), (100, 3500)]
)
def test_reductions_roundtrip_and_capture(rows, capacity):
    import numpy as np
    import torch
    import tensorrt as trt

    handle = ctypes.CDLL(
        os.environ["GRASPGENX_PLUGIN_LIBRARY"], mode=ctypes.RTLD_GLOBAL
    )
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    shapes = [(rows, 64), (4, rows), (capacity + 1,), (1,)]
    tensors = [
        network.add_input(f"in{i}", trt.float16 if i == 0 else trt.int64, shape)
        for i, shape in enumerate(shapes)
    ]
    registry = trt.get_plugin_registry()
    plugins = []
    for name in ("GraspGenXSegmentMax", "GraspGenXValidMean"):
        creator = registry.get_creator(name, "1", "")
        assert creator is not None
        plugins.append(
            creator.create_plugin(
                name, trt.PluginFieldCollection([]), trt.TensorRTPhase.BUILD
            )
        )
    pool = network.add_plugin_v3(tensors, [], plugins[0]).get_output(0)
    pool.name = "pooled"
    mean = network.add_plugin_v3([tensors[0], tensors[3]], [], plugins[1]).get_output(0)
    mean.name = "mean"
    network.mark_output(pool)
    network.mark_output(mean)
    config = builder.create_builder_config()
    config.builder_optimization_level = 0
    blob = builder.build_serialized_network(network, config)
    assert blob is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(bytes(blob))
    context = engine.create_execution_context()
    features = torch.randn(rows, 64, device="cuda", dtype=torch.float16)
    order = torch.arange(rows - 1, -1, -1, device="cuda", dtype=torch.int64).repeat(
        4, 1
    )
    count = torch.zeros(1, device="cuda", dtype=torch.int64)
    ptr = torch.arange(capacity + 1, device="cuda", dtype=torch.int64).clamp_max(rows)
    outputs = [
        torch.empty(capacity, 64, device="cuda", dtype=torch.float16),
        torch.empty(64, device="cuda", dtype=torch.float16),
    ]
    values = [features, order, ptr, count, *outputs]
    for name, value in zip(["in0", "in1", "in2", "in3", "pooled", "mean"], values):
        assert context.set_tensor_address(name, value.data_ptr())
    stream = torch.cuda.current_stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    for valid in (0, 1, min(rows, capacity)):
        count.fill_(valid)
        graph.replay()
        expected = torch.zeros_like(outputs[0])
        expected[:valid] = features.flip(0)[:valid]
        torch.testing.assert_close(outputs[0], expected, rtol=0, atol=0)
        # Mean accumulates FP32 sequentially, then rounds once to FP16.
        reference = np.zeros(64, dtype=np.float32)
        for row in features[:valid].float().cpu().numpy():
            reference += row
        if valid:
            reference /= valid
        torch.testing.assert_close(
            outputs[1],
            torch.from_numpy(reference).to(device="cuda", dtype=torch.float16),
            rtol=0,
            atol=0,
        )

    # Builder profiling does not promise meaningful values for topology inputs.
    # Extreme counts/pointers must remain bounded by the declared tensor shapes.
    for extreme in (-(2**63), 2**63 - 1):
        count.fill_(extreme)
        ptr.fill_(extreme)
        order.fill_(extreme)
        graph.replay()
        stream.synchronize()
        assert torch.isfinite(outputs[0]).all()
        assert torch.isfinite(outputs[1]).all()
        assert torch.count_nonzero(outputs[0]) == 0
        if extreme < 0:
            assert torch.count_nonzero(outputs[1]) == 0
