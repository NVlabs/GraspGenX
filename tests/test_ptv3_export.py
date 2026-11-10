# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate direct export of a hierarchy containing native V3 reductions."""

import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
if not hasattr(torch.onnx, "ops"):
    pytest.skip("requires fast-inference export dependencies", allow_module_level=True)
from cpp_runtime.tools.ptv3_plugins import segment_max, valid_mean


@pytest.mark.parametrize(
    "capacities", [(3500, 2048, 1024, 512, 512), (100, 200, 37, 99, 1)]
)
def test_direct_encoder_export_retains_reduction_nodes(tmp_path, capacities):
    class Encoder(torch.nn.Module):
        def forward(self, features, *topology):
            for i in range(4):
                order, ptr, count = topology[i * 3 : i * 3 + 3]
                features = segment_max(features, order, ptr, count, capacities[i + 1])
            return valid_mean(features, topology[-1])

    path = tmp_path / "encoder.onnx"
    values = [torch.zeros(capacities[0], 64, dtype=torch.float16)]
    names = ["features"]
    for i in range(4):
        values.extend(
            [
                torch.zeros(4, capacities[i], dtype=torch.int64),
                torch.zeros(capacities[i + 1] + 1, dtype=torch.int64),
                torch.zeros(1, dtype=torch.int64),
            ]
        )
        names.extend([f"order{i}", f"ptr{i}", f"count{i}"])
    values.append(torch.zeros(1, dtype=torch.int64))
    names.append("mean_count")
    inputs = tuple(values)
    torch.onnx.export(
        Encoder().eval(),
        inputs,
        path,
        dynamo=True,
        opset_version=18,
        external_data=False,
        input_names=names,
        output_names=["embedding"],
    )
    model = onnx.load(path)
    onnx.checker.check_model(model)
    plugins = [n for n in model.graph.node if n.domain == "trt.plugins"]
    assert [n.op_type for n in plugins] == ["GraspGenXSegmentMax"] * 4 + [
        "GraspGenXValidMean"
    ]
    assert model.graph.output[0].type.tensor_type.shape.dim[0].dim_value == 64
    assert all(
        any(a.name == "plugin_version" and a.s == b"1" for a in node.attribute)
        for node in plugins
    )
