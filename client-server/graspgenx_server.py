#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start a GraspGenX ZMQ inference server.

Usage:
    # Serve with default settings (assets at /code/assets):
    python client-server/graspgenx_server.py \\
        --config checkpoints/gen/config.yaml \\
        --assets_dir /code/assets

    # Pre-load a specific gripper:
    python client-server/graspgenx_server.py \\
        --config checkpoints/gen/config.yaml \\
        --assets_dir /code/assets \\
        --default_gripper franka_panda

    # Custom port:
    python client-server/graspgenx_server.py \\
        --config checkpoints/gen/config.yaml \\
        --assets_dir /code/assets \\
        --port 5557
"""

import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start a GraspGenX ZMQ inference server (cross-embodiment)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the GraspGenX model config YAML (e.g. checkpoints/gen/config.yaml)",
    )
    parser.add_argument(
        "--assets_dir",
        type=str,
        default="/code/assets",
        help="Root directory containing x_grippers/ and proc_grippers/ subdirectories (default: /code/assets)",
    )
    parser.add_argument(
        "--default_gripper",
        type=str,
        default=None,
        help="Default gripper to pre-load and use when clients omit gripper_name (e.g. franka_panda)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Address to bind the ZMQ socket (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5556,
        help="Port to bind the ZMQ socket (default: 5556)",
    )
    parser.add_argument(
        "--fast-inference",
        action="store_true",
        help="Use the native C++ fast-inference runtime "
        "(requires `uv sync --extra fast-inference`). Falls "
        "back to eager PyTorch fp32 if unavailable.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    from graspgenx.serving.zmq_server import GraspGenXZMQServer

    server = GraspGenXZMQServer(
        config_path=args.config,
        assets_dir=args.assets_dir,
        host=args.host,
        port=args.port,
        default_gripper=args.default_gripper,
        use_fast_inference=args.fast_inference,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
