# GraspGenX MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server that enables LLMs (Claude, Cursor, etc.) to generate **cross-embodiment** 6-DOF robotic grasp poses using GraspGenX.

```
┌─────────────────────┐        MCP (stdio)        ┌─────────────────────┐       ZMQ (tcp)       ┌─────────────────────┐
│  LLM / AI Agent     │  ◀─── tool calls ───────▶  │  MCP Server         │  ── pc + gripper ───▶ │  GraspGenX Server   │
│  (Cursor, Claude…)  │                            │  (this package)     │  ◀── grasps ────────  │  (GPU, model loaded)│
└─────────────────────┘                            └─────────────────────┘                       └─────────────────────┘
```

The MCP server is a lightweight bridge — it requires no CUDA, no model weights, and no dependency on the `graspgenx` package (the ZMQ wire protocol is inlined). It connects to a running GraspGenX ZMQ inference server and exposes its capabilities as MCP tools that any LLM agent can call.

GraspGenX serves grippers through one shared cross-embodiment model. You name the gripper per request (`gripper_name`), or describe it directly with raw **sweep-volume parameters** — so this bridge exposes both a name-based and a params-based path.

## Available Tools

| Tool | Description |
|------|-------------|
| `generate_grasps_from_mesh` | Generate 6-DOF grasp poses from a 3D mesh file (.obj, .stl, .ply, .glb) for a named gripper. Samples a point cloud from the mesh surface and runs GraspGenX inference. |
| `generate_grasps_from_point_cloud` | Generate 6-DOF grasp poses from a point cloud file (.npy, .npz, .ply, .pcd) for a named gripper. |
| `generate_grasps_from_sweep_volume` | Generate 6-DOF grasp poses for a gripper described by **raw sweep-volume parameters** (12 numbers) instead of a named gripper — no server-side assets needed. |
| `visualize_grasps` | Generate grasps and visualize them interactively in a 3D [viser](https://viser.studio/) web viewer. Accepts a mesh or point cloud file plus a gripper name. Grasps are color-coded by confidence (green=high, red=low). |
| `graspgenx_health_check` | Check if the GraspGenX inference server is running and responsive. |
| `graspgenx_server_info` | Get metadata about the server: default gripper, loaded grippers, supported actions, model config. |

## Prerequisites

The GraspGenX ZMQ server must be running. See the [client-server/README.md](../client-server/README.md) for setup instructions.

**Quick start (local):**

```bash
# From the GraspGenX repo root:
uv run --extra serve python client-server/graspgenx_server.py \
    --config ext/graspgenx_checkpoints/release \
    --assets_dir assets \
    --default_gripper franka_panda
```

## Installation

### Using uv (recommended)

```bash
cd GraspGenX/mcp
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -e .
```

### Using pip

```bash
cd GraspGenX/mcp
pip install -e .
```

## Configuration

### Configure for Cursor

Add the following to `.cursor/mcp.json` in your workspace (or to your global Cursor settings). Make sure to edit the `--directory` entry.

```json
{
  "mcpServers": {
    "graspgenx": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/absolute/path/to/GraspGenX/mcp",
        "mcp-server-graspgenx"
      ],
      "env": {
        "GRASPGENX_HOST": "localhost",
        "GRASPGENX_PORT": "5556"
      }
    }
  }
}
```

Or if you installed with pip:

```json
{
  "mcpServers": {
    "graspgenx": {
      "command": "python",
      "args": ["-m", "mcp_server_graspgenx"],
      "env": {
        "GRASPGENX_HOST": "localhost",
        "GRASPGENX_PORT": "5556"
      }
    }
  }
}
```

### Configure for Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "graspgenx": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/absolute/path/to/GraspGenX/mcp",
        "mcp-server-graspgenx"
      ],
      "env": {
        "GRASPGENX_HOST": "localhost",
        "GRASPGENX_PORT": "5556"
      }
    }
  }
}
```

### Configure for VS Code

Add to `.vscode/mcp.json` in your workspace:

```json
{
  "mcp": {
    "servers": {
      "graspgenx": {
        "command": "uv",
        "args": [
          "run",
          "--directory", "/absolute/path/to/GraspGenX/mcp",
          "mcp-server-graspgenx"
        ],
        "env": {
          "GRASPGENX_HOST": "localhost",
          "GRASPGENX_PORT": "5556"
        }
      }
    }
  }
}
```

### Custom Server Address

If the GraspGenX ZMQ server is on a different host or port, set the environment variables:

- `GRASPGENX_HOST` — default: `localhost`
- `GRASPGENX_PORT` — default: `5556`

Or pass them as CLI arguments:

```bash
mcp-server-graspgenx --host 192.168.1.100 --port 5557
```

## Example LLM Interactions

Once configured, an LLM can naturally call GraspGenX:

> **User:** "Generate grasps for the box mesh at `/models/sample_data/meshes/box.obj` with a Franka Panda gripper"
>
> **LLM → `generate_grasps_from_mesh`:** `{"mesh_file": "/models/sample_data/meshes/box.obj", "gripper_name": "franka_panda", "mesh_scale": 1.0}`
>
> **Response:** "Generated 100 grasps. Gripper: franka_panda. Confidence range: 0.7234 – 0.9812. Top grasp at position (0.012, -0.003, 0.045) with confidence 0.9812..."

> **User:** "Which grippers has the server loaded?"
>
> **LLM → `graspgenx_server_info`**
>
> **Response:** `{"default_gripper": "franka_panda", "loaded_grippers": ["franka_panda", "robotiq_2f_140"], ...}`

> **User:** "Is the grasp server running?"
>
> **LLM → `graspgenx_health_check`**
>
> **Response:** "GraspGenX server status: ok"

## Debugging

Use the MCP inspector to test the server:

```bash
cd GraspGenX/mcp
npx @modelcontextprotocol/inspector uv run mcp-server-graspgenx
```
