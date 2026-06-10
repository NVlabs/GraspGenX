# GraspGenX Agent Guide

Cross-embodiment grasp generation framework. Single model handles any robot gripper via swept-volume conditioning.

## Quick Start

```bash
# Inference — CUDA (uv)
uv sync

# Inference — AMD ROCm
uv sync --extra rocm

# Training (Docker only, CUDA)
bash docker/build.sh

# End-to-end pipeline (cuRobo + Newton, CUDA only)
uv sync --extra end2end
python end2end/setup_end2end_deps.py  # clones cuRobo assets, builds UR10e URDF
```

## Essential Commands

```bash
# Tests (skip slow end2end)
uv run --no-sync pytest -m "not end2end"

# End-to-end tests (GPU required)
uv run pytest tests/test_end2end_demos.py -m end2end -v -s

# List available grippers
uv run python scripts/list_grippers.py

# Demo: object point cloud
uv run python scripts/demo_object_pc.py \
    --sample_data_dir assets/sample_data/real_world \
    --gripper_name robotiq_2f_85 --plot_top_mesh

# Demo: scene point cloud
uv run python scripts/demo_scene_pc.py \
    --sample_data_dir assets/sample_data/real_world \
    --gripper_name robotiq_2f_85

# Demo: object mesh
uv run python scripts/demo_object_mesh.py \
    --mesh_file assets/sample_data/object_mesh/banana.obj \
    --gripper_name inspire_hand --plot_top_mesh

# Gripper config wizard (interactive GUI)
uv run python scripts/gripper_config_wizard.py \
    --urdf /path/to/gripper.urdf --name <name> --port 8081
```

## Architecture

- `graspgenx/` — Core package (models, samplers, dataset, serving)
- `scripts/` — Demo and utility scripts
- `end2end/` — Full pick-and-place pipeline (GraspGenX → cuRobo → Newton/MuJoCo)
- `client-server/` — ZMQ server for remote inference
- `assets/` — Sample data, procedural grippers, configs
- `ext/` — Auto-cloned dependencies (gripper_descriptions, checkpoints) — gitignored

## Runtime Dependencies (Auto-Cloned)

On first import, `graspgenx` clones:
1. `gripper_descriptions` → `ext/gripper_descriptions/` (URDFs, meshes, configs)
2. `graspgenx_checkpoints` → `ext/graspgenx_checkpoints/` (model weights)

Override with env vars:
```bash
export GRASPGENX_GRIPPER_CFG_DIR=/path/to/gripper_descriptions
export GRASPGENX_CHECKPOINT_DIR=/path/to/checkpoints
```

## Environment Variables

- `PYOPENGL_PLATFORM=egl` + `PYGLET_HEADLESS=true` — Required for end2end rendering (headless GPU)
- `GRASPGENX_BIN_COLLISION` — Bin collision mode: `primitives` (default), `coacd`, `solid`

## Testing

- Pytest with markers: `integration` (GPU/server), `end2end` (slow GPU demos)
- Fixtures in `tests/conftest.py`: device, random_seed, sample_point_cloud, sample_pose
- End2end tests skip UR10e case if merged URDF not built — re-run `setup_end2end_deps.py`

## Code Style

- Black (88 char line length, target py310)
- isort (black profile)
- DCO sign-off required on all commits (`git commit -s`)

## AMD ROCm Support

The repo supports AMD GPUs via ROCm 7.2. Install with `uv sync --extra rocm`.

### What Works on ROCm

| Feature | Status | Notes |
|---------|--------|-------|
| Base inference (point clouds, meshes, scenes) | **Works** | `ptv3_vanilla` backbone is pure PyTorch |
| All demo scripts | **Works** | Standard PyTorch device API |
| ZMQ server/client | **Works** | Pure Python networking + PyTorch |
| Gripper config wizard | **Works** | Browser GUI, no GPU |
| Tests (non-end2end) | **Works** | `uv run pytest -m "not end2end"` |
| `pointnet2_ops` (PointNet++) | **Fails gracefully** | Caught at import, warning printed |

### What Does NOT Work on ROCm

| Feature | Blocker |
|---------|---------|
| End-to-end pipeline (`end2end/`) | `nvidia-curobo`, `newton`, `warp` are CUDA-only |
| Docker training | NGC base image is NVIDIA-only |
| Multi-GPU training (NCCL) | Needs RCCL; env vars may warn |

### ROCm Install

```bash
uv sync --extra rocm
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Conflict Resolution

`rocm` and `end2end` extras are mutually exclusive — `uv` enforces this automatically.

## Gotchas

- `ext/` is gitignored — auto-populated on first import
- End2end demos need `PYOPENGL_PLATFORM=egl PYGLET_HEADLESS=true` prefix
- Checkpoint path points to directory with `gen/` and `dis/` subdirs, not a config file
- Pinned deps: `diffusers==0.11.1`, `huggingface-hub==0.25.2` (compatibility)
- Training only tested in Docker (`nvcr.io/nvidia/pytorch:25.03-py3`)
- `uv.lock` is gitignored — regenerate with `uv sync`
