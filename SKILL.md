---
name: graspgenx
description: >-
  Run and operate GraspGenX — a cross-embodiment foundation model for 6-DOF grasp
  generation. Use when installing GraspGenX (uv or Docker), running inference
  demos (segmented object point cloud, scene point cloud, or object mesh),
  choosing a gripper or planner (graspmoe vs diffusion), starting or calling the
  ZMQ client-server, driving grasping through the MCP server, integrating a new
  gripper (config wizard / sweep-volume params), or wiring grasps into an
  end-to-end cuRobo + Newton pick-and-place pipeline. Also covers checkpoint/asset
  setup and the depth-quality requirement for real-world use.
---

# GraspGenX

GraspGenX is a **cross-embodiment** grasp model: one model, conditioned on a
gripper's *swept volume*, predicts 6-DOF grasps for **any** gripper — including
grippers unseen in training. Output is a set of `(K, 4, 4)` SE(3) grasp poses
plus `(K,)` confidence scores; higher score = better, and visualizers color
grasps red→green by score.

**Repo layout (what lives where):**

| Path | Purpose |
|------|---------|
| `graspgenx/` | The package: model, samplers, `serving/` (ZMQ), `x_grippers.py`, utils |
| `scripts/` | Demos, `list_grippers.py`, `gripper_config_wizard.py`, client-server examples |
| `client-server/` | ZMQ inference server + CLI client |
| `mcp/` | MCP server bridging LLM tool-calls → the ZMQ server |
| `end2end/` | GraspGenX → cuRobo → Newton/MuJoCo pick-and-place pipeline |
| `assets/` | `sample_data/`, `proc_grippers/` |
| `ext/` | Auto-cloned checkpoints + gripper descriptions (see below) |

**Convention:** run every command with `uv run python …` from the repo root.

## Checkpoints & gripper assets (automatic)

On the **first `import graspgenx`**, two Hugging Face repos are shallow-cloned
(needs `git` + `git-lfs` on PATH) into `ext/`:

- Model checkpoints → `ext/graspgenx_checkpoints/release/{gen,dis}/`
  (each has `config.yaml` + `epoch_*.pth`).
- Gripper descriptions → `ext/gripper_descriptions/`.

Override the locations if you already have them elsewhere:

```bash
export GRASPGENX_CHECKPOINT_DIR=/path/to/graspgenx_checkpoints
export GRASPGENX_GRIPPER_CFG_DIR=/path/to/gripper_descriptions   # must already exist
```

Verify the assets resolve:

```bash
uv run python -c "from graspgenx import get_gripper_descriptions_root; print(get_gripper_descriptions_root())"
uv run python scripts/list_grippers.py        # all available grippers
```

---

## 1. Installation

Pick **uv** for inference, **Docker** for training (Docker also does inference).

### uv (recommended for inference)

```bash
git clone https://github.com/NVlabs/GraspGenX.git && cd GraspGenX && uv sync
```

Python >= 3.10. `pip install -e .` also works inside a conda or venv. Add the
`fast-inference` extra only for the native C++ runtime.

**Optional extras** (add only what you need):

| Command | Adds |
|---------|------|
| `uv sync --extra serve` | ZMQ client-server (`pyzmq`, `msgpack`, `msgpack-numpy`) |
| `uv sync --extra fast-inference` | Build dependencies for the native C++ runtime |
| `uv sync --extra training` | Training and dataset utilities |
| `uv sync --extra visualization` | Viser demos and offline rendering |
| `uv sync --extra end2end` | cuRobo + Newton/MuJoCo + CoACD + USD + fcl (pick-and-place) |
| `uv sync --extra dev` | `black`, `isort`, `flake8` |

### Native inference

Build the shared library and plans on the target GPU:

```bash
uv run --quiet --extra fast-inference graspgenx-install-fast-inference \
    --diffusion-grasps 100 --obb-grasps 50
```

The optimized runtime is implemented in `cpp_runtime/`; Python calls its C ABI.
It uses DDPM20, 100 diffusion + 50 OBB candidates, and capacities
`[3500,2048,1024,512,512]` by default. Custom counts and
`--ptv3-capacities INPUT POOL1 POOL2 POOL3 POOL4` require matching plans.
See [Fast inference](FAST_INFERENCE.md) for the supported input and execution
contract. Add `--fast-inference --num_grasps 100` to the point-cloud demos to
use the plans built above.

### Docker (training + inference)

```bash
bash docker/build.sh          # builds image  x_grasp:3.0  and  x_grasp:latest
```

Base image `nvcr.io/nvidia/pytorch:25.03-py3`, `WORKDIR /code/`. The image has
**no CMD/ENTRYPOINT and there is no docker-compose file** — mount the repo and
run commands yourself:

```bash
docker run --gpus all -v "$PWD":/code -it x_grasp:latest bash
# inside: python scripts/demo_object_pc.py ...   (uv is not required in Docker)
```

For headless rendering (no display) in Docker or over SSH, prefix commands with
`PYOPENGL_PLATFORM=egl`.

---

## 2. Inference inputs and controls

Every inference picks three things:

1. **Input modality** — segmented object point cloud, full scene point cloud, or
   object mesh.
2. **`--gripper_name`** — one of 26 built-in grippers (see `list_grippers.py`):
   `robotiq_2f_85`, `robotiq_2f_140`, `robotiq_3f`, `robotiq_hande`,
   `franka_panda`, `franka_umi`, `inspire_hand`, `surge_hand`, `barrett_hand`,
   `unitree_g1`, `galaxea_g1`, `ezgripper`, `arx_x5`, `abb_yumi`, `bd_spot`,
   `dh_ag95`, `fetch_robot`, `onrobot_RG2`, `onrobot_RG6`, `piper_hand`,
   `sawyer_hand`, `schunk_wsg50`, `sharpa_wave`, `tesollo_delto2f`, `wuji_hand`,
   `xarm_hand`. (Plus any gripper you add — see §6.)
3. **`--planner`** — `graspmoe` (default; diffusion ∪ oriented-bounding-box
   grasps, all scored by the discriminator) or `diffusion` (diffusion sampler
   only, faster).

**Visualization:** demos serve an interactive [viser](https://viser.studio/) GUI
at **`http://localhost:8080`** (open it in a browser; click *Next* to cycle
objects, top grasp shown in blue). The port is **fixed at 8080** — there is *no*
`--port` flag on the demo scripts; if 8080 is taken, the actual port is printed
in the log. Over SSH, forward it: `ssh -N -L 8080:localhost:8080 <host>`.

Sample inputs under `assets/sample_data/`: `real_world/{00,01}/` (depth + seg +
meta scenes), `object_pc/*.json`, `object_mesh/{banana,box}.obj`,
`hope_objects/*.obj`.

---

## 3. Inference demos

### `demo_object_pc.py` — segmented object point clouds

Runs inference on one pre-segmented object at a time.

```bash
uv run python scripts/demo_object_pc.py \
    --sample_data_dir assets/sample_data/real_world \
    --gripper_name robotiq_2f_85 \
    --plot_top_mesh
```

`--gripper_name` accepts **multiple** grippers (compare side by side). Common
flags: `--planner {graspmoe,diffusion}`, `--grasp_threshold 0.7` (`-1.0` = keep
top-k instead), `--num_grasps 200`, `--return_topk --topk_num_grasps 100`,
`--plot_top_mesh` (gripper mesh at the top grasp), `--moe_obb_density
{sparse,dense,dense-topandside}` (default `dense-topandside`).

### `demo_scene_pc.py` — objects within a full scene

Loads a full-scene point cloud with per-object segmentation, runs inference on
every object, and **filters grasps that collide with the rest of the scene**.

```bash
uv run python scripts/demo_scene_pc.py \
    --sample_data_dir assets/sample_data/real_world \
    --gripper_name robotiq_2f_85 \
    --scene 00
```

Collision filtering is **on by default**; disable with `--no-filter_collisions`.
`--scene 00` restricts to a single `real_world` scene. `--plot_top_mesh` is on by
default here (`--no-plot_top_mesh` to disable). Add `--fast-inference` to use
the shared native runtime, with `--num_grasps` matching the installed plans.

### `demo_object_mesh.py` — object meshes

Samples surface points from a mesh, runs inference, visualizes mesh + grasps.

```bash
uv run python scripts/demo_object_mesh.py \
    --mesh_file assets/sample_data/object_mesh/banana.obj \
    --mesh_scale 1.0 \
    --gripper_name inspire_hand \
    --grasp_threshold -1.0 --return_topk --topk_num_grasps 100 --plot_top_mesh
```

Formats: `.obj`, `.stl`, `.ply`, USD (`.usd/.usda/.usdc/.usdz`). For
**headless/batch** (no viewer) that writes grasps to a YAML in the Isaac grasp
format:

```bash
uv run python scripts/demo_object_mesh.py --mesh_file <mesh> --gripper_name <g> \
    --no-visualization --output_file /tmp/grasps.yml
```

### Other inference entrypoints

- `scripts/batch_inference_scene.py` — headless, one batched pass over a scene,
  optional `.npz` output (`--output_file`).
- `scripts/inference_xgrasp.py` — Hydra-driven large-scale eval to H5 (config
  overrides, not argparse).

---

## 4. MCP server (LLM-driven grasping)

`mcp/` is a lightweight bridge so an LLM (Claude Desktop, Cursor, VS Code) can
call GraspGenX as tools. It needs **no CUDA and no model weights** — it forwards
requests over ZMQ to a running GraspGenX server:

```
LLM  ──MCP (stdio)──▶  mcp/ bridge  ──ZMQ (tcp)──▶  GraspGenX server (GPU)
```

**Prerequisite:** the ZMQ server (§5) must already be running.

Install and register:

```bash
cd mcp && uv venv --python 3.10 .venv && source .venv/bin/activate && uv pip install -e .
```

Tools exposed: `generate_grasps_from_mesh`, `generate_grasps_from_point_cloud`,
`generate_grasps_from_sweep_volume` (params-only, no server assets),
`visualize_grasps`, `graspgenx_health_check`, `graspgenx_server_info`. Being
cross-embodiment, the grasp tools take a `gripper_name` per call.

Client config (Cursor `.cursor/mcp.json` shown; Claude Desktop / VS Code are the
same shape — see `mcp/README.md` for all three):

```json
{
  "mcpServers": {
    "graspgenx": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/GraspGenX/mcp", "mcp-server-graspgenx"],
      "env": { "GRASPGENX_HOST": "localhost", "GRASPGENX_PORT": "5556" }
    }
  }
}
```

---

## 5. ZMQ client-server

Serve GraspGenX once (model loaded on GPU) and let any process — same machine or
across the network, no GPU/torch needed client-side — request grasps.

### Start the server

```bash
uv sync --extra serve        # one-time: ZMQ deps
uv run python client-server/graspgenx_server.py \
    --config ext/graspgenx_checkpoints/release \
    --assets_dir assets \
    --port 5556 \
    --default_gripper franka_panda        # optional; used when a client omits gripper_name
```

`--config` is the checkpoint **root** directory (the one containing `gen/` and
`dis/`), *not* the inner `config.yaml`. Add `--fast-inference` to delegate the
complete optimized pipeline to the native C++ runtime. The
model loads eagerly at startup; per-gripper samplers load lazily and are cached.

### Wire actions (four modes)

| Action | Input | Gripper spec | Threshold/top-k |
|--------|-------|--------------|-----------------|
| `infer` | object point cloud | `gripper_name` (name-based) | server-side |
| `infer_object` | segmented object PC | sweep-volume params (12 numbers) | client-side |
| `infer_scene_depth` | depth (m) + intrinsics + instance mask | sweep-volume params | client-side |
| `infer_scene_pc` | scene PC + instance mask | sweep-volume params | client-side |

The three sweep-volume modes return **all** generated grasps with scores and
reject `gripper_name`/`grasp_threshold`/`topk` on the wire — the client applies
thresholding/top-k. `health` → `{"status":"ok"}`; `metadata` → default/loaded
grippers + model config.

### CLI client

```bash
# General CLI client (mesh or point cloud → grasps, optional viser):
uv run python client-server/graspgenx_client.py \
    --mesh_file assets/sample_data/object_mesh/box.obj --gripper_name franka_panda \
    --host localhost --port 5556 --visualize
```

### Python API

```python
from graspgenx.serving import GraspGenXClient, SweepVolumeParams

with GraspGenXClient(host="localhost", port=5556) as client:
    print(client.server_metadata)
    grasps, conf = client.infer(point_cloud, gripper_name="franka_panda")          # (K,4,4), (K,)
    # params-only (no server assets for the gripper):
    params = SweepVolumeParams.from_gripper_config("franka_panda")                  # or build 12 numbers directly
    grasps, conf = client.infer_object(object_pc, params, planner="graspmoe")
    per_instance = client.infer_scene_pc(scene_pc, instance_mask, params)          # {id: (grasps, conf)}
```

`infer_scene_depth` / `infer_scene_pc` return a `{instance_id: (grasps, conf)}`
dict; pass `return_branch_tags=True` to also get per-grasp `"diff"`/`"obb"` tags.

---

## 6. Integrating a new gripper

GraspGenX runs zero-shot on any gripper given a URDF + meshes. The only thing it
needs beyond a stock URDF is a `config.json` describing the open/close joint
states and the **swept-volume boxes** that condition the model. An interactive
wizard generates it.

### Run the wizard

```bash
uv run python scripts/gripper_config_wizard.py \
    --urdf /path/to/your/gripper.urdf \
    --name <gripper_name> \
    --port 8081
```

Open `http://localhost:8081` and step through six panels (click *Confirm* to
advance): **1** align base frame (`+Z` = approach along the fingers, `+X` =
closing direction) → **2** confirm fully-open joint pose → **3** drag the blue
box to enclose the inner finger volume at open → **4** set the closed pose, drag
the orange box for the half-open volume → **5** review the open↔close animation
→ **6** pick gripper type + symmetry and save.

### What gets written

Into `<gripper_descriptions_root>/…/assets/x_grippers/<name>/` (root resolved via
`GRASPGENX_GRIPPER_CFG_DIR`, else `ext/gripper_descriptions/`):

- `gripper.urdf` + `meshes/` — copied from your source.
- `vis_mesh.obj` — merged visual mesh at the open pose.
- `config.json` — the model conditioning. Key fields:

```jsonc
{
  "open":  { "<joint>": <float>, ... },     // fully-open joint state
  "close": { "<joint>": <float>, ... },     // closed joint state
  "fingertip": [x, y, z],                    // fingertip point (z = depth)
  "sweep_volume": {
    "extents":  [ex, ey, ez], "offset":  [ox, oy, oz],   // open box
    "extents2": [ex, ey, ez], "offset2": [ox, oy, oz]    // half-open box
  },
  "type": "parallel_2f",   // parallel_2f=0, revolute_2f=1, revolute_3f=2
  "base_rotation": [[4x4]], "symmetric": true, "bbox": [...], "standoff": [...]
}
```

Collision mesh, point-cloud, and TSDF caches are generated lazily on first use —
nothing else to run.

### Verify, then use it

```bash
uv run python scripts/vis_gripper_desc.py --gripper <gripper_name> --port 8081   # animates open↔close + boxes
uv run python scripts/demo_object_pc.py \
    --sample_data_dir assets/sample_data/real_world \
    --gripper_name <gripper_name> --plot_top_mesh
```

**Params-only alternative:** you can skip on-disk assets entirely and feed the
gripper as 12 raw sweep-volume numbers at inference time via `SweepVolumeParams`
(the `infer_object`/scene actions in §5, or the MCP
`generate_grasps_from_sweep_volume` tool).

If you onboard a gripper not already in the shared set, please upstream the saved
`assets/x_grippers/<name>/` directory via a PR.

---

## 7. End-to-end: grasps → cuRobo motion planning → sim

`end2end/` wires GraspGenX into a closed-loop pick-and-place pipeline:

**GraspGenX** grasps per object → **cuRobo** plans a collision-free
approach→grasp→lift trajectory to a reachable, collision-free grasp →
**Newton/MuJoCo** replays it under gravity + contacts → renders **MP4** + exports
**USD** for Isaac Sim / Omniverse.

### Install

```bash
uv sync --extra end2end                 # cuRobo, Newton/MuJoCo, CoACD, USD, fcl
python end2end/setup_end2end_deps.py    # clones cuRobo source into ext/curobo, builds merged URDFs
```

### Run a demo (from repo root, headless)

```bash
PYOPENGL_PLATFORM=egl PYGLET_HEADLESS=true uv run python end2end/e2e_grasp_demo.py \
  --robot_config end2end/robots/franka_panda.yaml \
  --env_config   end2end/envs/single_bin_demo.yaml \
  --task clutter_pick_and_drop --playback_mode dynamic --no-viser \
  --num_grasps 200 --topk 80 --grasp_threshold 0.7 --planner graspmoe \
  --seed 0 --export-trajectory end2end/runs/franka_single/trajectory.json
```

Four demos ship: Franka single-object → bin (above), Franka 3-object clutter →
bin (`envs/franka_clutter3_demo.yaml`, add `--max_retries_per_object 2`), and
UR10e pick-and-lift with an arx_x5 (`robots/ur10e_arx_x5.yaml`) or a Robotiq
2F-140 (`robots/ur10e_robotiq_2f_140.yaml`, the script default), both on
`envs/tabletop_single_nobin.yaml` with `--mesh_file … --hold_after_close_frames
150`. Each writes a `trajectory.json`; render it decoupled from the sim:

```bash
PYOPENGL_PLATFORM=egl uv run python end2end/render_trajectory_mp4.py \
  --trajectory end2end/runs/franka_single/trajectory.json \
  --output end2end/runs/franka_single/demo.mp4 --resolution 320x240 --no-texture
PYOPENGL_PLATFORM=egl uv run python end2end/export_trajectory_usd.py \
  --trajectory end2end/runs/franka_single/trajectory.json \
  --output end2end/runs/franka_single/demo.usda
```

### How the grasps drive cuRobo

The bridge between a GraspGenX grasp and a cuRobo plan is small and reusable:

1. **Frame convert.** A predicted grasp is a world-frame `(4,4)` pose in the
   *canonical grasp frame*. Each robot YAML supplies a `grasp_to_tool_transform`
   (grasp frame → the arm's tool0/TCP frame); apply it so cuRobo targets the
   tool link, not the grasp point. (See `robot_profiles.py`,
   `end2end/visualize_scene_grasps.py`.)
2. **Collision-filter first.** Score each grasp's *gripper mesh* against the
   table/bin/neighbors/target with fcl and **hand cuRobo only the collision-free
   grasps** — cuRobo never plans to a geometrically colliding grasp.
3. **Plan.** Build the goal with `curobo_compat.grasp_goals(...)` (returns a
   `curobo.types.Pose` or, on the lab fork, a `GoalToolPose`) and call
   `MotionPlanner.plan_grasp(...)`. Pass **all** top-K feasible grasps at once
   and cuRobo picks the most reachable; it returns approach→grasp→lift (falling
   back to approach+grasp when no feasible retraction exists). `curobo_compat.py`
   papers over the two `plan_grasp` signatures.

Minimal shape of the handoff (see `e2e_grasp_demo.py` / `clutter_task.py`):

```python
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo_compat import grasp_goals

# grasps_world: (K,4,4) from GraspGenX, already filtered to collision-free
tool_poses = [world_grasp @ grasp_to_tool_transform for world_grasp in grasps_world]
goal = grasp_goals(tool_poses, target_link=tool_frame)   # Pose / GoalToolPose
result = planner.plan_grasp(goal, start_state)           # approach → grasp → lift
```

### ⚠️ Real-world requirements & limitations

The demos use complete mesh-sampled or clean simulated point clouds. Deploying
on a real robot adds several requirements, and each is a source of error that
bounds grasp performance:

- **Calibrated camera.** You need a camera with known **intrinsics** (focal
  length, principal point — for back-projecting depth to a metric point cloud)
  and **extrinsics** (the camera→robot-base transform — so planned grasps land in
  the robot's frame). `infer_scene_depth` takes intrinsics directly; extrinsics
  are what put the grasp where cuRobo can reach it.

- **Good depth.** Grasp quality is bounded by depth quality; noisy or incomplete
  depth (reflective/transparent/thin objects, stereo holes) degrades grasps. Feed
  it good depth via **[FoundationStereo](https://github.com/NVlabs/FoundationStereo)**
  (learned stereo → dense accurate depth from a stereo pair) or a **LIDAR /
  time-of-flight** depth camera for direct high-quality range data.

- **Instance segmentation of the target.** The model consumes a *segmented*
  object point cloud (or a per-object instance mask). Segment the target with an
  instance segmenter such as **SAM2 or SAM3**, then use the mask to carve the
  object's points out of the scene cloud — that segmented cloud is the input to
  `infer` / `infer_object` (or feed the mask to `infer_scene_depth` /
  `infer_scene_pc`).

- **Downstream execution error.** Even with perfect grasps, **camera-calibration
  error** (intrinsics/extrinsics) and **robot control / execution error** (joint
  tracking, TCP calibration, gripper timing) shift where the gripper actually
  arrives versus where the grasp was predicted — both degrade real-world success
  independent of the model. Budget for them when a sim-perfect grasp misses on
  hardware.

---

## Debugging & gotchas

- **Two-process order for MCP/serving:** start the **ZMQ server first**, then the
  MCP bridge or example client. Use `graspgenx_health_check` (MCP) or
  `--host/--port` to confirm reachability.
- **Viewer:** demos are fixed to viser `:8080` (no `--port` flag); forward the
  port over SSH.
- **Planner choice:** `graspmoe` (default) is highest quality; `diffusion` is
  faster.
- **`grasp_threshold`:** `0.7` keeps confident grasps; `-1.0` disables the
  threshold and returns top-k instead.
- **Tests:** `uv run --no-sync pytest -m "not end2end"` (fast);
  `uv run pytest tests/test_end2end_demos.py -m end2end -v -s` (slow, GPU).
