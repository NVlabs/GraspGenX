# GraspGenX Standalone Server

GraspGenX can be run as a standalone ZMQ server so that any application — on the same machine or across the network — can request **cross-embodiment** 6-DOF grasp predictions without importing the model code or needing a GPU on the client side.

```
┌──────────────────────┐         ZMQ (tcp)            ┌──────────────────────┐
│   Client (any lang)  │  ── point cloud + gripper ─▶ │  GraspGenX Server    │
│   - Python / C++ / … │  ◀── grasps + scores ──────  │  - GPU, model loaded │
│   - No CUDA needed   │                              │  - One process, many │
│                      │                              │    grippers          │
└──────────────────────┘                              └──────────────────────┘
```

Unlike GraspGen (single gripper per server), GraspGenX loads a **single cross-embodiment model** and supports **any gripper** at inference time. The client specifies `gripper_name` per request; the server loads the gripper's **sweep volume v2** data from the assets directory lazily, then caches the sampler.

## Three Modes of Use

Beyond the legacy name-based `infer` action, the server exposes three
**sweep-volume-params** modes — the gripper is described by 12 raw numbers per
request, so the server needs no assets for it:

| Mode | Action / client method | Input | Output (frame) |
|------|------------------------|-------|----------------|
| 1. Segmented object PC | `infer_object` | (N,3) object point cloud + sweep-volume params | `(grasps, scores)` in the input PC's frame |
| 2. Scene depth image | `infer_scene_depth` | (H,W) depth [m] + (3,3) intrinsics + (H,W) instance mask + params | `{instance_id: (grasps, scores)}` in the **camera frame** |
| 3. Scene point cloud | `infer_scene_pc` | (N,3) or (H,W,3) point cloud [m] + instance mask + params | `{instance_id: (grasps, scores)}` in the point cloud's frame |

Instance mask semantics (modes 2–3): the pixel/point value is the instance id;
`0` means background/ignore. The server returns **all** generated grasps with
scores; thresholding/top-k happen client-side. See
[Example Scripts](#example-scripts-with-viser-visualization) for runnable,
visualized versions of each mode.

## Install

The serving layer is gated behind an optional extra so inference-only users don't pay the ZMQ/msgpack deps. On the server-side machine:

```bash
# From the GraspGenX repo root (inside the uv venv from the main install):
uv pip install pyzmq msgpack msgpack-numpy
# — or, more durably (survives uv sync):
uv sync --extra serve
```

Add `--extra fast-inference` to the `uv sync` command when the server should use
the native C++ runtime.

The client-side machine only needs `pyzmq`, `msgpack`, `msgpack-numpy`, `numpy`, and `trimesh` — no PyTorch or CUDA.

## Start the server

```bash
# Activate the GraspGenX uv venv first.
python client-server/graspgenx_server.py \
    --config <repo>/ext/graspgenx_checkpoints/release \
    --assets_dir <repo>/assets \
    --default_gripper franka_panda \
    --port 5556
```

`--config` is the **checkpoint root** containing `gen/` and `dis/` subdirectories — *not* the inner `config.yaml`. For the default checkpoint shipped with the release, that's `<repo>/ext/graspgenx_checkpoints/release/`.

`--default_gripper` is optional. If set, the gripper is pre-loaded at startup and used when a client omits `gripper_name`.

## Run the client

```bash
# Run inference from a mesh file:
python client-server/graspgenx_client.py \
    --mesh_file /path/to/box.obj --mesh_scale 1.0 \
    --gripper_name franka_panda \
    --host localhost --port 5556

# Or from a point cloud file (.pcd / .ply / .xyz / .npy):
python client-server/graspgenx_client.py \
    --pcd_file assets/sample_data/object_mesh/banana.obj \
    --gripper_name robotiq_2f_140 \
    --host localhost --port 5556

# Params-only mode 1 (no gripper assets on the server) — pass raw
# sweep-volume params from a JSON file instead of a gripper name:
python client-server/graspgenx_client.py \
    --mesh_file /path/to/box.obj \
    --sweep_volume_json /path/to/sweep_params.json \
    --host localhost --port 5556

# Add --visualize to render the result in a viser web viewer at :8080.
```

## Python Client API

```python
from graspgenx.serving import GraspGenXClient

with GraspGenXClient(host="localhost", port=5556) as client:
    # Server info (cached after first call).
    print(client.server_metadata)
    # {
    #   "default_gripper": "franka_panda",
    #   "loaded_grippers": ["franka_panda"],
    #   "model": {
    #     "generator_backbone": "ptv3vanilla",
    #     "discriminator_backbone": "ptv3vanilla",
    #     "grasp_repr": "r3_so3",
    #     "num_diffusion_iters_eval": 10,
    #   },
    #   "assets_dir": "/.../graspgenx/assets",
    # }

    # Run inference — specify the gripper for each request, or rely on default.
    grasps, confidences = client.infer(
        point_cloud,                  # (N, 3) numpy float32
        gripper_name="franka_panda",  # gripper whose sweep volume v2 conditions the model
        num_grasps=100,               # diffusion samples
        grasp_threshold=-1.0,         # -1.0 ⇒ use top-k instead of threshold
        topk_num_grasps=100,          # return top-k by confidence
    )
    # grasps:       (M, 4, 4) float32 — SE(3) poses
    # confidences:  (M,)      float32 — discriminator scores in [0, 1]

    # Try a different gripper on the same object — no server restart needed.
    grasps_rq, conf_rq = client.infer(point_cloud, gripper_name="robotiq_2f_140")
```

## Sweep-Volume-Params API (no gripper assets needed)

The released checkpoint conditions on the gripper's **sweep volume v2** — two
axis-aligned boxes in the gripper base frame (+Z = approach, +X = closing
direction) enclosing the inner finger volume at the open and half-open
states. Three actions accept those 12 numbers directly instead of a gripper
name, so the server needs **no assets** for the gripper — including grippers
outside its training set:

```python
from graspgenx.serving import GraspGenXClient, SweepVolumeParams

# From your own annotation (meters, gripper base frame) ...
params = SweepVolumeParams(
    extents_open=[0.085, 0.02, 0.046], offset_open=[0.0, 0.0, 0.11],
    extents_mid=[0.042, 0.02, 0.046],  offset_mid=[0.0, 0.0, 0.11],
    # Optional: fingertip depth (base → fingertips along +Z). Defaults to the
    # top plane of the open box (~1 cm high); only the GraspMoE OBB branch
    # uses it.
    fingertip_depth=0.136,
)
# ... or looked up from a shipped gripper's config.json (server-side installs):
params = SweepVolumeParams.from_gripper_config("robotiq_2f_85")

with GraspGenXClient(host="localhost", port=5556) as client:
    # Mode 1 — segmented object point cloud → grasps in the input PC's frame.
    grasps, scores = client.infer_object(
        object_pc,               # (N, 3) float32, meters
        params,                  # SweepVolumeParams | dict | flat (12,) array
        planner="graspmoe",      # or "diffusion"; moe_* knobs are kwargs
    )

    # Mode 2 — depth image → {instance_id: (grasps, scores)} in the CAMERA frame.
    results = client.infer_scene_depth(
        depth,                   # (H, W) float32 depth in METERS (<=0 = invalid)
        intrinsics,              # (3, 3) pinhole K
        instance_mask,           # (H, W) int; pixel value = instance id, 0 = ignore
        params,
    )

    # Mode 3 — scene point cloud → {instance_id: (grasps, scores)} in the
    # point cloud's own frame (non-finite points ignored).
    results = client.infer_scene_pc(
        scene_pc,                # (N, 3) or organized (H, W, 3) float32, meters
        instance_mask,           # int array with N elements (any reshapeable shape)
        params,
    )
    for instance_id, (grasps, scores) in results.items():
        ...                      # grasps (Ki, 4, 4), scores (Ki,)
```

Notes:

- These actions are **params-only** — pass `gripper_name` to the legacy
  `infer` action instead if you want name-based lookup.
- `planner="graspmoe"` (default) unions diffusion samples with OBB-swept
  candidates, all scored by the discriminator; per-grasp provenance is
  available via `return_branch_tags=True` ("diff" | "obb").
- **The server returns every generated grasp with its score.** The
  `grasp_threshold` / `topk_num_grasps` arguments of the Python client are
  applied **client-side**; raw-wire callers must threshold themselves (the
  server rejects those fields on the new actions).
- The server runs eager PyTorch **fp32** by default. The native C++ runtime is
  enabled with `graspgenx_server.py --fast-inference`; it runs
  the default released 20-step DDPM trajectory as one CUDA Graph and
  installs the combined dual-PTV3 CUDA runtime plus the native TensorRT scorer.
  PointNet2 and Torch-TensorRT are not required.
  The `metadata` response reports the active precision under `"precision"`.
- Instances with fewer than `min_object_points` (default 100) valid points,
  or with no grasps found, are absent from the returned dict.
- Collision filtering against the rest of the scene is **not** applied
  (it needs a gripper collision mesh, which raw sweep params don't provide).

## Example Scripts (with viser visualization)

`scripts/client_server_example_{1,2,3}.py` run one mode each against a live
server on the shipped sample scenes and render the results in a
[viser](https://viser.studio/) web viewer (grasps colored red→green by score,
grouped per object under `obj/<label>/grasps/{diff,obb}_*` in the scene tree).

```bash
# Terminal 1 — start the server with the native C++ fast runtime:
uv run python client-server/graspgenx_server.py \
    --config ext/graspgenx_checkpoints/release --assets_dir assets --port 5556 \
    --fast-inference

# Terminal 2 — run a mode, then open the URL the script prints:

# Mode 1 (example 3) — single segmented object point cloud:
uv run python scripts/client_server_example_3.py --gripper_name robotiq_2f_85 --port 5556

# Mode 2 (example 1) — scene depth image + intrinsics + instance mask:
uv run python scripts/client_server_example_1.py --gripper_name robotiq_2f_85 --port 5556

# Mode 3 (example 2) — scene point cloud + instance mask:
uv run python scripts/client_server_example_2.py --gripper_name robotiq_2f_85 --port 5556
```

`--gripper_name` is resolved to sweep-volume params **locally** (only the 12
numbers travel over the wire); pass `--sweep_volume_json file.json` instead
for a gripper with no local assets. Common flags:

- `--scene_dir assets/sample_data/real_world/01` — the other shipped scene.
- `--grasp_threshold 0.7` (default, client-side; `-1.0` disables) and
  `--topk_num_grasps 100` (default; `-1` keeps everything the server sent).
- `--planner diffusion` (default `graspmoe`) and
  `--moe_obb_density dense-topandside` for a denser OBB sweep.
- Example 3 only: `--object_index N` picks the object;
  `--pcd_npy cloud.npy` sends your own (N,3) point cloud.

## Supported Grippers

Anything with a directory under your assets root. The default `gripper_descriptions` checkout ships with:

`abb_yumi`, `arx_x5`, `barrett_hand`, `bd_spot`, `dh_ag95`, `ezgripper`, `fetch_robot`, `franka_panda`, `franka_umi`, `galaxea_g1`, `inspire_hand`, `onrobot_RG2`, `onrobot_RG6`, `piper_hand`, `robotiq_2f_85`, `robotiq_2f_140`, `robotiq_3f`, `robotiq_hande`, `sawyer_hand`, `schunk_wsg50`, `sharpa_wave`, `surge_hand`, `tesollo_delto2f`, `unitree_g1`, `wuji_hand`, `xarm_hand`, `yam_4310`.

Plus 32 procedural grippers under `assets/proc_grippers/` (the training set; see the main README for the full list).

## Protocol Reference

Wire format: **msgpack** (with `msgpack_numpy.patch()` so numpy arrays travel natively) over a **ZMQ REQ/REP** socket.

| Request | Fields | Response |
|---------|--------|----------|
| `{"action": "health"}` | — | `{"status": "ok"}` |
| `{"action": "metadata"}` | — | `{"default_gripper": str?, "loaded_grippers": [str], "model": {...}, "assets_dir": str}` |
| `{"action": "infer", ...}` | `point_cloud` (N,3 float32), `gripper_name` (str, optional if server has a default), `num_grasps` (int=100), `grasp_threshold` (float=-1.0), `topk_num_grasps` (int=100) | `{"grasps": (K,4,4) float32, "confidences": (K,) float32, "gripper_name": str, "timing": {"infer_ms": float}}` |
| `{"action": "infer_object", ...}` | `point_cloud` (N,3 float32), `sweep_volume_params` (map or flat (12,) float32 — required, no gripper_name), `planner` ("graspmoe"\|"diffusion"), `num_grasps`, `moe_*` knobs | `{"grasps": (K,4,4), "confidences": (K,), "branch_tags": ["diff"\|"obb", ...], "timing": {...}}` — frame of the input PC; **ALL** generated grasps (`grasp_threshold`/`topk_num_grasps` are rejected — threshold client-side) |
| `{"action": "infer_scene_depth", ...}` | `depth` (H,W float32 **meters**), `intrinsics` (3,3), `instance_mask` (H,W int, 0 = ignore), `sweep_volume_params`, `min_object_points` (int=100), planner + sampling knobs | `{"instance_ids": (M,) int32, "grasps": [(Ki,4,4)], "confidences": [(Ki,)], "branch_tags": [[str]], "skipped_instance_ids": (S,) int32, "timing": {...}}` — **camera frame**; parallel lists, aligned with `instance_ids` |
| `{"action": "infer_scene_pc", ...}` | `point_cloud` ((N,3) or (H,W,3) float32), `instance_mask` (int array with N elements), `sweep_volume_params`, `min_object_points`, planner + sampling knobs | same shape as `infer_scene_depth`, grasps in the input PC's frame |

`sweep_volume_params` map keys: `extents_open`, `offset_open`, `extents_mid`, `offset_mid` (each (3,) float, meters, gripper base frame); optional `gripper_type` (0 parallel_2f / 1 revolute_2f / 2 revolute_3f) and `fingertip_depth` (float). The flat (12,) form is `[extents_open, offset_open, extents_mid, offset_mid]`. Scene responses use parallel lists instead of an id-keyed map because msgpack's default `strict_map_key` rejects integer keys — the Python client reassembles them into `{instance_id: (grasps, scores)}`.

On error: `{"error": "<ExceptionType>: <message>"}`. The Python client raises `RuntimeError`.

The protocol is dead-simple to drive from C++/Rust/etc. with any ZMQ + msgpack bindings.
