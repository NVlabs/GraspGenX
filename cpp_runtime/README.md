# GraspGenX native C++ runtime

This folder contains the complete fixed-shape, PyTorch-free GraspGenX inference
runtime. The deployed library accepts one object point cloud and returns a
configured number of ranked world-frame grasps. It links CUDA, cuBLAS, and
TensorRT; it does not link or import PyTorch, LibTorch, ATen, ONNX Runtime, or
Python.

PyTorch and ONNX are used only by the offline checkpoint-to-plan tools.
All deployed custom CUDA kernels live in this runtime's `src/` directory and
are compiled only as part of this runtime.

## Runtime contract

- one object per call;
- one contiguous FP32 XYZ cloud with at least 100 points at the API boundary;
- deterministic normalization to the fixed 3,500-point engine shape: smaller
  clouds are filtered and repeated, while larger clouds are downsampled before
  filtering;
- a positive, runtime-construction-time diffusion candidate count (default 100);
- a non-negative analytic top-down OBB candidate count (default 50);
- at most 4,096 combined candidates;
- released 20-step DDPM denoising;
- FP16 PTV3 and denoiser plans;
- strict FP32 discriminator/scorer;
- independent generator and discriminator PTV3 weights with shared topology;
- external 512D generator and discriminator gripper embeddings, so the large
  plans are reusable across grippers; and
- score-sorted `[4,4]` world-frame poses, scores, source indices, object center,
  PTV3 stage counts, and topology-overflow flags.

Candidate counts are static TensorRT-plan and CUDA-Graph shapes, but they are
not C++ compile-time constants. Build matching dense plans and a Sobol buffer,
then set the counts in `RuntimeConfig`. The same library binary supports every
configuration; this keeps the hot path fixed-shape without making users rebuild
the runtime.

When updating this C++ header/library pair, rebuild C++ consumers together.
The existing C-ABI config layouts and inference entry points are unchanged.

The hot path uses persistent buffers and a multi-stream CUDA Graph. OBB runs on
a side stream while PTV3 topology is completed. The two learned PTV3 branches
run concurrently after the shared topology is ready. Denoising begins as soon
as the generator embedding is ready; scoring waits for denoising, OBB, and the
discriminator embedding.

Between recurrent steps, one CUDA kernel performs the denoising update and next
6-to-512 sample projection together. The intermediate state is still rounded
to FP16 exactly as in the unfused path, so this removes launches without
changing the validated output.

## Grasp-volume admission

Set `RuntimeConfig.minimum_points_in_volume` to a positive value, then call
`Runtime::SetGraspVolume(bounds)` with host FP32
`[xmin,ymin,zmin,xmax,ymax,zmax]` in the gripper frame, including standoff.
Enabled filtering rejects inference until bounds have been supplied. The default
threshold is zero to preserve existing unfiltered C++ and C-ABI callers.
`SetMinimumPointsInVolume` accepts 0 through 3500 and updates the captured graph's
device parameter without recapture; zero disables this optional filter.

The GPU transforms the supplied object points into each candidate's gripper
frame, counts finite points inside the inclusive bounds, and rejects candidates
with insufficient occupancy or malformed poses/scores. It counts input points
after the optional downsample to 3500 and before encoder filtering/repetition,
so encoder upsampling cannot inflate occupancy. Inputs are sampled points, not
a guarantee of distinct physical surface samples.

Passing candidates form a descending-score prefix in `DeviceOutputs`.
`DeviceOutputs.valid` is an aligned CUDA uint8 array of 0/1 values; rejected
candidates follow with confidence zero. A valid zero-score candidate precedes
rejected candidates. Poses and source indices retain their correspondence.
Filtering/ranking stays inside the CUDA graph without point/pose readback.
The setters wait for the runtime's last inference event before updating bounds
or threshold. Serialize host calls and order inference requests on their CUDA
streams; one runtime instance owns one shared set of input/output buffers.

The offline gripper exporter includes exact `grasp_volume` bounds in the NPZ
and a raw FP32 file for named grippers, using `config.json` sweep volume and
standoff with the same convention as `x_grippers`. Raw descriptor exports can
provide all six bounds with `--grasp-volume`.

Configure `-DGRASPGENX_BUILD_TESTS=ON` to build the GPU filter/ranking fixtures
and `graspgenx_volume_runtime_test`. Run the latter with a DDPM20 artifact
directory to verify the actual captured inference graph and setters.

## Source installation

TensorRT plans are specific to the GPU architecture and TensorRT version used
to build them. The host needs a CUDA toolkit with `nvcc` plus matching TensorRT
runtime and development files (`libnvinfer` and `NvInfer.h`). Everything else
is driven by one command from the repository root:

```bash
uv run --quiet --extra fast-inference graspgenx-install-fast-inference \
  --diffusion-grasps 200 --obb-grasps 50
```

The installer:

1. resolves or downloads the released checkpoint;
2. exports ONNX and native constant buffers;
3. builds the count-specific TensorRT plans on the current GPU;
4. incrementally builds `libgraspgenx_runtime.so` into
   `graspgenx/fast_inference/lib/`, where Python discovers it automatically; and
5. runs a Python-device-pointer to C++ CUDA Graph smoke test.

Child-process logs stay hidden unless a command fails; pass `--verbose` to
stream ONNX, TensorRT, and CMake output. Complete artifacts are reused. Common
variants remain one command:

```bash
# A different fixed candidate shape.
uv run --quiet --extra fast-inference graspgenx-install-fast-inference \
  --diffusion-grasps 500 --obb-grasps 50

# Nonstandard toolkit locations.
uv run --quiet --extra fast-inference graspgenx-install-fast-inference \
  --cuda-root /opt/cuda-13.0 --tensorrt-root /opt/tensorrt

# Rebuild plans after a checkpoint, TensorRT, or GPU change.
uv run --quiet --extra fast-inference graspgenx-install-fast-inference --force-artifacts

# Spend more build time searching for faster TensorRT plans (level 2 is default).
uv run --quiet --extra fast-inference graspgenx-install-fast-inference \
  --tensorrt-optimization-level 5 --force-artifacts
```

Use `--dry-run` to inspect the underlying commands. `--skip-artifacts` and
`--skip-native-build` are available for CI or externally staged inputs.

Generated files remain ignored under `cpp_runtime/artifacts/`. Production uses:

```text
artifacts/ddpm20/
  engines/
    denoiser_core_d100_fp16.plan
    scorer_core_c150_fp32.plan
    ptv3/ptv3_{generator,discriminator}_stage{0..4}_fp16.plan
  weights/
    observation_weight.bin
    sample_encoder_{weight,bias}.bin
    prediction_{weight,bias}.bin
    sobol_initial_sample_d100.bin
    schedules/20/
      timestep_conditioning.bin
      denoising_coefficients.bin
```

Changing Python-side grippers requires no extra installation: their small
embeddings are encoded once and passed to the shared runtime. A standalone C++
application can export raw embeddings separately:

```bash
.venv/bin/python cpp_runtime/tools/export_gripper_embeddings.py \
  --gripper-name robotiq_2f_85 \
  --output cpp_runtime/artifacts/ddpm20/robotiq_2f_85_embeddings.npz
```

The exporter writes raw little-endian FP32 files and a hash manifest. Changing
grippers does not rebuild any TensorRT plan.

## C++ API

```cpp
#include <graspgenx/runtime.h>

graspgenx::RuntimeConfig config;
config.artifact_directory = "cpp_runtime/artifacts/ddpm20";
config.gripper_depth = 0.136F;
config.diffusion_grasps = 100;
config.obb_grasps = 50;
graspgenx::Runtime runtime(config);

runtime.SetGripperEmbeddings(generator_embedding, discriminator_embedding);
runtime.InferDeviceAsync(points_nx3, point_count, seed, sequence_offset, stream);

graspgenx::DeviceOutputs output = runtime.outputs();
// output.grasps                    device FP32 [output.grasp_count,4,4]
// output.confidence                device FP32 [output.grasp_count]
// output.source_indices            device int32 [output.grasp_count]
// output.ptv3_stage_counts         device int64 [5]
// output.ptv3_topology_overflow    device bool [4]
```

The runtime uses the released DDPM20 model with no separate step-count input.
All recurrent denoiser executions are captured in the single inference CUDA
Graph.

`InferDeviceAsync` accepts any cloud with at least 100 points. It downsamples
clouds above 3,500 directly on CUDA and launches the graph on the caller
stream. `InferHostAsync` provides the same contract for CPU memory. For a
zero-copy producer path, write 100–3,500 rows directly to
`device_input_points()` and call `InferAsync` with the point count. One runtime
instance supports one in-flight request; use separate instances for concurrent
independent requests.

An overflow flag means that a cloud exceeded one of the fixed PTV3 capacities
`[3500,1536,512,128,64]` or the 10-bit-per-axis spatial sort range, and its
result is invalid. At the fixed scale/grid contract, the spatial range permits
an approximately 3.13 m object span on each axis. The synchronous Python API
raises on overflow. Asynchronous C++ callers should copy/check the four device
flags before consuming a result.

Python inference calls this runtime through the stable C ABI in
`include/graspgenx/c_api.h`. PyTorch supplies device pointers and its current
CUDA stream; C++ owns TensorRT, intermediate buffers, internal streams, and the
captured graph. The shared library intentionally has no PyBind, LibTorch, or
ATen dependency. Native applications can use either the C++ API above or the C
ABI.

Every runtime resource belongs to one `ExecutionDomain`. The current variant is
`PrimaryDomain`, which materializes the selected device's primary CUDA context
before creating streams, events, cuBLAS, TensorRT contexts, or graphs. The
domain is a value type backed by `std::variant`, leaving a direct seam for a
future create-or-borrow `GreenDomain` without virtual dispatch in the pipeline.

## Validation and benchmark

```bash
cpp_runtime/build-native/graspgenx_benchmark_runtime \
  cpp_runtime/artifacts/ddpm20 \
  cpp_runtime/artifacts/ddpm20/robotiq_2f_85_embeddings.generator_embedding.fp32.bin \
  cpp_runtime/artifacts/ddpm20/robotiq_2f_85_embeddings.discriminator_embedding.fp32.bin \
  0.136 100 50
```

RTX 5090 measurements for the default 100-diffusion + 50-OBB configuration use
three independent processes, each with 5,000 warmups and 5,000 CUDA-Graph
replays:

| DDPM schedule | Denoising | Point cloud to 150 ranked grasps |
|---:|---:|---:|
| default released 20-step | **1.380 ms** | **2.216 ms** |

Memory in the same pure C++ process:

| Allocation view | Memory |
|---|---:|
| Explicit persistent CUDA arena | 14.10 MiB for 20 steps |
| Sum of TensorRT context workspace sizes | 63.91 MiB |
| `cudaMemGetInfo` delta, including context/libraries/weights/graphs | 430 MiB |
| NVML process-resident GPU memory | **928 MiB** |

`ldd` on the native shared object contains CUDA, cuBLAS, TensorRT, and the
standard C++ runtime only; there are no Torch/ATen/c10 dependencies.
