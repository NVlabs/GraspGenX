# GraspGenX native C++ runtime

This folder contains the complete PyTorch-free GraspGenX inference runtime,
with fixed shapes per selected profile. The deployed library accepts one object
point cloud and returns a configured number of ranked world-frame grasps. It links CUDA, cuBLAS, and
TensorRT; it does not link or import PyTorch, LibTorch, ATen, ONNX Runtime, or
Python.

PyTorch and ONNX are used only by the offline checkpoint-to-plan tools.
Builds use strongly typed TensorRT networks, including on TensorRT 10.x;
precision-enabling builder flags are not used. Runtime tests currently cover
TensorRT 10.16.1.11. TensorRT 11.x execution has not been qualified.
All deployed custom CUDA kernels live in this runtime's `src/` directory and
are compiled only as part of this runtime.

## TensorRT plans and plugins

The exporter produces one complete ONNX graph per encoder directly.

The runtime uses four strongly typed TensorRT plans: generator encoder,
discriminator encoder, denoiser, and scorer. Each encoder contains all five
learned stages. Native `IPluginV3` reductions (`GraspGenXSegmentMax` and
`GraspGenXValidMean`, version 1) wrap the runtime CUDA kernels for pooling and
final aggregation. Both branches consume shared topology inputs prepared outside
TensorRT. The encoders retain separate weights and execute on separate streams.

Plugin creators are registered by `libgraspgenx_runtime.so`. Build this library
before exporting/building plans; the installer handles that order. The plan
builder loads it before ONNX parsing, and applications load it before plan
deserialization. Keep the library loaded while TensorRT engines exist. Plugin
execution uses the supplied CUDA stream without allocation or synchronization.
Explicit ONNX tensor types determine precision. Encoder and denoiser matrix
operations and storage use FP16, with FP32 normalization and nonlinear
arithmetic. Attention softmax uses FP32. The scorer uses FP32. Rebuild plans and the native library for the
target GPU and TensorRT version.

## Runtime contract

- one object per call;
- one contiguous FP32 XYZ cloud with at least 100 points at the API boundary;
- deterministic filtering/resampling to the encoder input capacity (3,500 by
  default); raw clouds above 3,500 are downsampled to 3,500 before filtering;
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

Build C++ consumers against the headers from the same source version as the
shared library. The Python wrapper and native library must also use matching
sources.

The hot path uses persistent buffers and a multi-stream CUDA Graph. OBB runs on
a side stream while PTV3 topology is completed. The two learned PTV3 branches
run concurrently after the shared topology is ready. Denoising begins as soon
as the generator embedding is ready; scoring waits for denoising, OBB, and the
discriminator embedding.

Between recurrent steps, one CUDA kernel performs the denoising update and next
6-to-512 sample projection together. Recurrent and final steps share one CUDA
update implementation, with the next-step projection omitted at compile time
for the final step. Initial and recurrent sample encoding share the same
projection helper. The intermediate state is rounded to FP16 before encoding.

## Input limits and concurrency

| Entry point | Accepted point count | Processing |
|---|---|---|
| `InferDeviceAsync` / `InferHostAsync` | At least 100; no explicit upper API limit | Above 3,500, select 3,500 points before filtering |
| `device_input_points()` + `InferAsync` | 100 through 3,500 | Caller fills the persistent input buffer |

All inputs are contiguous FP32 XYZ arrays for one object. After filtering,
points are repeated or downsampled to fill the selected first-stage neural input.
Accepting a larger cloud does not increase the selected encoder resolution.
Downsampling selects evenly spaced input indices, rather than spatially uniform
or farthest-point samples, so point ordering affects the retained geometry.

The default PTV3 capacities are `[3500,2048,1024,512,512]`; each runtime reads its
actual capacities from its plans. Exceeding a pooled-stage
capacity or the 10-bit-per-axis spatial sort range invalidates the result.
The spatial range permits approximately 3.13 m of object span per axis with
the current scale/grid settings. C++ callers must check
`DeviceOutputs.ptv3_topology_overflow` before consuming outputs.

Each runtime owns shared input/output buffers and supports one in-flight request.
Serialize host calls and order CUDA work; use separate runtime instances for
concurrent independent requests. Output pointers refer to runtime-owned buffers
that subsequent inference overwrites.

Only the released 20-step DDPM schedule is supported. Candidate counts are fixed
at construction, with at least one diffusion candidate, zero or more OBB
candidates, and at most 4,096 combined. Counts must match the exported plans and
Sobol buffer. TensorRT plans must match the target GPU architecture and TensorRT
version; regenerate target artifacts when those change.

## Configurable PTV3 capacities

Select the five static capacities in stage order: input, pool 1, pool 2, pool 3,
and pool 4. The input stage accepts 100–3,500; each pooled stage accepts 1–3,500.
Capacities are independent integers; they do not have to be powers of two or
strictly decreasing. A larger downstream capacity cannot recover groups lost
to an earlier overflow. The default is `[3500,2048,1024,512,512]`.

```bash
uv run --quiet --extra fast-inference graspgenx-install-fast-inference \
    --ptv3-capacities 3500 2048 1024 512 512
```

The installer exports matching plans into
`cpp_runtime/artifacts/ddpm20-p3500-2048-1024-512-512`. Python selects this default
directory automatically. For a custom profile, select its directory when running:

```bash
export GRASPGENX_CPP_RUNTIME_ARTIFACTS="$PWD/cpp_runtime/artifacts/ddpm20-p3500-2048-1024-512-512"
```

Python also accepts `artifact_directory=` in `install_fast_runtime` and
`NativeCppRuntime`; C++ uses `RuntimeConfig.artifact_directory`. The runtime
reads both branches' actual tensor shapes and validates their capacities,
channels, attention packing, and tensor types before allocating topology buffers
or capturing inference. One shared library serves all profiles without a native
rebuild. Inspect the loaded profile with C++ `runtime.stage_capacities()` or
Python `runtime.stage_capacities`.

Changing capacities requires new TensorRT plans and a new runtime instance.
`engines/ptv3_profile.json` records completed build capacities so the installer
cannot silently reuse another profile or a partially completed build. An
explicit artifact directory containing another profile requires
`--force-artifacts` to replace it. The native runtime validates the plans
themselves at construction.

For maximum pooled-count coverage, use
`--ptv3-capacities 3500 3500 3500 3500 3500`. Pooling cannot create more than
3,500 groups from 3,500 encoded points, so this removes pooled-count overflow.
The spatial encoding range still applies. Larger capacities increase memory and
padded computation. Smaller first-stage capacities reduce neural input
resolution; the raw input API still requires at least 100 points and stages at
most 3,500 points before filtering.

## Grasp-volume admission

Python's `NativeCppRuntime` uses the same native admission check. Call
`set_grasp_volume(bounds)` with host `[2,3]` bounds and
`set_minimum_points_in_volume(count)` (or pass the count to the constructor).
`infer()` returns an aligned CUDA Boolean `valid` mask. These controls and the
mask copy are exposed through the C ABI; Python does not implement a separate
volume filter.

Set `RuntimeConfig.minimum_points_in_volume` to a positive value, then call
`Runtime::SetGraspVolume(bounds)` with host FP32
`[xmin,ymin,zmin,xmax,ymax,zmax]` in the gripper frame, including standoff.
Enabled filtering rejects inference until bounds have been supplied. The default
threshold is zero, which disables grasp-volume filtering.
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
  --diffusion-grasps 100 --obb-grasps 50
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
artifacts/ddpm20-p3500-2048-1024-512-512/
  engines/
    ptv3_profile.json
    denoiser_core_d100_fp16.plan
    scorer_core_c150_fp32.plan
    ptv3/ptv3_{generator,discriminator}_fp16.plan
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
  --output cpp_runtime/artifacts/ddpm20-p3500-2048-1024-512-512/robotiq_2f_85_embeddings.npz
```

The exporter writes raw little-endian FP32 files and a hash manifest. Changing
grippers does not rebuild any TensorRT plan.

## C++ API

```cpp
#include <graspgenx/runtime.h>

graspgenx::RuntimeConfig config;
config.artifact_directory = "cpp_runtime/artifacts/ddpm20-p3500-2048-1024-512-512";
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
// output.ptv3_topology_overflow     device bool [4]
// output.valid                      device uint8 [output.grasp_count]
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

An overflow flag means that a cloud exceeded one of the selected PTV3 pooled
capacities or the 10-bit-per-axis spatial sort range, and its
result is invalid. At the fixed scale/grid contract, the spatial range permits
an approximately 3.13 m object span on each axis. `GraspGenXSampler` raises on
overflow. Direct C++, C ABI, and `NativeCppRuntime.infer()` callers must check
the four device flags before consuming a result.

Python inference calls this runtime through the stable C ABI in
`include/graspgenx/c_api.h`. PyTorch supplies device pointers and its current
CUDA stream; C++ owns TensorRT, intermediate buffers, internal streams, and the
captured graph. The shared library intentionally has no PyBind, LibTorch, or
ATen dependency. Native applications can use either the C++ API above or the C
ABI.

The runtime uses the selected device's primary CUDA context for streams,
events, cuBLAS, TensorRT contexts, and CUDA Graphs.

## Validation and benchmark

Build the benchmark target in the directory configured by the installer:

```bash
cmake --build cpp_runtime/build-native --target graspgenx_benchmark_runtime

GRASPGENX_BENCHMARK_ITERATIONS=300 \
cpp_runtime/build-native/graspgenx_benchmark_runtime \
  cpp_runtime/artifacts/ddpm20-p3500-2048-1024-512-512 \
  cpp_runtime/artifacts/ddpm20-p3500-2048-1024-512-512/robotiq_2f_85_embeddings.generator_embedding.fp32.bin \
  cpp_runtime/artifacts/ddpm20-p3500-2048-1024-512-512/robotiq_2f_85_embeddings.discriminator_embedding.fp32.bin \
  0.136 100 50
```

### Default-profile performance

The native C++ benchmark on NVIDIA RTX 6000 Ada uses TensorRT 10.16.1.11,
capacities `[3500,2048,1024,512,512]`, 20 DDPM steps, and 100 diffusion + 50 OBB
candidates. PTV3 and denoising use FP16; the scorer uses FP32. Input and output
buffers stay on the GPU. The benchmark uses a synthetic 3,500-point object and
CUDA event timing. Values below are means across three independent processes,
with 300 warmups and 300 measured CUDA Graph replays per process.

| Timing boundary | Time |
|---|---:|
| Complete point-cloud-to-ranked-grasps pipeline | 3.178 ms |

These timings exclude initialization, engine builds, and host-to-device input
transfer. They depend on the hardware and generated plans and do not measure
physical grasp success.

### Kernel profiling

The complete pipeline is replayed with one host CUDA Graph launch. Kernel
counts depend on the generated plans; profile the selected artifacts to measure them.

Set `GRASPGENX_BENCHMARK_CAPTURE_RANGE=1` when profiling the native benchmark
with Nsight Systems `--capture-range=cudaProfilerApi --capture-range-end=stop`.
This captures the measured replay loop after warmup and excludes the separate
component-timing pass. Use `--cuda-graph-trace=node` to inspect individual kernels.
