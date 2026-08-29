# Fast inference

GraspGenX inference accepts one object represented by at least 100 XYZ points.
The input is deterministically normalized to the fixed 3,500-point engine
shape. Candidate counts are selected when the runtime is constructed and stay
fixed for its TensorRT plans and CUDA Graph. The default produces 100 diffusion
grasps and 50 analytic OBB grasps, then scores all 150 candidates.

## Why the PyTorch path was slow

The original execution performed dynamic point-cloud bookkeeping separately
for the generator and discriminator encoders, launched the denoiser from Python
at every diffusion step, and materialized intermediate diffusion history.
Small point operations, tensor packing, synchronization, and framework launch
overhead were therefore a significant part of latency.

## Current pipeline

```text
N object points (N >= 100)
        |
        v
Downsample if needed, then CUDA filtering, resampling, and centering
        |
        +-----------------------------+
        |                             |
        v                             v
Shared PTV3 topology and packing   CUDA OBB proposals
        |                             |
        +----------------+            v
        |                |          50 poses
        v                v            |
FP16 generator PTV3  FP16 discriminator PTV3
        |                |            |
        v                |            |
20 fused DDPM steps      |            |
        |                |            |
        v                |            |
     100 poses           |            |
        +----------------+------------+
                         |
                         v
             FP32 discriminator and ranking
                         |
                         v
                150 ranked grasp poses
```

The main acceleration decisions are:

- **Fixed runtime shape.** One object, configured candidate counts, and
  fixed-capacity PTV3 stages remove dynamic allocation and make the full
  inference schedule graph-capturable. A different count uses count-specific
  dense TensorRT plans and a new runtime instance, not a C++ rebuild.
- **Shared point topology.** Generator and discriminator PTV3 encoders keep
  independent learned weights but reuse the same CUDA serialization, pooling,
  neighbor, and packing topology. Their learned branches run concurrently.
- **Custom CUDA point operations.** Filtering, resampling, Morton ordering,
  sparse-neighbor construction, segment reduction, topology packing, and final
  pose conversion avoid dynamic PyTorch operators and host synchronization.
- **TensorRT compute cores.** PTV3 and the denoiser use FP16 TensorRT plans.
  The final dense discriminator/scorer remains FP32 because its scores were
  more sensitive to reduced precision.
- **Fused recurrent boundary.** DDPM update, the next sample projection, and
  final latent-to-pose conversion are fused around the TensorRT denoiser. Only
  two persistent latent buffers are retained; diffusion history is not stored.
- **Persistent execution.** Buffers, TensorRT contexts, topology capacities,
  and a multi-stream CUDA Graph are created once and reused on every call.
- **Parallel analytic proposals.** Fifty OBB candidates run on a side stream
  alongside the learned pipeline, then share the discriminator and ranking
  pass with the diffusion candidates.
- **Gripper-independent plans.** Compact generator and discriminator gripper
  embeddings are loaded at runtime, so changing grippers does not rebuild the
  large TensorRT engines.

The default sampler uses the released 20-step DDPM trajectory. Tuned 5- and
10-step schedules remain selectable without changing the engine interface.

## Runtime boundary

The production hot path is implemented once in C++ with CUDA, cuBLAS, and
TensorRT; it has no PyTorch or Python dependency. The optimized Python API is a
thin C-ABI caller that supplies PyTorch device pointers and the current CUDA
stream to that same runtime. The eager PyTorch path remains available for
training and debugging.

On an RTX 5090, steady-state CUDA-Graph replay for the default 100+50 candidate,
20-step pipeline takes 2.216 ms from a device-resident point cloud to 150 ranked
device-resident grasps. This excludes model loading, engine construction, CUDA
compilation, and host-to-device input transfer. See
[`cpp_runtime/README.md`](cpp_runtime/README.md) for the build instructions,
runtime API, component timings, and memory measurements.
