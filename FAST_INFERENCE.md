# Fast inference

GraspGenX inference accepts one object represented by at least 100 XYZ points.
The input is deterministically normalized to the selected encoder input
capacity (3,500 points by default). Candidate counts are selected when the runtime is constructed and stay
fixed for its TensorRT plans and CUDA Graph. The default produces 100 diffusion
grasps and 50 analytic OBB grasps, then scores all 150 candidates.

## Supported inputs and limits

The native runtime uses the released DDPM20 sampler. It processes one object
per call and one in-flight request per runtime instance. Inputs below 100
points are rejected. Clouds above 3,500 points are downsampled by evenly spaced
input indices before filtering; smaller clouds are filtered and repeated to
the selected first-stage capacity. More raw input points do not increase the
selected neural resolution.
The zero-copy C++ buffer accepts only 100–3,500 points. Set all five stage
capacities with `--ptv3-capacities INPUT POOL1 POOL2 POOL3 POOL4` during
installation; the default is `[3500,2048,1024,512,512]`.
Input capacity ranges from 100 to 3,500, and each pooled capacity
from 1 to 3,500; matching plans determine runtime buffer sizes without a library
rebuild. See [capacity profiles](cpp_runtime/README.md#configurable-ptv3-capacities).

Fixed encoder capacities and spatial encoding can overflow even for an accepted
point count; overflow makes the result invalid. Candidate counts require matching
plans and cannot exceed 4,096 combined. Plans depend on the target GPU
architecture and TensorRT version. See the
[native runtime limits](cpp_runtime/README.md#input-limits-and-concurrency) for
capacity, spatial-range, sampling, output-lifetime, and concurrency details.

## Inference pipeline

```text
N object points (N >= 100)
        |
        v
Downsample raw input if needed, then CUDA filtering, profile resampling, and centering
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
20 DDPM steps      |            |
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

The runtime consists of:

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
  The final dense discriminator/scorer uses FP32.
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

The DDPM sampler uses only the released 20-step trajectory; no step-count
input is exposed.

## Runtime boundary

The production hot path is implemented once in C++ with CUDA, cuBLAS, and
TensorRT; it has no PyTorch or Python dependency. The optimized Python API is a
thin C-ABI caller that supplies PyTorch device pointers and the current CUDA
stream to that same runtime. The eager PyTorch path remains available for
training and debugging.

See the [native runtime guide](cpp_runtime/README.md) for installation, API,
capacity configuration, grasp-volume admission, and benchmarking.
