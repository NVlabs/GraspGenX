# Fast inference boundary

Optimized Python inference is a thin caller of the native C++ runtime. There is
one deployed implementation of PTV3, diffusion, OBB proposal generation,
scoring, ranking, and CUDA Graph capture: `cpp_runtime`.

Python owns only CUDA input/output tensors and the caller stream. `cpp_runtime.py`
passes their device pointers through the stable C ABI in
`cpp_runtime/include/graspgenx/c_api.h`; the shared library does not link
LibTorch, ATen, or Python. TensorRT contexts, persistent intermediate buffers,
events, internal streams, and graphs remain C++-owned.

The boundary accepts one CUDA point cloud shaped `[N,3]` or `[1,N,3]` with
`N >= 100`. Clouds above 3,500 points are deterministically downsampled in the
native runtime. Diffusion and OBB counts are fixed when the C++ runtime is
constructed and must match its TensorRT plans. Their sum cannot exceed 4,096.

Clouds are filtered and resampled to the selected first-stage capacity (3,500
by default).
Downsampling uses evenly spaced input indices and depends on point ordering.
Each runtime supports one in-flight request. Encoder capacity or spatial-range
overflow invalidates a result. `GraspGenXSampler` checks the flags and raises
on overflow; direct `NativeCppRuntime.infer()` callers must check
`result.ptv3_topology_overflow` before consuming outputs.
See the [native runtime limits](../../cpp_runtime/README.md#input-limits-and-concurrency)
for the complete input, concurrency, and target-artifact contract.

If the native library or a required artifact is missing, installation leaves
the eager model untouched. From a source checkout, install the complete native
path with:

```bash
uv run --quiet --extra fast-inference graspgenx-install-fast-inference \
    --diffusion-grasps 100 \
    --obb-grasps 50
```

The installer puts the library and artifacts in locations that the package
discovers automatically. Set these variables only for external/custom locations:

```bash
export GRASPGENX_CPP_RUNTIME_LIBRARY=/path/to/libgraspgenx_runtime.so
export GRASPGENX_CPP_RUNTIME_ARTIFACTS=/path/to/artifacts
```

The runtime uses the released DDPM20 model and defaults to
`[3500,2048,1024,512,512]`. Configure stage sizes at export/
installation with `--ptv3-capacities 3500 2048 1024 512 512`, then select that
profile's artifact directory. The same library reads and validates plan shapes
at construction; `runtime.stage_capacities` reports its active profile. All
stages support capacities up to 3,500; the input minimum is 100 and pooled-stage
minimum is one. See [capacity profiles](../../cpp_runtime/README.md#configurable-ptv3-capacities).

- `runtime.py` installs and shares the C++ runtime on a model.
- `cpp_runtime.py` is the small ctypes/device-pointer boundary.
- `../../cpp_runtime/tools/` contains the offline PyTorch export helpers and
  TensorRT plan builder; deployed fast inference imports none of them.
- `../../cpp_runtime/src/cuda_kernels.cu` contains the native CUDA kernels.
  Eager PyTorch inference does not compile or load this source.

Application code should import only:

```python
from graspgenx.fast_inference import install_fast_runtime
```

Use `scripts/benchmark_native_tensorrt.py` to measure the complete Python to C++
boundary. Use `cpp_runtime/build-native/graspgenx_benchmark_runtime` for native component
profiles.

The native library registers V3 pooling and mean-reduction plugins. Four strongly
typed plans cover both complete encoders, the denoiser, and the scorer.

Requests for native inference raise an initialization error if the shared library
or matching plans cannot be loaded. Python keeps input/output tensors and a
caller stream; all optimized inference runs in `cpp_runtime/`.
