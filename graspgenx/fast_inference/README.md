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

If the native library or a required artifact is missing, installation leaves
the eager model untouched. From a source checkout, install the complete native
path with:

```bash
uv run --extra fast-inference graspgenx-install-fast-inference
```

The installer puts the library and artifacts in locations that the package
discovers automatically. Set these variables only for external/custom locations:

```bash
export GRASPGENX_CPP_RUNTIME_LIBRARY=/path/to/libgraspgenx_runtime.so
export GRASPGENX_CPP_RUNTIME_ARTIFACTS=/path/to/artifacts
```

- `runtime.py` installs and shares the C++ runtime on a model.
- `cpp_runtime.py` is the small ctypes/device-pointer boundary.
- `../../cpp_runtime/tools/` contains the offline PyTorch export helpers and
  TensorRT plan builder; deployed fast inference imports none of them.
- `../../cpp_runtime/src/ptv3_cuda_kernel.cu` contains the native CUDA kernels.
  Eager PyTorch inference does not compile or load this source.

Application code should import only:

```python
from graspgenx.fast_inference import install_fast_runtime
```

Use `scripts/benchmark_native_tensorrt.py` to measure the complete Python to C++
boundary. Use `cpp_runtime/graspgenx_benchmark_runtime` for native component
profiles.
