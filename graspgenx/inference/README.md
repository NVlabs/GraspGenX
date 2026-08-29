# PyTorch inference

This package contains the small eager PyTorch inference layer. The PTV3 model
lives with the other models in `graspgenx/models/ptv3`; this package only owns
point-cloud preprocessing. Importing it does not compile custom CUDA code or
load, configure, or require TensorRT.

The native optimized implementation is intentionally separate:

```python
from graspgenx.fast_inference import install_fast_runtime
```

That package is only a thin Python bridge to `cpp_runtime`; the C++ runtime owns
TensorRT, CUDA Graphs, persistent buffers, and the complete optimized pipeline.

Files in this package:

- `preprocessing.py`: point-cloud filtering and fixed-count resampling for the
  eager sampler.

The fast runtime does not call this Python implementation on its hot path. CMake
compiles its CUDA kernels from `cpp_runtime/src/ptv3_cuda_kernel.cu`.
