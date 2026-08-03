# AGENTS.md — GraspGenX Coding Conventions

## Build / Lint / Test

### Package management
```bash
uv pip install -e ".[dev]"        # editable install with dev deps
uv sync --extra end2end            # full end-to-end stack (optional)
uv pip install -e .                # base inference-only install
```

### Run all tests
```bash
pytest
pytest -v                          # verbose
pytest -x                          # stop on first failure
pytest -k "keyword"                # filter by test name
```

### Run a single test file
```bash
pytest tests/test_installation.py
pytest tests/test_inference_installation.py -v
```

### Run a single test
```bash
pytest tests/test_installation.py::test_graspgenx_importable
pytest tests/test_compute_utils.py::TestFmt::test_bytes
```

### Markers (defined in pyproject.toml)
- `pytest.mark.integration` — tests requiring GPU or live server
- `pytest.mark.end2end` — slow end-to-end pick-and-place demos
- `pytest.mark.skipif(not torch.cuda.is_available(), ...)` — GPU-gated tests

### Lint / format
```bash
ruff check .                       # lint (preferred; not in pyproject but standard)
ruff format --check .              # format check
black --check .                    # line-length 88, target py310
isort --check --profile black .    # import sorting (black profile)
flake8                             # configured via pyproject or .flake8
```

### Lint / format (auto-fix)
```bash
ruff check --fix .
ruff format .
black .
isort --profile black .
```

### Type checking
```bash
mypy graspgenx/                    # if configured
```

### Single GPU test (CUDA required)
```bash
pytest tests/test_inference_installation.py::test_inference_100_grasps -v
```

---

## Code Style

### File header
Every `.py` file starts with the SPDX license header:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```
Optionally followed by `#!/usr/bin/env python3` for executable scripts.

### Imports
- `isort` with `--profile black` (line-length 88). Group order:

```python
# SPDX header

from __future__ import annotations          # optional (preferred for type annotations)

# Standard library
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

# Third-party
import numpy as np
import torch
import torch.nn as nn
import omegaconf

# Local
from graspgenx.models.grasp_gen import GraspGen
from graspgenx.utils.logging_config import get_logger

# Logger (module-level, after all imports)
logger = get_logger(__name__)
```

### Formatting
- **Black** with line-length 88, target Python 3.10.
- **Indentation**: 4 spaces (no tabs).
- Quotes: double quotes `"` preferred over single `'`.

### Type annotations
- Use Python 3.10+ style: `str | None` (not `Optional[str]`).
- `from __future__ import annotations` for forward references (used in some files).
- Typed function signatures everywhere, including return types.
- `DictConfig` from omegaconf for Hydra configs.

### Naming conventions
| Category | Convention | Example |
|----------|-----------|---------|
| Classes | PascalCase | `GraspGenXSampler`, `GraspGenGenerator` |
| Functions/methods | snake_case | `run_inference`, `load_state_dict` |
| Variables | snake_case | `grasp_sampler`, `object_pc` |
| Constants | UPPER_CASE | `OBJ_NPOINTS`, `OBJ_MLPS` |
| Private helpers | `_` prefix | `_fmt`, `_parse_proc_meminfo` |
| Private members | `_` prefix | `self._samplers` |
| Test classes | `Test` prefix | `TestFmt`, `TestLogSystemMemory` |
| Test functions | snake_case, `test_` prefix | `test_graspgenx_importable` |

### Docstrings
Google-style with `Args:`, `Returns:`, `Raises:` sections:

```python
def run_inference(
    object_pc: np.ndarray | torch.Tensor,
    grasp_threshold: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-liner description.

    Args:
        object_pc: Point cloud to generate grasps for.
        grasp_threshold: Threshold for valid grasps.

    Returns:
        grasps: Generated grasp poses.
        grasp_conf: Confidence scores for the grasps.
    """
```

### Error handling
- Raise specific exceptions: `ValueError`, `FileNotFoundError`, `NotImplementedError`.
- Use `try/except` with bounded scope (avoid bare `except:`).
- Log before re-raising or handling: `logger.warning(...)` / `logger.info(...)`.
- Guard optional imports with try/except and set a module-level fallback:

```python
try:
    from graspgenx.dataset.renderer import render_pc
except ImportError as e:
    render_pc = None
    logger.warning(f"Could not import renderer: {e}")
```

- Use `raise ... from e` for exception chaining where appropriate.
- Assertions for preconditions: `assert len(pts) > 0`.
- `@dataclass` + `Enum` for structured error codes (see `dataset/exceptions.py`).

### Logging
- Module-level logger via `get_logger(__name__)` (from `utils/logging_config.py`).
- Uses `logging.INFO` level by default.
- Always log import failures, checkpoint loading, and resource diagnostics.

### Model conventions
- Classes inherit `nn.Module`.
- Factory classmethod `@classmethod def from_config(cls, cfg: DictConfig)`.
- Inference methods decorated with `@torch.inference_mode()`.
- `forward()` returns `(outputs, losses, stats)` tuple convention.
- Config objects are `omegaconf.DictConfig`.
- Checkpoint dirs: `gen/` and `dis/` subdirectories with `config.yaml` + `epoch_*.pth`.

### Test conventions
- Tests in `tests/` mirror package structure.
- Use `pytest` fixtures in `conftest.py` for `device`, `random_seed`, `sample_point_cloud`.
- `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")` for GPU tests.
- `@pytest.mark.parametrize` for multi-backbone/configuration tests.
- `pytest.skip(reason)` for runtime condition skipping (e.g. missing checkpoints).
- Import the module inside the test for importability checks.
- Test classes for grouping (e.g. `TestFmt`), plain functions for simple tests.
- Smoke tests verify "doesn't crash"; assertion tests verify outputs.

### Data structures
- `@dataclass` for configuration/info objects (e.g. `GripperInfo`, `XGripperInfo`, `ErrorInfo`).
- `Enum` for error codes (`DataLoaderError`).
- `DictConfig` from `omegaconf` for model/training configs (Hydra).

### CUDA / device patterns
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = tensor.to(device)
model = model.cuda().eval()             # in production code
model = model.to(device).eval()         # in test code
```

### Point cloud conventions
- Shape `(N, 3)` for raw points, `(N, 4)` for homogeneous, `(N, 4, 4)` for pose matrices.
- Grasp poses are `(K, 4, 4)` homogeneous transformation matrices.
- Confidence scores are `(K,)` tensors.

---

## Platform Notes

### DGX Spark (aarch64 / NVIDIA GB10 / CUDA 13.0)

**Verified working:** `uv venv --python 3.12` with `uv pip install -e ".[dev]"`.

| Item | Status | Notes |
|------|--------|-------|
| torch 2.12.0+cu130 | ✅ | Must use cu130 index; upper cap removed in `spark` branch |
| torchvision 0.27.0+cu130 | ✅ | Upper cap removed |
| ptv3vanilla backbone | ✅ | Pure PyTorch, no compiler needed |
| scene-synthesizer | ❌ skipped | `usd-core` has no aarch64 wheel; guarded by `platform_machine != 'aarch64'` |
| pointnet2_ops | ⚠️ JIT only | No prebuilt aarch64 wheel; falls back to JIT compilation which may fail. Use `ptv3vanilla` backbone instead. |
| spconv-cu120 | ❌ | No aarch64 wheel; `ptv3vanilla` backbone doesn't need it |
| end2end extras | ❌ | cuRobo, Newton, CoACD etc. have no aarch64 wheels |
| pytest with ROS | ⚠️ workaround | ROS Jazzy installs a `launch_testing` plugin that breaks pytest collection. Run with: `python3 -c "import sys; sys.path=[p for p in sys.path if '/opt/ros/' not in p]; import pytest; pytest.main(...)"` |

**Backbone guidance on aarch64:** Use `object_backbone: ptv3vanilla` and `gripper_backbone: z_offset` (the defaults in inference tests). Avoid PointNet++ unless you build `pointnet2_ops` from source with the CUDA toolkit.
