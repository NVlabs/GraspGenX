# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin PyTorch boundary for the PyTorch-free native GraspGenX runtime.

PyTorch owns only the caller stream plus input/output storage. TensorRT
contexts, persistent intermediates, internal streams, and CUDA Graphs remain
owned by ``libgraspgenx_runtime``. The ABI is a small C interface so the native
library never links LibTorch, ATen, or Python.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from graspgenx._constants import MIN_OBJECT_POINT_COUNT
from graspgenx.fast_inference._constants import denoising_steps
from graspgenx.fast_inference._profiles import (
    DEFAULT_STAGE_CAPACITIES,
    profile_artifact_subdirectory,
)

ENGINE_POINT_COUNT = 3500
_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_LIBRARY_HANDLES: dict[Path, ctypes.CDLL] = {}

_LIBRARY_ENV = "GRASPGENX_CPP_RUNTIME_LIBRARY"
_ARTIFACT_ENV = "GRASPGENX_CPP_RUNTIME_ARTIFACTS"


class _ConfigV2(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("artifact_directory", ctypes.c_char_p),
        ("initial_gripper_depth", ctypes.c_float),
        ("denoiser_type", ctypes.c_int),
        ("diffusion_grasps", ctypes.c_int),
        ("obb_grasps", ctypes.c_int),
    ]


def _library_candidates(explicit: str | os.PathLike[str] | None):
    if explicit is not None:
        yield Path(explicit).expanduser()
        return
    configured = os.environ.get(_LIBRARY_ENV)
    if configured:
        yield Path(configured).expanduser()
    yield _ROOT / "graspgenx/fast_inference/lib/libgraspgenx_runtime.so"
    yield _ROOT / "cpp_runtime/build-native/libgraspgenx_runtime.so"
    yield _ROOT / "cpp_runtime/build/libgraspgenx_runtime.so"
    discovered = ctypes.util.find_library("graspgenx_runtime")
    if discovered:
        yield Path(discovered)


def resolve_runtime_library(
    library_path: str | os.PathLike[str] | None = None,
) -> Path:
    checked = []
    for candidate in _library_candidates(library_path):
        checked.append(str(candidate))
        if candidate.is_file():
            return candidate.resolve()
        # ``find_library`` may return a soname resolved by the dynamic loader.
        if candidate.parent == Path(".") and "/" not in str(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find libgraspgenx_runtime.so. Build cpp_runtime and set "
        f"{_LIBRARY_ENV}; checked: {checked}"
    )


def resolve_artifact_directory(
    artifact_directory: str | os.PathLike[str] | None,
    *,
    diffusion_grasps: int,
    obb_grasps: int,
    denoiser: str = "ddpm",
) -> Path:
    step_count = denoising_steps(denoiser)
    value = artifact_directory or os.environ.get(_ARTIFACT_ENV)
    directory = (
        Path(value).expanduser()
        if value is not None
        else _ROOT
        / "cpp_runtime/artifacts"
        / profile_artifact_subdirectory(denoiser, DEFAULT_STAGE_CAPACITIES)
    ).resolve()
    required = required_runtime_artifacts(
        directory,
        diffusion_grasps=diffusion_grasps,
        obb_grasps=obb_grasps,
        denoiser=denoiser,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        requested = (int(diffusion_grasps), int(obb_grasps))
        denoiser_label = denoiser.upper()
        available = _available_runtime_domains(directory, denoiser)
        available_text = (
            ", ".join(
                f"{diffusion} diffusion + {obb} OBB" for diffusion, obb in available
            )
            if available
            else "none"
        )
        issue = (
            "artifact domain mismatch"
            if available and requested not in available
            else "artifacts are incomplete"
        )
        raise FileNotFoundError(
            f"Native GraspGenX {issue}. Requested domain: "
            f"{step_count} {denoiser_label} steps, {requested[0]} diffusion + "
            f"{requested[1]} OBB grasps ({sum(requested)} total candidates). "
            f"Available complete domains for {step_count} {denoiser_label} steps: "
            f"{available_text}. Build the requested domain with "
            "`uv run --quiet --extra fast-inference graspgenx-install-fast-inference "
            f"--denoiser {denoiser} "
            f"--diffusion-grasps {requested[0]} --obb-grasps {requested[1]}`. "
            f"Artifact directory: {directory}. Missing: "
            f"{[str(path) for path in missing]}"
        )
    return directory


def required_runtime_artifacts(
    directory: str | os.PathLike[str],
    *,
    diffusion_grasps: int,
    obb_grasps: int,
    denoiser: str = "ddpm",
) -> tuple[Path, ...]:
    """Return every count- and schedule-specific file consumed by C++."""
    directory = Path(directory)
    engines = directory / "engines"
    weights = directory / "weights"
    schedule = weights / "schedules" / str(denoising_steps(denoiser))
    ptv3_plans = tuple(
        engines / "ptv3" / f"ptv3_{branch}_fp16.plan"
        for branch in ("generator", "discriminator")
    )
    return (
        *ptv3_plans,
        directory / "engines" / f"denoiser_core_d{int(diffusion_grasps)}_fp16.plan",
        engines / f"scorer_core_c{int(diffusion_grasps) + int(obb_grasps)}_fp32.plan",
        weights / "observation_weight.bin",
        weights / "sample_encoder_weight.bin",
        weights / "sample_encoder_bias.bin",
        weights / "prediction_weight.bin",
        weights / "prediction_bias.bin",
        weights / f"sobol_initial_sample_d{int(diffusion_grasps)}.bin",
        schedule / "timestep_conditioning.bin",
        schedule / "denoising_coefficients.bin",
    )


_ARTIFACT_REPORT_PATTERN = re.compile(
    r"^(?:build|export)_report_d(?P<diffusion>\d+)_o(?P<obb>\d+)\.json$"
)


def _available_runtime_domains(
    directory: Path, denoiser: str = "ddpm"
) -> tuple[tuple[int, int], ...]:
    """Return complete fixed-count domains described by artifact reports."""
    domains = set()
    for report_directory in (directory / "engines", directory / "onnx"):
        for report in report_directory.glob("*_report_d*_o*.json"):
            match = _ARTIFACT_REPORT_PATTERN.match(report.name)
            if match is None:
                continue
            domain = (int(match["diffusion"]), int(match["obb"]))
            required = required_runtime_artifacts(
                directory,
                diffusion_grasps=domain[0],
                obb_grasps=domain[1],
                denoiser=denoiser,
            )
            if all(path.is_file() for path in required):
                domains.add(domain)
    return tuple(sorted(domains))


def _configure_library(library):
    void_p = ctypes.c_void_p
    library.graspgenx_runtime_last_error.argtypes = []
    library.graspgenx_runtime_last_error.restype = ctypes.c_char_p
    library.graspgenx_runtime_create_v2.argtypes = [
        ctypes.POINTER(_ConfigV2),
        ctypes.POINTER(void_p),
    ]
    library.graspgenx_runtime_create_v2.restype = ctypes.c_int
    library.graspgenx_runtime_destroy.argtypes = [void_p]
    library.graspgenx_runtime_destroy.restype = None
    library.graspgenx_runtime_device.argtypes = [void_p, ctypes.POINTER(ctypes.c_int)]
    library.graspgenx_runtime_device.restype = ctypes.c_int
    library.graspgenx_runtime_grasp_count.argtypes = [
        void_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    library.graspgenx_runtime_grasp_count.restype = ctypes.c_int
    capacities = library.graspgenx_runtime_stage_capacities
    capacities.argtypes = [void_p, ctypes.POINTER(ctypes.c_int64)]
    capacities.restype = ctypes.c_int
    library.graspgenx_runtime_set_gripper_device_async.argtypes = [
        void_p,
        void_p,
        void_p,
        ctypes.c_float,
        void_p,
    ]
    library.graspgenx_runtime_set_gripper_device_async.restype = ctypes.c_int
    library.graspgenx_runtime_infer_device_async.argtypes = [
        void_p,
        void_p,
        ctypes.c_int64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        void_p,
        void_p,
        void_p,
        void_p,
        void_p,
        void_p,
        void_p,
        void_p,
    ]
    library.graspgenx_runtime_infer_device_async.restype = ctypes.c_int
    library.graspgenx_runtime_set_grasp_volume.argtypes = [
        void_p,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.graspgenx_runtime_set_grasp_volume.restype = ctypes.c_int
    library.graspgenx_runtime_set_minimum_points_in_volume.argtypes = [
        void_p,
        ctypes.c_int,
    ]
    library.graspgenx_runtime_set_minimum_points_in_volume.restype = ctypes.c_int
    library.graspgenx_runtime_copy_valid_async.argtypes = [void_p, void_p, void_p]
    library.graspgenx_runtime_copy_valid_async.restype = ctypes.c_int
    return library


@dataclass(frozen=True)
class NativeGripperEmbeddings:
    generator: torch.Tensor
    discriminator: torch.Tensor
    depth: float


@dataclass(frozen=True)
class NativeInferenceResult:
    grasps: torch.Tensor
    confidence: torch.Tensor
    source_indices: torch.Tensor
    object_center: torch.Tensor
    obb_bounds: torch.Tensor
    ptv3_stage_counts: torch.Tensor
    ptv3_topology_overflow: torch.Tensor
    valid: torch.Tensor


class NativeCppRuntime:
    """One-object native runtime with persistent PyTorch output storage.

    One instance supports one ordered in-flight sequence. Returned tensors are
    overwritten by the next call; application-level sampler methods return
    indexed tensors with independent storage.
    """

    def __init__(
        self,
        *,
        artifact_directory: str | os.PathLike[str] | None = None,
        library_path: str | os.PathLike[str] | None = None,
        diffusion_grasps: int = 100,
        obb_grasps: int = 50,
        device: int | torch.device | None = None,
        seed: int | None = None,
        denoiser: str = "ddpm",
        minimum_points_in_volume: int = 0,
    ):
        self.diffusion_grasps = int(diffusion_grasps)
        self.obb_grasps = int(obb_grasps)
        self.denoiser = str(denoiser)
        self.denoising_steps = denoising_steps(self.denoiser)
        if self.diffusion_grasps < 1:
            raise ValueError("diffusion_grasps must be positive")
        if self.obb_grasps < 0 or self.diffusion_grasps + self.obb_grasps > 4096:
            raise ValueError(
                "obb_grasps must be non-negative and combined candidates <= 4096"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("the native GraspGenX runtime requires CUDA")
        if device is None:
            device_index = torch.cuda.current_device()
        else:
            resolved = torch.device(device)
            if resolved.type != "cuda":
                raise ValueError(f"native runtime device must be CUDA, got {resolved}")
            device_index = (
                torch.cuda.current_device()
                if resolved.index is None
                else int(resolved.index)
            )
        self.artifact_directory = resolve_artifact_directory(
            artifact_directory,
            diffusion_grasps=self.diffusion_grasps,
            obb_grasps=self.obb_grasps,
            denoiser=self.denoiser,
        )
        self.library_path = resolve_runtime_library(library_path)
        # TensorRT's global creator registry retains pointers into this DSO.
        # Keep it loaded even after the last runtime instance is closed.
        if self.library_path not in _RUNTIME_LIBRARY_HANDLES:
            _RUNTIME_LIBRARY_HANDLES[self.library_path] = ctypes.CDLL(
                str(self.library_path)
            )
        self._library = _configure_library(_RUNTIME_LIBRARY_HANDLES[self.library_path])
        self._handle = ctypes.c_void_p()
        config = _ConfigV2(
            ctypes.sizeof(_ConfigV2),
            str(self.artifact_directory).encode(),
            ctypes.c_float(0.1034),
            0,
            self.diffusion_grasps,
            self.obb_grasps,
        )
        with torch.cuda.device(device_index):
            self._check(
                self._library.graspgenx_runtime_create_v2(
                    ctypes.byref(config), ctypes.byref(self._handle)
                ),
                "create native runtime",
            )
        native_device = ctypes.c_int(-1)
        self._check(
            self._library.graspgenx_runtime_device(
                self._handle, ctypes.byref(native_device)
            ),
            "query native runtime device",
        )
        native_capacities = (ctypes.c_int64 * 5)()
        self._check(
            self._library.graspgenx_runtime_stage_capacities(
                self._handle, native_capacities
            ),
            "query PTV3 capacities",
        )
        self.stage_capacities = tuple(native_capacities)
        self.device = torch.device("cuda", native_device.value)
        self.grasp_count = self.diffusion_grasps + self.obb_grasps
        native_count = ctypes.c_int(-1)
        self._check(
            self._library.graspgenx_runtime_grasp_count(
                self._handle, ctypes.byref(native_count)
            ),
            "query native grasp count",
        )
        if native_count.value != self.grasp_count:
            raise RuntimeError(
                f"native grasp count {native_count.value} != {self.grasp_count}"
            )
        self._result = NativeInferenceResult(
            grasps=torch.empty(
                (self.grasp_count, 4, 4), device=self.device, dtype=torch.float32
            ),
            confidence=torch.empty(
                self.grasp_count, device=self.device, dtype=torch.float32
            ),
            source_indices=torch.empty(
                self.grasp_count, device=self.device, dtype=torch.int32
            ),
            object_center=torch.empty(3, device=self.device, dtype=torch.float32),
            obb_bounds=torch.empty(8, device=self.device, dtype=torch.float32),
            ptv3_stage_counts=torch.empty(5, device=self.device, dtype=torch.int64),
            ptv3_topology_overflow=torch.empty(
                4, device=self.device, dtype=torch.uint8
            ),
            valid=torch.empty(self.grasp_count, device=self.device, dtype=torch.bool),
        )
        self.seed = int(torch.initial_seed() if seed is None else seed) & (
            (1 << 64) - 1
        )
        self.sequence_offset = 0
        self.set_minimum_points_in_volume(minimum_points_in_volume)

    def set_grasp_volume(self, bounds):
        """Set host FP32 [2,3] capture bounds, including gripper standoff."""
        values = np.ascontiguousarray(bounds, dtype=np.float32)
        if values.shape != (2, 3):
            raise ValueError("grasp volume bounds must have shape [2,3]")
        self._check(
            self._library.graspgenx_runtime_set_grasp_volume(
                self._handle, values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ),
            "set grasp volume",
        )

    def set_minimum_points_in_volume(self, minimum_points: int):
        minimum_points = int(minimum_points)
        if not 0 <= minimum_points <= ENGINE_POINT_COUNT:
            raise ValueError("minimum_points_in_volume must be in [0,3500]")
        self._check(
            self._library.graspgenx_runtime_set_minimum_points_in_volume(
                self._handle, minimum_points
            ),
            "set minimum points in volume",
        )

    def close(self):
        handle = getattr(self, "_handle", None)
        if handle is not None and handle.value:
            # Runtime destruction owns CUDA resources and is therefore ordered
            # only after all previously enqueued work is complete.
            torch.cuda.synchronize(self.device)
            self._library.graspgenx_runtime_destroy(handle)
            handle.value = None

    def __del__(self):  # pragma: no cover - interpreter shutdown is best effort
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter teardown is best effort
            return

    def _check(self, status: int, operation: str):
        if status == 0:
            return
        detail = self._library.graspgenx_runtime_last_error()
        message = detail.decode(errors="replace") if detail else "unknown error"
        raise RuntimeError(f"{operation}: {message}")

    @staticmethod
    def _pointer(tensor: torch.Tensor):
        return ctypes.c_void_p(tensor.data_ptr())

    def infer(
        self,
        points: torch.Tensor,
        gripper: NativeGripperEmbeddings,
        *,
        seed: int | None = None,
        sequence_offset: int | None = None,
    ) -> NativeInferenceResult:
        if points.ndim == 3 and points.shape[0] == 1:
            points = points[0]
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"points must have shape [N,3] or [1,N,3], got {points.shape}"
            )
        if len(points) < MIN_OBJECT_POINT_COUNT:
            raise ValueError(
                f"GraspGenX requires at least {MIN_OBJECT_POINT_COUNT} points; "
                f"received {len(points)}"
            )
        points = points.to(device=self.device, dtype=torch.float32).contiguous()
        generator = gripper.generator
        discriminator = gripper.discriminator
        if (
            generator.device != self.device
            or generator.dtype != torch.float16
            or tuple(generator.shape) != (512,)
            or not generator.is_contiguous()
        ):
            raise ValueError(
                "generator gripper embedding must be contiguous CUDA FP16 [512]"
            )
        if (
            discriminator.device != self.device
            or discriminator.dtype != torch.float32
            or tuple(discriminator.shape) != (512,)
            or not discriminator.is_contiguous()
        ):
            raise ValueError(
                "discriminator gripper embedding must be contiguous CUDA FP32 [512]"
            )
        caller_stream = torch.cuda.current_stream(self.device)
        stream_pointer = ctypes.c_void_p(caller_stream.cuda_stream)
        self._check(
            self._library.graspgenx_runtime_set_gripper_device_async(
                self._handle,
                self._pointer(generator),
                self._pointer(discriminator),
                ctypes.c_float(float(gripper.depth)),
                stream_pointer,
            ),
            "set native gripper",
        )
        call_seed = self.seed if seed is None else int(seed) & ((1 << 64) - 1)
        call_offset = (
            self.sequence_offset
            if sequence_offset is None
            else int(sequence_offset) & ((1 << 64) - 1)
        )
        output = self._result
        self._check(
            self._library.graspgenx_runtime_infer_device_async(
                self._handle,
                self._pointer(points),
                len(points),
                ctypes.c_uint64(call_seed),
                ctypes.c_uint64(call_offset),
                self._pointer(output.grasps),
                self._pointer(output.confidence),
                self._pointer(output.source_indices),
                self._pointer(output.object_center),
                self._pointer(output.obb_bounds),
                self._pointer(output.ptv3_stage_counts),
                self._pointer(output.ptv3_topology_overflow),
                stream_pointer,
            ),
            "native device inference",
        )
        self._check(
            self._library.graspgenx_runtime_copy_valid_async(
                self._handle, self._pointer(output.valid), stream_pointer
            ),
            "copy native validity",
        )
        if sequence_offset is None:
            random_pairs = self.denoising_steps * self.diffusion_grasps * 3
            self.sequence_offset = (self.sequence_offset + random_pairs) & (
                (1 << 64) - 1
            )
        return output


def encode_gripper_embeddings(
    model, descriptor, depth: float
) -> NativeGripperEmbeddings:
    """Encode a released 12D sweep-volume descriptor once for native inference."""
    generator = model.grasp_generator
    discriminator = model.grasp_discriminator
    if generator.gripper_backbone != "sweep_volume_v2" or (
        discriminator.gripper_backbone != "sweep_volume_v2"
    ):
        raise ValueError(
            "native C++ artifacts currently require sweep_volume_v2 gripper encoders"
        )
    device = next(generator.gripper_encoder.parameters()).device
    value = torch.as_tensor(descriptor, dtype=torch.float32, device=device).reshape(
        1, 12
    )
    with torch.inference_mode():
        generator_value = (
            generator.gripper_encoder(value).reshape(512).to(torch.float16).contiguous()
        )
        discriminator_value = (
            discriminator.gripper_encoder(value).reshape(512).float().contiguous()
        )
    return NativeGripperEmbeddings(
        generator=generator_value,
        discriminator=discriminator_value,
        depth=float(depth),
    )


__all__ = [
    "ENGINE_POINT_COUNT",
    "NativeCppRuntime",
    "NativeGripperEmbeddings",
    "NativeInferenceResult",
    "encode_gripper_embeddings",
    "required_runtime_artifacts",
    "resolve_artifact_directory",
    "resolve_runtime_library",
]
