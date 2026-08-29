// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "graspgenx/c_api.h"

#include "graspgenx/runtime.h"

#include <cuda_runtime_api.h>

#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

thread_local std::string g_last_error;

struct RuntimeHandle {
  explicit RuntimeHandle(graspgenx::RuntimeConfig config)
      : diffusion_grasps(config.diffusion_grasps),
        obb_grasps(config.obb_grasps),
        runtime(std::move(config)) {}

  int diffusion_grasps = 0;
  int obb_grasps = 0;
  graspgenx::Runtime runtime;
};

RuntimeHandle& Get(graspgenx_runtime_handle handle) {
  if (handle == nullptr) throw std::invalid_argument("runtime handle is null");
  return *static_cast<RuntimeHandle*>(handle);
}

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

template <typename Function>
int Guard(Function&& function) noexcept {
  try {
    function();
    g_last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    g_last_error = error.what();
  } catch (...) {
    g_last_error = "unknown native GraspGenX runtime error";
  }
  return 1;
}

template <typename T>
void CopyOptional(T* destination, const T* source, std::size_t count,
                  cudaStream_t stream, const char* operation) {
  if (destination == nullptr) return;
  if (source == nullptr) throw std::runtime_error(std::string(operation) +
                                                  ": source is null");
  CheckCuda(cudaMemcpyAsync(destination, source, count * sizeof(T),
                            cudaMemcpyDeviceToDevice, stream),
            operation);
}

}  // namespace

extern "C" {

const char* graspgenx_runtime_last_error(void) { return g_last_error.c_str(); }

int graspgenx_runtime_create(const graspgenx_runtime_c_config* config,
                            graspgenx_runtime_handle* output) {
  return Guard([&] {
    if (config == nullptr) throw std::invalid_argument("config is null");
    if (output == nullptr) throw std::invalid_argument("output handle is null");
    if (config->artifact_directory == nullptr ||
        config->artifact_directory[0] == '\0') {
      throw std::invalid_argument("artifact_directory is required");
    }
    graspgenx::RuntimeConfig native;
    native.artifact_directory =
        std::filesystem::path(config->artifact_directory);
    native.gripper_depth = config->initial_gripper_depth;
    native.diffusion_steps = config->diffusion_steps;
    native.diffusion_grasps = config->diffusion_grasps;
    native.obb_grasps = config->obb_grasps;
    auto runtime = std::make_unique<RuntimeHandle>(std::move(native));
    *output = runtime.release();
  });
}

void graspgenx_runtime_destroy(graspgenx_runtime_handle runtime) {
  delete static_cast<RuntimeHandle*>(runtime);
}

int graspgenx_runtime_device(graspgenx_runtime_handle runtime,
                            int* output_device) {
  return Guard([&] {
    if (output_device == nullptr) throw std::invalid_argument("output is null");
    *output_device = Get(runtime).runtime.device();
  });
}

int graspgenx_runtime_grasp_count(graspgenx_runtime_handle runtime,
                                 int* output_count) {
  return Guard([&] {
    if (output_count == nullptr) throw std::invalid_argument("output is null");
    const auto& value = Get(runtime);
    *output_count = value.diffusion_grasps + value.obb_grasps;
  });
}

int graspgenx_runtime_set_gripper_device_async(
    graspgenx_runtime_handle runtime,
    const void* generator_embedding_fp16_512,
    const float* discriminator_embedding_fp32_512,
    float gripper_depth,
    void* stream) {
  return Guard([&] {
    Get(runtime).runtime.SetGripperDeviceAsync(
        generator_embedding_fp16_512, discriminator_embedding_fp32_512,
        gripper_depth, static_cast<cudaStream_t>(stream));
  });
}

int graspgenx_runtime_infer_device_async(
    graspgenx_runtime_handle runtime,
    const float* points,
    int64_t point_count,
    uint64_t seed,
    uint64_t sequence_offset,
    float* output_grasps,
    float* output_confidence,
    int32_t* output_source_indices,
    float* output_object_center,
    float* output_obb_bounds,
    int64_t* output_ptv3_stage_counts,
    uint8_t* output_ptv3_topology_overflow,
    void* stream_pointer) {
  return Guard([&] {
    auto& value = Get(runtime);
    auto stream = static_cast<cudaStream_t>(stream_pointer);
    value.runtime.InferDeviceAsync(points, point_count, seed, sequence_offset,
                                   stream);
    const graspgenx::DeviceOutputs output = value.runtime.outputs();
    const std::size_t count = static_cast<std::size_t>(output.grasp_count);
    CopyOptional(output_grasps, output.grasps, count * 16, stream,
                 "copy output grasps");
    CopyOptional(output_confidence, output.confidence, count, stream,
                 "copy output confidence");
    CopyOptional(output_source_indices, output.source_indices, count, stream,
                 "copy output source indices");
    CopyOptional(output_object_center, output.object_center, 3, stream,
                 "copy output object center");
    CopyOptional(output_obb_bounds, output.obb_bounds, 8, stream,
                 "copy output OBB bounds");
    CopyOptional(output_ptv3_stage_counts, output.ptv3_stage_counts, 5, stream,
                 "copy output PTV3 stage counts");
    static_assert(sizeof(bool) == sizeof(uint8_t));
    CopyOptional(output_ptv3_topology_overflow,
                 reinterpret_cast<const uint8_t*>(
                     output.ptv3_topology_overflow),
                 4, stream, "copy output PTV3 overflow");
  });
}

}  // extern "C"
