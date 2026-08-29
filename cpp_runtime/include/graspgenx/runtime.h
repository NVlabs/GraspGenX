// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace graspgenx {

struct RuntimeConfig {
  std::filesystem::path artifact_directory;
  float gripper_depth = 0.1034F;
  int diffusion_steps = 20;
  // These select count-specific dense plans and become immutable when the
  // Runtime captures its graph. They do not require recompiling the library.
  int diffusion_grasps = 100;
  int obb_grasps = 50;
};

struct DeviceOutputs {
  const float* grasps = nullptr;       // [grasp_count,4,4], descending score
  const float* confidence = nullptr;   // [grasp_count]
  const std::int32_t* source_indices = nullptr;  // diffusion followed by OBB
  const float* object_center = nullptr;          // [3]
  const float* obb_bounds = nullptr;             // [8]
  const std::int64_t* ptv3_stage_counts = nullptr;  // [5]
  const bool* ptv3_topology_overflow = nullptr;     // [4], invalid if any true
  int grasp_count = 0;
};

struct RuntimeMemory {
  std::size_t persistent_cuda_bytes = 0;
  std::size_t tensorrt_device_memory_bytes = 0;
};

struct RuntimeProfile {
  float preprocessing_topology_obb_ms = 0.0F;
  float preprocessing_ms = 0.0F;
  float topology_ms = 0.0F;
  float obb_noise_ms = 0.0F;
  float dual_ptv3_ms = 0.0F;
  float denoising_steps_ms = 0.0F;
  float scorer_rank_ms = 0.0F;
  float full_pipeline_ms = 0.0F;
};

class Runtime {
 public:
  static constexpr std::int64_t kEnginePointCount = 3500;
  static constexpr std::int64_t kMinimumPointCount = 100;

  explicit Runtime(RuntimeConfig config);
  ~Runtime();

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
  Runtime(Runtime&&) noexcept;
  Runtime& operator=(Runtime&&) noexcept;

  // The neural plans consume encoded gripper features. Keeping both gripper
  // encoders outside the plans lets the same plans serve different grippers.
  void SetGripperEmbeddings(const float* generator_embedding_512,
                            const float* discriminator_embedding_512);

  // Steady-state Python/interop path. Inputs are device-resident generator
  // FP16 [512] and discriminator FP32 [512] embeddings. The copies and depth
  // update are ordered on stream before the next inference graph launch.
  void SetGripperDeviceAsync(const void* generator_embedding_fp16_512,
                             const float* discriminator_embedding_fp32_512,
                             float gripper_depth, cudaStream_t stream);

  // Zero-copy path: fill the first point_count rows of this persistent
  // [3500,3] FP32 CUDA buffer, then call InferAsync on the producer stream.
  // This path accepts [100,3500] rows.
  float* device_input_points() noexcept;

  // Variable-count input. At least 100 points are required. Inputs below
  // 3500 are filtered and repeated to the fixed engine shape; larger inputs
  // are deterministically downsampled to 3500 before filtering. Device and
  // host pointers are contiguous FP32 [point_count,3] arrays.
  void InferAsync(std::int64_t point_count, std::uint64_t seed,
                  std::uint64_t sequence_offset, cudaStream_t stream);
  void InferDeviceAsync(const float* points, std::int64_t point_count,
                        std::uint64_t seed, std::uint64_t sequence_offset,
                        cudaStream_t stream);
  void InferHostAsync(const float* points, std::int64_t point_count,
                      std::uint64_t seed, std::uint64_t sequence_offset,
                      cudaStream_t stream);

  DeviceOutputs outputs() const noexcept;
  RuntimeMemory memory() const noexcept;
  int device() const noexcept;
  RuntimeProfile Profile(int warmup_iterations = 100,
                         int measured_iterations = 1000);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace graspgenx
