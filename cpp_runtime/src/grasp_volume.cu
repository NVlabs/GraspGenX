// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "grasp_volume.h"

#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <cmath>

namespace graspgenx {
namespace {
constexpr int kThreads = 128;

__device__ bool rigid_pose(const float* pose, const float* center) {
  for (int i = 0; i < 16; ++i)
    if (!isfinite(pose[i])) return false;
  for (int i = 0; i < 3; ++i)
    if (!isfinite(center[i])) return false;
  if (fabsf(pose[12]) + fabsf(pose[13]) + fabsf(pose[14]) +
      fabsf(pose[15] - 1.F) > 1.e-4F) return false;
  for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b) {
      float dot = 0.F;
      for (int r = 0; r < 3; ++r) dot += pose[r * 4 + a] * pose[r * 4 + b];
      if (fabsf(dot - (a == b ? 1.F : 0.F)) > 1.e-3F) return false;
    }
  const float det = pose[0] * (pose[5] * pose[10] - pose[6] * pose[9]) -
                    pose[1] * (pose[4] * pose[10] - pose[6] * pose[8]) +
                    pose[2] * (pose[4] * pose[9] - pose[5] * pose[8]);
  return fabsf(det - 1.F) <= 1.e-3F;
}

__global__ void filter_volume(
    const float* diffusion, const float* obb, const float* center,
    const float* points, const std::int64_t* point_count,
    const float* bounds, const int* minimum_points,
    float* confidence, int diffusion_count) {
  const int candidate = blockIdx.x;
  const float score = confidence[candidate];
  // All threads take these uniform branches, including the disabled case.
  if (!isfinite(score) || score < 0.F || score > 1.F) {
    if (threadIdx.x == 0) confidence[candidate] = -1.F;
    return;
  }
  const int minimum = *minimum_points;
  if (minimum == 0) return;
  const auto count = *point_count;
  const float* pose = candidate < diffusion_count
      ? diffusion + candidate * 16 : obb + (candidate - diffusion_count) * 16;
  __shared__ bool pose_ok;
  if (threadIdx.x == 0) pose_ok = rigid_pose(pose, center);
  __syncthreads();
  if (!pose_ok || count < minimum || count > 3500 || minimum < 0) {
    if (threadIdx.x == 0) confidence[candidate] = -1.F;
    return;
  }
  int inside = 0;
  for (std::int64_t i = threadIdx.x; i < count; i += blockDim.x) {
    const float x = points[i * 3] - (pose[3] + center[0]);
    const float y = points[i * 3 + 1] - (pose[7] + center[1]);
    const float z = points[i * 3 + 2] - (pose[11] + center[2]);
    if (!isfinite(x) || !isfinite(y) || !isfinite(z)) continue;
    // Inverse of the proper rigid pose: R^T (point - translation).
    const float px = pose[0] * x + pose[4] * y + pose[8] * z;
    const float py = pose[1] * x + pose[5] * y + pose[9] * z;
    const float pz = pose[2] * x + pose[6] * y + pose[10] * z;
    inside += px >= bounds[0] && px <= bounds[3] &&
              py >= bounds[1] && py <= bounds[4] &&
              pz >= bounds[2] && pz <= bounds[5];
  }
  using Reduce = cub::BlockReduce<int, kThreads>;
  __shared__ typename Reduce::TempStorage storage;
  const int total = Reduce(storage).Sum(inside);
  if (threadIdx.x == 0 && total < minimum) confidence[candidate] = -1.F;
}

__global__ void expose_validity(float* confidence, std::uint8_t* valid, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float score = confidence[i];
    valid[i] = isfinite(score) && score >= 0.F && score <= 1.F;
    if (!valid[i]) confidence[i] = 0.F;
  }
}
}  // namespace

cudaError_t LaunchFilterGraspVolume(
    const float* diffusion_grasps, const float* obb_grasps,
    const float* center, const float* points, const std::int64_t* point_count,
    const float* bounds, const int* minimum_points, float* confidence,
    int diffusion_count, int obb_count, cudaStream_t stream) {
  if (diffusion_count < 1 || obb_count < 0 || diffusion_count > 4096 - obb_count)
    return cudaErrorInvalidValue;
  filter_volume<<<diffusion_count + obb_count, kThreads, 0, stream>>>(
      diffusion_grasps, obb_grasps, center, points, point_count, bounds,
      minimum_points, confidence, diffusion_count);
  return cudaPeekAtLastError();
}

cudaError_t LaunchGraspValidity(float* confidence, std::uint8_t* valid,
                                int count, cudaStream_t stream) {
  if (count < 1 || count > 4096) return cudaErrorInvalidValue;
  expose_validity<<<(count + 255) / 256, 256, 0, stream>>>(confidence, valid, count);
  return cudaPeekAtLastError();
}
}  // namespace graspgenx
