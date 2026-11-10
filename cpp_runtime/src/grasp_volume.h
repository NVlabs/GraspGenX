// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace graspgenx {

// Scorer poses are centered; input points and center use the input cloud frame.
// Rejected scores become -1 before ranking, below even valid zero scores.
cudaError_t LaunchFilterGraspVolume(
    const float* diffusion_grasps, const float* obb_grasps,
    const float* center, const float* points, const std::int64_t* point_count,
    const float* bounds, const int* minimum_points, float* confidence,
    int diffusion_count, int obb_count, cudaStream_t stream);

// Run after ranking to expose aligned validity and restore probability scores.
cudaError_t LaunchGraspValidity(float* confidence, std::uint8_t* valid,
                                int count, cudaStream_t stream);

}  // namespace graspgenx
