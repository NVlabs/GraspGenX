// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace graspgenx {

// All pointers are device pointers. FP16 arguments use CUDA's two-byte half
// storage but remain void* here so C++ callers do not need CUDA vector headers.
cudaError_t LaunchDenoiseSampleEncodeFp16(
    const void* sample, const void* weight, const void* bias,
    const void* observation_projection, const void* timestep_conditioning,
    void* sample_output, void* conditioning_output, std::int64_t rows,
    std::int64_t object_count, cudaStream_t stream);

cudaError_t LaunchDenoiseStepFp16(
    const void* prediction_hidden, const void* sample,
    const void* variance_noise, const void* weight, const void* bias,
    const void* coefficients, void* output, std::int64_t rows,
    cudaStream_t stream);

cudaError_t LaunchDenoiseStepAndSampleEncodeFp16(
    const void* prediction_hidden, const void* sample,
    const void* variance_noise, const void* prediction_weight,
    const void* prediction_bias, const void* coefficients, void* output,
    const void* sample_weight, const void* sample_bias,
    const void* observation_projection, const void* timestep_conditioning,
    void* sample_output, void* conditioning_output, std::int64_t rows,
    std::int64_t object_count, cudaStream_t stream);

cudaError_t LaunchFinalizeSo3Fp16(
    const void* latent, float* matrices, void* canonical_rt,
    std::int64_t grasp_count, float kappa, cudaStream_t stream);

cudaError_t LaunchObbTopdownFloatDeviceDepth(
    const float* points, std::int64_t point_count, float* matrices,
    float* canonical_rt, float* bounds, const float* gripper_depth, float kappa,
    std::int64_t grasp_count, cudaStream_t stream);

cudaError_t LaunchSegmentMaxFp16(
    const void* features, const std::int64_t* indices,
    const std::int64_t* index_ptr, const std::int64_t* segment_count,
    void* output, std::int64_t output_capacity, std::int64_t channels,
    std::int64_t input_capacity, cudaStream_t stream);

cudaError_t LaunchMeanFp16(
    const void* features, const std::int64_t* valid_count, void* output,
    std::int64_t channels, std::int64_t input_capacity, cudaStream_t stream);

cudaError_t LaunchFilterResampleDynamicFloat(
    const float* points, const std::int64_t* point_count,
    std::int64_t maximum_point_count, std::int64_t output_count,
    float threshold, bool* keep, std::int64_t* selected_indices,
    std::int32_t* selected_count, float* output, cudaStream_t stream);

cudaError_t LaunchResampleFixedFloat(
    const float* points, std::int64_t point_count, float* output,
    std::int64_t output_count, cudaStream_t stream);

cudaError_t LaunchCenterScaleAndGridFloat(
    const float* points, std::int64_t point_count, float kappa,
    float grid_size, float* centered_points, float* center,
    float* scaled_coordinates, void* fp16_features,
    std::int32_t* grid_coordinates, std::int32_t* maximum_coordinate,
    cudaStream_t stream);

cudaError_t LaunchSerializationAndSort(
    const std::int32_t* grid_coordinates, std::int64_t capacity,
    std::int32_t* maximum_coordinate, std::int64_t* codes,
    std::int64_t* order, std::int64_t* inverse,
    std::int64_t order_count, cudaStream_t stream);

cudaError_t LaunchNeighborMapPadded(
    const std::int32_t* grid_coordinates, const std::int64_t* morton_codes,
    const std::int64_t* morton_order, const std::int64_t* valid_count,
    const std::int64_t* offsets, std::int64_t* neighbor_indices,
    bool* neighbor_mask, std::int64_t capacity, std::int64_t offset_count,
    cudaStream_t stream);

cudaError_t LaunchSerializedPoolingPadded(
    const float* coordinates, const std::int32_t* grid_coordinates,
    const std::int64_t* codes, const std::int64_t* order,
    const std::int64_t* input_valid_count, std::int64_t input_capacity,
    std::int64_t output_capacity, int pooling_depth,
    std::int64_t* pool_index_ptr, std::int64_t* cluster,
    std::int64_t* segment_ids, float* pooled_coordinates,
    std::int32_t* pooled_grid_coordinates, std::int64_t* pooled_codes,
    std::int64_t* pooled_order, std::int64_t* pooled_inverse,
    std::int64_t* output_valid_count, bool* overflow,
    std::int64_t output_order_count, std::int64_t derived_order_count,
    const std::int32_t* maximum_coordinate,
    std::int32_t maximum_supported_coordinate,
    std::int32_t accumulated_pooling_depth, cudaStream_t stream,
    cudaStream_t order_stream, cudaEvent_t pool_data_ready,
    cudaEvent_t order_ready);

cudaError_t LaunchPackAttentionTopology(
    const std::int64_t* serialized_order,
    const std::int64_t* serialized_inverse,
    const std::int64_t* valid_count, std::int64_t* attention_order,
    std::int64_t* attention_inverse, std::int32_t* valid_key_counts,
    std::int64_t capacity, std::int64_t padded_capacity,
    std::int64_t patch_size, std::int64_t order_count, cudaStream_t stream);

cudaError_t LaunchNormalNoiseFp16(
    const std::uint64_t* seed, const std::uint64_t* sequence_offset,
    void* output, std::int64_t element_count, cudaStream_t stream);

cudaError_t LaunchPrepareScorerInputs(
    const void* diffusion_canonical_fp16, const float* obb_canonical,
    const void* discriminator_embedding_fp16, float* canonical_fp32,
    float* object_embedding_fp32, std::int64_t diffusion_count,
    std::int64_t obb_count, cudaStream_t stream);

cudaError_t LaunchRankAndTranslate(
    const float* diffusion_grasps, const float* obb_grasps,
    const float* confidence, const float* center, float* sorted_grasps,
    float* sorted_confidence, std::int32_t* source_indices,
    std::int64_t diffusion_count, std::int64_t obb_count,
    cudaStream_t stream);

}  // namespace graspgenx
