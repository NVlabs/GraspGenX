// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define GRASPGENX_C_API __declspec(dllexport)
#else
#define GRASPGENX_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* graspgenx_runtime_handle;

enum {
  GRASPGENX_DENOISER_DDPM = 0,
};

// Size-tagged configuration for the fixed DDPM20 native runtime.
typedef struct graspgenx_runtime_c_config_v2 {
  size_t struct_size;
  const char* artifact_directory;
  float initial_gripper_depth;
  int denoiser_type;
  int diffusion_grasps;
  int obb_grasps;
} graspgenx_runtime_c_config_v2;

// Every fallible function returns zero on success. On failure, the returned
// nonzero status can be expanded with graspgenx_runtime_last_error().
GRASPGENX_C_API const char* graspgenx_runtime_last_error(void);
GRASPGENX_C_API int graspgenx_runtime_create_v2(
    const graspgenx_runtime_c_config_v2* config,
    graspgenx_runtime_handle* output);
GRASPGENX_C_API void graspgenx_runtime_destroy(
    graspgenx_runtime_handle runtime);

GRASPGENX_C_API int graspgenx_runtime_device(
    graspgenx_runtime_handle runtime, int* output_device);
GRASPGENX_C_API int graspgenx_runtime_grasp_count(
    graspgenx_runtime_handle runtime, int* output_count);
// Host output array of five int64 values.
GRASPGENX_C_API int graspgenx_runtime_stage_capacities(
    graspgenx_runtime_handle runtime, int64_t* output_capacities_5);

// Host bounds [xmin,ymin,zmin,xmax,ymax,zmax]. Controls wait for prior inference.
GRASPGENX_C_API int graspgenx_runtime_set_grasp_volume(
    graspgenx_runtime_handle runtime, const float* bounds_6);
GRASPGENX_C_API int graspgenx_runtime_set_minimum_points_in_volume(
    graspgenx_runtime_handle runtime, int minimum_points);
// Copy the last inference's validity mask to CUDA uint8 [grasp_count] storage.
// Enqueue on the inference stream before submitting another inference.
GRASPGENX_C_API int graspgenx_runtime_copy_valid_async(
    graspgenx_runtime_handle runtime, uint8_t* output_valid, void* stream);

GRASPGENX_C_API int graspgenx_runtime_set_gripper_device_async(
    graspgenx_runtime_handle runtime,
    const void* generator_embedding_fp16_512,
    const float* discriminator_embedding_fp32_512,
    float gripper_depth,
    void* stream);

// points and all non-null outputs are CUDA device pointers. The graph launch
// and output copies are enqueued on stream, which must be in the Runtime's
// primary execution domain. Output pointers may be null when not requested.
GRASPGENX_C_API int graspgenx_runtime_infer_device_async(
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
    void* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#undef GRASPGENX_C_API
