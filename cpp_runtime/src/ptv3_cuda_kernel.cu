// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Native preprocessing, PTV3 topology, diffusion, OBB, and ranking kernels.

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

using NativeHalf = __half;

constexpr int kThreads = 256;
constexpr int kFilterNeighbors = 20;
constexpr int kFilterGroupSize = 32;
constexpr int kFilterThreads = 128;
constexpr int kFilterQueriesPerBlock = kFilterThreads / kFilterGroupSize;
constexpr float kPi = 3.14159265358979323846F;
constexpr int kObbSortCapacity = 4096;
constexpr int kObbSortThreads = 1024;
constexpr int kDenoiseSampleDim = 6;
constexpr int kDenoiseHiddenDim = 512;
constexpr int kFixedAlgorithmCapacity = 4096;
constexpr int kFixedAlgorithmItemsPerThread =
    kFixedAlgorithmCapacity / kThreads;

// These fixed-size block collectives cover every deployment capacity
// (3500, 1536, 512, 128, 64). All callers launch exactly kThreads threads.
__global__ void center_scale_grid_kernel(
    const float* __restrict__ points,
    float* __restrict__ centered_points,
    float* __restrict__ center,
    float* __restrict__ scaled_coordinates,
    NativeHalf* __restrict__ fp16_features,
    int32_t* __restrict__ grid_coordinates,
    int32_t* __restrict__ maximum_coordinate,
    int64_t point_count,
    float kappa,
    float inverse_grid_size) {
  __shared__ float reduction_x[kThreads];
  __shared__ float reduction_y[kThreads];
  __shared__ float reduction_z[kThreads];
  __shared__ int32_t maximum_grid[kThreads];
  float sum_x = 0.0F;
  float sum_y = 0.0F;
  float sum_z = 0.0F;
  for (int64_t point = threadIdx.x; point < point_count; point += blockDim.x) {
    sum_x += points[point * 3 + 0];
    sum_y += points[point * 3 + 1];
    sum_z += points[point * 3 + 2];
  }
  reduction_x[threadIdx.x] = sum_x;
  reduction_y[threadIdx.x] = sum_y;
  reduction_z[threadIdx.x] = sum_z;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction_x[threadIdx.x] += reduction_x[threadIdx.x + stride];
      reduction_y[threadIdx.x] += reduction_y[threadIdx.x + stride];
      reduction_z[threadIdx.x] += reduction_z[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x < 3) {
    center[threadIdx.x] =
        (threadIdx.x == 0 ? reduction_x[0]
         : threadIdx.x == 1 ? reduction_y[0]
                            : reduction_z[0])
        / static_cast<float>(point_count);
  }
  __syncthreads();

  float minimum_x = 3.402823466e+38F;
  float minimum_y = 3.402823466e+38F;
  float minimum_z = 3.402823466e+38F;
  for (int64_t point = threadIdx.x; point < point_count; point += blockDim.x) {
    const float x = points[point * 3 + 0] - center[0];
    const float y = points[point * 3 + 1] - center[1];
    const float z = points[point * 3 + 2] - center[2];
    const float sx = x * kappa;
    const float sy = y * kappa;
    const float sz = z * kappa;
    centered_points[point * 3 + 0] = x;
    centered_points[point * 3 + 1] = y;
    centered_points[point * 3 + 2] = z;
    scaled_coordinates[point * 3 + 0] = sx;
    scaled_coordinates[point * 3 + 1] = sy;
    scaled_coordinates[point * 3 + 2] = sz;
    fp16_features[point * 3 + 0] = static_cast<NativeHalf>(sx);
    fp16_features[point * 3 + 1] = static_cast<NativeHalf>(sy);
    fp16_features[point * 3 + 2] = static_cast<NativeHalf>(sz);
    minimum_x = fminf(minimum_x, sx);
    minimum_y = fminf(minimum_y, sy);
    minimum_z = fminf(minimum_z, sz);
  }
  reduction_x[threadIdx.x] = minimum_x;
  reduction_y[threadIdx.x] = minimum_y;
  reduction_z[threadIdx.x] = minimum_z;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction_x[threadIdx.x] =
          fminf(reduction_x[threadIdx.x], reduction_x[threadIdx.x + stride]);
      reduction_y[threadIdx.x] =
          fminf(reduction_y[threadIdx.x], reduction_y[threadIdx.x + stride]);
      reduction_z[threadIdx.x] =
          fminf(reduction_z[threadIdx.x], reduction_z[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  int32_t thread_maximum = 0;
  for (int64_t point = threadIdx.x; point < point_count; point += blockDim.x) {
    const int32_t gx = static_cast<int32_t>(
        (scaled_coordinates[point * 3 + 0] - reduction_x[0])
        * inverse_grid_size);
    const int32_t gy = static_cast<int32_t>(
        (scaled_coordinates[point * 3 + 1] - reduction_y[0])
        * inverse_grid_size);
    const int32_t gz = static_cast<int32_t>(
        (scaled_coordinates[point * 3 + 2] - reduction_z[0])
        * inverse_grid_size);
    grid_coordinates[point * 3 + 0] = gx;
    grid_coordinates[point * 3 + 1] = gy;
    grid_coordinates[point * 3 + 2] = gz;
    thread_maximum = max(thread_maximum, max(gx, max(gy, gz)));
  }
  maximum_grid[threadIdx.x] = thread_maximum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      maximum_grid[threadIdx.x] =
          max(maximum_grid[threadIdx.x], maximum_grid[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    maximum_coordinate[0] = maximum_grid[0];
  }
}

__global__ void fixed_stable_sort_pairs_kernel(
    const int64_t* __restrict__ keys,
    int64_t* __restrict__ order,
    int capacity,
    const int32_t* __restrict__ maximum_coordinate,
    int depth_reduction) {
  using BlockSort = cub::BlockRadixSort<
      uint32_t, kThreads, kFixedAlgorithmItemsPerThread, uint16_t, 5>;
  __shared__ typename BlockSort::TempStorage sort_storage;
  uint32_t thread_keys[kFixedAlgorithmItemsPerThread];
  uint16_t thread_indices[kFixedAlgorithmItemsPerThread];
  const uint32_t maximum = static_cast<uint32_t>(maximum_coordinate[0]);
  const int original_depth = maximum == 0 ? 0 : 32 - __clz(maximum);
  const int depth = original_depth > depth_reduction
      ? original_depth - depth_reduction
      : 0;
  const int end_bit = min(depth * 3, 30);
  const int row = static_cast<int>(blockIdx.x);
  keys += static_cast<int64_t>(row) * capacity;
  order += static_cast<int64_t>(row) * capacity;
#pragma unroll
  for (int item = 0; item < kFixedAlgorithmItemsPerThread; ++item) {
    const int index = threadIdx.x * kFixedAlgorithmItemsPerThread + item;
    if (index < capacity) {
      thread_keys[item] = static_cast<uint32_t>(keys[index]);
      thread_indices[item] = static_cast<uint16_t>(index);
    } else {
      thread_keys[item] = ~uint32_t{0};
      thread_indices[item] = 0;
    }
  }
  // CUB's block radix sort is stable, so duplicate voxel codes retain their
  // input order without packing the original index into a narrower key.
  BlockSort(sort_storage).Sort(thread_keys, thread_indices, 0, end_bit);
#pragma unroll
  for (int item = 0; item < kFixedAlgorithmItemsPerThread; ++item) {
    const int output_index =
        threadIdx.x * kFixedAlgorithmItemsPerThread + item;
    if (output_index < capacity) {
      const int source = thread_indices[item];
      order[output_index] = source;
    }
  }
}

__global__ void fixed_stable_sort_pairs_128_kernel(
    const int64_t* __restrict__ keys,
    int64_t* __restrict__ order,
    int capacity,
    const int32_t* __restrict__ maximum_coordinate,
    int depth_reduction) {
  using BlockSort = cub::BlockRadixSort<uint32_t, 128, 1, uint16_t, 5>;
  __shared__ typename BlockSort::TempStorage sort_storage;
  const int row = static_cast<int>(blockIdx.x);
  const uint32_t maximum = static_cast<uint32_t>(maximum_coordinate[0]);
  const int original_depth = maximum == 0 ? 0 : 32 - __clz(maximum);
  const int depth = original_depth > depth_reduction
      ? original_depth - depth_reduction
      : 0;
  const int end_bit = min(depth * 3, 30);
  keys += static_cast<int64_t>(row) * capacity;
  order += static_cast<int64_t>(row) * capacity;
  uint32_t key[1];
  uint16_t index[1];
  key[0] = threadIdx.x < capacity
      ? static_cast<uint32_t>(keys[threadIdx.x])
      : UINT32_MAX;
  index[0] = static_cast<uint16_t>(threadIdx.x);
  BlockSort(sort_storage).Sort(key, index, 0, end_bit);
  if (threadIdx.x < capacity) {
    order[threadIdx.x] = index[0];
  }
}

template <int BlockThreads, int ItemsPerThread>
__global__ void fixed_pool_boundaries_kernel(
    const int64_t* __restrict__ serialized_code,
    const int64_t* __restrict__ order,
    const int64_t* __restrict__ input_valid_count,
    int64_t* __restrict__ output_valid_count,
    bool* __restrict__ overflow,
    int64_t* __restrict__ index_ptr,
    int input_capacity,
    int output_capacity,
    int shift,
    const int32_t* __restrict__ maximum_coordinate,
    int maximum_supported_coordinate) {
  using BlockScan = cub::BlockScan<int, BlockThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  int flags[ItemsPerThread];
  int local_prefix[ItemsPerThread];
  int local_total = 0;
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int index = threadIdx.x * ItemsPerThread + item;
    const int boundary = index < input_capacity
        && (index == 0
            || (serialized_code[order[index]] >> shift)
                != (serialized_code[order[index - 1]] >> shift));
    flags[item] = boundary;
    local_prefix[item] = local_total;
    local_total += boundary;
  }
  int thread_prefix;
  int total_runs;
  BlockScan(scan_storage).ExclusiveSum(
      local_total, thread_prefix, total_runs);
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int index = threadIdx.x * ItemsPerThread + item;
    if (index < input_capacity && flags[item]) {
      const int run = thread_prefix + local_prefix[item];
      if (run <= output_capacity) {
        index_ptr[run] = index;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int segments = total_runs;
    const int valid = static_cast<int>(input_valid_count[0]);
    if (valid < input_capacity) {
      --segments;
    }
    overflow[0] = segments > output_capacity
        || (maximum_coordinate != nullptr
            && maximum_coordinate[0] > maximum_supported_coordinate);
    const int output_segments =
        segments < output_capacity ? segments : output_capacity;
    output_valid_count[0] = output_segments;
    if (segments <= output_capacity) {
      index_ptr[output_segments] = valid;
    }
  }
}

template <int BlockThreads, int ItemsPerThread>
__global__ void fixed_derived_pool_order_kernel(
    const int64_t* __restrict__ input_order,
    const int64_t* __restrict__ cluster,
    const int64_t* __restrict__ input_valid_count,
    const int64_t* __restrict__ output_valid_count,
    int64_t* __restrict__ pooled_order,
    int64_t* __restrict__ pooled_inverse,
    int input_capacity,
    int output_capacity,
    int first_row) {
  using BlockScan = cub::BlockScan<int, BlockThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  const int valid = static_cast<int>(input_valid_count[0]);
  const int segments = static_cast<int>(output_valid_count[0]);
  const int row = first_row + static_cast<int>(blockIdx.x);
  for (int index = threadIdx.x; index < output_capacity;
       index += blockDim.x) {
    pooled_order[row * output_capacity + index] = index;
    pooled_inverse[row * output_capacity + index] = index;
  }
  __syncthreads();
  int flags[ItemsPerThread];
  int local_prefix[ItemsPerThread];
  int local_total = 0;
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int position = threadIdx.x * ItemsPerThread + item;
    int boundary = 0;
    if (position < valid) {
      const int64_t parent =
          cluster[input_order[row * input_capacity + position]];
      boundary = position == 0
          || parent
              != cluster[input_order[row * input_capacity + position - 1]];
    }
    flags[item] = boundary;
    local_prefix[item] = local_total;
    local_total += boundary;
  }
  int thread_prefix;
  int total_runs;
  BlockScan(scan_storage).ExclusiveSum(
      local_total, thread_prefix, total_runs);
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int position = threadIdx.x * ItemsPerThread + item;
    if (position < valid && flags[item]) {
      const int output_position = thread_prefix + local_prefix[item];
      if (output_position < segments) {
        const int64_t parent =
            cluster[input_order[row * input_capacity + position]];
        pooled_order[row * output_capacity + output_position] = parent;
        pooled_inverse[row * output_capacity + parent] = output_position;
      }
    }
  }
}

using FixedBlockScan = cub::BlockScan<int, kThreads>;

template <typename scalar_t>
__device__ __forceinline__ void compact_resample_impl(
    const scalar_t* __restrict__ points,
    const bool* __restrict__ keep,
    int64_t* __restrict__ selected_indices,
    int32_t* __restrict__ selected_count,
    scalar_t* __restrict__ output,
    int input_count,
    int output_count,
    FixedBlockScan::TempStorage& scan_storage) {
  int flags[kFixedAlgorithmItemsPerThread];
  int local_prefix[kFixedAlgorithmItemsPerThread];
  int local_total = 0;
#pragma unroll
  for (int item = 0; item < kFixedAlgorithmItemsPerThread; ++item) {
    const int index = threadIdx.x * kFixedAlgorithmItemsPerThread + item;
    const int selected = index < input_count && keep[index];
    flags[item] = selected;
    local_prefix[item] = local_total;
    local_total += selected;
  }
  int thread_prefix;
  int total_selected;
  FixedBlockScan(scan_storage).ExclusiveSum(
      local_total, thread_prefix, total_selected);
#pragma unroll
  for (int item = 0; item < kFixedAlgorithmItemsPerThread; ++item) {
    const int index = threadIdx.x * kFixedAlgorithmItemsPerThread + item;
    if (index < input_count && flags[item]) {
      selected_indices[thread_prefix + local_prefix[item]] = index;
    }
  }
  if (threadIdx.x == 0) {
    selected_count[0] = total_selected;
  }
  __syncthreads();
  for (int output_index = threadIdx.x; output_index < output_count;
       output_index += blockDim.x) {
    int input_index;
    if (total_selected <= 0) {
      input_index = output_index % input_count;
    } else if (total_selected >= output_count) {
      input_index = selected_indices[
          (static_cast<int64_t>(output_index) * total_selected) / output_count];
    } else {
      input_index = selected_indices[output_index % total_selected];
    }
    output[output_index * 3 + 0] = points[input_index * 3 + 0];
    output[output_index * 3 + 1] = points[input_index * 3 + 1];
    output[output_index * 3 + 2] = points[input_index * 3 + 2];
  }
}

template <typename scalar_t>
__global__ void compact_resample_kernel(
    const scalar_t* __restrict__ points,
    const bool* __restrict__ keep,
    int64_t* __restrict__ selected_indices,
    int32_t* __restrict__ selected_count,
    scalar_t* __restrict__ output,
    int input_count,
    int output_count) {
  __shared__ FixedBlockScan::TempStorage scan_storage;
  compact_resample_impl(
      points, keep, selected_indices, selected_count, output, input_count,
      output_count, scan_storage);
}

template <typename scalar_t>
__global__ void compact_resample_dynamic_kernel(
    const scalar_t* __restrict__ points,
    const bool* __restrict__ keep,
    int64_t* __restrict__ selected_indices,
    int32_t* __restrict__ selected_count,
    scalar_t* __restrict__ output,
    const int64_t* __restrict__ input_count,
    int output_count) {
  __shared__ FixedBlockScan::TempStorage scan_storage;
  compact_resample_impl(
      points, keep, selected_indices, selected_count, output,
      static_cast<int>(input_count[0]), output_count, scan_storage);
}

template <typename scalar_t>
__global__ void resample_fixed_kernel(
    const scalar_t* __restrict__ points,
    int64_t input_count,
    scalar_t* __restrict__ output,
    int64_t output_count) {
  const int64_t output_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= output_count) {
    return;
  }
  int64_t input_index;
  if (input_count >= output_count) {
    // floor(output_index * input_count / output_count), written without a
    // potentially overflowing multiplication for very large input clouds.
    const int64_t quotient = input_count / output_count;
    const int64_t remainder = input_count % output_count;
    input_index = output_index * quotient
        + (output_index * remainder) / output_count;
  } else {
    input_index = output_index % input_count;
  }
  output[output_index * 3 + 0] = points[input_index * 3 + 0];
  output[output_index * 3 + 1] = points[input_index * 3 + 1];
  output[output_index * 3 + 2] = points[input_index * 3 + 2];
}

__device__ __forceinline__ uint64_t splitmix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

__global__ void normal_noise_fp16_kernel(
    const uint64_t* __restrict__ seed,
    const uint64_t* __restrict__ sequence_offset,
    NativeHalf* __restrict__ output,
    int64_t element_count) {
  const int64_t pair =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t first = pair * 2;
  if (first >= element_count) {
    return;
  }
  const uint64_t base = sequence_offset[0] + static_cast<uint64_t>(pair);
  const uint64_t random_a = splitmix64(seed[0] ^ (base * 2));
  const uint64_t random_b = splitmix64(seed[0] ^ (base * 2 + 1));
  const float uniform_a =
      (static_cast<float>((random_a >> 40) + 1ULL)) * (1.0F / 16777217.0F);
  const float uniform_b =
      static_cast<float>(random_b >> 40) * (1.0F / 16777216.0F);
  const float radius = sqrtf(-2.0F * logf(uniform_a));
  float sine;
  float cosine;
  sincosf(2.0F * kPi * uniform_b, &sine, &cosine);
  output[first] = static_cast<NativeHalf>(radius * cosine);
  if (first + 1 < element_count) {
    output[first + 1] = static_cast<NativeHalf>(radius * sine);
  }
}

__global__ void prepare_scorer_inputs_kernel(
    const NativeHalf* __restrict__ diffusion_canonical,
    const float* __restrict__ obb_canonical,
    const NativeHalf* __restrict__ discriminator_embedding,
    float* __restrict__ canonical,
    float* __restrict__ object_embedding,
    int64_t diffusion_count,
    int64_t obb_count) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t diffusion_elements = diffusion_count * 6;
  const int64_t candidate_elements = (diffusion_count + obb_count) * 6;
  if (index < candidate_elements) {
    canonical[index] = index < diffusion_elements
        ? static_cast<float>(diffusion_canonical[index])
        : obb_canonical[index - diffusion_elements];
  }
  if (index < kDenoiseHiddenDim) {
    object_embedding[index] =
        static_cast<float>(discriminator_embedding[index]);
  }
}

template <int ItemsPerThread>
__global__ void rank_translate_kernel(
    const float* __restrict__ diffusion_grasps,
    const float* __restrict__ obb_grasps,
    const float* __restrict__ confidence,
    const float* __restrict__ center,
    float* __restrict__ sorted_grasps,
    float* __restrict__ sorted_confidence,
    int32_t* __restrict__ source_indices,
    int64_t diffusion_count,
    int64_t candidate_count) {
  using Sort = cub::BlockRadixSort<float, kThreads, ItemsPerThread, int32_t>;
  __shared__ typename Sort::TempStorage sort_storage;
  float scores[ItemsPerThread];
  int32_t indices[ItemsPerThread];
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int32_t source = threadIdx.x * ItemsPerThread + item;
    float score = source < candidate_count
        ? confidence[source]
        : -3.402823466e+38F;
    if (!isfinite(score)) {
      score = -3.402823466e+38F;
    }
    scores[item] = score;
    indices[item] = source;
  }
  Sort(sort_storage).SortDescending(scores, indices);
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int32_t rank = threadIdx.x * ItemsPerThread + item;
    if (rank >= candidate_count) {
      continue;
    }
    const int32_t source = indices[item];
    const float* matrix = source < diffusion_count
        ? diffusion_grasps + static_cast<int64_t>(source) * 16
        : obb_grasps
            + static_cast<int64_t>(source - diffusion_count) * 16;
    float* destination = sorted_grasps + static_cast<int64_t>(rank) * 16;
    for (int element = 0; element < 16; ++element) {
      float value = matrix[element];
      if (element == 3) value += center[0];
      if (element == 7) value += center[1];
      if (element == 11) value += center[2];
      destination[element] = value;
    }
    sorted_confidence[rank] = scores[item];
    source_indices[rank] = source;
  }
}

__device__ __forceinline__ uint64_t morton3d(
    uint32_t x, uint32_t y, uint32_t z, int depth) {
  uint64_t code = 0;
#pragma unroll
  for (int bit = 0; bit < 16; ++bit) {
    if (bit >= depth) {
      break;
    }
    const uint64_t mask = uint64_t{1} << bit;
    code |= (static_cast<uint64_t>(x) & mask) << (2 * bit + 2);
    code |= (static_cast<uint64_t>(y) & mask) << (2 * bit + 1);
    code |= (static_cast<uint64_t>(z) & mask) << (2 * bit + 0);
  }
  return code;
}

// Skilling's Hilbert transpose algorithm. Packing every dimension's bit
// vector in a register turns the original nested PyTorch slicing/XOR loop into
// scalar bit operations inside one CUDA thread.
__device__ __forceinline__ uint64_t hilbert3d(
    uint32_t x, uint32_t y, uint32_t z, int depth) {
  uint64_t g0 = x;
  uint64_t g1 = y;
  uint64_t g2 = z;

#pragma unroll
  for (int bit = 0; bit < 16; ++bit) {
    if (bit >= depth) {
      break;
    }
    const int position = depth - 1 - bit;
    const uint64_t tail = (uint64_t{1} << position) - 1;

    // dimension 0: the exchange is a no-op when the current bit is zero.
    if ((g0 >> position) & 1) {
      g0 ^= tail;
    }

    if ((g1 >> position) & 1) {
      g0 ^= tail;
    } else {
      const uint64_t flip = (g0 ^ g1) & tail;
      g1 ^= flip;
      g0 ^= flip;
    }

    if ((g2 >> position) & 1) {
      g0 ^= tail;
    } else {
      const uint64_t flip = (g0 ^ g2) & tail;
      g2 ^= flip;
      g0 ^= flip;
    }
  }

  uint64_t gray = 0;
#pragma unroll
  for (int bit = 0; bit < 16; ++bit) {
    if (bit >= depth) {
      break;
    }
    const int position = depth - 1 - bit;
    gray = (gray << 1) | ((g0 >> position) & 1);
    gray = (gray << 1) | ((g1 >> position) & 1);
    gray = (gray << 1) | ((g2 >> position) & 1);
  }

  // Gray-to-binary prefix XOR for at most 48 meaningful bits.
  uint64_t binary = gray;
  binary ^= binary >> 1;
  binary ^= binary >> 2;
  binary ^= binary >> 4;
  binary ^= binary >> 8;
  binary ^= binary >> 16;
  binary ^= binary >> 32;
  return binary;
}



__global__ void serialization_codes_single_object_kernel(
    const int32_t* __restrict__ grid_coord,
    const int32_t* __restrict__ max_coord,
    int64_t* __restrict__ output,
    int64_t point_count) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= point_count) {
    return;
  }
  const uint32_t maximum = static_cast<uint32_t>(max_coord[0]);
  const int depth = maximum == 0 ? 0 : 32 - __clz(maximum);
  const uint32_t x = static_cast<uint32_t>(grid_coord[index * 3 + 0]);
  const uint32_t y = static_cast<uint32_t>(grid_coord[index * 3 + 1]);
  const uint32_t z = static_cast<uint32_t>(grid_coord[index * 3 + 2]);
  const uint64_t morton = morton3d(x, y, z, depth);
  output[0 * point_count + index] = static_cast<int64_t>(morton);
  output[1 * point_count + index] = static_cast<int64_t>(morton3d(y, x, z, depth));
}


__global__ void neighbor_lookup_morton_padded_kernel(
    const int32_t* __restrict__ grid_coord,
    const int64_t* __restrict__ morton_codes,
    const int64_t* __restrict__ morton_order,
    const int64_t* __restrict__ valid_count,
    const int64_t* __restrict__ offsets,
    int64_t* __restrict__ found_indices,
    bool* __restrict__ found_mask,
    int64_t point_capacity,
    int64_t offset_count) {
  const int64_t flat_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t lookup_count = point_capacity * offset_count;
  if (flat_index >= lookup_count) {
    return;
  }
  const int64_t point = flat_index / offset_count;
  const int64_t offset = flat_index - point * offset_count;
  const int64_t points = valid_count[0];
  if (point >= points) {
    found_indices[flat_index] = 0;
    found_mask[flat_index] = false;
    return;
  }
  const int64_t x = static_cast<int64_t>(grid_coord[point * 3 + 0])
      + offsets[offset * 3 + 0];
  const int64_t y = static_cast<int64_t>(grid_coord[point * 3 + 1])
      + offsets[offset * 3 + 1];
  const int64_t z = static_cast<int64_t>(grid_coord[point * 3 + 2])
      + offsets[offset * 3 + 2];
  const bool in_bounds =
      x >= 0 && x <= 65535 && y >= 0 && y <= 65535 && z >= 0 && z <= 65535;
  const int64_t query = in_bounds
      ? static_cast<int64_t>(morton3d(
            static_cast<uint32_t>(x),
            static_cast<uint32_t>(y),
            static_cast<uint32_t>(z),
            16))
      : -1;
  int64_t low = 0;
  int64_t high = points;
  if (in_bounds) {
    while (low < high) {
      const int64_t middle = low + ((high - low) >> 1);
      const int64_t candidate = morton_codes[morton_order[middle]];
      if (candidate < query) {
        low = middle + 1;
      } else {
        high = middle;
      }
    }
  }
  const bool found = in_bounds && low < points
      && morton_codes[morton_order[low]] == query;
  const int64_t clamped = low < points ? low : points - 1;
  found_indices[flat_index] = morton_order[clamped];
  found_mask[flat_index] = found;
}

__global__ void pack_attention_topology_kernel(
    const int64_t* __restrict__ serialized_order,
    const int64_t* __restrict__ serialized_inverse,
    const int64_t* __restrict__ valid_count,
    int64_t* __restrict__ attention_order,
    int64_t* __restrict__ attention_inverse,
    int32_t* __restrict__ valid_key_counts,
    int64_t capacity,
    int64_t padded_capacity,
    int64_t patch_size,
    int64_t order_count) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t points = valid_count[0];
  const int64_t padded_points = points <= patch_size
      ? points
      : ((points + patch_size - 1) / patch_size) * patch_size;
  if (index < order_count * padded_capacity) {
    const int64_t row = index / padded_capacity;
    const int64_t column = index - row * padded_capacity;
    if (column < padded_points) {
      const int64_t source_column =
          column < points ? column : column - patch_size;
      attention_order[index] =
          serialized_order[row * capacity + source_column];
    } else {
      attention_order[index] = 0;
    }
  }
  if (index < order_count * capacity) {
    const int64_t row = index / capacity;
    const int64_t column = index - row * capacity;
    if (column < points) {
      attention_inverse[index] =
          serialized_inverse[row * capacity + column];
    } else {
      attention_inverse[index] = 0;
    }
  }
  const int64_t key_count = padded_capacity / patch_size;
  if (index < key_count) {
    valid_key_counts[index] = points <= patch_size
        ? (index == 0 ? static_cast<int32_t>(points) : 0)
        : static_cast<int32_t>(patch_size);
  }
}



__global__ void inverse_order_kernel(
    const int64_t* __restrict__ order,
    int64_t* __restrict__ inverse,
    int64_t element_count,
    int64_t row_size) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < element_count) {
    const int64_t row = index / row_size;
    const int64_t column = index - row * row_size;
    inverse[row * row_size + order[index]] = column;
  }
}


__global__ void serialized_pool_segments_padded_kernel(
    const float* __restrict__ coord,
    const int32_t* __restrict__ grid_coord,
    const int64_t* __restrict__ serialized_code,
    const int64_t* __restrict__ indices,
    const int64_t* __restrict__ index_ptr,
    const int64_t* __restrict__ output_valid_count,
    float* __restrict__ pooled_coord,
    int32_t* __restrict__ pooled_grid_coord,
    int64_t* __restrict__ pooled_code,
    int64_t* __restrict__ pooled_order,
    int64_t* __restrict__ pooled_inverse,
    int64_t* __restrict__ cluster,
    int64_t* __restrict__ segment_ids,
    int64_t input_capacity,
    int64_t output_capacity,
    int64_t order_count,
    int pooling_depth,
    const int32_t* __restrict__ maximum_coordinate,
    int accumulated_pooling_depth) {
  const int64_t segment =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (segment >= output_capacity) {
    return;
  }
  pooled_order[segment] = segment;
  pooled_inverse[segment] = segment;
  if (segment >= output_valid_count[0]) {
    pooled_coord[segment * 3 + 0] = 0.0F;
    pooled_coord[segment * 3 + 1] = 0.0F;
    pooled_coord[segment * 3 + 2] = 0.0F;
    pooled_grid_coord[segment * 3 + 0] = 0;
    pooled_grid_coord[segment * 3 + 1] = 0;
    pooled_grid_coord[segment * 3 + 2] = 0;
    for (int64_t order = 0; order < order_count; ++order) {
      pooled_code[order * output_capacity + segment] =
          INT64_MAX;
    }
    return;
  }
  const int64_t begin = index_ptr[segment];
  const int64_t end = index_ptr[segment + 1];
  // When the fixed profile truncates an overflowing segment list, initialize
  // the omitted sources to a valid parent before derived orders consume the
  // cluster map. The overflow flag still marks the topology as unsupported.
  if (segment == 0 && output_valid_count[0] == output_capacity) {
    const int64_t omitted_begin = index_ptr[output_capacity];
    for (int64_t position = omitted_begin; position < input_capacity;
         ++position) {
      cluster[indices[position]] = 0;
    }
  }
  float sum_x = 0.0F;
  float sum_y = 0.0F;
  float sum_z = 0.0F;
  for (int64_t position = begin; position < end; ++position) {
    const int64_t source = indices[position];
    sum_x += coord[source * 3 + 0];
    sum_y += coord[source * 3 + 1];
    sum_z += coord[source * 3 + 2];
    segment_ids[position] = segment;
    cluster[source] = segment;
  }
  const float inverse_count = 1.0F / static_cast<float>(end - begin);
  pooled_coord[segment * 3 + 0] = sum_x * inverse_count;
  pooled_coord[segment * 3 + 1] = sum_y * inverse_count;
  pooled_coord[segment * 3 + 2] = sum_z * inverse_count;
  const int64_t head = indices[begin];
  const int32_t gx = grid_coord[head * 3 + 0] >> pooling_depth;
  const int32_t gy = grid_coord[head * 3 + 1] >> pooling_depth;
  const int32_t gz = grid_coord[head * 3 + 2] >> pooling_depth;
  pooled_grid_coord[segment * 3 + 0] = gx;
  pooled_grid_coord[segment * 3 + 1] = gy;
  pooled_grid_coord[segment * 3 + 2] = gz;
  const int shift = pooling_depth * 3;
  const int64_t derived_codes = order_count < 2 ? order_count : 2;
  for (int64_t order = 0; order < derived_codes; ++order) {
    pooled_code[order * output_capacity + segment] =
        serialized_code[order * input_capacity + head] >> shift;
  }
  if (order_count > 2) {
    const uint32_t maximum = static_cast<uint32_t>(maximum_coordinate[0]);
    const int original_depth = maximum == 0 ? 0 : 32 - __clz(maximum);
    const int depth = original_depth > accumulated_pooling_depth
        ? original_depth - accumulated_pooling_depth
        : 0;
    pooled_code[2 * output_capacity + segment] = static_cast<int64_t>(
        hilbert3d(gx, gy, gz, depth));
    pooled_code[3 * output_capacity + segment] = static_cast<int64_t>(
        hilbert3d(gy, gx, gz, depth));
  }
}

template <typename scalar_t>
__global__ void sparse_gather_kernel(
    const scalar_t* __restrict__ feat,
    const int64_t* __restrict__ found_indices,
    const bool* __restrict__ found_mask,
    scalar_t* __restrict__ gathered,
    int64_t point_count,
    int64_t offset_count,
    int64_t channels) {
  const int64_t element_count = point_count * offset_count * channels;
  for (int64_t flat_index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       flat_index < element_count;
       flat_index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t channel = flat_index % channels;
    const int64_t neighbor_slot = flat_index / channels;
    gathered[flat_index] = found_mask[neighbor_slot]
        ? feat[found_indices[neighbor_slot] * channels + channel]
        : scalar_t{0};
  }
}

template <typename scalar_t, int BlockThreads>
__global__ void layer_norm_residual_inplace_kernel(
    scalar_t* __restrict__ value,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    int64_t rows,
    int channels,
    float epsilon) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  static_assert(BlockThreads == kThreads);
  using BlockReduce = cub::BlockReduce<float, BlockThreads>;
  __shared__ typename BlockReduce::TempStorage reduction;
  __shared__ float mean;
  __shared__ float inverse_std;
  float local_sum = 0.0F;
  for (int channel = threadIdx.x; channel < channels; channel += BlockThreads) {
    local_sum += static_cast<float>(value[row * channels + channel]);
  }
  const float sum = BlockReduce(reduction).Sum(local_sum);
  if (threadIdx.x == 0) {
    mean = sum / static_cast<float>(channels);
  }
  __syncthreads();
  float local_squared = 0.0F;
  for (int channel = threadIdx.x; channel < channels; channel += BlockThreads) {
    const float centered =
        static_cast<float>(value[row * channels + channel]) - mean;
    local_squared = fmaf(centered, centered, local_squared);
  }
  const float squared_sum = BlockReduce(reduction).Sum(local_squared);
  if (threadIdx.x == 0) {
    inverse_std = rsqrtf(
        squared_sum / static_cast<float>(channels) + epsilon);
  }
  __syncthreads();
  for (int channel = threadIdx.x; channel < channels; channel += BlockThreads) {
    const int64_t index = row * channels + channel;
    const float normalized =
        (static_cast<float>(value[index]) - mean) * inverse_std;
    value[index] = static_cast<scalar_t>(
        static_cast<float>(residual[index])
        + normalized * static_cast<float>(weight[channel])
        + static_cast<float>(bias[channel]));
  }
}

__global__ void denoise_sample_encode_kernel(
    const NativeHalf* __restrict__ sample,
    const NativeHalf* __restrict__ weight,
    const NativeHalf* __restrict__ bias,
    const NativeHalf* __restrict__ observation_projection,
    const NativeHalf* __restrict__ timestep_conditioning,
    NativeHalf* __restrict__ sample_output,
    NativeHalf* __restrict__ conditioning_output,
    int64_t rows,
    int64_t object_count) {
  const int64_t output_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t condition_elements = object_count * 2048;
  if (output_index < condition_elements) {
    const int channel = output_index % 2048;
    conditioning_output[output_index] = static_cast<NativeHalf>(
        static_cast<float>(observation_projection[output_index])
        + static_cast<float>(timestep_conditioning[channel]));
  }
  if (output_index >= rows * kDenoiseHiddenDim) {
    return;
  }
  const int64_t row = output_index / kDenoiseHiddenDim;
  const int channel = output_index % kDenoiseHiddenDim;
  float value = static_cast<float>(bias[channel]);
#pragma unroll
  for (int input = 0; input < kDenoiseSampleDim; ++input) {
    value = fmaf(
        static_cast<float>(sample[row * kDenoiseSampleDim + input]),
        static_cast<float>(weight[channel * kDenoiseSampleDim + input]),
        value);
  }
  sample_output[output_index] = static_cast<NativeHalf>(fmaxf(value, 0.0F));
}

__global__ void denoise_step_kernel(
    const NativeHalf* __restrict__ prediction_hidden,
    const NativeHalf* __restrict__ sample,
    const NativeHalf* __restrict__ variance_noise,
    const NativeHalf* __restrict__ weight,
    const NativeHalf* __restrict__ bias,
    const NativeHalf* __restrict__ coefficients,
    NativeHalf* __restrict__ output,
    int64_t rows) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int output_channel = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (row >= rows || output_channel >= kDenoiseSampleDim) {
    return;
  }
  float prediction = 0.0F;
#pragma unroll
  for (int input = lane; input < kDenoiseHiddenDim; input += 32) {
    prediction = fmaf(
        static_cast<float>(
            prediction_hidden[row * kDenoiseHiddenDim + input]),
        static_cast<float>(
            weight[output_channel * kDenoiseHiddenDim + input]),
        prediction);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    prediction += __shfl_down_sync(0xffffffffU, prediction, offset);
  }
  if (lane != 0) {
    return;
  }
  prediction += static_cast<float>(bias[output_channel]);
  const int64_t sample_index = row * kDenoiseSampleDim + output_channel;
  const float current = static_cast<float>(sample[sample_index]);
  const float predicted_original = fminf(
      1.0F,
      fmaxf(
          -1.0F,
          current * static_cast<float>(coefficients[output_channel])
              + prediction
                  * static_cast<float>(
                      coefficients[kDenoiseSampleDim + output_channel])));
  const float previous =
      predicted_original
          * static_cast<float>(
              coefficients[2 * kDenoiseSampleDim + output_channel])
      + current
          * static_cast<float>(
              coefficients[3 * kDenoiseSampleDim + output_channel])
      + static_cast<float>(variance_noise[sample_index])
          * static_cast<float>(
              coefficients[4 * kDenoiseSampleDim + output_channel]);
  output[sample_index] = static_cast<NativeHalf>(previous);
}

// Every recurrent boundary except the last consumes the six-value denoiser output
// immediately in the next 6 -> 512 sample projection. Keeping the rounded
// FP16 state in shared memory preserves the original arithmetic while removing
// one launch and one global read from each transition between denoiser steps.
__global__ void denoise_step_sample_encode_kernel(
    const NativeHalf* __restrict__ prediction_hidden,
    const NativeHalf* __restrict__ sample,
    const NativeHalf* __restrict__ variance_noise,
    const NativeHalf* __restrict__ prediction_weight,
    const NativeHalf* __restrict__ prediction_bias,
    const NativeHalf* __restrict__ coefficients,
    NativeHalf* __restrict__ output,
    const NativeHalf* __restrict__ sample_weight,
    const NativeHalf* __restrict__ sample_bias,
    const NativeHalf* __restrict__ observation_projection,
    const NativeHalf* __restrict__ timestep_conditioning,
    NativeHalf* __restrict__ sample_output,
    NativeHalf* __restrict__ conditioning_output,
    int64_t rows,
    int64_t object_count) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  __shared__ NativeHalf rounded_sample[kDenoiseSampleDim];

  if (row < rows && warp < kDenoiseSampleDim) {
    float prediction = 0.0F;
#pragma unroll
    for (int input = lane; input < kDenoiseHiddenDim; input += 32) {
      prediction = fmaf(
          static_cast<float>(
              prediction_hidden[row * kDenoiseHiddenDim + input]),
          static_cast<float>(
              prediction_weight[warp * kDenoiseHiddenDim + input]),
          prediction);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      prediction += __shfl_down_sync(0xffffffffU, prediction, offset);
    }
    if (lane == 0) {
      prediction += static_cast<float>(prediction_bias[warp]);
      const int64_t sample_index = row * kDenoiseSampleDim + warp;
      const float current = static_cast<float>(sample[sample_index]);
      const float predicted_original = fminf(
          1.0F,
          fmaxf(
              -1.0F,
              current * static_cast<float>(coefficients[warp])
                  + prediction
                      * static_cast<float>(
                          coefficients[kDenoiseSampleDim + warp])));
      const float previous =
          predicted_original
              * static_cast<float>(
                  coefficients[2 * kDenoiseSampleDim + warp])
          + current
              * static_cast<float>(
                  coefficients[3 * kDenoiseSampleDim + warp])
          + static_cast<float>(variance_noise[sample_index])
              * static_cast<float>(
                  coefficients[4 * kDenoiseSampleDim + warp]);
      const NativeHalf rounded = static_cast<NativeHalf>(previous);
      output[sample_index] = rounded;
      rounded_sample[warp] = rounded;
    }
  }

  if (blockIdx.x == 0) {
    const int64_t condition_elements = object_count * 2048;
    for (int64_t index = threadIdx.x; index < condition_elements;
         index += blockDim.x) {
      conditioning_output[index] = static_cast<NativeHalf>(
          static_cast<float>(observation_projection[index])
          + static_cast<float>(timestep_conditioning[index % 2048]));
    }
  }
  __syncthreads();

  const int channel = threadIdx.x;
  if (row >= rows || channel >= kDenoiseHiddenDim) {
    return;
  }
  float value = static_cast<float>(sample_bias[channel]);
#pragma unroll
  for (int input = 0; input < kDenoiseSampleDim; ++input) {
    value = fmaf(
        static_cast<float>(rounded_sample[input]),
        static_cast<float>(sample_weight[channel * kDenoiseSampleDim + input]),
        value);
  }
  sample_output[row * kDenoiseHiddenDim + channel] =
      static_cast<NativeHalf>(fmaxf(value, 0.0F));
}

template <typename scalar_t>
__device__ __forceinline__ void knn_l1_filter_impl(
    const scalar_t* __restrict__ points,
    bool* __restrict__ keep,
    int64_t point_count,
    float threshold) {
  // One warp cooperates on each query. Every lane keeps a sorted local top-20,
  // then twenty warp reductions select the global nearest distances without a
  // serial 640-value merge or an N x N distance matrix.
  const int group = threadIdx.x / kFilterGroupSize;
  const int lane = threadIdx.x % kFilterGroupSize;
  const int64_t query =
      static_cast<int64_t>(blockIdx.x) * kFilterQueriesPerBlock + group;
  if (query >= point_count) {
    return;
  }
  float nearest[kFilterNeighbors];
#pragma unroll
  for (int item = 0; item < kFilterNeighbors; ++item) {
    nearest[item] = 3.402823466e+38F;
  }
  const float qx = static_cast<float>(points[query * 3 + 0]);
  const float qy = static_cast<float>(points[query * 3 + 1]);
  const float qz = static_cast<float>(points[query * 3 + 2]);
  for (int64_t candidate = lane; candidate < point_count;
       candidate += kFilterGroupSize) {
    if (candidate == query) {
      continue;
    }
    const float distance =
        fabsf(qx - static_cast<float>(points[candidate * 3 + 0]))
        + fabsf(qy - static_cast<float>(points[candidate * 3 + 1]))
        + fabsf(qz - static_cast<float>(points[candidate * 3 + 2]));
    if (distance < nearest[kFilterNeighbors - 1]) {
      int slot = kFilterNeighbors - 1;
#pragma unroll
      for (int item = kFilterNeighbors - 1; item > 0; --item) {
        if (distance < nearest[item - 1]) {
          nearest[item] = nearest[item - 1];
          slot = item - 1;
        }
      }
      nearest[slot] = distance;
    }
  }
  const int neighbor_count = static_cast<int>(
      point_count - 1 < kFilterNeighbors ? point_count - 1 : kFilterNeighbors);
  int cursor = 0;
  float distance_sum = 0.0F;
  constexpr unsigned kFullWarp = 0xffffffffU;
  for (int item = 0; item < neighbor_count; ++item) {
    const float candidate = nearest[cursor];
    float warp_min = candidate;
#pragma unroll
    for (int offset = kFilterGroupSize / 2; offset > 0; offset >>= 1) {
      warp_min = fminf(
          warp_min, __shfl_down_sync(kFullWarp, warp_min, offset));
    }
    warp_min = __shfl_sync(kFullWarp, warp_min, 0);
    const unsigned matching = __ballot_sync(kFullWarp, candidate == warp_min);
    const int winner = __ffs(matching) - 1;
    if (lane == winner) {
      ++cursor;
    }
    if (lane == 0) {
      distance_sum += warp_min;
    }
  }
  if (lane == 0) {
    keep[query] =
        neighbor_count > 0 && distance_sum / neighbor_count < threshold;
  }
}

template <typename scalar_t>
__global__ void knn_l1_filter_kernel(
    const scalar_t* __restrict__ points,
    bool* __restrict__ keep,
    int64_t point_count,
    float threshold) {
  knn_l1_filter_impl(points, keep, point_count, threshold);
}

template <typename scalar_t>
__global__ void knn_l1_filter_dynamic_kernel(
    const scalar_t* __restrict__ points,
    bool* __restrict__ keep,
    const int64_t* __restrict__ point_count,
    float threshold) {
  knn_l1_filter_impl(points, keep, point_count[0], threshold);
}

template <typename scalar_t>
__global__ void resample_filtered_points_kernel(
    const scalar_t* __restrict__ points,
    const int64_t* __restrict__ selected_indices,
    const int32_t* __restrict__ selected_count,
    scalar_t* __restrict__ output,
    int64_t input_count,
    int64_t output_count) {
  const int64_t output_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= output_count) {
    return;
  }
  const int64_t count = static_cast<int64_t>(selected_count[0]);
  int64_t input_index;
  if (count <= 0) {
    input_index = output_index % input_count;
  } else if (count >= output_count) {
    input_index = selected_indices[(output_index * count) / output_count];
  } else {
    input_index = selected_indices[output_index % count];
  }
  output[output_index * 3 + 0] = points[input_index * 3 + 0];
  output[output_index * 3 + 1] = points[input_index * 3 + 1];
  output[output_index * 3 + 2] = points[input_index * 3 + 2];
}

template <typename scalar_t>
__global__ void finalize_so3_kernel(
    const scalar_t* __restrict__ latent,
    float* __restrict__ matrices,
    scalar_t* __restrict__ canonical_rt,
    int64_t grasp_count,
    float inverse_kappa) {
  const int64_t grasp =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (grasp >= grasp_count) {
    return;
  }

  const float tx_scaled = static_cast<float>(latent[grasp * 6 + 0]);
  const float ty_scaled = static_cast<float>(latent[grasp * 6 + 1]);
  const float tz_scaled = static_cast<float>(latent[grasp * 6 + 2]);
  const float vx_scaled = static_cast<float>(latent[grasp * 6 + 3]);
  const float vy_scaled = static_cast<float>(latent[grasp * 6 + 4]);
  const float vz_scaled = static_cast<float>(latent[grasp * 6 + 5]);

  // Match rt_to_matrix(r3_so3): construct the unit quaternion for pi*v and
  // then use the same quaternion-to-matrix arithmetic. This is more accurate
  // than materializing a matrix in PyTorch through many small elementwise
  // kernels and gives the public FP32 4x4 pose in this same launch.
  const float scaled_norm = sqrtf(
      vx_scaled * vx_scaled + vy_scaled * vy_scaled + vz_scaled * vz_scaled);
  const float angle = kPi * scaled_norm;
  const float half_angle = 0.5F * angle;
  float sin_half;
  float cos_half;
  sincosf(half_angle, &sin_half, &cos_half);
  const float quaternion_scale =
      scaled_norm > 0.0F ? sin_half / scaled_norm : 0.0F;
  const float qi = vx_scaled * quaternion_scale;
  const float qj = vy_scaled * quaternion_scale;
  const float qk = vz_scaled * quaternion_scale;
  const float qr = cos_half;
  const float two_s = 2.0F / (qr * qr + qi * qi + qj * qj + qk * qk);

  float* matrix = matrices + grasp * 16;
  matrix[0] = 1.0F - two_s * (qj * qj + qk * qk);
  matrix[1] = two_s * (qi * qj - qk * qr);
  matrix[2] = two_s * (qi * qk + qj * qr);
  matrix[3] = tx_scaled * inverse_kappa;
  matrix[4] = two_s * (qi * qj + qk * qr);
  matrix[5] = 1.0F - two_s * (qi * qi + qk * qk);
  matrix[6] = two_s * (qj * qk - qi * qr);
  matrix[7] = ty_scaled * inverse_kappa;
  matrix[8] = two_s * (qi * qk - qj * qr);
  matrix[9] = two_s * (qj * qk + qi * qr);
  matrix[10] = 1.0F - two_s * (qi * qi + qj * qj);
  matrix[11] = tz_scaled * inverse_kappa;
  matrix[12] = 0.0F;
  matrix[13] = 0.0F;
  matrix[14] = 0.0F;
  matrix[15] = 1.0F;

  // matrix_to_rt immediately applies the SO(3) principal logarithm. Since
  // diffusion clips each component to [-1, 1], ||v|| < 2 and the exact
  // canonical form is v for ||v|| <= 1, otherwise -(2-||v||) * v/||v||.
  // Translation's divide/multiply-by-kappa round trip is exactly the original
  // scaled latent at model precision.
  float canonical_scale = 1.0F;
  if (scaled_norm > 1.0F) {
    canonical_scale = -(2.0F - scaled_norm) / scaled_norm;
  }
  canonical_rt[grasp * 6 + 0] = static_cast<scalar_t>(tx_scaled);
  canonical_rt[grasp * 6 + 1] = static_cast<scalar_t>(ty_scaled);
  canonical_rt[grasp * 6 + 2] = static_cast<scalar_t>(tz_scaled);
  canonical_rt[grasp * 6 + 3] =
      static_cast<scalar_t>(vx_scaled * canonical_scale);
  canonical_rt[grasp * 6 + 4] =
      static_cast<scalar_t>(vy_scaled * canonical_scale);
  canonical_rt[grasp * 6 + 5] =
      static_cast<scalar_t>(vz_scaled * canonical_scale);
}

__global__ void obb_pca_orientation_kernel(
    const float* __restrict__ points,
    float* __restrict__ bounds,
    int64_t point_count) {
  __shared__ float sum_x[kThreads];
  __shared__ float sum_y[kThreads];
  __shared__ float sum_xx[kThreads];
  __shared__ float sum_xy[kThreads];
  __shared__ float sum_yy[kThreads];
  float local_x = 0.0F;
  float local_y = 0.0F;
  float local_xx = 0.0F;
  float local_xy = 0.0F;
  float local_yy = 0.0F;
  for (int64_t index = threadIdx.x; index < point_count; index += blockDim.x) {
    const float x = points[index * 3 + 0];
    const float y = points[index * 3 + 1];
    local_x += x;
    local_y += y;
    local_xx += x * x;
    local_xy += x * y;
    local_yy += y * y;
  }
  sum_x[threadIdx.x] = local_x;
  sum_y[threadIdx.x] = local_y;
  sum_xx[threadIdx.x] = local_xx;
  sum_xy[threadIdx.x] = local_xy;
  sum_yy[threadIdx.x] = local_yy;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sum_x[threadIdx.x] += sum_x[threadIdx.x + stride];
      sum_y[threadIdx.x] += sum_y[threadIdx.x + stride];
      sum_xx[threadIdx.x] += sum_xx[threadIdx.x + stride];
      sum_xy[threadIdx.x] += sum_xy[threadIdx.x + stride];
      sum_yy[threadIdx.x] += sum_yy[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float inverse_count = 1.0F / static_cast<float>(point_count);
    const float mean_x = sum_x[0] * inverse_count;
    const float mean_y = sum_y[0] * inverse_count;
    const float covariance_xx = sum_xx[0] * inverse_count - mean_x * mean_x;
    const float covariance_xy = sum_xy[0] * inverse_count - mean_x * mean_y;
    const float covariance_yy = sum_yy[0] * inverse_count - mean_y * mean_y;
    const float angle =
        0.5F * atan2f(2.0F * covariance_xy, covariance_xx - covariance_yy);
    sincosf(angle, &bounds[7], &bounds[6]);
  }
}

__global__ void robust_axis_bounds_kernel(
    const float* __restrict__ points,
    float* __restrict__ bounds,
    int64_t point_count) {
  // One block sorts one coordinate. 3500 deployment points fit in a padded
  // 4096-value bitonic network, giving the exact linear-interpolated 2nd and
  // 98th percentiles without a global radix-sort workspace or CPU round trip.
  __shared__ float values[kObbSortCapacity];
  const int coordinate = blockIdx.x;
  for (int index = threadIdx.x; index < kObbSortCapacity;
       index += blockDim.x) {
    float value = __int_as_float(0x7f800000);
    if (index < point_count) {
      const float x = points[static_cast<int64_t>(index) * 3 + 0];
      const float y = points[static_cast<int64_t>(index) * 3 + 1];
      if (coordinate == 0) {
        value = bounds[6] * x + bounds[7] * y;
      } else if (coordinate == 1) {
        value = -bounds[7] * x + bounds[6] * y;
      } else {
        value = points[static_cast<int64_t>(index) * 3 + 2];
      }
    }
    values[index] = value;
  }
  __syncthreads();

  for (int subsequence = 2; subsequence <= kObbSortCapacity;
       subsequence <<= 1) {
    for (int stride = subsequence >> 1; stride > 0; stride >>= 1) {
      for (int index = threadIdx.x; index < kObbSortCapacity;
           index += blockDim.x) {
        const int partner = index ^ stride;
        if (partner > index) {
          const bool ascending = (index & subsequence) == 0;
          const float left = values[index];
          const float right = values[partner];
          if ((left > right) == ascending) {
            values[index] = right;
            values[partner] = left;
          }
        }
      }
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) {
    const float maximum_rank = static_cast<float>(point_count - 1);
    const float low_rank = 0.02F * maximum_rank;
    const float high_rank = 0.98F * maximum_rank;
    const int low_index = static_cast<int>(floorf(low_rank));
    const int high_index = static_cast<int>(floorf(high_rank));
    const float low_fraction = low_rank - static_cast<float>(low_index);
    const float high_fraction = high_rank - static_cast<float>(high_index);
    bounds[coordinate] =
        values[low_index]
        + low_fraction * (values[low_index + 1] - values[low_index]);
    bounds[3 + coordinate] =
        values[high_index]
        + high_fraction * (values[high_index + 1] - values[high_index]);
  }
}

__device__ void write_obb_topdown_candidate(
    const float* __restrict__ bounds,
    float* __restrict__ matrices,
    float* __restrict__ canonical_rt,
    float gripper_depth,
    float kappa,
    int64_t grasp_count,
    int64_t grasp) {
  if (grasp >= grasp_count) {
    return;
  }
  // Preserve the released 10-yaw x 5-height layout for the default count.
  // Other counts use the same five heights, with the final yaw optionally
  // containing fewer candidates. Counts below five prioritize yaw coverage.
  const int64_t height_count = grasp_count >= 5 ? 5 : 1;
  const int64_t yaw_count =
      (grasp_count + height_count - 1) / height_count;
  const int64_t yaw_index = grasp / height_count;
  const int64_t height_index = grasp % height_count;
  const float yaw = 2.0F * kPi * static_cast<float>(yaw_index)
      / static_cast<float>(yaw_count);
  const float z_offset = height_count == 1
      ? 0.0F
      : -0.08F
          + 0.08F * static_cast<float>(height_index)
              / static_cast<float>(height_count - 1);
  float sine;
  float cosine;
  sincosf(yaw, &sine, &cosine);

  const float center_local_x = 0.5F * (bounds[0] + bounds[3]);
  const float center_local_y = 0.5F * (bounds[1] + bounds[4]);
  const float tx = bounds[6] * center_local_x - bounds[7] * center_local_y;
  const float ty = bounds[7] * center_local_x + bounds[6] * center_local_y;
  // The released sparse OBB proposal is placed at the robust top surface,
  // offset along local -Z from fingertip to gripper base. Since top-down local
  // Z points toward -world-Z, this adds the gripper depth in world Z.
  const float tz =
      bounds[5] + z_offset + gripper_depth;

  float* matrix = matrices + grasp * 16;
  matrix[0] = cosine;
  matrix[1] = sine;
  matrix[2] = 0.0F;
  matrix[3] = tx;
  matrix[4] = sine;
  matrix[5] = -cosine;
  matrix[6] = 0.0F;
  matrix[7] = ty;
  matrix[8] = 0.0F;
  matrix[9] = 0.0F;
  matrix[10] = -1.0F;
  matrix[11] = tz;
  matrix[12] = 0.0F;
  matrix[13] = 0.0F;
  matrix[14] = 0.0F;
  matrix[15] = 1.0F;

  // A top-down yaw is a pi rotation about [cos(yaw/2), sin(yaw/2), 0].
  // matrix_to_axis_angle chooses the sign whose largest-magnitude quaternion
  // component is positive; reproduce that convention for discriminator input.
  float axis_x;
  float axis_y;
  sincosf(0.5F * yaw, &axis_y, &axis_x);
  const bool flip = fabsf(axis_x) >= fabsf(axis_y) ? axis_x < 0.0F : axis_y < 0.0F;
  if (flip) {
    axis_x = -axis_x;
    axis_y = -axis_y;
  }
  canonical_rt[grasp * 6 + 0] = tx * kappa;
  canonical_rt[grasp * 6 + 1] = ty * kappa;
  canonical_rt[grasp * 6 + 2] = tz * kappa;
  canonical_rt[grasp * 6 + 3] = axis_x;
  canonical_rt[grasp * 6 + 4] = axis_y;
  canonical_rt[grasp * 6 + 5] = 0.0F;
}

__global__ void obb_topdown_candidates_kernel(
    const float* __restrict__ bounds,
    float* __restrict__ matrices,
    float* __restrict__ canonical_rt,
    float gripper_depth,
    float kappa,
    int64_t grasp_count) {
  const int64_t grasp =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  write_obb_topdown_candidate(bounds, matrices, canonical_rt, gripper_depth,
                              kappa, grasp_count, grasp);
}

__global__ void obb_topdown_candidates_device_depth_kernel(
    const float* __restrict__ bounds,
    float* __restrict__ matrices,
    float* __restrict__ canonical_rt,
    const float* __restrict__ gripper_depth,
    float kappa,
    int64_t grasp_count) {
  const int64_t grasp =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  write_obb_topdown_candidate(bounds, matrices, canonical_rt, *gripper_depth,
                              kappa, grasp_count, grasp);
}

template <typename scalar_t>
__global__ void segment_max_padded_kernel(
    const scalar_t* __restrict__ feat,
    const int64_t* __restrict__ indices,
    const int64_t* __restrict__ index_ptr,
    const int64_t* __restrict__ segment_count,
    scalar_t* __restrict__ output,
    int64_t output_capacity,
    int64_t channels) {
  const int64_t flat =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t element_count = output_capacity * channels;
  if (flat >= element_count) {
    return;
  }
  const int64_t segment = flat / channels;
  const int64_t channel = flat - segment * channels;
  if (segment >= segment_count[0]) {
    output[flat] = static_cast<scalar_t>(0.0F);
    return;
  }
  const int64_t begin = index_ptr[segment];
  const int64_t end = index_ptr[segment + 1];
  float maximum = -3.402823466e+38F;
  for (int64_t item = begin; item < end; ++item) {
    const float value = static_cast<float>(
        feat[indices[item] * channels + channel]);
    maximum = value > maximum ? value : maximum;
  }
  output[flat] = static_cast<scalar_t>(
      begin < end ? maximum : 0.0F);
}

template <typename scalar_t>
__global__ void mean_padded_kernel(
    const scalar_t* __restrict__ feat,
    const int64_t* __restrict__ valid_count,
    scalar_t* __restrict__ output,
    int64_t channels) {
  const int64_t channel =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= channels) {
    return;
  }
  const int64_t rows = valid_count[0];
  float sum = 0.0F;
  for (int64_t row = 0; row < rows; ++row) {
    sum += static_cast<float>(feat[row * channels + channel]);
  }
  output[channel] = static_cast<scalar_t>(sum / static_cast<float>(rows));
}

}  // namespace

namespace graspgenx {

cudaError_t LaunchDenoiseSampleEncodeFp16(
    const void* sample, const void* weight, const void* bias,
    const void* observation_projection, const void* timestep_conditioning,
    void* sample_output, void* conditioning_output, int64_t rows,
    int64_t object_count, cudaStream_t stream) {
  const int64_t element_count = std::max<int64_t>(
      rows * kDenoiseHiddenDim, object_count * 2048);
  denoise_sample_encode_kernel<<<
      (element_count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      static_cast<const NativeHalf*>(sample),
      static_cast<const NativeHalf*>(weight),
      static_cast<const NativeHalf*>(bias),
      static_cast<const NativeHalf*>(observation_projection),
      static_cast<const NativeHalf*>(timestep_conditioning),
      static_cast<NativeHalf*>(sample_output),
      static_cast<NativeHalf*>(conditioning_output), rows, object_count);
  return cudaPeekAtLastError();
}

cudaError_t LaunchDenoiseStepFp16(
    const void* prediction_hidden, const void* sample,
    const void* variance_noise, const void* weight, const void* bias,
    const void* coefficients, void* output, int64_t rows,
    cudaStream_t stream) {
  denoise_step_kernel<<<rows, 32 * kDenoiseSampleDim, 0, stream>>>(
      static_cast<const NativeHalf*>(prediction_hidden),
      static_cast<const NativeHalf*>(sample),
      static_cast<const NativeHalf*>(variance_noise),
      static_cast<const NativeHalf*>(weight),
      static_cast<const NativeHalf*>(bias),
      static_cast<const NativeHalf*>(coefficients),
      static_cast<NativeHalf*>(output), rows);
  return cudaPeekAtLastError();
}

cudaError_t LaunchDenoiseStepAndSampleEncodeFp16(
    const void* prediction_hidden, const void* sample,
    const void* variance_noise, const void* prediction_weight,
    const void* prediction_bias, const void* coefficients, void* output,
    const void* sample_weight, const void* sample_bias,
    const void* observation_projection, const void* timestep_conditioning,
    void* sample_output, void* conditioning_output, int64_t rows,
    int64_t object_count, cudaStream_t stream) {
  denoise_step_sample_encode_kernel<<<rows, kDenoiseHiddenDim, 0, stream>>>(
      static_cast<const NativeHalf*>(prediction_hidden),
      static_cast<const NativeHalf*>(sample),
      static_cast<const NativeHalf*>(variance_noise),
      static_cast<const NativeHalf*>(prediction_weight),
      static_cast<const NativeHalf*>(prediction_bias),
      static_cast<const NativeHalf*>(coefficients),
      static_cast<NativeHalf*>(output),
      static_cast<const NativeHalf*>(sample_weight),
      static_cast<const NativeHalf*>(sample_bias),
      static_cast<const NativeHalf*>(observation_projection),
      static_cast<const NativeHalf*>(timestep_conditioning),
      static_cast<NativeHalf*>(sample_output),
      static_cast<NativeHalf*>(conditioning_output), rows, object_count);
  return cudaPeekAtLastError();
}

cudaError_t LaunchFinalizeSo3Fp16(
    const void* latent, float* matrices, void* canonical_rt,
    int64_t grasp_count, float kappa, cudaStream_t stream) {
  const int blocks = static_cast<int>((grasp_count + kThreads - 1) / kThreads);
  finalize_so3_kernel<NativeHalf><<<blocks, kThreads, 0, stream>>>(
      static_cast<const NativeHalf*>(latent), matrices,
      static_cast<NativeHalf*>(canonical_rt), grasp_count, 1.0F / kappa);
  return cudaPeekAtLastError();
}

cudaError_t LaunchObbTopdownFloat(
    const float* points, int64_t point_count, float* matrices,
    float* canonical_rt, float* bounds, float gripper_depth, float kappa,
    int64_t grasp_count, cudaStream_t stream) {
  obb_pca_orientation_kernel<<<1, kThreads, 0, stream>>>(
      points, bounds, point_count);
  robust_axis_bounds_kernel<<<3, kObbSortThreads, 0, stream>>>(
      points, bounds, point_count);
  if (grasp_count > 0) {
    obb_topdown_candidates_kernel<<<
        (grasp_count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        bounds, matrices, canonical_rt, gripper_depth, kappa, grasp_count);
  }
  return cudaPeekAtLastError();
}

cudaError_t LaunchObbTopdownFloatDeviceDepth(
    const float* points, int64_t point_count, float* matrices,
    float* canonical_rt, float* bounds, const float* gripper_depth, float kappa,
    int64_t grasp_count, cudaStream_t stream) {
  obb_pca_orientation_kernel<<<1, kThreads, 0, stream>>>(
      points, bounds, point_count);
  robust_axis_bounds_kernel<<<3, kObbSortThreads, 0, stream>>>(
      points, bounds, point_count);
  if (grasp_count > 0) {
    obb_topdown_candidates_device_depth_kernel<<<
        (grasp_count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        bounds, matrices, canonical_rt, gripper_depth, kappa, grasp_count);
  }
  return cudaPeekAtLastError();
}

cudaError_t LaunchSegmentMaxFp16(
    const void* features, const int64_t* indices, const int64_t* index_ptr,
    const int64_t* segment_count, void* output, int64_t output_capacity,
    int64_t channels, cudaStream_t stream) {
  const int64_t elements = output_capacity * channels;
  segment_max_padded_kernel<NativeHalf><<<
      (elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      static_cast<const NativeHalf*>(features), indices, index_ptr,
      segment_count, static_cast<NativeHalf*>(output), output_capacity,
      channels);
  return cudaPeekAtLastError();
}

cudaError_t LaunchMeanFp16(
    const void* features, const int64_t* valid_count, void* output,
    int64_t channels, cudaStream_t stream) {
  mean_padded_kernel<NativeHalf><<<
      (channels + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      static_cast<const NativeHalf*>(features), valid_count,
      static_cast<NativeHalf*>(output), channels);
  return cudaPeekAtLastError();
}

cudaError_t LaunchFilterResampleFloat(
    const float* points, int64_t point_count, int64_t output_count,
    float threshold, bool* keep, int64_t* selected_indices,
    int32_t* selected_count, float* output, cudaStream_t stream) {
  if (point_count <= 0 || point_count > kFixedAlgorithmCapacity
      || output_count <= 0 || output_count > kFixedAlgorithmCapacity) {
    return cudaErrorInvalidValue;
  }
  const int blocks = static_cast<int>(
      (point_count + kFilterQueriesPerBlock - 1) / kFilterQueriesPerBlock);
  knn_l1_filter_kernel<float><<<blocks, kFilterThreads, 0, stream>>>(
      points, keep, point_count, threshold);
  compact_resample_kernel<float><<<1, kThreads, 0, stream>>>(
      points, keep, selected_indices, selected_count, output,
      static_cast<int>(point_count), static_cast<int>(output_count));
  return cudaPeekAtLastError();
}

cudaError_t LaunchFilterResampleDynamicFloat(
    const float* points, const int64_t* point_count,
    int64_t maximum_point_count, int64_t output_count,
    float threshold, bool* keep, int64_t* selected_indices,
    int32_t* selected_count, float* output, cudaStream_t stream) {
  if (maximum_point_count <= 0
      || maximum_point_count > kFixedAlgorithmCapacity
      || output_count <= 0 || output_count > kFixedAlgorithmCapacity) {
    return cudaErrorInvalidValue;
  }
  const int blocks = static_cast<int>(
      (maximum_point_count + kFilterQueriesPerBlock - 1)
      / kFilterQueriesPerBlock);
  knn_l1_filter_dynamic_kernel<float><<<blocks, kFilterThreads, 0, stream>>>(
      points, keep, point_count, threshold);
  compact_resample_dynamic_kernel<float><<<1, kThreads, 0, stream>>>(
      points, keep, selected_indices, selected_count, output, point_count,
      static_cast<int>(output_count));
  return cudaPeekAtLastError();
}

cudaError_t LaunchResampleFixedFloat(
    const float* points, int64_t point_count, float* output,
    int64_t output_count, cudaStream_t stream) {
  if (point_count <= 0 || output_count <= 0) {
    return cudaErrorInvalidValue;
  }
  const int blocks =
      static_cast<int>((output_count + kThreads - 1) / kThreads);
  resample_fixed_kernel<float><<<blocks, kThreads, 0, stream>>>(
      points, point_count, output, output_count);
  return cudaPeekAtLastError();
}

cudaError_t LaunchCenterScaleAndGridFloat(
    const float* points, int64_t point_count, float kappa, float grid_size,
    float* centered_points, float* center, float* scaled_coordinates,
    void* fp16_features, int32_t* grid_coordinates,
    int32_t* maximum_coordinate, cudaStream_t stream) {
  center_scale_grid_kernel<<<1, kThreads, 0, stream>>>(
      points, centered_points, center, scaled_coordinates,
      static_cast<NativeHalf*>(fp16_features), grid_coordinates,
      maximum_coordinate, point_count, kappa, 1.0F / grid_size);
  return cudaPeekAtLastError();
}

cudaError_t LaunchSerializationAndSort(
    const int32_t* grid_coordinates, int64_t capacity,
    int32_t* maximum_coordinate, int64_t* codes, int64_t* order,
    int64_t* inverse, int64_t order_count, cudaStream_t stream) {
  const int blocks = static_cast<int>((capacity + kThreads - 1) / kThreads);
  serialization_codes_single_object_kernel<<<blocks, kThreads, 0, stream>>>(
      grid_coordinates, maximum_coordinate, codes, capacity);
  if (capacity > kFixedAlgorithmCapacity) {
    return cudaErrorInvalidValue;
  }
  fixed_stable_sort_pairs_kernel<<<order_count, kThreads, 0, stream>>>(
      codes, order, static_cast<int>(capacity), maximum_coordinate, 0);
  inverse_order_kernel<<<order_count * blocks, kThreads, 0, stream>>>(
      order, inverse, order_count * capacity, capacity);
  return cudaPeekAtLastError();
}

cudaError_t LaunchNeighborMapPadded(
    const int32_t* grid_coordinates, const int64_t* morton_codes,
    const int64_t* morton_order, const int64_t* valid_count,
    const int64_t* offsets, int64_t* neighbor_indices, bool* neighbor_mask,
    int64_t capacity, int64_t offset_count, cudaStream_t stream) {
  const int64_t lookup_count = capacity * offset_count;
  const int blocks = static_cast<int>(
      (lookup_count + kThreads - 1) / kThreads);
  neighbor_lookup_morton_padded_kernel<<<blocks, kThreads, 0, stream>>>(
      grid_coordinates, morton_codes, morton_order, valid_count, offsets,
      neighbor_indices, neighbor_mask, capacity, offset_count);
  return cudaPeekAtLastError();
}

cudaError_t LaunchSerializedPoolingPadded(
    const float* coordinates, const int32_t* grid_coordinates,
    const int64_t* codes, const int64_t* order,
    const int64_t* input_valid_count, int64_t input_capacity,
    int64_t output_capacity, int pooling_depth,
    int64_t* pool_index_ptr, int64_t* cluster,
    int64_t* segment_ids, float* pooled_coordinates,
    int32_t* pooled_grid_coordinates, int64_t* pooled_codes,
    int64_t* pooled_order, int64_t* pooled_inverse,
    int64_t* output_valid_count, bool* overflow, int64_t output_order_count,
    int64_t derived_order_count,
    const int32_t* maximum_coordinate, int32_t maximum_supported_coordinate,
    int32_t accumulated_pooling_depth, cudaStream_t stream,
    cudaStream_t order_stream, cudaEvent_t pool_data_ready,
    cudaEvent_t order_ready) {
  if (output_capacity > 6 * kThreads) {
    return cudaErrorInvalidValue;
  }
  if (input_capacity > kFixedAlgorithmCapacity
      || output_capacity > kFixedAlgorithmCapacity) {
    return cudaErrorInvalidValue;
  }
  if (input_capacity <= 128) {
    fixed_pool_boundaries_kernel<128, 1><<<1, 128, 0, stream>>>(
        codes, order, input_valid_count, output_valid_count, overflow,
        pool_index_ptr, static_cast<int>(input_capacity),
        static_cast<int>(output_capacity), pooling_depth * 3,
        maximum_coordinate, maximum_supported_coordinate);
  } else if (input_capacity <= 512) {
    fixed_pool_boundaries_kernel<256, 2><<<1, 256, 0, stream>>>(
        codes, order, input_valid_count, output_valid_count, overflow,
        pool_index_ptr, static_cast<int>(input_capacity),
        static_cast<int>(output_capacity), pooling_depth * 3,
        maximum_coordinate, maximum_supported_coordinate);
  } else if (input_capacity <= 1536) {
    fixed_pool_boundaries_kernel<256, 6><<<1, 256, 0, stream>>>(
        codes, order, input_valid_count, output_valid_count, overflow,
        pool_index_ptr, static_cast<int>(input_capacity),
        static_cast<int>(output_capacity), pooling_depth * 3,
        maximum_coordinate, maximum_supported_coordinate);
  } else {
    fixed_pool_boundaries_kernel<256, 16><<<1, 256, 0, stream>>>(
        codes, order, input_valid_count, output_valid_count, overflow,
        pool_index_ptr, static_cast<int>(input_capacity),
        static_cast<int>(output_capacity), pooling_depth * 3,
        maximum_coordinate, maximum_supported_coordinate);
  }
  const int output_blocks = static_cast<int>(
      (output_capacity + kThreads - 1) / kThreads);
  serialized_pool_segments_padded_kernel<<<
      output_blocks, kThreads, 0, stream>>>(
      coordinates, grid_coordinates, codes, order, pool_index_ptr,
      output_valid_count, pooled_coordinates, pooled_grid_coordinates,
      pooled_codes, pooled_order, pooled_inverse, cluster, segment_ids,
      input_capacity, output_capacity, output_order_count, pooling_depth,
      maximum_coordinate, accumulated_pooling_depth);
  cudaError_t status = cudaEventRecord(pool_data_ready, stream);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaStreamWaitEvent(order_stream, pool_data_ready);
  if (status != cudaSuccess) {
    return status;
  }
  const int64_t remaining_derived_orders = derived_order_count - 1;
  if (remaining_derived_orders < 0) {
    return cudaErrorInvalidValue;
  }
  if (input_capacity <= 128) {
    fixed_derived_pool_order_kernel<128, 1><<<
        remaining_derived_orders, 128, 0, order_stream>>>(
        order, cluster, input_valid_count, output_valid_count, pooled_order,
        pooled_inverse, static_cast<int>(input_capacity),
        static_cast<int>(output_capacity), 1);
  } else if (input_capacity <= 512) {
    fixed_derived_pool_order_kernel<256, 2><<<
        remaining_derived_orders, 256, 0, order_stream>>>(
        order, cluster, input_valid_count, output_valid_count, pooled_order,
        pooled_inverse, static_cast<int>(input_capacity),
        static_cast<int>(output_capacity), 1);
  } else if (input_capacity <= 1536) {
    fixed_derived_pool_order_kernel<256, 6><<<
        remaining_derived_orders, 256, 0, order_stream>>>(
        order, cluster, input_valid_count, output_valid_count, pooled_order,
        pooled_inverse, static_cast<int>(input_capacity),
        static_cast<int>(output_capacity), 1);
  } else {
    fixed_derived_pool_order_kernel<256, 16><<<
        remaining_derived_orders, 256, 0, order_stream>>>(
        order, cluster, input_valid_count, output_valid_count, pooled_order,
        pooled_inverse, static_cast<int>(input_capacity),
        static_cast<int>(output_capacity), 1);
  }
  if (output_order_count > derived_order_count) {
    const int64_t extra_orders = output_order_count - derived_order_count;
    if (output_capacity <= 128) {
      fixed_stable_sort_pairs_128_kernel<<<
          extra_orders, 128, 0, order_stream>>>(
          pooled_codes + derived_order_count * output_capacity,
          pooled_order + derived_order_count * output_capacity,
          static_cast<int>(output_capacity), maximum_coordinate,
          accumulated_pooling_depth);
    } else {
      fixed_stable_sort_pairs_kernel<<<
          extra_orders, kThreads, 0, order_stream>>>(
          pooled_codes + derived_order_count * output_capacity,
          pooled_order + derived_order_count * output_capacity,
          static_cast<int>(output_capacity), maximum_coordinate,
          accumulated_pooling_depth);
    }
    inverse_order_kernel<<<
        extra_orders * output_blocks, kThreads, 0, order_stream>>>(
        pooled_order + derived_order_count * output_capacity,
        pooled_inverse + derived_order_count * output_capacity,
        extra_orders * output_capacity, output_capacity);
  }
  status = cudaEventRecord(order_ready, order_stream);
  if (status != cudaSuccess) {
    return status;
  }
  return cudaPeekAtLastError();
}

cudaError_t LaunchPackAttentionTopology(
    const int64_t* serialized_order, const int64_t* serialized_inverse,
    const int64_t* valid_count, int64_t* attention_order,
    int64_t* attention_inverse, int32_t* valid_key_counts, int64_t capacity,
    int64_t padded_capacity, int64_t patch_size, int64_t order_count,
    cudaStream_t stream) {
  const int64_t key_count = padded_capacity / patch_size;
  const int64_t work = std::max<int64_t>(
      order_count * padded_capacity,
      std::max<int64_t>(order_count * capacity, key_count));
  pack_attention_topology_kernel<<<
      (work + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      serialized_order, serialized_inverse, valid_count, attention_order,
      attention_inverse, valid_key_counts, capacity, padded_capacity,
      patch_size, order_count);
  return cudaPeekAtLastError();
}

cudaError_t LaunchNormalNoiseFp16(
    const uint64_t* seed, const uint64_t* sequence_offset, void* output,
    int64_t element_count, cudaStream_t stream) {
  const int64_t pair_count = (element_count + 1) / 2;
  normal_noise_fp16_kernel<<<
      (pair_count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      seed, sequence_offset, static_cast<NativeHalf*>(output), element_count);
  return cudaPeekAtLastError();
}

cudaError_t LaunchPrepareScorerInputs(
    const void* diffusion_canonical_fp16, const float* obb_canonical,
    const void* discriminator_embedding_fp16, float* canonical_fp32,
    float* object_embedding_fp32, int64_t diffusion_count,
    int64_t obb_count, cudaStream_t stream) {
  const int64_t candidate_count = diffusion_count + obb_count;
  const int64_t work = candidate_count * 6 > 512
      ? candidate_count * 6
      : 512;
  prepare_scorer_inputs_kernel<<<
      (work + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      static_cast<const NativeHalf*>(diffusion_canonical_fp16), obb_canonical,
      static_cast<const NativeHalf*>(discriminator_embedding_fp16),
      canonical_fp32, object_embedding_fp32, diffusion_count, obb_count);
  return cudaPeekAtLastError();
}

cudaError_t LaunchRankAndTranslate(
    const float* diffusion_grasps, const float* obb_grasps,
    const float* confidence, const float* center, float* sorted_grasps,
    float* sorted_confidence, int32_t* source_indices,
    int64_t diffusion_count, int64_t obb_count, cudaStream_t stream) {
  const int64_t candidate_count = diffusion_count + obb_count;
#define GRASPGENX_LAUNCH_RANK(items)                                      \
  rank_translate_kernel<items><<<1, kThreads, 0, stream>>>(               \
      diffusion_grasps, obb_grasps, confidence, center, sorted_grasps,    \
      sorted_confidence, source_indices, diffusion_count, candidate_count)
  switch ((candidate_count + kThreads - 1) / kThreads) {
    case 1: GRASPGENX_LAUNCH_RANK(1); break;
    case 2: GRASPGENX_LAUNCH_RANK(2); break;
    case 3: GRASPGENX_LAUNCH_RANK(3); break;
    case 4: GRASPGENX_LAUNCH_RANK(4); break;
    case 5: GRASPGENX_LAUNCH_RANK(5); break;
    case 6: GRASPGENX_LAUNCH_RANK(6); break;
    case 7: GRASPGENX_LAUNCH_RANK(7); break;
    case 8: GRASPGENX_LAUNCH_RANK(8); break;
    case 9: GRASPGENX_LAUNCH_RANK(9); break;
    case 10: GRASPGENX_LAUNCH_RANK(10); break;
    case 11: GRASPGENX_LAUNCH_RANK(11); break;
    case 12: GRASPGENX_LAUNCH_RANK(12); break;
    case 13: GRASPGENX_LAUNCH_RANK(13); break;
    case 14: GRASPGENX_LAUNCH_RANK(14); break;
    case 15: GRASPGENX_LAUNCH_RANK(15); break;
    case 16: GRASPGENX_LAUNCH_RANK(16); break;
    default: return cudaErrorInvalidValue;
  }
#undef GRASPGENX_LAUNCH_RANK
  return cudaPeekAtLastError();
}

}  // namespace graspgenx
