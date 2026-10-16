// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "graspgenx/runtime.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

void Check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

std::optional<double> QueryProcessGpuMib() {
  FILE* pipe = popen(
      "nvidia-smi --query-compute-apps=pid,used_gpu_memory "
      "--format=csv,noheader,nounits 2>/dev/null",
      "r");
  if (pipe == nullptr) return std::nullopt;
  const int process_id = static_cast<int>(getpid());
  char line[256];
  std::optional<double> result;
  while (fgets(line, sizeof(line), pipe) != nullptr) {
    int candidate_pid = -1;
    double memory_mib = 0.0;
    if (std::sscanf(line, "%d, %lf", &candidate_pid, &memory_mib) == 2 &&
        candidate_pid == process_id) {
      result = memory_mib;
      break;
    }
  }
  pclose(pipe);
  return result;
}

std::vector<float> ReadFloatFile(const char* path, std::size_t count) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() !=
                     static_cast<std::streamoff>(count * sizeof(float))) {
    throw std::runtime_error(std::string("Invalid FP32 tensor file: ") + path);
  }
  std::vector<float> result(count);
  stream.seekg(0, std::ios::beg);
  stream.read(reinterpret_cast<char*>(result.data()),
              static_cast<std::streamsize>(count * sizeof(float)));
  if (!stream) {
    throw std::runtime_error(std::string("Could not read FP32 tensor: ") + path);
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 4 && argc != 5 && argc != 7) {
    std::cerr
        << "usage: graspgenx_benchmark_runtime ARTIFACT_DIRECTORY "
           "[GENERATOR_EMBEDDING_BIN DISCRIMINATOR_EMBEDDING_BIN "
           "GRIPPER_DEPTH] [DIFFUSION_GRASPS OBB_GRASPS]\n";
    return 2;
  }
  try {
    const char* iteration_text = std::getenv("GRASPGENX_BENCHMARK_ITERATIONS");
    const int benchmark_iterations =
        iteration_text == nullptr ? 1000 : std::max(1, std::stoi(iteration_text));
    std::size_t free_before = 0;
    std::size_t total = 0;
    Check(cudaMemGetInfo(&free_before, &total), "cudaMemGetInfo before");
    graspgenx::RuntimeConfig config;
    config.artifact_directory = argv[1];
    const bool has_embeddings = argc == 5 || argc == 7;
    if (has_embeddings) config.gripper_depth = std::stof(argv[4]);
    if (argc == 4) {
      config.diffusion_grasps = std::stoi(argv[2]);
      config.obb_grasps = std::stoi(argv[3]);
    } else if (argc == 7) {
      config.diffusion_grasps = std::stoi(argv[5]);
      config.obb_grasps = std::stoi(argv[6]);
    }
    graspgenx::Runtime runtime(config);
    if (has_embeddings) {
      const auto generator_embedding = ReadFloatFile(argv[2], 512);
      const auto discriminator_embedding = ReadFloatFile(argv[3], 512);
      runtime.SetGripperEmbeddings(generator_embedding.data(),
                                   discriminator_embedding.data());
    }
    constexpr int kPointCount = 3500;
    std::vector<float> points(kPointCount * 3);
    for (int index = 0; index < kPointCount; ++index) {
      const float u = (static_cast<float>(index) + 0.5F) / kPointCount;
      const float z = 0.08F * (2.0F * u - 1.0F);
      const float angle = 2.39996323F * index;
      const float radius = 0.04F * std::sqrt(std::max(0.0F, 1.0F - z * z));
      points[index * 3 + 0] = radius * std::cos(angle);
      points[index * 3 + 1] = radius * std::sin(angle);
      points[index * 3 + 2] = z;
    }
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    Check(cudaStreamCreate(&stream), "cudaStreamCreate");
    Check(cudaEventCreate(&start), "cudaEventCreate start");
    Check(cudaEventCreate(&stop), "cudaEventCreate stop");
    runtime.InferHostAsync(points.data(), kPointCount, 7, 0, stream);
    Check(cudaStreamSynchronize(stream), "initial inference");
    const std::uint64_t random_pairs =
        static_cast<std::uint64_t>(
            graspgenx::DenoisingStepCount(config.denoiser)) *
        static_cast<std::uint64_t>(config.diffusion_grasps) * 3;
    for (int iteration = 0; iteration < benchmark_iterations; ++iteration) {
      runtime.InferAsync(kPointCount, 7,
                         static_cast<std::uint64_t>(iteration) * random_pairs,
                         stream);
    }
    Check(cudaStreamSynchronize(stream), "warmup");
    Check(cudaEventRecord(start, stream), "record start");
    for (int iteration = 0; iteration < benchmark_iterations; ++iteration) {
      runtime.InferAsync(kPointCount, 7,
                         static_cast<std::uint64_t>(iteration) * random_pairs,
                         stream);
    }
    Check(cudaEventRecord(stop, stream), "record stop");
    Check(cudaEventSynchronize(stop), "synchronize stop");
    float milliseconds = 0.0F;
    Check(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed time");
    const auto profile = runtime.Profile(benchmark_iterations,
                                         benchmark_iterations);
    std::vector<float> scores(runtime.outputs().grasp_count);
    std::vector<std::int32_t> source_indices(runtime.outputs().grasp_count);
    std::array<std::int64_t, 5> stage_counts{};
    std::array<bool, 4> topology_overflow{};
    Check(cudaMemcpy(scores.data(), runtime.outputs().confidence,
                     scores.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "copy scores");
    Check(cudaMemcpy(source_indices.data(), runtime.outputs().source_indices,
                     source_indices.size() * sizeof(std::int32_t),
                     cudaMemcpyDeviceToHost),
          "copy source indices");
    Check(cudaMemcpy(stage_counts.data(),
                     runtime.outputs().ptv3_stage_counts,
                     sizeof(stage_counts), cudaMemcpyDeviceToHost),
          "copy PTV3 stage counts");
    Check(cudaMemcpy(topology_overflow.data(),
                     runtime.outputs().ptv3_topology_overflow,
                     sizeof(topology_overflow), cudaMemcpyDeviceToHost),
          "copy PTV3 overflow flags");
    if (std::any_of(topology_overflow.begin(), topology_overflow.end(),
                    [](bool value) { return value; })) {
      throw std::runtime_error("PTV3 topology capacity overflow");
    }
    if (!std::is_sorted(scores.begin(), scores.end(), std::greater<float>())) {
      throw std::runtime_error("ranked confidence is not descending");
    }
    std::sort(source_indices.begin(), source_indices.end());
    for (std::size_t index = 0; index < source_indices.size(); ++index) {
      if (source_indices[index] != static_cast<std::int32_t>(index)) {
        throw std::runtime_error("source indices are not a permutation");
      }
    }
    std::size_t free_after = 0;
    Check(cudaMemGetInfo(&free_after, &total), "cudaMemGetInfo after");
    const graspgenx::RuntimeMemory memory = runtime.memory();
    const auto process_gpu_mib = QueryProcessGpuMib();
    const float inference_ms = milliseconds / benchmark_iterations;
    std::cout << "denoiser: ddpm\n"
              << "denoising_steps: "
              << graspgenx::DenoisingStepCount(config.denoiser) << '\n'
              << "diffusion_grasps: " << config.diffusion_grasps << '\n'
              << "obb_grasps: " << config.obb_grasps << '\n'
              << "candidate_count: " << runtime.outputs().grasp_count << '\n'
              << "pointcloud_to_ranked_grasps_ms: " << inference_ms << '\n'
              << "persistent_cuda_mib: "
              << static_cast<double>(memory.persistent_cuda_bytes) / (1 << 20)
              << '\n'
              << "tensorrt_context_mib: "
              << static_cast<double>(memory.tensorrt_device_memory_bytes) /
                     (1 << 20)
              << '\n'
              << "cuda_free_memory_delta_mib: "
              << static_cast<double>(free_before - free_after) / (1 << 20)
              << '\n';
    if (process_gpu_mib.has_value()) {
      std::cout << "process_gpu_resident_mib: " << *process_gpu_mib << '\n';
    }
    std::cout
              << "best_score: " << scores.front() << '\n'
              << "worst_score: " << scores.back() << '\n'
              << "ptv3_stage_counts: ";
    for (std::size_t index = 0; index < stage_counts.size(); ++index) {
      std::cout << (index == 0 ? "" : ",") << stage_counts[index];
    }
    std::cout << '\n';
    std::cout << "split_preprocessing_topology_obb_ms: "
              << profile.preprocessing_topology_obb_ms << '\n'
              << "split_preprocessing_ms: " << profile.preprocessing_ms << '\n'
              << "split_topology_ms: " << profile.topology_ms << '\n'
              << "split_obb_noise_ms: " << profile.obb_noise_ms << '\n'
              << "split_dual_ptv3_ms: " << profile.dual_ptv3_ms << '\n'
              << "split_denoising_ms: " << profile.denoising_steps_ms << '\n'
              << "split_scorer_rank_ms: " << profile.scorer_rank_ms << '\n'
              << "profile_full_pipeline_ms: " << profile.full_pipeline_ms
              << '\n';
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
