// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <graspgenx/runtime.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(bool ok, const char* why) {
  if (!ok) throw std::runtime_error(why);
}
void cuda_check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template <class F> void rejected(F call) {
  try { call(); } catch (const std::logic_error&) { return; }
  throw std::runtime_error("invalid volume request was accepted");
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: graspgenx_volume_runtime_test DDPM20_ARTIFACT_DIRECTORY\n";
    return 1;
  }
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
  try {
    graspgenx::RuntimeConfig config;
    config.artifact_directory = argv[1];
    config.minimum_points_in_volume = 1;
    graspgenx::Runtime runtime(config);
    cudaStream_t stream{};
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<float> points(100 * 3);
    for (int i = 0; i < 100; ++i) {
      points[i * 3] = (i % 10 - 4.5F) * .004F;
      points[i * 3 + 1] = (i / 10 - 4.5F) * .004F;
      points[i * 3 + 2] = .001F * (i % 3);
    }
    rejected([&] { runtime.InferHostAsync(points.data(), 100, 3, 5, stream); });
    rejected([&] { runtime.SetMinimumPointsInVolume(-1); });
    rejected([&] { runtime.SetMinimumPointsInVolume(3501); });
    std::array<float, 6> bounds{-100, -100, -100, 100, 100, 100};
    auto invalid = bounds;
    invalid[3] = invalid[0];
    rejected([&] { runtime.SetGraspVolume(invalid.data()); });
    runtime.SetGraspVolume(bounds.data());
    const auto output = runtime.outputs();
    check(output.grasp_count == 150 && output.valid, "missing validity output");
    auto infer = [&](bool expected) {
      runtime.InferHostAsync(points.data(), 100, 3, 5, stream);
      cuda_check(cudaStreamSynchronize(stream));
      std::vector<std::uint8_t> valid(output.grasp_count);
      std::vector<float> scores(output.grasp_count);
      cuda_check(cudaMemcpy(valid.data(), output.valid, valid.size(), cudaMemcpyDeviceToHost));
      cuda_check(cudaMemcpy(scores.data(), output.confidence, scores.size() * sizeof(float), cudaMemcpyDeviceToHost));
      check(std::all_of(valid.begin(), valid.end(), [=](auto v) { return v == expected; }),
            "captured graph ignored updated volume/threshold");
      if (!expected)
        check(std::all_of(scores.begin(), scores.end(), [](float s) { return s == 0; }),
              "rejected confidence must be zero");
      return scores;
    };
    runtime.SetMinimumPointsInVolume(100);
    const auto baseline = infer(true);
    // Setters must wait for queued inference without recapturing the graph.
    runtime.InferHostAsync(points.data(), 100, 3, 5, stream);
    runtime.SetMinimumPointsInVolume(101);
    infer(false);  // Encoder repetition must not inflate the supplied point count.
    runtime.SetMinimumPointsInVolume(1);
    bounds = {1000, 1000, 1000, 1001, 1001, 1001};
    runtime.InferHostAsync(points.data(), 100, 3, 5, stream);
    runtime.SetGraspVolume(bounds.data());
    infer(false);
    runtime.SetMinimumPointsInVolume(0);
    check(infer(true) == baseline, "disabled filtering changed deterministic scores");
    cuda_check(cudaStreamDestroy(stream));
    std::cout << "DDPM20 volume/graph update checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
