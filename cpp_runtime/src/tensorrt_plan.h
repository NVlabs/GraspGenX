// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <string>

namespace graspgenx::detail {

template <typename T>
struct TensorRtDeleter {
  void operator()(T* object) const noexcept { delete object; }
};

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override;
  const std::string& last_message() const noexcept { return last_message_; }

 private:
  std::string last_message_;
};

class TensorRtPlan {
 public:
  explicit TensorRtPlan(const std::filesystem::path& path);

  TensorRtPlan(const TensorRtPlan&) = delete;
  TensorRtPlan& operator=(const TensorRtPlan&) = delete;

  void Bind(const char* name, void* address);
  void Enqueue(cudaStream_t stream);
  std::size_t device_memory_bytes() const noexcept;
  nvinfer1::Dims Shape(const char* name) const;
  void CheckTensor(const char* name, std::initializer_list<std::int64_t> shape,
                   nvinfer1::DataType type, bool output = false) const;

 private:
  std::filesystem::path path_;
  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime, TensorRtDeleter<nvinfer1::IRuntime>> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine,
                  TensorRtDeleter<nvinfer1::ICudaEngine>>
      engine_;
  std::unique_ptr<nvinfer1::IExecutionContext,
                  TensorRtDeleter<nvinfer1::IExecutionContext>>
      context_;
};

}  // namespace graspgenx::detail
