// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorrt_plan.h"

#include <fstream>
#include <stdexcept>
#include <vector>

namespace graspgenx::detail {
namespace {

std::vector<char> ReadFile(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("Could not open TensorRT plan: " + path.string());
  }
  const auto end = stream.tellg();
  if (end <= 0) {
    throw std::runtime_error("TensorRT plan is empty: " + path.string());
  }
  std::vector<char> bytes(static_cast<std::size_t>(end));
  stream.seekg(0, std::ios::beg);
  stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!stream) {
    throw std::runtime_error("Could not read TensorRT plan: " + path.string());
  }
  return bytes;
}

}  // namespace

void Logger::log(Severity severity, const char* message) noexcept {
  if (severity <= Severity::kWARNING) {
    last_message_ = message == nullptr ? "" : message;
  }
}

TensorRtPlan::TensorRtPlan(const std::filesystem::path& path) : path_(path) {
  const std::vector<char> bytes = ReadFile(path);
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) {
    throw std::runtime_error("Could not create TensorRT runtime");
  }
  engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
  if (!engine_) {
    throw std::runtime_error("Could not deserialize " + path.string() + ": " +
                             logger_.last_message());
  }
  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    throw std::runtime_error("Could not create execution context for " +
                             path.string());
  }
}

void TensorRtPlan::Bind(const char* name, void* address) {
  if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kNONE) {
    throw std::runtime_error(std::string("TensorRT plan has no tensor named ") +
                             name);
  }
  if (!context_->setTensorAddress(name, address)) {
    throw std::runtime_error(std::string("TensorRT rejected tensor address for ") +
                             name);
  }
}

void TensorRtPlan::Enqueue(cudaStream_t stream) {
  if (!context_->enqueueV3(stream)) {
    throw std::runtime_error("TensorRT enqueueV3 failed: " +
                             logger_.last_message());
  }
}

std::size_t TensorRtPlan::device_memory_bytes() const noexcept {
  return engine_->getDeviceMemorySizeV2();
}

nvinfer1::Dims TensorRtPlan::Shape(const char* name) const {
  const auto shape = engine_->getTensorShape(name);
  if (shape.nbDims < 0)
    throw std::invalid_argument(path_.string() + ": missing tensor " + name);
  return shape;
}

void TensorRtPlan::CheckTensor(const char* name,
                              std::initializer_list<std::int64_t> expected,
                              nvinfer1::DataType type, bool output) const {
  const auto shape = Shape(name);
  bool matches = shape.nbDims == static_cast<int>(expected.size());
  int index = 0;
  for (auto dimension : expected) {
    if (index >= shape.nbDims || shape.d[index] != dimension) matches = false;
    ++index;
  }
  const auto mode = output ? nvinfer1::TensorIOMode::kOUTPUT
                           : nvinfer1::TensorIOMode::kINPUT;
  if (!matches || engine_->getTensorDataType(name) != type ||
      engine_->getTensorIOMode(name) != mode)
    throw std::invalid_argument(
        path_.string() + ": incompatible shape, type, or mode for tensor " + name);
}

}  // namespace graspgenx::detail
