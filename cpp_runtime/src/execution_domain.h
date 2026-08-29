// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace graspgenx::detail {

inline void CheckDomainCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

inline void CheckDomainDriver(CUresult status, const char* operation) {
  if (status != CUDA_SUCCESS) {
    const char* detail = nullptr;
    cuGetErrorString(status, &detail);
    throw std::runtime_error(std::string(operation) + ": " +
                             (detail == nullptr ? "unknown CUDA driver error"
                                                : detail));
  }
}

class UniqueCudaStream {
 public:
  explicit UniqueCudaStream(cudaStream_t stream = nullptr) noexcept
      : stream_(stream) {}
  ~UniqueCudaStream() {
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

  UniqueCudaStream(const UniqueCudaStream&) = delete;
  UniqueCudaStream& operator=(const UniqueCudaStream&) = delete;
  UniqueCudaStream(UniqueCudaStream&& other) noexcept
      : stream_(std::exchange(other.stream_, nullptr)) {}
  UniqueCudaStream& operator=(UniqueCudaStream&& other) noexcept {
    if (this != &other) {
      if (stream_ != nullptr) cudaStreamDestroy(stream_);
      stream_ = std::exchange(other.stream_, nullptr);
    }
    return *this;
  }

  cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_{};
};

// A CUDA Runtime stream is created in the device's primary context. Keeping
// this ownership in one value type gives Runtime a single execution-domain
// seam. A future GreenDomain can be added to ExecutionDomainState without
// changing pipeline code or introducing a virtual hierarchy.
struct PrimaryDomain {
  static PrimaryDomain Current() {
    CheckDomainDriver(cuInit(0), "initialize CUDA driver");
    int device = 0;
    CheckDomainCuda(cudaGetDevice(&device), "cudaGetDevice execution domain");
    CUdevice driver_device = 0;
    CheckDomainDriver(cuDeviceGet(&driver_device, device),
                      "resolve primary-domain device");
    CUcontext context = nullptr;
    CheckDomainDriver(cuDevicePrimaryCtxRetain(&context, driver_device),
                      "retain CUDA primary context");
    const CUresult activation = cuCtxSetCurrent(context);
    if (activation != CUDA_SUCCESS) {
      cuDevicePrimaryCtxRelease(driver_device);
      CheckDomainDriver(activation, "activate CUDA primary context");
    }
    return PrimaryDomain(device, driver_device, context);
  }

  PrimaryDomain(int device, CUdevice driver_device, CUcontext context)
      : device(device), driver_device_(driver_device), context_(context) {}

  ~PrimaryDomain() {
    if (context_ == nullptr) return;
    // Streams must be destroyed while their context is current and before the
    // primary-context retain is released.
    cuCtxSetCurrent(context_);
    streams.clear();
    cuDevicePrimaryCtxRelease(driver_device_);
  }

  PrimaryDomain(const PrimaryDomain&) = delete;
  PrimaryDomain& operator=(const PrimaryDomain&) = delete;
  PrimaryDomain(PrimaryDomain&& other) noexcept
      : device(other.device),
        streams(std::move(other.streams)),
        driver_device_(other.driver_device_),
        context_(std::exchange(other.context_, nullptr)) {}
  PrimaryDomain& operator=(PrimaryDomain&&) = delete;

  cudaStream_t CreateStream(unsigned int flags = cudaStreamNonBlocking) {
    Activate();
    cudaStream_t stream = nullptr;
    CheckDomainCuda(cudaStreamCreateWithFlags(&stream, flags),
                    "create primary-domain stream");
    streams.emplace_back(stream);
    return stream;
  }

  void Activate() const {
    CheckDomainDriver(cuCtxSetCurrent(context_),
                      "activate primary execution domain");
  }

  void ActivateNoThrow() const noexcept {
    if (context_ != nullptr) (void)cuCtxSetCurrent(context_);
  }

  void ValidateStream(cudaStream_t stream) const {
    Activate();
    if (stream == nullptr) return;
    CUcontext stream_context = nullptr;
    CheckDomainDriver(
        cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &stream_context),
        "query caller stream context");
    if (stream_context != context_) {
      throw std::invalid_argument(
          "caller CUDA stream does not belong to the runtime execution domain");
    }
  }

  int device = 0;
  std::vector<UniqueCudaStream> streams;

 private:
  CUdevice driver_device_{};
  CUcontext context_{};
};

using ExecutionDomainState = std::variant<PrimaryDomain>;

class ExecutionDomain {
 public:
  static ExecutionDomain Primary() {
    return ExecutionDomain(PrimaryDomain::Current());
  }

  cudaStream_t CreateStream(unsigned int flags = cudaStreamNonBlocking) {
    return std::visit(
        [flags](auto& domain) { return domain.CreateStream(flags); }, state_);
  }

  void Activate() const {
    std::visit([](const auto& domain) { domain.Activate(); }, state_);
  }

  void ActivateNoThrow() const noexcept {
    std::visit([](const auto& domain) { domain.ActivateNoThrow(); }, state_);
  }

  void ValidateStream(cudaStream_t stream) const {
    std::visit([stream](const auto& domain) { domain.ValidateStream(stream); },
               state_);
  }

  int device() const noexcept {
    return std::visit([](const auto& domain) { return domain.device; }, state_);
  }

 private:
  explicit ExecutionDomain(PrimaryDomain domain) : state_(std::move(domain)) {}

  ExecutionDomainState state_;
};

}  // namespace graspgenx::detail
