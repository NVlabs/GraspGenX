// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "graspgenx/runtime.h"

#include "cuda_kernels.h"
#include "execution_domain.h"
#include "tensorrt_plan.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace graspgenx {
namespace {

constexpr int kStageCount = 5;
constexpr int kPointCount = 3500;
constexpr int kMinimumPointCount = 100;
static_assert(Runtime::kEnginePointCount == kPointCount);
static_assert(Runtime::kMinimumPointCount == kMinimumPointCount);
constexpr std::array<int, kStageCount> kCapacities{kPointCount, 1536, 512, 128,
                                                    64};
constexpr std::array<int, kStageCount> kInputChannels{3, 64, 128, 256, 512};
constexpr std::array<int, kStageCount> kOutputChannels{64, 128, 256, 512, 512};
constexpr int kOrderCount = 4;
constexpr std::array<int, kStageCount> kUsedOrderCounts{2, 2, 2, 4, 2};
constexpr int kPatchMaximum = 1024;
constexpr int kEmbeddingChannels = 512;
constexpr int kSpatialCoordinateBits = 10;
constexpr float kKappa = 3.27F;
constexpr float kGridSize = 0.01F;
constexpr float kOutlierThreshold = 0.014F;

void ValidatePointCount(std::int64_t point_count) {
  if (point_count < kMinimumPointCount) {
    throw std::invalid_argument(
        "GraspGenX requires at least 100 object points; received "
        + std::to_string(point_count));
  }
}

std::int64_t DownsampleIndex(std::int64_t output_index,
                             std::int64_t input_count) {
  const std::int64_t quotient = input_count / kPointCount;
  const std::int64_t remainder = input_count % kPointCount;
  return output_index * quotient
      + (output_index * remainder) / kPointCount;
}

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

void CheckCublas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with status " +
                             std::to_string(static_cast<int>(status)));
  }
}

std::vector<char> ReadExact(const std::filesystem::path& path,
                            std::size_t expected_bytes) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("Could not open runtime tensor: " + path.string());
  }
  const auto end = stream.tellg();
  if (end < 0 || static_cast<std::size_t>(end) != expected_bytes) {
    throw std::runtime_error("Unexpected byte count for runtime tensor: " +
                             path.string());
  }
  std::vector<char> bytes(expected_bytes);
  stream.seekg(0, std::ios::beg);
  stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!stream) {
    throw std::runtime_error("Could not read runtime tensor: " + path.string());
  }
  return bytes;
}

class DeviceArena {
 public:
  ~DeviceArena() {
    for (void* pointer : allocations_) cudaFree(pointer);
  }

  template <typename T>
  T* Allocate(std::size_t count) {
    void* pointer = nullptr;
    const std::size_t bytes = std::max<std::size_t>(count * sizeof(T), 1);
    CheckCuda(cudaMalloc(&pointer, bytes), "cudaMalloc");
    allocations_.push_back(pointer);
    bytes_ += bytes;
    return static_cast<T*>(pointer);
  }

  void* AllocateBytes(std::size_t bytes) {
    return Allocate<std::uint8_t>(std::max<std::size_t>(bytes, 1));
  }

  std::size_t bytes() const noexcept { return bytes_; }

 private:
  std::vector<void*> allocations_;
  std::size_t bytes_ = 0;
};

int PatchSize(int capacity) { return std::min(capacity, kPatchMaximum); }

int PaddedCapacity(int capacity) {
  const int patch = PatchSize(capacity);
  return capacity <= patch ? capacity : ((capacity + patch - 1) / patch) * patch;
}

struct TopologyStage {
  int capacity = 0;
  int patch_size = 0;
  int padded_capacity = 0;
  float* coordinates = nullptr;
  std::int32_t* grid_coordinates = nullptr;
  std::int64_t* codes = nullptr;
  std::int64_t* order = nullptr;
  std::int64_t* inverse = nullptr;
  std::int64_t* valid_count = nullptr;
  std::int64_t* neighbor_k3 = nullptr;
  bool* neighbor_k3_mask = nullptr;
  std::int64_t* neighbor_k5 = nullptr;
  bool* neighbor_k5_mask = nullptr;
  std::int64_t* attention_order = nullptr;
  std::int64_t* attention_inverse = nullptr;
  std::int32_t* valid_key_counts = nullptr;
  std::int64_t* pool_index_ptr = nullptr;
  std::int64_t* pool_cluster = nullptr;
  bool* overflow = nullptr;
};

struct EncoderBranch {
  std::array<void*, kStageCount> inputs{};
  std::array<void*, kStageCount> outputs{};
  void* embedding = nullptr;
  std::array<std::unique_ptr<detail::TensorRtPlan>, kStageCount> plans;
};

}  // namespace

class Runtime::Impl {
 public:
  explicit Impl(RuntimeConfig config)
      : config_(std::move(config)), domain_(detail::ExecutionDomain::Primary()) {
    ValidateConfig();
    root_stream_ = domain_.CreateStream();
    generator_stream_ = domain_.CreateStream();
    discriminator_stream_ = domain_.CreateStream();
    obb_stream_ = domain_.CreateStream();
    topology_stream_ = domain_.CreateStream();
    topology_order_stream_ = domain_.CreateStream();
    topology_attention_stream_ = domain_.CreateStream();
    CreateEvent(&points_ready_);
    CreateEvent(&topology_ready_);
    CreateEvent(&generator_ready_);
    CreateEvent(&discriminator_ready_);
    CreateEvent(&obb_ready_);
    for (cudaEvent_t& event : topology_stage_ready_) CreateEvent(&event);
    for (cudaEvent_t& event : topology_pool_data_ready_) CreateEvent(&event);
    for (cudaEvent_t& event : topology_order_ready_) CreateEvent(&event);
    CreateEvent(&topology_maps_ready_);
    CreateEvent(&topology_attention_ready_);
    CheckCublas(cublasCreate(&cublas_), "cublasCreate");
    AllocateBuffers();
    LoadConstants();
    LoadPlans();
    BindPlans();
    SetGripperEmbeddings(nullptr, nullptr);
    WarmAndCapture();
  }

  ~Impl() {
    // Destruction can be initiated while another CUDA device/context is
    // current in the embedding application.
    domain_.ActivateNoThrow();
    if (graph_exec_ != nullptr) cudaGraphExecDestroy(graph_exec_);
    if (graph_ != nullptr) cudaGraphDestroy(graph_);
    if (cublas_ != nullptr) cublasDestroy(cublas_);
    DestroyEvent(points_ready_);
    DestroyEvent(topology_ready_);
    DestroyEvent(generator_ready_);
    DestroyEvent(discriminator_ready_);
    DestroyEvent(obb_ready_);
    for (cudaEvent_t event : topology_stage_ready_) DestroyEvent(event);
    for (cudaEvent_t event : topology_pool_data_ready_) DestroyEvent(event);
    for (cudaEvent_t event : topology_order_ready_) DestroyEvent(event);
    DestroyEvent(topology_maps_ready_);
    DestroyEvent(topology_attention_ready_);
  }

  void SetGripperEmbeddings(const float* generator_embedding,
                            const float* discriminator_embedding) {
    domain_.Activate();
    std::array<__half, kEmbeddingChannels> generator_half{};
    std::array<float, kEmbeddingChannels> discriminator_float{};
    for (int index = 0; index < kEmbeddingChannels; ++index) {
      generator_half[index] =
          __float2half(generator_embedding == nullptr ? 0.0F
                                                      : generator_embedding[index]);
      discriminator_float[index] = discriminator_embedding == nullptr
          ? 0.0F
          : discriminator_embedding[index];
    }
    CheckCuda(cudaMemcpy(observation_ + kEmbeddingChannels,
                         generator_half.data(), sizeof(generator_half),
                         cudaMemcpyHostToDevice),
              "copy generator gripper embedding");
    CheckCuda(cudaMemcpy(scorer_gripper_embedding_, discriminator_float.data(),
                         sizeof(discriminator_float), cudaMemcpyHostToDevice),
              "copy discriminator gripper embedding");
    CheckCuda(cudaMemcpy(gripper_depth_, &config_.gripper_depth,
                         sizeof(config_.gripper_depth), cudaMemcpyHostToDevice),
              "copy gripper depth");
  }

  void SetGripperDeviceAsync(const void* generator_embedding,
                             const float* discriminator_embedding,
                             float gripper_depth, cudaStream_t stream) {
    domain_.ValidateStream(stream);
    if (generator_embedding == nullptr || discriminator_embedding == nullptr) {
      throw std::invalid_argument("device gripper embeddings must not be null");
    }
    if (!std::isfinite(gripper_depth) || gripper_depth <= 0.0F) {
      throw std::invalid_argument("gripper_depth must be positive and finite");
    }
    CheckCuda(cudaMemcpyAsync(observation_ + kEmbeddingChannels,
                              generator_embedding,
                              kEmbeddingChannels * sizeof(__half),
                              cudaMemcpyDeviceToDevice, stream),
              "copy device generator gripper embedding");
    CheckCuda(cudaMemcpyAsync(scorer_gripper_embedding_,
                              discriminator_embedding,
                              kEmbeddingChannels * sizeof(float),
                              cudaMemcpyDeviceToDevice, stream),
              "copy device discriminator gripper embedding");
    CheckCuda(cudaMemcpyAsync(gripper_depth_, &gripper_depth,
                              sizeof(gripper_depth), cudaMemcpyHostToDevice,
                              stream),
              "copy device gripper depth");
  }

  void InferAsync(std::int64_t point_count, std::uint64_t seed,
                  std::uint64_t sequence_offset, cudaStream_t stream) {
    domain_.ValidateStream(stream);
    ValidatePointCount(point_count);
    if (point_count > kPointCount) {
      throw std::invalid_argument(
          "The zero-copy input buffer holds at most 3500 points; use "
          "InferDeviceAsync or InferHostAsync for larger clouds");
    }
    LaunchPrepared(point_count, seed, sequence_offset, stream);
  }

  void LaunchPrepared(std::int64_t point_count, std::uint64_t seed,
                      std::uint64_t sequence_offset, cudaStream_t stream) {
    CheckCuda(cudaMemcpyAsync(input_point_count_, &point_count,
                              sizeof(point_count), cudaMemcpyHostToDevice,
                              stream),
              "copy input point count");
    CheckCuda(cudaMemcpyAsync(seed_, &seed, sizeof(seed), cudaMemcpyHostToDevice,
                              stream),
              "copy seed");
    CheckCuda(cudaMemcpyAsync(sequence_offset_, &sequence_offset,
                              sizeof(sequence_offset), cudaMemcpyHostToDevice,
                              stream),
              "copy sequence offset");
    CheckCuda(cudaGraphLaunch(graph_exec_, stream), "cudaGraphLaunch inference");
  }

  void InferDeviceAsync(const float* points, std::int64_t point_count,
                        std::uint64_t seed, std::uint64_t sequence_offset,
                        cudaStream_t stream) {
    domain_.ValidateStream(stream);
    ValidatePointCount(point_count);
    if (points == nullptr) {
      throw std::invalid_argument("Device point cloud pointer is null");
    }
    const std::int64_t effective_count =
        std::min<std::int64_t>(point_count, kPointCount);
    if (point_count <= kPointCount) {
      CheckCuda(cudaMemcpyAsync(
                    input_points_, points,
                    static_cast<std::size_t>(point_count) * 3 * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream),
                "copy device point cloud");
    } else {
      CheckCuda(LaunchResampleFixedFloat(
                    points, point_count, input_points_, kPointCount, stream),
                "downsample device point cloud");
    }
    LaunchPrepared(effective_count, seed, sequence_offset, stream);
  }

  void InferHostAsync(const float* points, std::int64_t point_count,
                      std::uint64_t seed, std::uint64_t sequence_offset,
                      cudaStream_t stream) {
    domain_.ValidateStream(stream);
    ValidatePointCount(point_count);
    if (points == nullptr) {
      throw std::invalid_argument("Host point cloud pointer is null");
    }
    const float* copy_source = points;
    std::int64_t effective_count = point_count;
    if (point_count > kPointCount) {
      for (std::int64_t output = 0; output < kPointCount; ++output) {
        const std::int64_t input = DownsampleIndex(output, point_count);
        host_resampled_points_[output * 3 + 0] = points[input * 3 + 0];
        host_resampled_points_[output * 3 + 1] = points[input * 3 + 1];
        host_resampled_points_[output * 3 + 2] = points[input * 3 + 2];
      }
      copy_source = host_resampled_points_.data();
      effective_count = kPointCount;
    }
    CheckCuda(cudaMemcpyAsync(
                  input_points_, copy_source,
                  static_cast<std::size_t>(effective_count) * 3 * sizeof(float),
                  cudaMemcpyHostToDevice, stream),
              "copy host point cloud");
    LaunchPrepared(effective_count, seed, sequence_offset, stream);
  }

  DeviceOutputs outputs() const noexcept {
    return DeviceOutputs{sorted_grasps_, sorted_confidence_, source_indices_,
                         center_, obb_bounds_, topology_valid_counts_,
                         topology_overflow_,
                         config_.diffusion_grasps + config_.obb_grasps};
  }

  RuntimeMemory memory() const noexcept {
    return RuntimeMemory{arena_.bytes(), tensorrt_device_memory_bytes_};
  }

  int device() const noexcept { return domain_.device(); }

  RuntimeProfile Profile(int warmup_iterations, int measured_iterations) {
    domain_.Activate();
    if (warmup_iterations < 1 || measured_iterations < 1) {
      throw std::invalid_argument("Profile iteration counts must be positive");
    }
    CheckCuda(cudaDeviceSynchronize(), "synchronize before component capture");
    auto topology = CaptureComponent([this](cudaStream_t stream) {
      BuildTopology(stream);
      CheckCuda(cudaStreamWaitEvent(stream, obb_ready_),
                "topology profile waits OBB");
    });
    auto preprocessing = CaptureComponent([this](cudaStream_t stream) {
      PreparePoints(stream);
    });
    auto ptv3_topology = CaptureComponent([this](cudaStream_t stream) {
      BuildPtv3Topology(stream);
    });
    auto obb_noise = CaptureComponent([this](cudaStream_t stream) {
      RunObbAndNoise(stream);
    });
    auto encoders = CaptureComponent([this](cudaStream_t stream) {
      CheckCuda(cudaEventRecord(topology_ready_, stream),
                "profile topology ready");
      CheckCuda(cudaStreamWaitEvent(generator_stream_, topology_ready_),
                "profile generator wait");
      CheckCuda(cudaStreamWaitEvent(discriminator_stream_, topology_ready_),
                "profile discriminator wait");
      RunEncoder(&generator_, generator_stream_);
      CheckCuda(cudaEventRecord(generator_ready_, generator_stream_),
                "profile generator ready");
      RunEncoder(&discriminator_, discriminator_stream_);
      CheckCuda(cudaEventRecord(discriminator_ready_, discriminator_stream_),
                "profile discriminator ready");
      CheckCuda(cudaStreamWaitEvent(stream, generator_ready_),
                "profile join generator");
      CheckCuda(cudaStreamWaitEvent(stream, discriminator_ready_),
                "profile join discriminator");
    });
    auto denoising = CaptureComponent([this](cudaStream_t stream) {
      RunDiffusion(stream, false);
    });
    auto scoring = CaptureComponent([this](cudaStream_t stream) {
      CheckCuda(LaunchPrepareScorerInputs(
                    diffusion_canonical_, obb_canonical_,
                    discriminator_.embedding, scorer_canonical_,
                    scorer_object_embedding_, config_.diffusion_grasps,
                    config_.obb_grasps, stream),
                "profile prepare scorer");
      scorer_->Enqueue(stream);
      CheckCuda(LaunchRankAndTranslate(
                    diffusion_grasps_, obb_grasps_, scorer_confidence_, center_,
                    sorted_grasps_, sorted_confidence_, source_indices_,
                    config_.diffusion_grasps, config_.obb_grasps, stream),
                "profile rank grasps");
    });
    RuntimeProfile result;
    result.preprocessing_topology_obb_ms = MeasureGraph(
        topology.executable, warmup_iterations, measured_iterations);
    result.preprocessing_ms = MeasureGraph(
        preprocessing.executable, warmup_iterations, measured_iterations);
    result.topology_ms = MeasureGraph(
        ptv3_topology.executable, warmup_iterations, measured_iterations);
    result.obb_noise_ms = MeasureGraph(
        obb_noise.executable, warmup_iterations, measured_iterations);
    result.dual_ptv3_ms =
        MeasureGraph(encoders.executable, warmup_iterations, measured_iterations);
    result.denoising_steps_ms = MeasureGraph(
        denoising.executable, warmup_iterations, measured_iterations);
    result.scorer_rank_ms =
        MeasureGraph(scoring.executable, warmup_iterations, measured_iterations);
    result.full_pipeline_ms =
        MeasureGraph(graph_exec_, warmup_iterations, measured_iterations);
    return result;
  }

  float* input_points() noexcept { return input_points_; }

 private:
  struct CapturedComponent {
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    CapturedComponent() = default;
    CapturedComponent(const CapturedComponent&) = delete;
    CapturedComponent& operator=(const CapturedComponent&) = delete;
    CapturedComponent(CapturedComponent&& other) noexcept
        : graph(std::exchange(other.graph, nullptr)),
          executable(std::exchange(other.executable, nullptr)) {}
    ~CapturedComponent() {
      if (executable != nullptr) cudaGraphExecDestroy(executable);
      if (graph != nullptr) cudaGraphDestroy(graph);
    }
  };

  CapturedComponent CaptureComponent(
      const std::function<void(cudaStream_t)>& enqueue) {
    CapturedComponent captured;
    CheckCuda(cudaStreamBeginCapture(root_stream_, cudaStreamCaptureModeGlobal),
              "begin component capture");
    enqueue(root_stream_);
    CheckCuda(cudaStreamEndCapture(root_stream_, &captured.graph),
              "end component capture");
    CheckCuda(cudaGraphInstantiate(&captured.executable, captured.graph, 0),
              "instantiate component graph");
    return captured;
  }

  float MeasureGraph(cudaGraphExec_t executable, int warmup_iterations,
                     int measured_iterations) {
    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
      CheckCuda(cudaGraphLaunch(executable, root_stream_),
                "component graph warmup");
    }
    CheckCuda(cudaStreamSynchronize(root_stream_),
              "component graph warmup synchronize");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CheckCuda(cudaEventCreate(&start), "create profile start event");
    CheckCuda(cudaEventCreate(&stop), "create profile stop event");
    CheckCuda(cudaEventRecord(start, root_stream_), "record profile start");
    for (int iteration = 0; iteration < measured_iterations; ++iteration) {
      CheckCuda(cudaGraphLaunch(executable, root_stream_),
                "component graph replay");
    }
    CheckCuda(cudaEventRecord(stop, root_stream_), "record profile stop");
    CheckCuda(cudaEventSynchronize(stop), "synchronize profile stop");
    float elapsed = 0.0F;
    CheckCuda(cudaEventElapsedTime(&elapsed, start, stop),
              "profile elapsed time");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return elapsed / measured_iterations;
  }

  void ValidateConfig() {
    if (config_.artifact_directory.empty()) {
      throw std::invalid_argument("artifact_directory is required");
    }
    if (!std::isfinite(config_.gripper_depth) || config_.gripper_depth <= 0.0F) {
      throw std::invalid_argument("gripper_depth must be positive and finite");
    }
    if (config_.diffusion_steps != 5 && config_.diffusion_steps != 10 &&
        config_.diffusion_steps != 20) {
      throw std::invalid_argument("diffusion_steps must be one of 5, 10, or 20");
    }
    if (config_.diffusion_grasps < 1) {
      throw std::invalid_argument("diffusion_grasps must be positive");
    }
    if (config_.obb_grasps < 0) {
      throw std::invalid_argument("obb_grasps must be non-negative");
    }
    if (config_.diffusion_grasps + config_.obb_grasps > 4096) {
      throw std::invalid_argument(
          "the combined diffusion and OBB grasp count must not exceed 4096");
    }
  }

  static void CreateEvent(cudaEvent_t* event) {
    CheckCuda(cudaEventCreateWithFlags(event, cudaEventDisableTiming),
              "cudaEventCreate");
  }

  static void DestroyEvent(cudaEvent_t event) {
    if (event != nullptr) cudaEventDestroy(event);
  }

  template <typename T>
  T* Allocate(std::size_t count) {
    return arena_.Allocate<T>(count);
  }

  void AllocateBuffers() {
    input_points_ = Allocate<float>(kPointCount * 3);
    input_point_count_ = Allocate<std::int64_t>(1);
    filter_keep_ = Allocate<bool>(kPointCount);
    selected_indices_ = Allocate<std::int64_t>(kPointCount);
    selected_count_ = Allocate<std::int32_t>(1);
    filtered_points_ = Allocate<float>(kPointCount * 3);
    centered_points_ = Allocate<float>(kPointCount * 3);
    center_ = Allocate<float>(3);
    input_features_ = Allocate<__half>(kPointCount * 3);
    maximum_coordinate_ = Allocate<std::int32_t>(1);
    pool_segment_ids_ = Allocate<std::int64_t>(kCapacities[0]);
    topology_valid_counts_ = Allocate<std::int64_t>(kStageCount);
    topology_overflow_ = Allocate<bool>(kStageCount - 1);

    for (int index = 0; index < kStageCount; ++index) {
      TopologyStage& stage = topology_[index];
      stage.capacity = kCapacities[index];
      stage.patch_size = PatchSize(stage.capacity);
      stage.padded_capacity = PaddedCapacity(stage.capacity);
      stage.coordinates = Allocate<float>(stage.capacity * 3);
      stage.grid_coordinates = Allocate<std::int32_t>(stage.capacity * 3);
      stage.codes = Allocate<std::int64_t>(kOrderCount * stage.capacity);
      stage.order = Allocate<std::int64_t>(kOrderCount * stage.capacity);
      stage.inverse = Allocate<std::int64_t>(kOrderCount * stage.capacity);
      stage.valid_count = topology_valid_counts_ + index;
      stage.neighbor_k3 = Allocate<std::int64_t>(stage.capacity * 27);
      stage.neighbor_k3_mask = Allocate<bool>(stage.capacity * 27);
      if (index == 0) {
        stage.neighbor_k5 = Allocate<std::int64_t>(stage.capacity * 125);
        stage.neighbor_k5_mask = Allocate<bool>(stage.capacity * 125);
      }
      stage.attention_order =
          Allocate<std::int64_t>(kOrderCount * stage.padded_capacity);
      stage.attention_inverse =
          Allocate<std::int64_t>(kOrderCount * stage.capacity);
      stage.valid_key_counts =
          Allocate<std::int32_t>(stage.padded_capacity / stage.patch_size);
      if (index + 1 < kStageCount) {
        stage.pool_index_ptr = Allocate<std::int64_t>(kCapacities[index + 1] + 1);
        stage.pool_cluster = Allocate<std::int64_t>(stage.capacity);
        stage.overflow = topology_overflow_ + index;
      }
    }
    offsets_k3_ = Allocate<std::int64_t>(27 * 3);
    offsets_k5_ = Allocate<std::int64_t>(125 * 3);
    AllocateBranch(&generator_);
    AllocateBranch(&discriminator_);

    observation_ = Allocate<__half>(1024);
    generator_.embedding = observation_;
    discriminator_.embedding = Allocate<__half>(kEmbeddingChannels);
    observation_projection_ = Allocate<__half>(2048);
    conditioning_buffer_ = Allocate<__half>(2048);
    const std::size_t diffusion_grasps = config_.diffusion_grasps;
    const std::size_t obb_grasps = config_.obb_grasps;
    const std::size_t candidates = diffusion_grasps + obb_grasps;
    sample_embedding_ = Allocate<__half>(diffusion_grasps * 512);
    prediction_hidden_ = Allocate<__half>(diffusion_grasps * 512);
    initial_sample_ = Allocate<__half>(diffusion_grasps * 6);
    variance_noise_ = Allocate<__half>(
        config_.diffusion_steps * diffusion_grasps * 6);
    state_[0] = Allocate<__half>(diffusion_grasps * 6);
    state_[1] = Allocate<__half>(diffusion_grasps * 6);
    diffusion_grasps_ = Allocate<float>(diffusion_grasps * 16);
    diffusion_canonical_ = Allocate<__half>(diffusion_grasps * 6);
    obb_grasps_ = Allocate<float>(obb_grasps * 16);
    obb_canonical_ = Allocate<float>(obb_grasps * 6);
    obb_bounds_ = Allocate<float>(8);

    scorer_canonical_ = Allocate<float>(candidates * 6);
    scorer_object_embedding_ = Allocate<float>(512);
    scorer_gripper_embedding_ = Allocate<float>(512);
    gripper_depth_ = Allocate<float>(1);
    scorer_logits_ = Allocate<float>(candidates);
    scorer_confidence_ = Allocate<float>(candidates);
    sorted_grasps_ = Allocate<float>(candidates * 16);
    sorted_confidence_ = Allocate<float>(candidates);
    source_indices_ = Allocate<std::int32_t>(candidates);
    seed_ = Allocate<std::uint64_t>(1);
    sequence_offset_ = Allocate<std::uint64_t>(1);

    observation_weight_ = Allocate<__half>(2048 * 1024);
    timestep_conditioning_ =
        Allocate<__half>(config_.diffusion_steps * 2048);
    sample_encoder_weight_ = Allocate<__half>(512 * 6);
    sample_encoder_bias_ = Allocate<__half>(512);
    prediction_weight_ = Allocate<__half>(6 * 512);
    prediction_bias_ = Allocate<__half>(6);
    ddpm_coefficients_ =
        Allocate<__half>(config_.diffusion_steps * 5 * 6);
  }

  void AllocateBranch(EncoderBranch* branch) {
    branch->inputs[0] = input_features_;
    for (int stage = 0; stage < kStageCount; ++stage) {
      branch->outputs[stage] = Allocate<__half>(
          kCapacities[stage] * kOutputChannels[stage]);
      if (stage + 1 < kStageCount) {
        branch->inputs[stage + 1] = Allocate<__half>(
            kCapacities[stage + 1] * kOutputChannels[stage]);
      }
    }
  }

  void LoadConstants() {
    std::vector<std::int64_t> k3;
    std::vector<std::int64_t> k5;
    for (int x = -2; x <= 2; ++x) {
      for (int y = -2; y <= 2; ++y) {
        for (int z = -2; z <= 2; ++z) {
          k5.insert(k5.end(), {x, y, z});
          if (std::abs(x) <= 1 && std::abs(y) <= 1 && std::abs(z) <= 1) {
            k3.insert(k3.end(), {x, y, z});
          }
        }
      }
    }
    CheckCuda(cudaMemcpy(offsets_k3_, k3.data(), k3.size() * sizeof(k3[0]),
                         cudaMemcpyHostToDevice),
              "copy K3 offsets");
    CheckCuda(cudaMemcpy(offsets_k5_, k5.data(), k5.size() * sizeof(k5[0]),
                         cudaMemcpyHostToDevice),
              "copy K5 offsets");
    const std::int64_t initial_valid_count = kCapacities[0];
    CheckCuda(cudaMemcpy(topology_[0].valid_count, &initial_valid_count,
                         sizeof(initial_valid_count), cudaMemcpyHostToDevice),
              "copy initial valid count");
    const auto weights = config_.artifact_directory / "weights";
    auto schedule_weights =
        weights / "schedules" / std::to_string(config_.diffusion_steps);
    if (config_.diffusion_steps == 10 &&
        !std::filesystem::is_directory(schedule_weights)) {
      schedule_weights = weights;
    }
    UploadTensor(weights / "observation_weight.bin", observation_weight_,
                 2048 * 1024 * sizeof(__half));
    UploadTensor(schedule_weights / "timestep_conditioning.bin",
                 timestep_conditioning_,
                 config_.diffusion_steps * 2048 * sizeof(__half));
    UploadTensor(weights / "sample_encoder_weight.bin", sample_encoder_weight_,
                 512 * 6 * sizeof(__half));
    UploadTensor(weights / "sample_encoder_bias.bin", sample_encoder_bias_,
                 512 * sizeof(__half));
    UploadTensor(weights / "prediction_weight.bin", prediction_weight_,
                 6 * 512 * sizeof(__half));
    UploadTensor(weights / "prediction_bias.bin", prediction_bias_,
                 6 * sizeof(__half));
    UploadTensor(schedule_weights / "ddpm_coefficients.bin", ddpm_coefficients_,
                 config_.diffusion_steps * 5 * 6 * sizeof(__half));
    UploadTensor(
        weights / ("sobol_initial_sample_d" +
                   std::to_string(config_.diffusion_grasps) + ".bin"),
        initial_sample_, config_.diffusion_grasps * 6 * sizeof(__half));
  }

  void UploadTensor(const std::filesystem::path& path, void* destination,
                    std::size_t bytes) {
    const auto value = ReadExact(path, bytes);
    CheckCuda(cudaMemcpy(destination, value.data(), bytes, cudaMemcpyHostToDevice),
              "upload runtime tensor");
  }

  void LoadPlans() {
    const auto engines = config_.artifact_directory / "engines";
    for (int stage = 0; stage < kStageCount; ++stage) {
      generator_.plans[stage] = std::make_unique<detail::TensorRtPlan>(
          engines / "ptv3" /
          ("ptv3_generator_stage" + std::to_string(stage) + "_fp16.plan"));
      discriminator_.plans[stage] = std::make_unique<detail::TensorRtPlan>(
          engines / "ptv3" /
          ("ptv3_discriminator_stage" + std::to_string(stage) +
           "_fp16.plan"));
      tensorrt_device_memory_bytes_ +=
          generator_.plans[stage]->device_memory_bytes();
      tensorrt_device_memory_bytes_ +=
          discriminator_.plans[stage]->device_memory_bytes();
    }
    denoiser_ = std::make_unique<detail::TensorRtPlan>(
        engines / ("denoiser_core_d" +
                   std::to_string(config_.diffusion_grasps) + "_fp16.plan"));
    scorer_ = std::make_unique<detail::TensorRtPlan>(
        engines / ("scorer_core_c" +
                   std::to_string(config_.diffusion_grasps +
                                  config_.obb_grasps) +
                   "_fp32.plan"));
    tensorrt_device_memory_bytes_ += denoiser_->device_memory_bytes();
    tensorrt_device_memory_bytes_ += scorer_->device_memory_bytes();
  }

  void BindPlans() {
    BindEncoder(&generator_);
    BindEncoder(&discriminator_);
    denoiser_->Bind("sample_embedding", sample_embedding_);
    denoiser_->Bind("conditioning_projection", conditioning_buffer_);
    denoiser_->Bind("prediction_hidden", prediction_hidden_);
    scorer_->Bind("canonical_rt", scorer_canonical_);
    scorer_->Bind("object_embedding", scorer_object_embedding_);
    scorer_->Bind("gripper_embedding", scorer_gripper_embedding_);
    scorer_->Bind("logits", scorer_logits_);
    scorer_->Bind("grasp_confidence", scorer_confidence_);
  }

  void BindEncoder(EncoderBranch* branch) {
    for (int stage = 0; stage < kStageCount; ++stage) {
      TopologyStage& topology = topology_[stage];
      auto& plan = branch->plans[stage];
      plan->Bind("features", branch->inputs[stage]);
      plan->Bind("neighbor_indices_k3", topology.neighbor_k3);
      plan->Bind("neighbor_mask_k3", topology.neighbor_k3_mask);
      plan->Bind("attention_order", topology.attention_order);
      plan->Bind("attention_inverse", topology.attention_inverse);
      plan->Bind("valid_key_counts", topology.valid_key_counts);
      if (stage == 0) {
        plan->Bind("neighbor_indices_k5", topology.neighbor_k5);
        plan->Bind("neighbor_mask_k5", topology.neighbor_k5_mask);
      }
      plan->Bind("features_out", branch->outputs[stage]);
    }
  }

  void PreparePoints(cudaStream_t stream) {
    CheckCuda(LaunchFilterResampleDynamicFloat(
                  input_points_, input_point_count_, kPointCount, kPointCount,
                  kOutlierThreshold, filter_keep_, selected_indices_,
                  selected_count_, filtered_points_, stream),
              "filter and resample");
    CheckCuda(LaunchCenterScaleAndGridFloat(
                  filtered_points_, kPointCount, kKappa, kGridSize,
                  centered_points_, center_,
                  topology_[0].coordinates, input_features_,
                  topology_[0].grid_coordinates, maximum_coordinate_, stream),
              "center, scale and grid");
  }

  void RunObbAndNoise(cudaStream_t stream) {
    CheckCuda(LaunchObbTopdownFloatDeviceDepth(
                  centered_points_, kPointCount, obb_grasps_,
                  obb_canonical_, obb_bounds_, gripper_depth_,
                  kKappa, config_.obb_grasps, stream),
              "OBB proposals");
    CheckCuda(LaunchNormalNoiseFp16(
                  seed_, sequence_offset_, variance_noise_,
                  config_.diffusion_steps * config_.diffusion_grasps * 6,
                  stream),
              "asynchronous diffusion variance noise");
  }

  void BuildPtv3Topology(cudaStream_t stream) {
    CheckCuda(LaunchSerializationAndSort(
                  topology_[0].grid_coordinates, topology_[0].capacity,
                  maximum_coordinate_, topology_[0].codes,
                  topology_[0].order, topology_[0].inverse,
                  kUsedOrderCounts[0], stream),
              "initial serialization");
    for (int index = 0; index < kStageCount; ++index) {
      TopologyStage& stage = topology_[index];
      CheckCuda(cudaEventRecord(topology_stage_ready_[index], stream),
                "record topology stage ready");
      CheckCuda(cudaStreamWaitEvent(
                    topology_stream_, topology_stage_ready_[index]),
                "topology map wait");
      CheckCuda(cudaStreamWaitEvent(
                    topology_attention_stream_, topology_stage_ready_[index]),
                "topology attention data wait");
      if (index == 0) {
        CheckCuda(LaunchNeighborMapPadded(
                      stage.grid_coordinates, stage.codes, stage.order,
                      stage.valid_count, offsets_k5_, stage.neighbor_k5,
                      stage.neighbor_k5_mask, stage.capacity, 125,
                      topology_stream_),
                  "K5 neighbor map");
      }
      CheckCuda(LaunchNeighborMapPadded(
                    stage.grid_coordinates, stage.codes, stage.order,
                    stage.valid_count, offsets_k3_, stage.neighbor_k3,
                    stage.neighbor_k3_mask, stage.capacity, 27,
                    topology_stream_),
                "K3 neighbor map");
      if (index > 0) {
        CheckCuda(cudaStreamWaitEvent(
                      topology_attention_stream_, topology_order_ready_[index]),
                  "topology order wait");
      }
      CheckCuda(LaunchPackAttentionTopology(
                    stage.order, stage.inverse, stage.valid_count,
                    stage.attention_order, stage.attention_inverse,
                    stage.valid_key_counts, stage.capacity,
                    stage.padded_capacity, stage.patch_size,
                    kUsedOrderCounts[index], topology_attention_stream_),
                "attention topology");
      if (index + 1 < kStageCount) {
        TopologyStage& next = topology_[index + 1];
        CheckCuda(LaunchSerializedPoolingPadded(
                      stage.coordinates, stage.grid_coordinates, stage.codes,
                      stage.order, stage.valid_count, stage.capacity,
                      next.capacity, 1, stage.pool_index_ptr, stage.pool_cluster,
                      pool_segment_ids_,
                      next.coordinates, next.grid_coordinates, next.codes,
                      next.order, next.inverse, next.valid_count,
                      stage.overflow, kUsedOrderCounts[index + 1], 2,
                      maximum_coordinate_,
                      index == 0 ? (1 << kSpatialCoordinateBits) - 1
                                 : INT32_MAX,
                      index + 1, stream,
                      topology_order_stream_, topology_pool_data_ready_[index],
                      topology_order_ready_[index + 1]),
                  "serialized pooling topology");
      }
    }
    CheckCuda(cudaEventRecord(topology_maps_ready_, topology_stream_),
              "record topology maps ready");
    CheckCuda(cudaEventRecord(
                  topology_attention_ready_, topology_attention_stream_),
              "record topology attention ready");
    CheckCuda(cudaStreamWaitEvent(stream, topology_maps_ready_),
              "wait for topology maps");
    CheckCuda(cudaStreamWaitEvent(stream, topology_attention_ready_),
              "wait for topology attention");
  }

  void BuildTopology(cudaStream_t stream) {
    PreparePoints(stream);
    CheckCuda(cudaEventRecord(points_ready_, stream), "record points ready");
    CheckCuda(cudaStreamWaitEvent(obb_stream_, points_ready_),
              "OBB wait for centered points");
    RunObbAndNoise(obb_stream_);
    CheckCuda(cudaEventRecord(obb_ready_, obb_stream_),
              "record OBB and noise ready");
    BuildPtv3Topology(stream);
  }

  void RunEncoder(EncoderBranch* branch, cudaStream_t stream) {
    for (int stage = 0; stage < kStageCount; ++stage) {
      branch->plans[stage]->Enqueue(stream);
      if (stage + 1 < kStageCount) {
        CheckCuda(LaunchSegmentMaxFp16(
                      branch->outputs[stage], topology_[stage].order,
                      topology_[stage].pool_index_ptr,
                      topology_[stage + 1].valid_count,
                      branch->inputs[stage + 1], kCapacities[stage + 1],
                      kOutputChannels[stage], stream),
                  "PTV3 segment max");
      }
    }
    CheckCuda(LaunchMeanFp16(branch->outputs[kStageCount - 1],
                             topology_[kStageCount - 1].valid_count,
                             branch->embedding, kEmbeddingChannels, stream),
              "PTV3 final mean");
  }

  void ProjectObservation(cudaStream_t stream) {
    CheckCublas(cublasSetStream(cublas_, stream), "cublasSetStream");
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    CheckCublas(
        cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, 2048, 1, 1024,
                     &alpha, observation_weight_, CUDA_R_16F, 1024,
                     observation_, CUDA_R_16F, 1024, &beta,
                     observation_projection_, CUDA_R_16F, 2048,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "observation projection");
  }

  void RunDiffusion(cudaStream_t stream, bool externally_generated_noise) {
    ProjectObservation(stream);
    if (!externally_generated_noise) {
      CheckCuda(LaunchNormalNoiseFp16(
                    seed_, sequence_offset_, variance_noise_,
                    config_.diffusion_steps * config_.diffusion_grasps * 6,
                    stream),
                "diffusion variance noise");
    }
    CheckCuda(LaunchDenoiseSampleEncodeFp16(
                  initial_sample_, sample_encoder_weight_,
                  sample_encoder_bias_, observation_projection_,
                  timestep_conditioning_, sample_embedding_,
                  conditioning_buffer_, config_.diffusion_grasps, 1, stream),
              "initial diffusion sample encode");
    for (int step = 0; step < config_.diffusion_steps; ++step) {
      const void* sample = step == 0 ? static_cast<void*>(initial_sample_)
                                     : static_cast<void*>(state_[(step - 1) & 1]);
      void* output = state_[step & 1];
      denoiser_->Enqueue(stream);
      if (step == 0 && externally_generated_noise) {
        CheckCuda(cudaStreamWaitEvent(stream, obb_ready_),
                  "DDPM update waits variance noise");
      }
      if (step + 1 < config_.diffusion_steps) {
        CheckCuda(LaunchDenoiseStepAndSampleEncodeFp16(
                      prediction_hidden_, sample,
                      variance_noise_ + step * config_.diffusion_grasps * 6,
                      prediction_weight_, prediction_bias_,
                      ddpm_coefficients_ + step * 5 * 6, output,
                      sample_encoder_weight_, sample_encoder_bias_,
                      observation_projection_,
                      timestep_conditioning_ + (step + 1) * 2048,
                      sample_embedding_, conditioning_buffer_,
                      config_.diffusion_grasps, 1, stream),
                  "fused DDPM update and next sample encode");
      } else {
        CheckCuda(LaunchDenoiseStepFp16(
                      prediction_hidden_, sample,
                      variance_noise_ + step * config_.diffusion_grasps * 6,
                      prediction_weight_, prediction_bias_,
                      ddpm_coefficients_ + step * 5 * 6, output,
                      config_.diffusion_grasps, stream),
                  "final DDPM update");
      }
    }
    CheckCuda(LaunchFinalizeSo3Fp16(
                  state_[(config_.diffusion_steps - 1) & 1], diffusion_grasps_,
                  diffusion_canonical_, config_.diffusion_grasps, kKappa,
                  stream),
              "SO3 finalization");
  }

  void EnqueuePipeline(cudaStream_t root) {
    BuildTopology(root);
    CheckCuda(cudaEventRecord(topology_ready_, root), "record topology ready");
    CheckCuda(cudaStreamWaitEvent(generator_stream_, topology_ready_),
              "generator waits topology");
    CheckCuda(cudaStreamWaitEvent(discriminator_stream_, topology_ready_),
              "discriminator waits topology");
    RunEncoder(&generator_, generator_stream_);
    CheckCuda(cudaEventRecord(generator_ready_, generator_stream_),
              "record generator ready");
    RunEncoder(&discriminator_, discriminator_stream_);
    CheckCuda(cudaEventRecord(discriminator_ready_, discriminator_stream_),
              "record discriminator ready");

    CheckCuda(cudaStreamWaitEvent(root, generator_ready_),
              "diffusion waits generator");
    RunDiffusion(root, true);
    CheckCuda(cudaStreamWaitEvent(root, discriminator_ready_),
              "scorer waits discriminator");
    CheckCuda(cudaStreamWaitEvent(root, obb_ready_), "scorer waits OBB");
    CheckCuda(LaunchPrepareScorerInputs(
                  diffusion_canonical_, obb_canonical_, discriminator_.embedding,
                  scorer_canonical_, scorer_object_embedding_,
                  config_.diffusion_grasps, config_.obb_grasps, root),
              "prepare scorer inputs");
    scorer_->Enqueue(root);
    CheckCuda(LaunchRankAndTranslate(
                  diffusion_grasps_, obb_grasps_, scorer_confidence_, center_,
                  sorted_grasps_, sorted_confidence_, source_indices_,
                  config_.diffusion_grasps, config_.obb_grasps, root),
              "rank and translate grasps");
  }

  void WarmAndCapture() {
    CheckCuda(cudaMemsetAsync(input_points_, 0,
                              kPointCount * 3 * sizeof(float),
                              root_stream_),
              "clear warmup input");
    const std::int64_t point_count = kPointCount;
    CheckCuda(cudaMemcpyAsync(input_point_count_, &point_count,
                              sizeof(point_count), cudaMemcpyHostToDevice,
                              root_stream_),
              "initialize input point count");
    std::uint64_t zero = 0;
    CheckCuda(cudaMemcpyAsync(seed_, &zero, sizeof(zero), cudaMemcpyHostToDevice,
                              root_stream_),
              "initialize seed");
    CheckCuda(cudaMemcpyAsync(sequence_offset_, &zero, sizeof(zero),
                              cudaMemcpyHostToDevice, root_stream_),
              "initialize sequence offset");
    for (int iteration = 0; iteration < 3; ++iteration) {
      EnqueuePipeline(root_stream_);
    }
    CheckCuda(cudaStreamSynchronize(root_stream_), "pipeline warmup");
    CheckCuda(cudaStreamBeginCapture(root_stream_, cudaStreamCaptureModeGlobal),
              "cudaStreamBeginCapture");
    EnqueuePipeline(root_stream_);
    CheckCuda(cudaStreamEndCapture(root_stream_, &graph_),
              "cudaStreamEndCapture");
    CheckCuda(cudaGraphInstantiate(&graph_exec_, graph_, 0),
              "cudaGraphInstantiate");
    CheckCuda(cudaGraphLaunch(graph_exec_, root_stream_),
              "captured pipeline warmup");
    CheckCuda(cudaStreamSynchronize(root_stream_),
              "captured pipeline synchronize");
  }

  RuntimeConfig config_;
  detail::ExecutionDomain domain_;
  DeviceArena arena_;
  cudaStream_t root_stream_{};
  cudaStream_t generator_stream_{};
  cudaStream_t discriminator_stream_{};
  cudaStream_t obb_stream_{};
  cudaStream_t topology_stream_{};
  cudaStream_t topology_order_stream_{};
  cudaStream_t topology_attention_stream_{};
  cudaEvent_t points_ready_{};
  cudaEvent_t topology_ready_{};
  cudaEvent_t generator_ready_{};
  cudaEvent_t discriminator_ready_{};
  cudaEvent_t obb_ready_{};
  std::array<cudaEvent_t, kStageCount> topology_stage_ready_{};
  std::array<cudaEvent_t, kStageCount - 1> topology_pool_data_ready_{};
  std::array<cudaEvent_t, kStageCount> topology_order_ready_{};
  cudaEvent_t topology_maps_ready_{};
  cudaEvent_t topology_attention_ready_{};
  cudaGraph_t graph_{};
  cudaGraphExec_t graph_exec_{};
  cublasHandle_t cublas_{};

  float* input_points_{};
  std::int64_t* input_point_count_{};
  bool* filter_keep_{};
  std::int64_t* selected_indices_{};
  std::int32_t* selected_count_{};
  float* filtered_points_{};
  float* centered_points_{};
  float* center_{};
  __half* input_features_{};
  std::int32_t* maximum_coordinate_{};
  std::int64_t* pool_segment_ids_{};
  std::int64_t* topology_valid_counts_{};
  bool* topology_overflow_{};
  std::array<TopologyStage, kStageCount> topology_{};
  std::int64_t* offsets_k3_{};
  std::int64_t* offsets_k5_{};
  EncoderBranch generator_;
  EncoderBranch discriminator_;

  __half* observation_{};
  __half* observation_projection_{};
  __half* conditioning_buffer_{};
  __half* sample_embedding_{};
  __half* prediction_hidden_{};
  __half* initial_sample_{};
  __half* variance_noise_{};
  std::array<__half*, 2> state_{};
  float* diffusion_grasps_{};
  __half* diffusion_canonical_{};
  float* obb_grasps_{};
  float* obb_canonical_{};
  float* obb_bounds_{};
  float* scorer_canonical_{};
  float* scorer_object_embedding_{};
  float* scorer_gripper_embedding_{};
  float* gripper_depth_{};
  float* scorer_logits_{};
  float* scorer_confidence_{};
  float* sorted_grasps_{};
  float* sorted_confidence_{};
  std::int32_t* source_indices_{};
  std::uint64_t* seed_{};
  std::uint64_t* sequence_offset_{};
  std::array<float, kPointCount * 3> host_resampled_points_{};
  __half* observation_weight_{};
  __half* timestep_conditioning_{};
  __half* sample_encoder_weight_{};
  __half* sample_encoder_bias_{};
  __half* prediction_weight_{};
  __half* prediction_bias_{};
  __half* ddpm_coefficients_{};

  std::unique_ptr<detail::TensorRtPlan> denoiser_;
  std::unique_ptr<detail::TensorRtPlan> scorer_;
  std::size_t tensorrt_device_memory_bytes_ = 0;
};

Runtime::Runtime(RuntimeConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}
Runtime::~Runtime() = default;
Runtime::Runtime(Runtime&&) noexcept = default;
Runtime& Runtime::operator=(Runtime&&) noexcept = default;

void Runtime::SetGripperEmbeddings(const float* generator_embedding_512,
                                   const float* discriminator_embedding_512) {
  impl_->SetGripperEmbeddings(generator_embedding_512,
                              discriminator_embedding_512);
}

void Runtime::SetGripperDeviceAsync(
    const void* generator_embedding_fp16_512,
    const float* discriminator_embedding_fp32_512, float gripper_depth,
    cudaStream_t stream) {
  impl_->SetGripperDeviceAsync(generator_embedding_fp16_512,
                               discriminator_embedding_fp32_512,
                               gripper_depth, stream);
}

float* Runtime::device_input_points() noexcept { return impl_->input_points(); }

void Runtime::InferAsync(std::int64_t point_count, std::uint64_t seed,
                         std::uint64_t sequence_offset, cudaStream_t stream) {
  impl_->InferAsync(point_count, seed, sequence_offset, stream);
}

void Runtime::InferDeviceAsync(const float* points, std::int64_t point_count,
                               std::uint64_t seed,
                               std::uint64_t sequence_offset,
                               cudaStream_t stream) {
  impl_->InferDeviceAsync(points, point_count, seed, sequence_offset, stream);
}

void Runtime::InferHostAsync(const float* points, std::int64_t point_count,
                             std::uint64_t seed,
                             std::uint64_t sequence_offset,
                             cudaStream_t stream) {
  impl_->InferHostAsync(points, point_count, seed, sequence_offset, stream);
}

DeviceOutputs Runtime::outputs() const noexcept { return impl_->outputs(); }
RuntimeMemory Runtime::memory() const noexcept { return impl_->memory(); }
int Runtime::device() const noexcept { return impl_->device(); }

RuntimeProfile Runtime::Profile(int warmup_iterations,
                                int measured_iterations) {
  return impl_->Profile(warmup_iterations, measured_iterations);
}

}  // namespace graspgenx
