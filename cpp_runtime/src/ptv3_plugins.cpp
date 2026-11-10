// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <NvInfer.h>
#include <new>
#include "cuda_kernels.h"

namespace graspgenx {
namespace {
using namespace nvinfer1;

// Stateless reductions share the native runtime kernels and require no scratch
// allocation or host access to the device-resident valid counts.
template <bool Mean>
class Reduction final : public IPluginV3, public IPluginV3OneCore,
                        public IPluginV3OneBuild, public IPluginV3OneRuntime {
  PluginFieldCollection fields_{};
 public:
  static char const* name() {
    return Mean ? "GraspGenXValidMean" : "GraspGenXSegmentMax";
  }
  IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
    switch (type) {
      case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
      case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
      case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
    }
    return nullptr;
  }
  IPluginV3* clone() noexcept override { return new (std::nothrow) Reduction(); }
  char const* getPluginName() const noexcept override { return name(); }
  char const* getPluginVersion() const noexcept override { return "1"; }
  char const* getPluginNamespace() const noexcept override { return ""; }
  int32_t getNbOutputs() const noexcept override { return 1; }
  int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t ni,
                          DynamicPluginTensorDesc const*, int32_t no) noexcept override {
    return ni == (Mean ? 2 : 4) && no == 1 ? 0 : -1;
  }
  int32_t getOutputDataTypes(DataType* out, int32_t no, DataType const* in,
                             int32_t ni) const noexcept override {
    if (ni != (Mean ? 2 : 4) || no != 1 || in[0] != DataType::kHALF) return -1;
    out[0] = DataType::kHALF;
    return 0;
  }
  int32_t getOutputShapes(DimsExprs const* in, int32_t ni, DimsExprs const*,
                          int32_t ns, DimsExprs* out, int32_t no,
                          IExprBuilder& eb) noexcept override {
    if (ni != (Mean ? 2 : 4) || no != 1 || ns || in[0].nbDims != 2) return -1;
    if constexpr (Mean) {
      out[0].nbDims = 1;
      out[0].d[0] = in[0].d[1];
    } else {
      if (in[2].nbDims != 1) return -1;
      out[0].nbDims = 2;
      out[0].d[0] = eb.operation(DimensionOperation::kSUB, *in[2].d[0], *eb.constant(1));
      out[0].d[1] = in[0].d[1];
    }
    return 0;
  }
  bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* io,
                                 int32_t ni, int32_t no) noexcept override {
    if (ni != (Mean ? 2 : 4) || no != 1 || pos < 0 || pos > ni) return false;
    return io[pos].desc.format == TensorFormat::kLINEAR &&
      io[pos].desc.type == ((pos == 0 || pos == ni) ? DataType::kHALF : DataType::kINT64);
  }
  int32_t onShapeChange(PluginTensorDesc const* in, int32_t ni,
                        PluginTensorDesc const* out, int32_t no) noexcept override {
    if (ni != (Mean ? 2 : 4) || no != 1 || in[0].dims.nbDims != 2 ||
        in[0].dims.d[0] < 1 || in[0].dims.d[0] > 3500 || in[0].dims.d[1] < 1 || in[0].dims.d[1] > 512)
      return -1;
    for (int i = 0; i <= ni; ++i) {
      auto const& d = i == ni ? out[0] : in[i];
      if (d.format != TensorFormat::kLINEAR ||
          d.type != ((i == 0 || i == ni) ? DataType::kHALF : DataType::kINT64)) return -1;
    }
    auto const& count = in[Mean ? 1 : 3].dims;
    if (count.nbDims != 1 || count.d[0] != 1) return -1;
    if constexpr (Mean) {
      return out[0].dims.nbDims == 1 && out[0].dims.d[0] == in[0].dims.d[1] ? 0 : -1;
    } else {
      auto const& order = in[1].dims;
      auto const& ptr = in[2].dims;
      return order.nbDims == 2 && order.d[0] == 4 && order.d[1] == in[0].dims.d[0] &&
        ptr.nbDims == 1 && ptr.d[0] >= 2 && ptr.d[0] <= 3501 &&
        out[0].dims.nbDims == 2 && out[0].dims.d[0] == ptr.d[0] - 1 &&
        out[0].dims.d[1] == in[0].dims.d[1] ? 0 : -1;
    }
  }
  int32_t enqueue(PluginTensorDesc const* desc, PluginTensorDesc const* output,
                  void const* const* in, void* const* out, void*,
                  cudaStream_t stream) noexcept override {
    cudaError_t status;
    if constexpr (Mean) {
      status = LaunchMeanFp16(in[0], static_cast<int64_t const*>(in[1]),
                             out[0], desc[0].dims.d[1], desc[0].dims.d[0], stream);
    } else {
      status = LaunchSegmentMaxFp16(in[0], static_cast<int64_t const*>(in[1]),
          static_cast<int64_t const*>(in[2]), static_cast<int64_t const*>(in[3]),
          out[0], output[0].dims.d[0], desc[0].dims.d[1], desc[0].dims.d[0], stream);
    }
    return status == cudaSuccess ? 0 : -1;
  }
  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
  PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &fields_; }
  char const* getTimingCacheID() noexcept override { return "fp16-reduction-v1"; }
};

template <bool Mean> class Creator : public IPluginCreatorV3One {
  PluginFieldCollection fields_{};
 public:
  IPluginV3* createPlugin(char const*, PluginFieldCollection const* fields,
                          TensorRTPhase) noexcept override {
    if (fields && fields->nbFields != 0) return nullptr;
    return new (std::nothrow) Reduction<Mean>();
  }
  PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
  char const* getPluginName() const noexcept override { return Reduction<Mean>::name(); }
  char const* getPluginVersion() const noexcept override { return "1"; }
  char const* getPluginNamespace() const noexcept override { return ""; }
};
class SegmentMaxCreator final : public Creator<false> {};
class ValidMeanCreator final : public Creator<true> {};
REGISTER_TENSORRT_PLUGIN(SegmentMaxCreator);
REGISTER_TENSORRT_PLUGIN(ValidMeanCreator);
}  // namespace
}  // namespace graspgenx
