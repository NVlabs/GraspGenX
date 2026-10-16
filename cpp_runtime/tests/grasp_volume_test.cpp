// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// GPU filter/ranking fixtures independent of TensorRT engines.
#include "grasp_volume.h"
#include "cuda_kernels.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void cuda_check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template <class T> struct Buffer {
  T* p{};
  explicit Buffer(std::size_t n) { cuda_check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer&) = delete;
  void put(const std::vector<T>& v) { cuda_check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
  std::vector<T> get(std::size_t n) {
    std::vector<T> v(n); cuda_check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
  }
};

void fixture(int n, int minimum) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> center{10, 20, 30};
  const std::vector<float> bounds{0, 0, 0, .5F, 1, 1};
  // Only five supplied points count, even though the remaining allocation
  // deliberately contains 3495 additional points inside the volume.
  std::vector<float> points(3500 * 3);
  for (int i = 0; i < 3500; ++i)
    std::copy(center.begin(), center.end(), points.begin() + i * 3);
  const float first[]{10.25F,20.5F,30.5F, 10,20,30, 10.5F,21,31,
                      nan,20,30, 19,29,39};
  std::copy(std::begin(first), std::end(first), points.begin());
  std::vector<float> poses(n * 16), scores(n);
  std::vector<std::uint8_t> expected(n);
  const std::array<float, 10> pattern{.99F,0,.95F,.6F,.6F,.9F,.8F,nan,1.2F,0};
  for (int i = 0; i < n; ++i) {
    float* m = poses.data() + i * 16;
    m[0] = m[5] = m[10] = m[15] = 1;
    scores[i] = pattern[i % 10];
    switch (i % 10) {
      case 0: m[3] = 10; break;                         // empty volume
      case 2: m[0]=m[5]=0; m[1]=-1; m[4]=1; break;    // proper 90 degree rotation
      case 4: m[11]=.5F; break;                        // translated capture volume
      case 5: m[0]=-1; break;                          // reflection is not a pose
      case 6: m[7]=nan; break;                         // nonfinite translation
    }
    // Inclusive bounds give identity 3 points (including two boundary points),
    // the rotated pose 1 point, and the translated pose 2 points.
    const int occupancy = (i%10==1 || i%10==3 || i%10==9 || i%10==7 || i%10==8) ? 3
                         : i%10==2 ? 1 : i%10==4 ? 2 : 0;
    expected[i] = std::isfinite(scores[i]) && scores[i] >= 0 && scores[i] <= 1 &&
                  (minimum == 0 || occupancy >= minimum);
  }
  const int diffusion = std::max(1, n * 2 / 3), obb = n - diffusion;
  Buffer<float> dposes(n*16), dpoints(points.size()), dcenter(3), dbounds(6), dscores(n);
  Buffer<std::int64_t> dcount(1);
  Buffer<int> dminimum(1);
  Buffer<float> outposes(n*16), outscores(n);
  Buffer<std::int32_t> indices(n);
  Buffer<std::uint8_t> valid(n+2);
  dposes.put(poses); dpoints.put(points); dcenter.put(center); dbounds.put(bounds);
  dscores.put(scores); dcount.put({5}); dminimum.put({minimum});
  valid.put(std::vector<std::uint8_t>(n+2, 0x7B));
  cuda_check(graspgenx::LaunchFilterGraspVolume(dposes.p, dposes.p+diffusion*16,
      dcenter.p,dpoints.p,dcount.p,dbounds.p,dminimum.p,dscores.p,diffusion,obb,nullptr));
  cuda_check(graspgenx::LaunchRankAndTranslate(dposes.p,dposes.p+diffusion*16,
      dscores.p,dcenter.p,outposes.p,outscores.p,indices.p,diffusion,obb,nullptr));
  cuda_check(graspgenx::LaunchGraspValidity(outscores.p,valid.p+1,n,nullptr));
  auto result=outscores.get(n), result_poses=outposes.get(n*16);
  auto source=indices.get(n); auto mask=valid.get(n+2);
  check(mask.front()==0x7B && mask.back()==0x7B,"validity buffer overrun");
  std::vector<int> order(n); std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a,int b) {
    return (expected[a] ? scores[a] : -1.F) > (expected[b] ? scores[b] : -1.F);
  });
  for (int i=0;i<n;++i) {
    check(source[i]==order[i],"score/source ranking or tie order mismatch");
    check(mask[i+1]==expected[source[i]],"validity/source mismatch");
    check(result[i]==(expected[source[i]] ? scores[source[i]] : 0.F),"confidence mismatch");
    if (i) {
      check(mask[i] >= mask[i+1],"valid grasps must form a prefix");
      check(result[i-1]>=result[i],"confidence order mismatch");
    }
    for (int k=0;k<16;++k) {
      float value=poses[source[i]*16+k];
      if(k==3)value+=center[0];
      if(k==7)value+=center[1];
      if(k==11)value+=center[2];
      check((std::isnan(value) && std::isnan(result_poses[i*16+k])) || value==result_poses[i*16+k],
            "pose/source/center alignment mismatch");
    }
  }
}

int main() {
  int count=0;
  if(cudaGetDeviceCount(&count)!=cudaSuccess || !count) return 77;
  try {
    for(int n : {150,4096}) for(int minimum : {0,1,2,4,7}) fixture(n,minimum);
    std::cout<<"10 GPU volume/ranking fixtures passed\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
