cmake_minimum_required(VERSION 3.24)

project(graspgenx_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation prefix")
set(_TENSORRT_HINTS
    ${TENSORRT_ROOT}
    $ENV{TENSORRT_ROOT}
    /usr
    /usr/local)
find_path(
  TENSORRT_INCLUDE_DIR
  NvInfer.h
  HINTS ${_TENSORRT_HINTS}
  PATH_SUFFIXES include include/x86_64-linux-gnu x86_64-linux-gnu)
find_library(
  TENSORRT_RUNTIME_LIBRARY
  nvinfer
  HINTS ${_TENSORRT_HINTS}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu x86_64-linux-gnu)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_RUNTIME_LIBRARY)
  message(FATAL_ERROR "TensorRT headers and libnvinfer are required")
endif()
get_filename_component(
  TENSORRT_LIBRARY_DIRECTORY "${TENSORRT_RUNTIME_LIBRARY}" DIRECTORY)

add_library(
  graspgenx_runtime SHARED
  src/ptv3_cuda_kernel.cu
  src/c_api.cpp
  src/runtime.cpp
  src/tensorrt_plan.cpp)
target_include_directories(
  graspgenx_runtime
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(
  graspgenx_runtime
  PRIVATE CUDA::cublas CUDA::cudart CUDA::cuda_driver
          ${TENSORRT_RUNTIME_LIBRARY})
target_compile_options(
  graspgenx_runtime
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-lineinfo>)
set_target_properties(
  graspgenx_runtime
  PROPERTIES OUTPUT_NAME graspgenx_runtime
             BUILD_RPATH "${TENSORRT_LIBRARY_DIRECTORY}"
             CUDA_ARCHITECTURES native
             CUDA_STANDARD 17
             POSITION_INDEPENDENT_CODE ON)

add_executable(graspgenx_benchmark_runtime examples/benchmark_runtime.cpp)
target_link_libraries(
  graspgenx_benchmark_runtime PRIVATE graspgenx_runtime CUDA::cudart)
set_target_properties(
  graspgenx_benchmark_runtime
  PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")

install(
  TARGETS graspgenx_runtime
  LIBRARY DESTINATION lib)
install(
  FILES include/graspgenx/c_api.h include/graspgenx/runtime.h
  DESTINATION include/graspgenx)
