cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_ARCHITECTURES 89)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
project(kernline LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

find_package(CUDAToolkit REQUIRED)

# libkernels.so: one extern "C" launcher per kernel+variant (docs/DESIGN.md, D6).
file(GLOB KERNEL_SRCS CONFIGURE_DEPENDS kernels/*/*.cu)
add_library(kernels SHARED src/api.cu ${KERNEL_SRCS})
target_include_directories(kernels PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(kernels PRIVATE CUDA::cublas)

add_executable(bench bench/bench.cu)
target_link_libraries(bench PRIVATE kernels CUDA::nvml CUDA::nvtx3)

# Transformer block runner (D4): per-step and whole-block timings.
add_executable(block block/block.cu)
target_link_libraries(block PRIVATE kernels CUDA::nvml CUDA::nvtx3)

# Roofline ceilings + device limits for specs/<gpu>.json (D3). tools/ceilings/run.sh drives it.
add_executable(ceilings tools/ceilings/ceilings.cu)
target_include_directories(ceilings PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(ceilings PRIVATE CUDA::nvml)

# Cache simulator (docs/SIM.md): CPU only. simcore holds everything but the two mains.
file(GLOB SIM_SRCS CONFIGURE_DEPENDS sim/*.cpp sim/generators/*.cpp)
list(FILTER SIM_SRCS EXCLUDE REGEX "/sim/(main|test)\\.cpp$")
add_library(simcore STATIC ${SIM_SRCS})
target_include_directories(simcore PUBLIC ${CMAKE_SOURCE_DIR})
add_executable(cachesim_test sim/test.cpp)
target_link_libraries(cachesim_test PRIVATE simcore)
