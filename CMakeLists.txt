cmake_minimum_required(VERSION 3.20)
project(cuda_mandelbrot LANGUAGES CXX CUDA)

option(BUILD_CPU_ONLY "Build without CUDA and use CPU fallback" OFF)

set(CMAKE_CXX_STANDARD 17)

if (BUILD_CPU_ONLY)
  message(STATUS "Building CPU-only version with OpenMP")
  find_package(OpenMP REQUIRED)
  add_executable(mandelbrot
      src/utils.cpp
      src/mandelbrot_cpu.cpp
      src/main.cu
  )
  set_source_files_properties(src/main.cu PROPERTIES LANGUAGE CXX)
  target_include_directories(mandelbrot PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_compile_definitions(mandelbrot PRIVATE BUILD_CPU_ONLY=1)
  target_link_libraries(mandelbrot PRIVATE OpenMP::OpenMP_CXX)
else()
  find_package(CUDA REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  if (NOT DEFINED GPU_ARCH)
    set(GPU_ARCH "sm_70")
  endif()
  add_executable(mandelbrot
      src/main.cu
      src/mandelbrot_kernel.cu
      src/utils.cpp
      src/mandelbrot_cpu.cpp
  )
  target_include_directories(mandelbrot PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_compile_options(mandelbrot PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--gpu-architecture=${GPU_ARCH} --use_fast_math -O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
  )
  set_target_properties(mandelbrot PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_ARCHITECTURES "70;75;86")
endif()
