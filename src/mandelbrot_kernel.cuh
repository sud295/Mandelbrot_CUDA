#pragma once
#include <cuda_runtime.h>
#include <cstdint>

__global__ void mandelbrot_kernel(
    uint8_t* rgb, int width, int height,
    float x_min, float x_max, float y_min, float y_max,
    int max_iter);

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
