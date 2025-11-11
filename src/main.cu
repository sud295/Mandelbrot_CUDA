#include "mandelbrot_kernel.cuh"
#include "mandelbrot_cpu.hpp"
#include "utils.hpp"
#include <vector>
#include <iostream>

int main() {
    const int width = 1920, height = 1080, max_iter = 1000;
    const float x_center = -0.5f, y_center = 0.0f, view_width = 3.5f;
    const float aspect = (float)height / width;
    float x_min = x_center - 0.5f * view_width;
    float x_max = x_center + 0.5f * view_width;
    float y_min = y_center - 0.5f * view_width * aspect;
    float y_max = y_center + 0.5f * view_width * aspect;

    size_t bytes = (size_t)width * height * 3;
    std::vector<uint8_t> host_rgb(bytes);

#ifdef BUILD_CPU_ONLY
    mandelbrot_cpu(host_rgb.data(), width, height, x_min, x_max, y_min, y_max, max_iter);

#else
    uint8_t* dev_rgb = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_rgb, bytes));
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    mandelbrot_kernel<<<grid, block>>>(dev_rgb, width, height, x_min, x_max, y_min, y_max, max_iter);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_rgb.data(), dev_rgb, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dev_rgb));

#endif
    write_ppm("mandelbrot.ppm", host_rgb, width, height);
    std::cout << "Wrote mandelbrot.ppm (" << width << "x" << height << ")\n";
}
