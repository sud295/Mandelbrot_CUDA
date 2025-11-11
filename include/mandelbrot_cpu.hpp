#pragma once
#include <cstdint>

void mandelbrot_cpu(
    uint8_t* rgb, int width, int height,
    float x_min, float x_max, float y_min, float y_max,
    int max_iter);
