#include "mandelbrot_cpu.hpp"
#include <algorithm>
#include <cmath>

static inline void pixel_to_complex(
    int px, int py, int width, int height,
    float x_min, float x_max, float y_min, float y_max,
    float &cx, float &cy)
{
    cx = x_min + (x_max - x_min) * (float(px) / (width - 1));
    cy = y_min + (y_max - y_min) * (float(py) / (height - 1));
}

static inline float smooth_iter(float zx, float zy, int iters, int max_iter)
{
    float r2 = zx*zx + zy*zy;
    if (iters >= max_iter) return (float)iters;
    float nu = iters + 1.0f - std::log2(std::log(std::max(std::sqrt(r2), 1e-20f)));
    return nu;
}

void mandelbrot_cpu(
    uint8_t* rgb, int width, int height,
    float x_min, float x_max, float y_min, float y_max,
    int max_iter)
{
    // Parallelize rows
    #pragma omp parallel for schedule(static)
    for (int py = 0; py < height; ++py) {
        for (int px = 0; px < width; ++px) {
            float cx, cy;
            pixel_to_complex(px, py, width, height, x_min, x_max, y_min, y_max, cx, cy);

            float zx = 0.f, zy = 0.f;
            int iter = 0;
            while (iter < max_iter) {
                float zx2 = zx*zx - zy*zy + cx;
                float zy2 = 2.0f*zx*zy + cy;
                zx = zx2; zy = zy2;
                if (zx*zx + zy*zy > 4.0f) break;
                ++iter;
            }

            float t = smooth_iter(zx, zy, iter, max_iter) / max_iter;
            float r, g, b;
            if (iter >= max_iter) r = g = b = 0.f;
            else {
                float a = 3.0f * t;
                if (a < 1.0f) { r = a; g = 0.f; b = 0.f; }
                else if (a < 2.0f) { r = 1.f; g = a - 1.f; b = 0.f; }
                else { r = 1.f; g = 1.f; b = a - 2.f; }
            }

            int idx = (py * width + px) * 3;
            rgb[idx + 0] = (uint8_t)std::lround(255.0f * std::clamp(r, 0.f, 1.f));
            rgb[idx + 1] = (uint8_t)std::lround(255.0f * std::clamp(g, 0.f, 1.f));
            rgb[idx + 2] = (uint8_t)std::lround(255.0f * std::clamp(b, 0.f, 1.f));
        }
    }
}
