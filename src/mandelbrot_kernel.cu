#include "mandelbrot_kernel.cuh"
#include <cmath>

__device__ inline void pixel_to_complex(
    int px, int py, int width, int height,
    float x_min, float x_max, float y_min, float y_max,
    float &cx, float &cy)
{
    cx = x_min + (x_max - x_min)*(float(px)/(width - 1));
    cy = y_min + (y_max - y_min)*(float(py)/(height - 1));
}

__device__ inline float smooth_iter(float zx, float zy, int iters, int max_iter)
{
    float r2 = zx*zx + zy*zy;
    if (iters >= max_iter){
        return (float)iters;
    }
    float nu = iters + 1.0f - log2f(logf(fmaxf(sqrtf(r2), 1e-20f)));
    return nu;
}

__global__ void mandelbrot_kernel(
    uint8_t* rgb, int width, int height,
    float x_min, float x_max, float y_min, float y_max,
    int max_iter)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height){
        return;
    }

    float cx, cy;
    pixel_to_complex(px, py, width, height, x_min, x_max, y_min, y_max, cx, cy);

    float zx = 0, zy = 0;
    int iter = 0;

    while (iter < max_iter){
        float zx2 = zx*zx - zy*zy + cx;
        float zy2 = 2.0f*zx*zy + cy;
        zx = zx2;
        zy = zy2;
        if (zx*zx + zy*zy > 4.0f){
            break;
        }
        iter++;
    }

    float t = smooth_iter(zx, zy, iter, max_iter) / max_iter;
    float r, g, b;
    if (iter >= max_iter){
        r = g = b = 0;
    }
    else{
        float a = 3.0f * t;
        if (a < 1.0f){
            r = a; 
            g = 0; 
            b = 0; 
        }
        else if (a < 2.0f){ 
            r = 1; 
            g = a - 1; 
            b = 0;
        }
        else{ 
            r = 1; 
            g = 1; 
            b = a - 2;
        }
    }

    int idx = (py * width + px) * 3;
    rgb[idx + 0] = (uint8_t)(255 * r);
    rgb[idx + 1] = (uint8_t)(255 * g);
    rgb[idx + 2] = (uint8_t)(255 * b);
}
