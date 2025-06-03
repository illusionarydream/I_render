#ifndef DENOISE_CUH
#define DENOISE_CUH

#include <cuda_runtime.h>  // Include the necessary CUDA runtime header file
#include "math_materials.cuh"

__global__ void denoiseKernel(V3f *image,
                              const V3f *input_image,
                              const int width,
                              const int height,
                              const int kernel_size,
                              const int denoise_type,
                              const float sigma_spatial);

#endif  // DENOISE_CUH