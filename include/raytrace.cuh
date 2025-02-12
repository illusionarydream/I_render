#ifndef RAYTRACE_CUH
#define RAYTRACE_CUH

#include <cuda_runtime.h>  // Include the necessary CUDA runtime header file
#include "math_materials.cuh"
#include "mesh.cuh"

__global__ void initCurandStates_and_image(curandState *state,
                                           V3f *image,
                                           int seed,
                                           int width,
                                           int height,
                                           int samples_per_kernel,
                                           bool if_mixed);

__global__ void generateRayKernel(const M4f *Inv_Extrinsic,
                                  const M3f *Inv_Intrinsic,
                                  const V4f *camera_pos,
                                  const int width,
                                  const int height,
                                  Ray *rays);

__global__ void raytrace(const Mesh *mesh,
                         const int mesh_num,
                         const int width,
                         const int height,
                         const Ray *rays,
                         V3f *image,
                         // other parameters
                         bool if_depthmap = false,
                         bool if_normalmap = false,
                         bool if_pathtracing = false,
                         bool if_more_kernel = false,
                         // for path tracing
                         float russian_roulette = 0.90f,
                         int samples_per_pixel = 1,
                         int samples_per_kernel = 1,
                         // random seed support
                         curandState *state = nullptr);

#endif  // RAYTRACE_CUH