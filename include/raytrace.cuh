#ifndef RAYTRACE_CUH
#define RAYTRACE_CUH

#include <cuda_runtime.h>  // Include the necessary CUDA runtime header file
#include "math_materials.cuh"
#include "mesh.cuh"

__global__ void generateRayKernel(const M4f *Inv_Extrinsic,
                                  const M3f *Inv_Intrinsic,
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
                         bool if_depthmap = false);

#endif  // RAYTRACE_CUH