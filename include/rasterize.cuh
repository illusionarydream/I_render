#ifndef RASTERIZE_CUH
#define RASTERIZE_CUH

#include <cuda_runtime.h>
#include "math_materials.cuh"
#include "mesh.cuh"

// Z_buffer update data structure
struct ZBuffer_element {
    float depth;
    float3 normal;
    float3 position;
    int lock;
};

__device__ float atomicMinFloat(float *address, float value);

__global__ void initDepthBufferandImage(ZBuffer_element *depth_buffer,
                                        V3f *image,
                                        int width,
                                        int height,
                                        int super_sampling_ratio);

__global__ void transformTrianglesAndLights(
    Triangle *triangles,
    Light *lights,
    int width,
    int height,
    int num_triangles,
    int num_lights,
    int sample_square,
    M4f *Extrinsics,
    M4f *Inv_Extrinsics,
    M3f *Intrinsics,
    ZBuffer_element *depth_buffer);

__global__ void shading(const Triangle *triangles,
                        const Light *lights,
                        int width,
                        int height,
                        int num_triangles,
                        int num_lights,
                        const ZBuffer_element *buffer_elements,
                        V3f *image,
                        // other parameters
                        float ka,
                        float kd,
                        float ks,
                        float kn,
                        bool if_depthmap,
                        bool if_normalmap,
                        int super_sampling_ratio);

#endif  // RASTERIZE_CUH