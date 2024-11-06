#include "raytrace.cuh"

__global__ void generateRayKernel(const M4f *Inv_Extrinsic,
                                  const M3f *Inv_Intrinsic,
                                  const int width,
                                  const int height,
                                  Ray *rays) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // generate ray from camera to the pixel (x, y)
        V3f p_film((x + 0.5f) / width, (y + 0.5f) / height, 1.0f);
        V3f p_camera3 = (*Inv_Intrinsic) * p_film;
        auto p_camera = V4f(p_camera3[0], p_camera3[1], p_camera3[2], 1.0f);
        V4f p_world = (*Inv_Extrinsic) * p_camera;

        // set the origin and direction of the ray
        rays[y * width + x].orig = Inv_Extrinsic->col(3);
        rays[y * width + x].dir = normalize(p_world - rays[y * width + x].orig);
    }
}

__global__ void raytrace(const Mesh *mesh,
                         const int mesh_num,
                         const int width,
                         const int height,
                         const Ray *rays,
                         V3f *image,
                         // other parameters
                         bool if_depthmap = false) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // * output the depth map
    if (if_depthmap) {
        if (x < width && y < height) {
            // ray tracing
            // for each ray, find the intersection with the mesh
            float t = MAX;
            int idx = -1;

            // find the intersection with the mesh
            bool hit = (*mesh).hitting(rays[y * width + x], t, idx);

            // if hit, set the color of the pixel
            if (hit) {
                auto col = sigmoid((t - 2.0f) * 20);
                image[y * width + x] = V3f(col, col, col);
            } else {
                image[y * width + x] = V3f(0.0f, 0.0f, 0.0f);
            }
        }
    }
}
