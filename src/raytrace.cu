#include "raytrace.cuh"

__global__ void initCurandStates_and_image(curandState *state,
                                           V3f *image,
                                           int seed,
                                           int width,
                                           int height,
                                           int samples_per_kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < samples_per_kernel) {
        int random_idx = y * width * samples_per_kernel + x * samples_per_kernel + z;
        curand_init(seed, random_idx, 0, &state[random_idx]);
        if (z == 0) {
            image[y * width + x] = V3f(0.0f, 0.0f, 0.0f);
        }
    }
}

__global__ void generateRayKernel(const M4f *Inv_Extrinsic,
                                  const M3f *Inv_Intrinsic,
                                  const V4f *camera_pos,
                                  const int width,
                                  const int height,
                                  Ray *rays) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // generate ray from camera to the pixel (x, y)
        V3f p_film((x + 0.5f) / width, (y + 0.5f) / height, 1.0f);
        V3f p_camera3 = (*Inv_Intrinsic) * p_film;
        auto p_camera = V4f(p_camera3[0], p_camera3[1], -1.0f, 1.0f);
        V4f p_world = (*Inv_Extrinsic) * p_camera;

        // set the origin and direction of the ray
        rays[y * width + x].orig = *camera_pos;
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
                         bool if_depthmap,
                         bool if_normalmap,
                         bool if_pathtracing,
                         bool if_more_kernel,
                         // for path tracing
                         float russian_roulette,
                         int samples_per_pixel,
                         int samples_per_kernel,
                         // random seed support
                         curandState *state) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // * output the depth map
    if (if_depthmap) {
        if (x < width && y < height && z < samples_per_kernel) {
            // ray tracing
            // for each ray, find the intersection with the mesh
            float t = MAX;
            int idx = -1;

            // find the intersection with the mesh
            bool hit = (*mesh).hitting(rays[y * width + x], t, idx);

            // if hit, set the color of the pixel
            if (hit) {
                auto col = sigmoid(t);
                image[y * width + x] = V3f(col, col, col);
            } else {
                image[y * width + x] = V3f(0.0f, 0.0f, 0.0f);
            }
        }
    }

    // * output the normal map
    if (if_normalmap) {
        if (x < width && y < height && z < samples_per_kernel) {
            // ray tracing
            // for each ray, find the intersection with the mesh
            float t = MAX;
            int idx = -1;

            // find the intersection with the mesh
            bool hit = (*mesh).hitting(rays[y * width + x], t, idx);

            // if hit, set the color of the pixel
            if (hit) {
                V4f collsion = rays[y * width + x](t);
                V4f normal = toV4f((*mesh).get_triangle(idx).interpolate_normal(collsion.toV3f()));
                normal = normalize(normal);
                normal = (normal + V4f(1.0f, 1.0f, 1.0f, 1.0f)) / 2.0f;

                image[y * width + x] = V3f(normal[0], normal[1], normal[2]);
            } else {
                image[y * width + x] = V3f(0.0f, 0.0f, 0.0f);
            }
        }
    }

    // * output the path tracing
    if (if_pathtracing) {
        if (x < width && y < height && z < samples_per_kernel) {
            // pixel index
            int random_idx = y * width * samples_per_kernel + x * samples_per_kernel + z;

            // set the color of the pixel
            auto accum_col = V4f(0.0f, 0.0f, 0.0f, 1.0f);

            // more kernel
            int samples_times = if_more_kernel ? samples_per_pixel / samples_per_kernel : samples_per_pixel;

            // iterate over the samples
            for (int sample_idx = 0; sample_idx < samples_times; sample_idx++) {
                // set the color and the ray
                auto temp_col = V4f(1.0f, 1.0f, 1.0f, 1.0f);
                auto temp_ray = rays[y * width + x];

                // give the ray a little offset
                temp_ray.dir = normalize(temp_ray.dir + LITTLE_FUZZ * random_in_unit_sphere_V4f(&state[random_idx]));

                // recursive ray tracing
                while (true) {
                    // find the intersection with the mesh
                    float t = MAX;
                    int idx = -1;
                    bool hit = (*mesh).hitting_BVH(temp_ray, t, idx);

                    // if hit, set the color of the pixel
                    if (hit) {
                        // get the intersection point
                        V4f collsion = temp_ray(t);
                        Triangle collsion_triangle = (*mesh).get_triangle(idx);
                        V4f collision_normal = toV4f(collsion_triangle.interpolate_normal(collsion.toV3f()));
                        collision_normal = normalize(collision_normal);

                        // judge if the material is light
                        bool if_light = collsion_triangle.mat.if_light;

                        // get the new ray
                        if (if_light) {
                            // set the color of the pixel
                            temp_col = temp_col * collsion_triangle.mat.albedo;
                            break;
                        } else {
                            // russian roulette
                            if (random_float(&state[random_idx]) < russian_roulette) {
                                // get the new light ray
                                V4f albedo;
                                Ray new_ray;
                                collsion_triangle.mat.scatter(temp_ray,
                                                              collsion,
                                                              collision_normal,
                                                              new_ray,
                                                              albedo,
                                                              &state[random_idx]);
                                // calculate n * wi
                                float cos_theta = dot(collision_normal, new_ray.dir);
                                // set the color of the pixel
                                temp_col = temp_col * albedo * cos_theta / russian_roulette;
                                // set the new ray
                                temp_ray = new_ray;
                            } else {
                                // set the color of the pixel
                                temp_col = V4f(0.0f, 0.0f, 0.0f, 1.0f);
                                break;
                            }
                        }

                    } else {
                        // set the color of the pixel
                        temp_col = V4f(0.0f, 0.0f, 0.0f, 1.0f);
                        break;
                    }
                }
                accum_col = accum_col + temp_col / samples_per_pixel;
            }

            // set the color of the pixel
            if (if_more_kernel) {
                atomicAdd(&image[y * width + x][0], accum_col[0]);
                atomicAdd(&image[y * width + x][1], accum_col[1]);
                atomicAdd(&image[y * width + x][2], accum_col[2]);
            } else {
                image[y * width + x] = V3f(accum_col[0], accum_col[1], accum_col[2]);
            }
        }
    }
}