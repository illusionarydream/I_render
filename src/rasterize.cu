#include "rasterize.cuh"

// Z buffer update data structure
__device__ void atomicUpdateStruct(ZBuffer_element *address, ZBuffer_element newValue) {
    // 先尝试锁定该结构体
    while (atomicCAS(&address->lock, 0, 1) != 0) {
        // 如果未成功获取锁，则继续等待
    }

    // 在获得锁的情况下进行更新
    if (newValue.depth < address->depth) {
        // 更新 depth
        int *address_depth = (int *)&address->depth;
        atomicExch(address_depth, __float_as_int(newValue.depth));

        // 更新其他分量
        address->normal = newValue.normal;
        address->position = newValue.position;
    }

    // 更新完成后释放锁
    atomicExch(&address->lock, 0);
}

// atomicMin for float
__device__ float atomicMinFloat(float *address, float value) {
    int *address_as_int = (int *)address;
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        // 使用 __int_as_float 和 __float_as_int 将浮点数转为整数，保证操作原子性
        old = atomicCAS(address_as_int, assumed, __float_as_int(fminf(value, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}

__global__ void initDepthBufferandImage(ZBuffer_element *depth_buffer,
                                        V3f *image,
                                        int width,
                                        int height,
                                        int super_sampling_ratio) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        depth_buffer[idx].depth = MAX;
        depth_buffer[idx].normal = make_float3(0.0f, 0.0f, 0.0f);
        depth_buffer[idx].position = make_float3(0.0f, 0.0f, 0.0f);
        depth_buffer[idx].lock = 0;

        if (x % super_sampling_ratio == 0 && y % super_sampling_ratio == 0) {
            int orig_x = x / super_sampling_ratio;
            int orig_y = y / super_sampling_ratio;
            int orig_idx = orig_y * width / super_sampling_ratio + orig_x;

            image[orig_idx] = V3f(0.0f, 0.0f, 0.0f);
        }
    }
}

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
    ZBuffer_element *depth_buffer) {
    // * transform the triangles and lights to the view space
    // triangles: the triangles in the scene
    // lights: the lights in the scene
    // num_triangles: the number of triangles
    // num_lights: the number of lights
    // Extrinsics: the extrinsics matrix
    // Intrinsics: the intrinsics matrix
    // depth_buffer: the depth buffer
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * sample_square + x;

    // * transform the lights
    if (idx >= num_triangles && idx < num_triangles + num_lights) {
        idx = idx - num_triangles;

        // get the light
        Light light;
        light = lights[idx];
        // transform the position
        V4f position = light.position;
        V4f position_view = (*Extrinsics) * position;
        light.position_view = position_view / position_view[3];

        // set d_lights
        lights[idx] = light;

        return;
    }

    // * set shared memory
    int idx_block = threadIdx.y * blockDim.x + threadIdx.x;
    __shared__ V3f shared_triangles_vertices[BLOCK_SIZE_2D][3];  // vertices -> vertices_view
    __shared__ V3f shared_triangles_normals[BLOCK_SIZE_2D][3];   // normals -> normals_view

    // * move the data to shared memory
    if (idx < num_triangles) {
        // get the triangle
        Triangle triangle = triangles[idx];

        // move the vertices and normals
        for (int i = 0; i < 3; i++) {
            shared_triangles_vertices[idx_block][i] = triangle.vertices[i];
            shared_triangles_normals[idx_block][i] = triangle.normals[i];
        }
    }

    // * synchronize the threads
    __syncthreads();

    // * transform the triangles
    if (idx < num_triangles) {
        // get the triangle
        Triangle triangle = triangles[idx];

        // transform the vertices
        for (int i = 0; i < 3; i++) {
            V4f vertex_view = (*Extrinsics) * toV4f(shared_triangles_vertices[idx_block][i], 1.0f);
            shared_triangles_vertices[idx_block][i] = vertex_view.toV3f() / vertex_view[3];
        }

        // transform the normal
        for (int i = 0; i < 3; i++) {
            V4f normal = toV4f(shared_triangles_normals[idx_block][i], 0.0f);
            shared_triangles_normals[idx_block][i] = ((*Inv_Extrinsics) * normal).toV3f();
        }

        // * set z buffer
        V3f v1 = (*Intrinsics) * shared_triangles_vertices[idx_block][0];
        V3f v2 = (*Intrinsics) * shared_triangles_vertices[idx_block][1];
        V3f v3 = (*Intrinsics) * shared_triangles_vertices[idx_block][2];
        v1 = (v1 / v1[2]) * width - V3f(0.5f, 0.5f, 0.0f);
        v2 = (v2 / v2[2]) * width - V3f(0.5f, 0.5f, 0.0f);
        v3 = (v3 / v3[2]) * width - V3f(0.5f, 0.5f, 0.0f);

        // depth
        float depth1 = shared_triangles_vertices[idx_block][0][2];
        float depth2 = shared_triangles_vertices[idx_block][1][2];
        float depth3 = shared_triangles_vertices[idx_block][2][2];

        // bounding box
        int x_min = roundf(fmin(fmin(v1[0], v2[0]), v3[0]));
        int x_max = roundf(fmax(fmax(v1[0], v2[0]), v3[0]));
        int y_min = roundf(fmin(fmin(v1[1], v2[1]), v3[1]));
        int y_max = roundf(fmax(fmax(v1[1], v2[1]), v3[1]));

        for (int x = x_min; x <= x_max; x++) {
            for (int y = y_min; y <= y_max; y++) {
                if (x < 0 || x >= width || y < 0 || y >= height) {
                    continue;
                }

                // * check if the point is in the triangle

                // barcentric coordinates
                float u, v, w;
                bool if_in_triangle = barycentric(float(x), float(y),
                                                  v1[0], v1[1], v2[0], v2[1], v3[0], v3[1],
                                                  u, v, w);

                // the point is outside the triangle
                if (!if_in_triangle) {
                    continue;
                }

                // interpolate the depth
                float depth = -(u * depth1 + v * depth2 + w * depth3);
                if (depth < 0) {
                    continue;
                }

                // update the depth buffer
                int depth_idx = y * width + x;

                // other information for shading: view space normal and position
                V3f tmp_normal = shared_triangles_normals[idx_block][0] * u +
                                 shared_triangles_normals[idx_block][1] * v +
                                 shared_triangles_normals[idx_block][2] * w;
                V3f tmp_position = shared_triangles_vertices[idx_block][0] * u +
                                   shared_triangles_vertices[idx_block][1] * v +
                                   shared_triangles_vertices[idx_block][2] * w;

                float3 normal = make_float3(tmp_normal[0], tmp_normal[1], tmp_normal[2]);
                float3 position = make_float3(tmp_position[0], tmp_position[1], tmp_position[2]);

                atomicUpdateStruct(&depth_buffer[depth_idx], {depth, normal, position, 0});
            }
        }
    }

    // * synchronize the threads
    // __syncthreads();

    // * move the data back
    if (idx < num_triangles) {
        // get the triangle
        Triangle triangle = triangles[idx];

        // move the vertices and normals
        for (int i = 0; i < 3; i++) {
            triangle.vertices_view[i] = shared_triangles_vertices[idx_block][i];
            triangle.normals_view[i] = shared_triangles_normals[idx_block][i];
        }

        // set d_triangles
        triangles[idx] = triangle;
    }
}

__global__ void shading(const Triangle *triangles,
                        const Light *lights,
                        int width,
                        int height,
                        int num_triangles,
                        int num_lights,
                        const ZBuffer_element *depth_buffer,
                        V3f *image,
                        // other parameters
                        float ka,
                        float kd,
                        float ks,
                        float kn,
                        bool if_depthmap,
                        bool if_normalmap,
                        int super_sampling_ratio) {
    // * shading the triangles
    // d_triangles: the triangles in the scene
    // d_lights: the lights in the scene
    // num_triangles: the number of triangles
    // num_lights: the number of lights
    // depth_buffer: the depth buffer
    // image: the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    // * set shared memory
    int idx_block = threadIdx.y * blockDim.x + threadIdx.x;
    __shared__ Light shared_lights[MAX_light];
    __shared__ ZBuffer_element shared_depth_buffer[BLOCK_SIZE_2D];
    __shared__ V3f shared_image[BLOCK_SIZE_2D];

    // * move the data to shared memory
    // move the lights
    if (idx_block < num_lights) {
        shared_lights[idx_block] = lights[idx_block];
    }
    // move the depth buffer and image
    if (x < width && y < height) {
        shared_image[idx_block] = V3f(0.0f, 0.0f, 0.0f);
        shared_depth_buffer[idx_block] = depth_buffer[idx];
    }

    // * synchronize the threads
    __syncthreads();

    // * shading
    if (x < width && y < height) {
        // if the depth is MAX, then the pixel is not visible
        if (shared_depth_buffer[idx_block].depth > MAX - 10) {
            return;
        }

        // get the normal and position
        V3f normal = normalize(V3f(shared_depth_buffer[idx_block].normal.x,
                                   shared_depth_buffer[idx_block].normal.y,
                                   shared_depth_buffer[idx_block].normal.z));
        V3f position = V3f(shared_depth_buffer[idx_block].position.x,
                           shared_depth_buffer[idx_block].position.y,
                           shared_depth_buffer[idx_block].position.z);

        // * Blinn-Phong shading
        // ambient: ka * Ia
        shared_image[idx_block] = ka * V3f(1.0f, 1.0f, 1.0f);

        // diffuse and specular
        for (int i = 0; i < num_lights; i++) {
            // get the light direction
            V3f light_direction = shared_lights[i].position_view.toV3f() - position;

            float dist2 = dot(light_direction, light_direction);
            if (dist2 < MIN_surface) continue;
            float inv_dist2 = 1.0f / dist2;
            light_direction = light_direction * sqrt(inv_dist2);

            // printf("inv_dist2: %f\n", inv_dist2);

            // get the view direction
            V3f view_direction = V3f(0.0f, 0.0f, 0.0f) - position;
            view_direction = normalize(view_direction);

            // get the half vector
            V3f half_vector = light_direction + view_direction;
            half_vector = normalize(half_vector);

            // get the diffuse illumination: kd * I * max(0, dot(N, L))
            float diffuse = fmax(0.0f, dot(normal, light_direction)) * kd * inv_dist2;

            // get the specular illumination: ks * I * max(0, dot(N, H))^n
            float specular = pow(fmax(0.0f, dot(normal, half_vector)), kn) * ks * inv_dist2;

            // add the color
            shared_image[idx_block] = shared_image[idx_block] + shared_lights[i].emission.toV3f() * (diffuse + specular);
        }

        // dept
        if (if_depthmap) {
            shared_image[idx_block] = V3f(shared_depth_buffer[idx_block].depth, shared_depth_buffer[idx_block].depth, shared_depth_buffer[idx_block].depth) - V3f(7.0f, 7.0f, 7.0f);
        }

        // normal
        if (if_normalmap) {
            shared_image[idx_block] = (normal + V3f(1.0f, 1.0f, 1.0f)) / 2.0f;
        }
    }

    // * synchronize the threads
    // __syncthreads();

    // * move the data back
    if (x < width && y < height) {
        int orig_x = x / super_sampling_ratio;
        int orig_y = y / super_sampling_ratio;

        int orig_idx = orig_y * width / super_sampling_ratio + orig_x;

        float inv_super_sampling_ratio = 1.0f / (super_sampling_ratio * super_sampling_ratio);

        // super sampling
        atomicAdd(&image[orig_idx][0], shared_image[idx_block][0] * inv_super_sampling_ratio);
        atomicAdd(&image[orig_idx][1], shared_image[idx_block][1] * inv_super_sampling_ratio);
        atomicAdd(&image[orig_idx][2], shared_image[idx_block][2] * inv_super_sampling_ratio);
    }
}
