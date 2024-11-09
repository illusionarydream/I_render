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

__global__ void initDepthBuffer(ZBuffer_element *depth_buffer,
                                int width,
                                int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        depth_buffer[idx].depth = MAX;
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

    if (idx < num_triangles) {
        // * transform the triangles
        // get the triangle
        Triangle triangle = triangles[idx];

        // transform the vertices
        for (int i = 0; i < 3; i++) {
            V4f vertex = toV4f(triangle.vertices[i], 1.0f);
            V4f vertex_view = (*Extrinsics) * vertex;
            V3f vertex_view3 = vertex_view.toV3f() / vertex_view[3];
            triangle.vertices_view[i] = vertex_view3;
        }

        // transform the normal
        for (int i = 0; i < 3; i++) {
            V4f normal = toV4f(triangle.normals[i], 0.0f);
            V4f normal_view = (*Inv_Extrinsics) * normal;
            triangle.normals_view[i] = normal_view.toV3f();
        }

        // set z buffer
        V3f v1 = (*Intrinsics) * triangle.vertices_view[0];
        V3f v2 = (*Intrinsics) * triangle.vertices_view[1];
        V3f v3 = (*Intrinsics) * triangle.vertices_view[2];
        v1 = (v1 / v1[2]) * width - V3f(0.5f, 0.5f, 0.0f);
        v2 = (v2 / v2[2]) * width - V3f(0.5f, 0.5f, 0.0f);
        v3 = (v3 / v3[2]) * width - V3f(0.5f, 0.5f, 0.0f);

        float depth1 = triangle.vertices_view[0][2];
        float depth2 = triangle.vertices_view[1][2];
        float depth3 = triangle.vertices_view[2][2];

        int x_min = fmin(fmin(v1[0], v2[0]), v3[0]);
        int x_max = fmax(fmax(v1[0], v2[0]), v3[0]);
        int y_min = fmin(fmin(v1[1], v2[1]), v3[1]);
        int y_max = fmax(fmax(v1[1], v2[1]), v3[1]);

        for (int x = x_min; x <= x_max; x++) {
            for (int y = y_min; y <= y_max; y++) {
                if (x < 0 || x >= width || y < 0 || y >= height) {
                    continue;
                }

                // barcentric coordinates
                float u, v, w;
                barycentric(float(x), float(y), v1[0], v1[1], v2[0], v2[1], v3[0], v3[1], u, v, w);

                // the point is outside the triangle
                if (u < 0 || v < 0 || w < 0) {
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
                V3f tmp_normal = triangle.normals_view[0] * u + triangle.normals_view[1] * v + triangle.normals_view[2] * w;
                V3f tmp_position = triangle.vertices_view[0] * u + triangle.vertices_view[1] * v + triangle.vertices_view[2] * w;
                float3 normal = make_float3(tmp_normal[0], tmp_normal[1], tmp_normal[2]);
                float3 position = make_float3(tmp_position[0], tmp_position[1], tmp_position[2]);

                atomicUpdateStruct(&depth_buffer[depth_idx], {depth, normal, position, 0});
            }
        }

    } else if (idx < num_triangles + num_lights) {
        // * transform the lights
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
    }
}

__global__ void shading(const Triangle *d_triangles,
                        const Light *lights,
                        int width,
                        int height,
                        int num_triangles,
                        int num_lights,
                        const ZBuffer_element *depth_buffer,
                        V3f *image) {
    // * shading the triangles
    // d_triangles: the triangles in the scene
    // d_lights: the lights in the scene
    // num_triangles: the number of triangles
    // num_lights: the number of lights
    // depth_buffer: the depth buffer
    // image: the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;

        // get the depth buffer element
        ZBuffer_element element = depth_buffer[idx];

        // if the depth is MAX, then the pixel is not visible
        if (element.depth > MAX - 10) {
            return;
        }

        // get the normal and position
        V3f normal = V3f(element.normal.x, element.normal.y, element.normal.z);
        V3f position = V3f(element.position.x, element.position.y, element.position.z);

        // * Blinn-Phong shading
        V3f color = V3f(0.0f, 0.0f, 0.0f);

        // ambient: ka * Ia
        float ambient = 0.1f;
        color = color + ambient * V3f(1.0f, 1.0f, 1.0f);

        // diffuse and specular
        for (int i = 0; i < num_lights; i++) {
            Light light;
            light = lights[i];

            // get the light direction
            V3f light_direction = light.position_view.toV3f() - position;
            float dist2 = dot(light_direction, light_direction);
            if (dist2 < MIN_surface) continue;
            float inv_dist2 = 1.0f / dist2;
            light_direction = light_direction * sqrt(inv_dist2);

            // get the view direction
            V3f view_direction = V3f(0.0f, 0.0f, 0.0f) - position;
            view_direction = normalize(view_direction);

            // get the half vector
            V3f half_vector = light_direction + view_direction;
            half_vector = normalize(half_vector);

            // get the diffuse illumination: kd * I * max(0, dot(N, L))
            float diffuse = fmax(0.0f, dot(normal, light_direction)) * 0.8f * inv_dist2;

            // get the specular illumination: ks * I * max(0, dot(N, H))^n
            float specular = pow(fmax(0.0f, dot(normal, half_vector)), 32) * 0.8f * inv_dist2;

            // add the color
            color = color + light.emission.toV3f() * (diffuse + specular);
        }

        // set the color
        image[idx] = color;
    }
}