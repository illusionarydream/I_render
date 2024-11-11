#include "camera.cuh"

void Camera::setGPUParameters_raytrace(const Mesh& meshes,
                                       const int width,
                                       const int height) {
    // * set the parameters for GPU
    cudaMalloc(&d_meshes, sizeof(Mesh));
    cudaMalloc(&d_rays, width * height * sizeof(Ray));
    cudaMalloc(&d_image, width * height * sizeof(V3f));
    cudaMalloc((void**)&devStates, width * height * samples_per_pixel * sizeof(curandState));

    cudaMemcpy(d_meshes, &meshes, sizeof(Mesh), cudaMemcpyHostToDevice);
}

// render by ray tracing
void Camera::render_raytrace(const int width,
                             const int height,
                             const Mesh& meshes,
                             std::vector<V3f>& image) {
    // * render the scene by ray tracing
    // width: width of the image
    // height: height of the image
    // meshes: the meshes in the scene
    // rays: the rays generated by the camera
    // color: the color of the meshes
    // ! use cuda functions
    M4f* d_Inv_Extrinsics;
    M3f* d_Inv_Intrinsics;
    V4f* d_cam_pos;

    cudaMalloc(&d_Inv_Extrinsics, sizeof(M4f));
    cudaMalloc(&d_Inv_Intrinsics, sizeof(M3f));
    cudaMalloc(&d_cam_pos, sizeof(V4f));

    cudaMemcpy(d_Inv_Extrinsics, &Inv_Extrinsics, sizeof(M4f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Inv_Intrinsics, &Inv_Intrinsics, sizeof(M3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cam_pos, &cam_pos, sizeof(V4f), cudaMemcpyHostToDevice);

    // set the block size and grid size
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);

    // sample-level parallelism
    if (if_more_kernel) {
        block = dim3(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
        grid = dim3((width + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y,
                    (samples_per_kernel + block.z - 1) / block.z);
    }

    // * init the curand states and image
    if (if_show_info)
        printf("Camera::init_raytrace\n");
    initCurandStates_and_image<<<grid, block>>>(devStates,
                                                d_image,
                                                time(0),
                                                width,
                                                height,
                                                samples_per_kernel);

    // synchronize the device
    cudaDeviceSynchronize();

    // * generate ray
    if (if_show_info)
        printf("Camera::generateRay\n");

    // generate ray
    generateRayKernel<<<grid, block>>>(d_Inv_Extrinsics,
                                       d_Inv_Intrinsics,
                                       d_cam_pos,
                                       width,
                                       height,
                                       d_rays);

    // synchronize the device
    cudaDeviceSynchronize();

    // * raytrace
    if (if_show_info)
        printf("Camera::render_raytrace\n");

    raytrace<<<grid, block>>>(d_meshes,
                              meshes.get_num_triangles(),
                              width,
                              height,
                              d_rays,
                              d_image,
                              // other parameters
                              if_depthmap,
                              if_normalmap,
                              if_pathtracing,
                              if_more_kernel,
                              // for path tracing
                              russian_roulette,
                              samples_per_pixel,
                              samples_per_kernel,
                              // random seed support
                              devStates);

    // synchronize the device
    cudaDeviceSynchronize();

    // copy the data back
    cudaMemcpy(image.data(), d_image, image.size() * sizeof(V3f), cudaMemcpyDeviceToHost);

    // free the memory
    cudaFree(d_Inv_Extrinsics);
    cudaFree(d_Inv_Intrinsics);
    cudaFree(d_cam_pos);

    return;
}

void Camera::setGPUParameters_rasterize(const Mesh& meshes,
                                        const int width,
                                        const int height) {
    // * set the parameters for GPU
    // meshes: the meshes in the scene
    // lights: the lights in the scene
    // ! use cuda functions
    cudaMalloc(&d_triangles, meshes.get_num_triangles() * sizeof(Triangle));
    cudaMalloc(&d_lights, meshes.get_num_lights() * sizeof(Light));
    cudaMalloc(&d_buffer_elements, width * height * super_sampling_ratio * super_sampling_ratio * sizeof(ZBuffer_element));
    cudaMalloc(&d_image, width * height * sizeof(V3f));

    cudaMemcpy(d_triangles, meshes.triangles, meshes.get_num_triangles() * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lights, meshes.light, meshes.get_num_lights() * sizeof(Light), cudaMemcpyHostToDevice);
}

// render by rasterization
void Camera::render_rasterization(const int width,
                                  const int height,
                                  const Mesh& meshes,
                                  std::vector<V3f>& image) {
    // * render the scene by rasterization
    // width: width of the image
    // height: height of the image
    // meshes: the meshes in the scene
    // image: the image to store the color
    // ! use cuda functions
    // redefine the near, far, left, right, top, bottom

    // * data transfer
    int triangle_num = meshes.get_num_triangles();
    int light_num = meshes.get_num_lights();
    int total_num = triangle_num + light_num;
    // transformation matrix
    M4f* d_Extrinsics;
    M4f* d_Inv_Extrinsics;
    M3f* d_Intrinsics;

    // transformation matrix
    cudaMalloc(&d_Extrinsics, sizeof(M4f));
    cudaMalloc(&d_Inv_Extrinsics, sizeof(M4f));
    cudaMalloc(&d_Intrinsics, sizeof(M3f));
    // copy the data to the device
    cudaMemcpy(d_Extrinsics, &Extrinsics, sizeof(M4f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Inv_Extrinsics, &Inv_Extrinsics, sizeof(M4f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Intrinsics, &Intrinsics, sizeof(M3f), cudaMemcpyHostToDevice);

    // * initialize the depth buffer
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid((width * super_sampling_ratio + block.x - 1) / block.x, (height * super_sampling_ratio + block.y - 1) / block.y, 1);

    if (if_show_info)
        printf("Camera::init_rasterization\n");

    initDepthBufferandImage<<<grid, block>>>(d_buffer_elements,
                                             d_image,
                                             width * super_sampling_ratio,
                                             height * super_sampling_ratio,
                                             super_sampling_ratio);

    // synchronize the device
    cudaDeviceSynchronize();

    // * transfrom the triangles and the lights to the view space
    // set the block size and grid size
    block = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    int sample_square = sqrt(total_num) + 1;
    grid = dim3((sample_square + block.x - 1) / block.x, (sample_square + block.y - 1) / block.y, 1);

    if (if_show_info)
        printf("Camera::transformTrianglesAndLights\n");

    // transform the triangles and lights to the view space
    transformTrianglesAndLights<<<grid, block>>>(d_triangles,
                                                 d_lights,
                                                 width * super_sampling_ratio,
                                                 height * super_sampling_ratio,
                                                 triangle_num,
                                                 light_num,
                                                 sample_square,
                                                 d_Extrinsics,
                                                 d_Inv_Extrinsics,
                                                 d_Intrinsics,
                                                 // immediate variables
                                                 d_buffer_elements);
    // synchronize the device
    cudaDeviceSynchronize();

    // * do shading
    if (if_show_info)
        printf("Camera::shading\n");

    // set the block size and grid size
    block = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    grid = dim3((width * super_sampling_ratio + block.x - 1) / block.x, (height * super_sampling_ratio + block.y - 1) / block.y, 1);

    // shading
    shading<<<grid, block>>>(d_triangles,
                             d_lights,
                             width * super_sampling_ratio,
                             height * super_sampling_ratio,
                             triangle_num,
                             light_num,
                             d_buffer_elements,
                             d_image,
                             // other parameters
                             ka,
                             kd,
                             ks,
                             kn,
                             if_depthmap,
                             if_normalmap,
                             super_sampling_ratio);
    // synchronize the device
    cudaDeviceSynchronize();

    // * copy the data back
    cudaMemcpy(image.data(), d_image, width * height * sizeof(V3f), cudaMemcpyDeviceToHost);

    // * free the memory
    cudaFree(d_Extrinsics);
    cudaFree(d_Inv_Extrinsics);
    cudaFree(d_Intrinsics);

    return;
}