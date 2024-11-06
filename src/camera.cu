#include "camera.cuh"

// generate ray from camera
void Camera::generateRay(const int width, const int height, std::vector<Ray>& rays) {
    // * generate ray from camera to the pixel (x, y)
    // width: width of the image
    // height: height of the image
    // ray: the generated ray
    // ! use cuda functions
    // set the block size and grid size
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // move the data to the device
    V4f* d_cam_pos;
    M3f* d_Inv_Intrinsics;
    M4f* d_Inv_Extrinsics;
    Ray* d_rays;
    cudaMalloc(&d_cam_pos, sizeof(V4f));
    cudaMalloc(&d_rays, rays.size() * sizeof(Ray));
    cudaMalloc(&d_Inv_Intrinsics, sizeof(M3f));
    cudaMalloc(&d_Inv_Extrinsics, sizeof(M4f));

    cudaMemcpy(d_cam_pos, &cam_pos, sizeof(V4f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Inv_Intrinsics, &Inv_Intrinsics, sizeof(M3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Inv_Extrinsics, &Inv_Extrinsics, sizeof(M4f), cudaMemcpyHostToDevice);

    printf("Camera::generateRay\n");

    // generate ray
    generateRayKernel<<<grid, block>>>(d_Inv_Extrinsics, d_Inv_Intrinsics, d_cam_pos, width, height, d_rays);

    // copy the data back
    cudaMemcpy(rays.data(), d_rays, rays.size() * sizeof(Ray), cudaMemcpyDeviceToHost);

    // ? debug: print the rays
    // for (int i = 0; i < rays.size(); i++) {
    //     printf("Ray %d: orig(%f, %f, %f, %f), dir(%f, %f, %f, %f)\n", i, rays[i].orig[0], rays[i].orig[1], rays[i].orig[2], rays[i].orig[3], rays[i].dir[0], rays[i].dir[1], rays[i].dir[2], rays[i].dir[3]);
    // }
    return;
}

// render by ray tracing
void Camera::render_raytrace(const int width,
                             const int height,
                             const Mesh& meshes,
                             const std::vector<Ray>& rays,
                             std::vector<V3f>& image) {
    // * render the scene by ray tracing
    // width: width of the image
    // height: height of the image
    // meshes: the meshes in the scene
    // rays: the rays generated by the camera
    // color: the color of the meshes
    // ! use cuda functions
    // set the block size and grid size
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // move the data to the device
    Mesh* d_meshes;
    Ray* d_rays;
    V3f* d_image;
    cudaMalloc(&d_meshes, sizeof(Mesh));
    cudaMalloc(&d_rays, rays.size() * sizeof(Ray));
    cudaMalloc(&d_image, image.size() * sizeof(V3f));

    cudaMemcpy(d_meshes, &meshes, sizeof(Mesh), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rays, rays.data(), rays.size() * sizeof(Ray), cudaMemcpyHostToDevice);

    printf("Camera::render_raytrace\n");

    curandState* devStates;
    cudaMalloc((void**)&devStates, width * height * sizeof(curandState));
    initCurandStates<<<grid, block>>>(devStates, time(0), width, height);

    // raytrace
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
                              // for path tracing
                              russian_roulette,
                              samples_per_pixel,
                              // random seed support
                              devStates);

    // check the cuda error
    CHECK_CUDA_ERROR(cudaGetLastError());

    // copy the data back
    cudaMemcpy(image.data(), d_image, image.size() * sizeof(V3f), cudaMemcpyDeviceToHost);

    return;
}