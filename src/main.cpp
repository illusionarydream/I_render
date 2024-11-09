// main.cpp
#include <iostream>
#include "dataload.cuh"
#include "camera.cuh"
#include "window.cuh"

#define IMAGE_WIDTH 400
#define IMAGE_HEIGHT 400

int main() {
    // * read the mesh
    auto triangles = load_obj("/home/illusionary/文档/计算机图形学/I_render/datasets/bunny.obj");

    Triangle* d_triangles = triangles.data();
    Mesh meshes(d_triangles, triangles.size());

    // * set the mesh material
    Material material(1, V4f(1.0f, 1.0f, 1.0f, 1.0f));
    meshes.set_material(material);  // this step must be before add_triangles, because the added light will not have the material

    // * set the light
    // Material light_material(0, V4f(8.0f, 3.0f, 0.0f, 1.0f));
    // Triangle light(V3f(0.0f, 2.0f, 2.0f), V3f(0.0f, 2.0f, 0.0f), V3f(-2.0f, 2.0f, 0.0f));
    // light.set_material(light_material);
    // meshes.add_triangle(light);
    Light l1 = Light(V4f(10.0f, 10.0f, 10.0f, 1.0f), V4f(0.0f, 5.0f, 0.0f, 1.0f), 1.0f);
    meshes.add_light(l1);

    // * set the mirror triangle
    // Material mirror_material(2, V4f(1.0f, 1.0f, 1.0f, 1.0f));
    // Triangle mirror(V3f(10.0f, -5.0f, 10.0f), V3f(-10.0f, -5.0f, 10.0f), V3f(0.0f, 12.0f, -30.0f));
    // mirror.set_material(mirror_material);
    // meshes.add_triangle(mirror);

    // * build the BVH
    // meshes.build_BVH();

    // * set the camera
    Camera camera;
    camera.setIntrinsics(2.0f, 2.0f, 0.5f, 0.5f, 0.0f);
    // camera.if_normalmap = true;  // set to true to render normal map
    // camera.if_depthmap = true;  // set to true to render depth map
    // camera.if_pathtracing = true;  // set to true to render path tracing
    // camera.if_more_kernel = true;  // set to true to render more kernel

    // get circle camera
    auto start = std::chrono::high_resolution_clock::now();
    printf("Rendering circle camera\n");

    for (int i = 0; i < 360; i++) {
        // * show progress bar
        showProgressBar((float)i / 360.0f);

        // * get the camera position
        float theta = i * M_PI / 180.0f;
        float x = 8.0f * cos(theta);
        float z = 8.0f * sin(theta);
        camera.setExtrinsics(V4f(x, 0.0f, z, 1.0f), V4f(0.0f, 0.0f, 0.0f, 1.0f), V4f(0.0f, 1.0f, 0.0f, 0.0f));

        // * render by rasterization
        std::vector<V3f> image(IMAGE_HEIGHT * IMAGE_WIDTH);
        camera.render_rasterization(IMAGE_HEIGHT, IMAGE_WIDTH, meshes, image);

        // // * generate rays
        // std::vector<Ray> rays(IMAGE_WIDTH * IMAGE_HEIGHT);
        // camera.generateRay(IMAGE_HEIGHT, IMAGE_WIDTH, rays);

        // // * render the scene
        // std::vector<V3f> image(IMAGE_HEIGHT * IMAGE_WIDTH);
        // camera.render_raytrace(IMAGE_HEIGHT, IMAGE_WIDTH, meshes, rays, image);

        // * store the image
        camera.storeImage("/home/illusionary/文档/计算机图形学/I_render/output/circle/circle_" + std::to_string(i) + ".png", IMAGE_WIDTH, IMAGE_HEIGHT, image);

        // * clear the cuda memory
        cudaDeviceReset();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Total rendering time: " << duration.count() << " seconds" << std::endl;

    return 0;
}