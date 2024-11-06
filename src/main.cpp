// main.cpp
#include <iostream>
#include "dataload.cuh"
#include "camera.cuh"

#define IMAGE_WIDTH 1600
#define IMAGE_HEIGHT 1600

int main() {
    // * read the mesh
    auto triangles = load_obj("/home/illusionary/文档/计算机图形学/I_render/datasets/stanford-bunny.obj");
    Triangle* d_triangles = triangles.data();
    Mesh meshes(d_triangles, triangles.size());

    // * set the camera
    Camera camera;
    camera.setIntrinsics(5.0f, 5.0f, 0.4f, 0.2f, 0.0f);
    camera.setExtrinsics(V4f(0.0f, 0.0f, 2.0f, 1.0f), V4f(0.0f, 0.0f, 0.0f, 1.0f), V4f(0.0f, 1.0f, 0.0f, 0.0f));

    // * generate rays
    std::vector<Ray> rays(IMAGE_WIDTH * IMAGE_HEIGHT);
    camera.generateRay(IMAGE_HEIGHT, IMAGE_WIDTH, rays);

    // * render the scene
    std::vector<V3f> image(IMAGE_HEIGHT * IMAGE_WIDTH);
    camera.render_raytrace(IMAGE_HEIGHT, IMAGE_WIDTH, meshes, rays, image);
    camera.storeImage("output.jpg", IMAGE_WIDTH, IMAGE_HEIGHT, image);

    return 0;
}