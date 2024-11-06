// main.cpp
#include <iostream>
#include "dataload.cuh"
#include "camera.cuh"

#define IMAGE_WIDTH 400
#define IMAGE_HEIGHT 400

int main() {
    // * read the mesh
    auto triangles = load_obj("/home/illusionary/文档/计算机图形学/I_render/datasets/bunny.obj");

    Triangle* d_triangles = triangles.data();
    Mesh meshes(d_triangles, triangles.size());

    // * set the mesh material
    Material material(1, V4f(0.5f, 0.5f, 0.5f, 0.0f));
    meshes.set_material(material);  // this step must be before add_triangles, because the added light will not have the material

    // * set the light
    Material light_material(0, V4f(10.0f, 4.0f, 0.0f, 0.0f));
    Triangle light(V3f(0.0f, 3.0f, 4.0f), V3f(0.0f, 3.0f, 0.0f), V3f(4.0f, 3.0f, 0.0f),
                   V3f(0.0f, -1.0f, 0.0f), V3f(0.0f, -1.0f, 0.0f), V3f(0.0f, -1.0f, 0.0f));
    light.set_material(light_material);
    meshes.add_triangle(light);

    // * set the mirror triangle
    // Material mirror_material(2, V4f(1.0f, 1.0f, 1.0f, 0.0f));
    // Triangle mirror(V3f(0.0f, -3.0f, 8.0f), V3f(0.0f, -3.0f, 0.0f), V3f(8.0f, -3.0f, 0.0f),
    //                 V3f(0.0f, 1.0f, 0.0f), V3f(0.0f, 1.0f, 0.0f), V3f(0.0f, 1.0f, 0.0f));
    // mirror.set_material(mirror_material);
    // meshes.add_triangle(mirror);

    // * set the camera
    Camera camera;
    camera.setIntrinsics(2.0f, 2.0f, 0.5f, 0.3f, 0.0f);
    camera.setExtrinsics(V4f(0.0f, 0.0f, 8.0f, 1.0f), V4f(0.0f, 0.0f, 0.0f, 1.0f), V4f(0.0f, 1.0f, 0.0f, 0.0f));

    // * generate rays
    std::vector<Ray> rays(IMAGE_WIDTH * IMAGE_HEIGHT);
    camera.generateRay(IMAGE_HEIGHT, IMAGE_WIDTH, rays);

    // * render the scene
    // camera.if_normalmap = true;  // set to true to render normal map
    // camera.if_depthmap = true;  // set to true to render depth map
    camera.if_pathtracing = true;  // set to true to render path tracing

    std::vector<V3f> image(IMAGE_HEIGHT * IMAGE_WIDTH);
    camera.render_raytrace(IMAGE_HEIGHT, IMAGE_WIDTH, meshes, rays, image);
    camera.storeImage("output.jpg", IMAGE_WIDTH, IMAGE_HEIGHT, image);

    return 0;
}