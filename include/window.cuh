#ifndef WINDOW_HPP
#define WINDOW_HPP
#include <iostream>
#include <string>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "camera.cuh"

#define IMAGE_WIDTH 600
#define IMAGE_HEIGHT 600

void showProgressBar(float progress) {
    int barWidth = 70;
    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

class Window {
    // for rendering
    static Camera camera;
    Mesh meshes;
    static float radius;
    int render_type;  // 0-rasterization, 1-ray tracing, 2-mixed rendering
    static int sample_Max;

    // for camera
    static V4f camera_pos;
    static V4f camera_lookat;
    static V4f camera_up;

    // for mouse callback
    static float sensitivity;

    // basic parameters
    int width, height;

   public:
    // ! all the mesh and camera parameters are set here
    // ! rasterization
    Window(int _width = IMAGE_WIDTH,
           int _height = IMAGE_HEIGHT,
           int render_type = 0,
           std::string obj_path = "",
           std::string texture_path = "",
           std::string denoise_model_path = "") {
        this->render_type = render_type;
        if (render_type == 0) {
            // ! rasterization
            // * set basic parameters
            width = _width;
            height = _height;

            // * read the mesh
            bool if_texture = (texture_path != "");
            auto triangles = load_obj(obj_path, if_texture, V3f(0.0f, -1.5f, 0.0f), 3.0f);
            Triangle* d_triangles = triangles.data();
            meshes.add_triangles(d_triangles, triangles.size());

            // * read the texture
            int texture_width, texture_height;
            if (if_texture) {
                auto texture = load_texture(texture_path, texture_width, texture_height);
                V3f* d_texture = texture.data();
                meshes.add_texture(d_texture, texture_width, texture_height);
            }

            // * set the mesh material
            Material material(1, V4f(1.0f, 1.0f, 1.0f, 1.0f));
            meshes.set_material(material);  // this step must be before add_triangles, because the added light will not have the material

            // * set the backplane
            Material backplane_material(1, V4f(1.0f, 1.0f, 1.0f, 1.0f));
            meshes.add_ground(-1.5f, backplane_material);

            // * set the light
            Light l1 = Light(V4f(15.0f, 15.0f, 15.0f, 1.0f), V4f(0.0f, 5.0f, 0.0f, 1.0f), 1.0f);
            meshes.add_light(l1);
            Light l2 = Light(V4f(10.0f, 10.0f, 10.0f, 1.0f), V4f(-3.0f, 0.0f, -2.0f, 1.0f), 1.0f);
            meshes.add_light(l2);

            // * set the camera
            camera.settextrue(if_texture);
            camera.setIntrinsics(2.0f, 2.0f, 0.5f, 0.5f, 0.0f);
            camera.setExtrinsics(V4f(0.0f, 0.0f, radius, 1.0f), V4f(0.0f, 0.0f, -1.0f, 1.0f), V4f(0.0f, -1.0f, 0.0f, 0.0f));  // initial position of the camera
        } else if (render_type == 1) {
            // ! ray tracing
            // * set basic parameters
            width = _width;
            height = _height;

            // * load denoise model
            camera.model_path = denoise_model_path;
            if (camera.if_denoise == true && camera.denoise_type == 3) {  // 3 means using the denoise model
                camera.loadDenoiseModel();
            }

            // * read the mesh
            auto triangles1 = load_obj(obj_path, false, V3f(0.0f, -1.5f, -2.0f), 3.0f);
            auto triangles2 = load_obj(obj_path, false, V3f(0.0f, -1.5f, 0.0f), 3.0f);
            Triangle* d_triangles1 = triangles1.data();
            Triangle* d_triangles2 = triangles2.data();
            meshes.add_triangles(d_triangles1, triangles1.size());
            meshes.add_triangles(d_triangles2, triangles2.size());

            // * set the mesh material
            Material material1(3, V4f(1.0f, 0.5f, 1.0f, 1.0f), 0.9);
            meshes.set_material(material1, 0, triangles1.size());  // this step must be before add_triangles, because the added light will not have the material

            Material material2(2, V4f(1.0f, 1.0f, 0.5f, 1.0f), 0.1);
            meshes.set_material(material2, triangles1.size(), triangles2.size());

            // * set the light
            Triangle light_tri(V3f(10.0f, 5.0f, 10.0f),
                               V3f(-10.0f, 5.0f, 0.0f),
                               V3f(10.0f, 5.0f, -10.0f));
            Material light_material(0, V4f(1.0f, 1.5f, 1.5f, 1.0f));
            light_tri.set_material(light_material);
            meshes.add_triangle(light_tri);

            // * set the backplane
            Material backplane_material(1, V4f(1.0f, 1.0f, 1.0f, 1.0f));
            meshes.add_ground(-1.5f, backplane_material);

            // * build BVH
            meshes.build_BVH();

            // * set the camera
            camera.setRussianRoulette(0.95f);
            camera.if_pathtracing = true;
            camera.setIntrinsics(2.0f, 2.0f, 0.5f, 0.5f, 0.0f);
            camera.setExtrinsics(V4f(0.0f, 0.0f, radius, 1.0f), V4f(0.0f, 0.0f, 0.0f, 1.0f), V4f(0.0f, -1.0f, 0.0f, 0.0f));  // initial position of the camera

            // * set the camera sampling
            camera.setSamplePerPixel(5);  // ! set to a constant, should larger than 5
        } else if (render_type == 2) {
            // ! mixed rendering
            // * set basic parameters
            width = _width;
            height = _height;

            // * read the mesh
            bool if_texture = (texture_path != "");
            auto triangles = load_obj(obj_path, if_texture, V3f(0.0f, -1.5f, 0.0f), 3.0f);
            Triangle* d_triangles = triangles.data();
            meshes.add_triangles(d_triangles, triangles.size());

            // * read the texture
            int texture_width, texture_height;
            if (if_texture) {
                auto texture = load_texture(texture_path, texture_width, texture_height);
                V3f* d_texture = texture.data();
                meshes.add_texture(d_texture, texture_width, texture_height);
            }

            // * set the mesh material
            Material material(2, V4f(1.0f, 1.0f, 1.0f, 1.0f), 0.1);
            meshes.set_material(material);  // this step must be before add_triangles, because the added light will not have the material

            // * set the light for raytracing
            Triangle light_tri(V3f(10.0f, 5.0f, 10.0f),
                               V3f(-10.0f, 5.0f, 0.0f),
                               V3f(10.0f, 5.0f, -10.0f));
            Material light_material(0, V4f(1.0f, 1.0f, 1.0f, 1.0f));
            light_tri.set_material(light_material);
            meshes.add_triangle(light_tri);

            // * set the light for rasterization
            Light l1 = Light(V4f(15.0f, 15.0f, 15.0f, 1.0f), V4f(0.0f, 5.0f, 0.0f, 1.0f), 1.0f);
            meshes.add_light(l1);
            Light l2 = Light(V4f(10.0f, 10.0f, 10.0f, 1.0f), V4f(-3.0f, 0.0f, -2.0f, 1.0f), 1.0f);
            meshes.add_light(l2);

            // * set the backplane
            Material backplane_material(1, V4f(1.0f, 1.0f, 1.0f, 1.0f));
            meshes.add_ground(-1.5f, backplane_material);

            // * build BVH
            meshes.build_BVH();

            // * set the camera
            camera.settextrue(if_texture);
            camera.if_pathtracing = true;
            camera.setIntrinsics(2.0f, 2.0f, 0.5f, 0.5f, 0.0f);
            camera.setExtrinsics(V4f(0.0f, 0.0f, radius, 1.0f), V4f(0.0f, 0.0f, 0.0f, 1.0f), V4f(0.0f, -1.0f, 0.0f, 0.0f));  // initial position of the camera

            // * set the camera sampling
            camera.setSamplePerPixel(sample_Max - 250);
            camera.setRussianRoulette(0.60f);
            camera.setsuper_sampling_ratio(1);
        }
    }

    static void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
        static bool firstMouse = true;
        static float lastX = 400, lastY = 300;
        static float yaw = 180.0f, pitch = 0.0f;

        // the first time the mouse moves
        if (firstMouse) {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }

        // update the x and y offset of the mouse
        float xoffset = xpos - lastX;
        float yoffset = ypos - lastY;
        lastX = xpos;
        lastY = ypos;

        yaw += xoffset * sensitivity * 1.4;
        pitch += yoffset * sensitivity;

        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;

        // calculate the camera position
        camera_pos[0] = radius * cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        camera_pos[1] = radius * sin(glm::radians(pitch));
        camera_pos[2] = radius * sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        camera_pos[3] = 1.0f;

        // calculate the camera up vector
        glm::vec3 camera_x = glm::normalize(glm::cross(glm::vec3(camera_pos[0], camera_pos[1], camera_pos[2]), glm::vec3(0.0f, 1.0f, 0.0f)));
        glm::vec3 camera_y = glm::normalize(glm::cross(glm::vec3(camera_pos[0], camera_pos[1], camera_pos[2]), camera_x));

        camera_up[0] = camera_y.x;
        camera_up[1] = camera_y.y;
        camera_up[2] = camera_y.z;
        camera_up[3] = 0.0f;

        // set the camera extrinsics
        camera.setExtrinsics(camera_pos, camera_lookat, camera_up);

        // * adapt the camera sampling by moving velocity
        // int move_velocity = xoffset * xoffset + yoffset * yoffset;
        // int samples_per_pixel = 50 + sample_Max / (1 + 40 * move_velocity);

        // camera.setSamplePerPixel(samples_per_pixel);
    }

    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        radius -= yoffset * sensitivity;
        if (radius < 5.0f) radius = 5.0f;    // 防止摄像机距离过近
        if (radius > 20.0f) radius = 20.0f;  // 防止摄像机距离过远

        // update the camera position
        camera_pos = radius * normalize(camera_pos);
        camera_pos[3] = 1.0f;

        // set the camera extrinsics
        camera.setExtrinsics(camera_pos, camera_lookat, camera_up);
    }

    void renderLoop(GLFWwindow* window) {
        glfwGetFramebufferSize(window, &width, &height);
        std::vector<V3f> image(height * width);

        // prepare the camera parameters
        if (render_type == 1)
            camera.setGPUParameters_raytrace(meshes, width, height);
        else if (render_type == 0)
            camera.setGPUParameters_rasterize(meshes, width, height);
        else if (render_type == 2) {
            camera.setGPUParameters_raytrace(meshes, width, height);
            camera.setGPUParameters_rasterize(meshes, width, height);
        }

        while (!glfwWindowShouldClose(window)) {
            // eliminate the flicker
            if (camera.samples_per_pixel == sample_Max) {
                glfwPollEvents();
                continue;
            }

            if (render_type == 1)
                camera.render_raytrace(width, height, meshes, image);
            else if (render_type == 0)
                camera.render_rasterization(width, height, meshes, image);
            else if (render_type == 2)
                camera.render_mixed(width, height, meshes, image);

            // Gradually restores rendering quality
            // camera.samples_per_pixel = MIN(camera.samples_per_pixel + 20, sample_Max);

            glClear(GL_COLOR_BUFFER_BIT);
            glDrawPixels(width, height, GL_RGB, GL_FLOAT, image.data());
            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    void renderSingleFrame(std::string filename) {
        std::vector<V3f> image(height * width);

        // set the camera position
        // camera_pos = toV4f(radius * normalize(V3f(1.0f, 0.0f, -0.5f)), 1.0f);
        // camera_lookat = V4f(0.0f, 0.0f, 0.0f, 1.0f);
        // camera.setExtrinsics(camera_pos, camera_lookat, V4f(0.0f, 1.0f, 0.0f, 0.0f));  // initial position of the camera

        // prepare the camera parameters
        // if (render_type == 1)
        //     camera.setGPUParameters_raytrace(meshes, width, height);
        // else if (render_type == 0)
        //     camera.setGPUParameters_rasterize(meshes, width, height);
        // else if (render_type == 2) {
        //     camera.setGPUParameters_raytrace(meshes, width, height);
        //     camera.setGPUParameters_rasterize(meshes, width, height);
        // }

        // render the scene
        if (render_type == 1)
            camera.render_raytrace(width, height, meshes, image);
        else if (render_type == 0)
            camera.render_rasterization(width, height, meshes, image);
        else if (render_type == 2)
            camera.render_mixed(width, height, meshes, image);

        camera.storeImage(filename, width, height, image);
    }

    void renderMultipleFrame(std::string filename_prefix) {
        const int theta_steps = 6;  // horizontal angle divisions
        const int phi_steps = 4;    // vertical angle divisions
        const float PI = 3.1415926f;

        V3f center(0.0f, 0.0f, 0.0f);  // Target point the camera looks at

        // prepare the camera parameters
        if (render_type == 1)
            camera.setGPUParameters_raytrace(meshes, width, height);
        else if (render_type == 0)
            camera.setGPUParameters_rasterize(meshes, width, height);
        else if (render_type == 2) {
            camera.setGPUParameters_raytrace(meshes, width, height);
            camera.setGPUParameters_rasterize(meshes, width, height);
        }

        // base index for the frames
        int frame_idx = 0;

        for (int i = 0; i < phi_steps; ++i) {
            float phi = PI * (i + 1) / (phi_steps + 1);  // avoid poles

            for (int j = 0; j < theta_steps; ++j) {
                float theta = 2.0f * PI * j / theta_steps;

                // Spherical to Cartesian
                float x = radius * std::sin(phi) * std::cos(theta);
                float y = radius * std::cos(phi);
                float z = radius * std::sin(phi) * std::sin(theta);
                V3f eye(x, y, z);

                printf("Rendering frame %d: Camera position: (%f, %f, %f)\n", frame_idx, eye[0], eye[1], eye[2]);

                // Set global camera parameters
                camera_pos = toV4f(eye, 1.0f);
                camera_lookat = toV4f(center, 1.0f);

                // Calculate up vector
                glm::vec3 view_dir = glm::normalize(glm::vec3(center[0] - x, center[1] - y, center[2] - z));
                glm::vec3 world_up(0.0f, 1.0f, 0.0f);
                glm::vec3 right = glm::normalize(glm::cross(world_up, view_dir));
                glm::vec3 up = glm::normalize(glm::cross(view_dir, right));

                camera_up[0] = up.x;
                camera_up[1] = up.y;
                camera_up[2] = up.z;
                camera_up[3] = 0.0f;

                // Set camera pose
                camera.setExtrinsics(camera_pos, camera_lookat, camera_up);

                // Construct output filename, e.g., "output_0000.png"
                char buffer[256];
                sprintf(buffer, "%s_%04d.png", filename_prefix.c_str(), frame_idx++);

                // Render the scene and save the image
                renderSingleFrame(buffer);
            }
        }
    }

    void start() {
        if (!glfwInit()) {
            fprintf(stderr, "Failed to initialize GLFW\n");
            return;
        }

        // * initialize the window
        GLFWwindow* window = glfwCreateWindow(IMAGE_WIDTH, IMAGE_HEIGHT, "Rendering Window", NULL, NULL);
        if (!window) {
            fprintf(stderr, "Failed to open GLFW window\n");
            glfwTerminate();
            return;
        }

        glfwMakeContextCurrent(window);
        glewExperimental = true;
        if (glewInit() != GLEW_OK) {
            fprintf(stderr, "Failed to initialize GLEW\n");
            return;
        }

        // * initialize the callback function
        glfwSetCursorPosCallback(window, mouse_callback);  // mouse move callback
        glfwSetScrollCallback(window, scroll_callback);    // mouse scroll callback

        // * MAIN LOOP
        renderLoop(window);

        // * clean up
        glfwDestroyWindow(window);
        glfwTerminate();

        return;
    }
};

// * static members initialization
Camera Window::camera;
float Window::sensitivity = 0.4f;
float Window::radius = 10.0f;
int Window::sample_Max = 100;
V4f Window::camera_pos;
V4f Window::camera_lookat = V4f(0.0f, 0.0f, 0.0f, 1.0f);
V4f Window::camera_up;

#endif  // WINDOW_HPP