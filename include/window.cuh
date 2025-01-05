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
    bool render_type;  // 0-rasterization, 1-ray tracing
    static int sample_Max;

    // for mouse callback
    static float sensitivity;

    // basic parameters
    int width, height;

   public:
    // ! all the mesh and camera parameters are set here
    // ! rasterization
    Window(int _width = IMAGE_WIDTH, int _height = IMAGE_HEIGHT, bool render_type = 0, std::string obj_path = "", std::string texture_path = "") {
        if (render_type == 0)
            this->render_type = 0;
        else
            this->render_type = 1;
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
        } else {
            // ! ray tracing
            // * set basic parameters
            width = _width;
            height = _height;

            // * read the mesh
            auto triangles = load_obj(obj_path, false, V3f(0.0f, -1.5f, 0.0f), 3.0f);
            Triangle* d_triangles = triangles.data();
            meshes.add_triangles(d_triangles, triangles.size());

            // * set the mesh material
            Material material(1, V4f(1.0f, 1.0f, 1.0f, 1.0f));
            meshes.set_material(material);  // this step must be before add_triangles, because the added light will not have the material

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
            camera.if_pathtracing = true;
            camera.setIntrinsics(2.0f, 2.0f, 0.5f, 0.5f, 0.0f);
            camera.setExtrinsics(V4f(0.0f, 0.0f, radius, 1.0f), V4f(0.0f, 0.0f, 0.0f, 1.0f), V4f(0.0f, -1.0f, 0.0f, 0.0f));  // initial position of the camera

            // * set the camera sampling
            camera.setSamplePerPixel(sample_Max - 200);
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

        // set the camera position
        V4f camera_pos;
        V4f camera_lookat(0.0f, 0.0f, 0.0f, 1.0f);
        V4f camera_up;

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
        int move_velocity = xoffset * xoffset + yoffset * yoffset;
        int samples_per_pixel = 20 + sample_Max / (1 + 10 * move_velocity);
        camera.setSamplePerPixel(samples_per_pixel);
    }

    void renderLoop(GLFWwindow* window) {
        glfwGetFramebufferSize(window, &width, &height);
        std::vector<V3f> image(height * width);

        // prepare the camera parameters
        if (render_type == 1)
            camera.setGPUParameters_raytrace(meshes, width, height);
        else
            camera.setGPUParameters_rasterize(meshes, width, height);

        while (!glfwWindowShouldClose(window)) {
            // eliminate the flicker
            if (camera.samples_per_pixel == sample_Max) {
                glfwPollEvents();
                continue;
            }

            if (render_type == 1)
                camera.render_raytrace(height, width, meshes, image);
            else
                camera.render_rasterization(height, width, meshes, image);

            // Gradually restores rendering quality
            camera.samples_per_pixel = MIN(camera.samples_per_pixel + 50, sample_Max);

            glClear(GL_COLOR_BUFFER_BIT);
            glDrawPixels(width, height, GL_RGB, GL_FLOAT, image.data());
            glfwSwapBuffers(window);
            glfwPollEvents();
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
        glfwSetCursorPosCallback(window, mouse_callback);

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
int Window::sample_Max = 350;

#endif  // WINDOW_HPP