#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "math_materials.cuh"
#include "mesh.cuh"
#include "dataload.cuh"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "raytrace.cuh"
#include "rasterize.cuh"

class Camera {
   public:
    // * data members
    V4f cam_pos;
    M3f Intrinsics;
    M4f Extrinsics;
    M3f Inv_Intrinsics;
    M4f Inv_Extrinsics;

    bool if_depthmap = false;
    bool if_normalmap = false;
    bool if_pathtracing = false;
    bool if_more_kernel = false;
    bool if_show_info = false;

    float russian_roulette = 0.80f;
    int samples_per_pixel = 100;
    int samples_per_kernel = 20;

    // * for rasterization
    float ka = 0.1;
    float kd = 0.1;
    float ks = 1.2;
    float kn = 32.0;

    // * pre-gpu parameters
    // rasterize
    Triangle* d_triangles;
    Light* d_lights;
    ZBuffer_element* d_buffer_elements;

    // raytrace
    Mesh* d_meshes;
    Ray* d_rays;
    curandState* devStates;  // for random seed support

    // all
    V3f* d_image;

    int super_sampling_ratio = 4;  // cannot larger than 4

    // * constructors
    Camera() {
        cam_pos = V4f(0.0f, 0.0f, 0.0f, 1.0f);
        Intrinsics.reset(1.0f, 0.0f, 0.5f,
                         0.0f, 1.0f, 0.5f,
                         0.0f, 0.0f, 1.0f);
        Inv_Intrinsics.reset(1.0f, 0.0f, -0.5f,
                             0.0f, 1.0f, -0.5f,
                             0.0f, 0.0f, 1.0f);
        Extrinsics.reset(1.0f, 0.0f, 0.0f, 0.0f,
                         0.0f, 1.0f, 0.0f, 0.0f,
                         0.0f, 0.0f, 1.0f, 0.0f,
                         0.0f, 0.0f, 0.0f, 1.0f);
        Inv_Extrinsics.reset(1.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 1.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 1.0f, 0.0f,
                             0.0f, 0.0f, 0.0f, 1.0f);
        d_triangles = nullptr;
        d_lights = nullptr;
        d_buffer_elements = nullptr;
    }
    Camera(const M3f& intrinsics, const M4f& extrinsics) : Intrinsics(intrinsics), Extrinsics(extrinsics) {}
    ~Camera() {}

    // * member functions
    void setSamplePerPixel(const int samples_per_pixel) {
        this->samples_per_pixel = samples_per_pixel;
    }

    void setIntrinsics(float f_sx = 1.0f, float f_sy = 1.0f, float c_x = 1.0f, float c_y = 1.0f, float s = 0.0f) {
        // * from camera coordinate system to image coordinate system
        // f_sx: focal length in x direction
        // f_sy: focal length in y direction
        // c_x: principal point in x direction
        // c_y: principal point in y direction
        Intrinsics.reset(f_sx, s, c_x,
                         0.0f, f_sy, c_y,
                         0.0f, 0.0f, 1.0f);

        // get the inverse of Intrinsics
        Eigen::Matrix3f Intrinsics_eigen;
        Intrinsics_eigen << f_sx, s, c_x,
            0.0f, f_sy, c_y,
            0.0f, 0.0f, 1.0f;
        auto Inv_Intrinsics_eigen = Intrinsics_eigen.inverse();
        Inv_Intrinsics.reset(Inv_Intrinsics_eigen(0, 0), Inv_Intrinsics_eigen(0, 1), Inv_Intrinsics_eigen(0, 2),
                             Inv_Intrinsics_eigen(1, 0), Inv_Intrinsics_eigen(1, 1), Inv_Intrinsics_eigen(1, 2),
                             Inv_Intrinsics_eigen(2, 0), Inv_Intrinsics_eigen(2, 1), Inv_Intrinsics_eigen(2, 2));
    }

    void setExtrinsics(const V4f& cam_pos, const V4f& cam_lookat, const V4f& cam_up) {
        // * from world coordinate system to camera coordinate system
        // cam_pos: camera position
        // cam_lookat: camera lookat, the point camera is looking at
        // cam_up: camera up
        // camera coordinate system
        V4f z_cam = normalize(cam_lookat - cam_pos);  // lookat the negative z direction
        V4f y_cam = normalize(cam_up);
        V4f x_cam = normalize(cross(z_cam, y_cam));

        Extrinsics.reset(x_cam[0], x_cam[1], x_cam[2], dot(x_cam, cam_pos) * -1.0,
                         y_cam[0], y_cam[1], y_cam[2], dot(y_cam, cam_pos) * -1.0,
                         -z_cam[0], -z_cam[1], -z_cam[2], dot(z_cam, cam_pos),
                         0.0f, 0.0f, 0.0f, 1.0f);

        // get the inverse of Extrinsics
        Eigen::Matrix4f Extrinsics_eigen;
        Extrinsics_eigen << x_cam[0], x_cam[1], x_cam[2], dot(x_cam, cam_pos) * -1.0,
            y_cam[0], y_cam[1], y_cam[2], dot(y_cam, cam_pos) * -1.0,
            -z_cam[0], -z_cam[1], -z_cam[2], dot(z_cam, cam_pos),
            0.0f, 0.0f, 0.0f, 1.0f;

        auto Inv_Extrinsics_eigen = Extrinsics_eigen.inverse();

        Inv_Extrinsics.reset(Inv_Extrinsics_eigen(0, 0), Inv_Extrinsics_eigen(0, 1), Inv_Extrinsics_eigen(0, 2), Inv_Extrinsics_eigen(0, 3),
                             Inv_Extrinsics_eigen(1, 0), Inv_Extrinsics_eigen(1, 1), Inv_Extrinsics_eigen(1, 2), Inv_Extrinsics_eigen(1, 3),
                             Inv_Extrinsics_eigen(2, 0), Inv_Extrinsics_eigen(2, 1), Inv_Extrinsics_eigen(2, 2), Inv_Extrinsics_eigen(2, 3),
                             Inv_Extrinsics_eigen(3, 0), Inv_Extrinsics_eigen(3, 1), Inv_Extrinsics_eigen(3, 2), Inv_Extrinsics_eigen(3, 3));

        this->cam_pos = cam_pos;
    }

    void setGPUParameters_raytrace(const Mesh& meshes,
                                   const int width,
                                   const int height);

    void setGPUParameters_rasterize(const Mesh& meshes,
                                    const int width,
                                    const int height);

    // store the image
    void storeImage(const std::string& filename,
                    const int width,
                    const int height,
                    const std::vector<V3f>& image) {
        if (if_show_info)
            printf("Camera::storeImage\n");
        // store the image
        cv::Mat img(height, width, CV_32FC3, (void*)image.data());
        img.convertTo(img, CV_8UC3, 255.0);
        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
        cv::imwrite(filename, img);
    }

    // render the scene
    void render_raytrace(const int width,
                         const int height,
                         const Mesh& meshes,
                         std::vector<V3f>& image);

    void render_rasterization(const int width,
                              const int height,
                              const Mesh& meshes,
                              std::vector<V3f>& image);
};

#endif  // CAMERA_CUH