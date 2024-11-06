#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "math_materials.cuh"
#include "mesh.cuh"
#include "dataload.cuh"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "raytrace.cuh"

class Camera {
   public:
    // * data members
    M3f Intrinsics;
    M4f Extrinsics;
    M3f Inv_Intrinsics;
    M4f Inv_Extrinsics;

    bool if_depthmap = false;

    // * constructors
    Camera() {
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
    }
    Camera(const M3f& intrinsics, const M4f& extrinsics) : Intrinsics(intrinsics), Extrinsics(extrinsics) {}
    ~Camera() {}

    // * member functions
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
        V4f x_cam = normalize(cross(cam_up, z_cam));  // x_cam = cam_up(y) x z_cam
        V4f y_cam = normalize(cross(z_cam, x_cam));   // y_cam = z_cam x x_cam

        Inv_Extrinsics.reset(x_cam[0], x_cam[1], x_cam[2], dot(x_cam, cam_pos) * -1.0,
                             y_cam[0], y_cam[1], y_cam[2], dot(y_cam, cam_pos) * -1.0,
                             z_cam[0], z_cam[1], z_cam[2], dot(z_cam, cam_pos) * -1.0,
                             0.0f, 0.0f, 0.0f, 1.0f);

        // get the inverse of Extrinsics
        Eigen::Matrix4f Inv_Extrinsics_eigen;
        Inv_Extrinsics_eigen << x_cam[0], x_cam[1], x_cam[2], dot(x_cam, cam_pos) * -1.0,
            y_cam[0], y_cam[1], y_cam[2], dot(y_cam, cam_pos) * -1.0,
            z_cam[0], z_cam[1], z_cam[2], dot(z_cam, cam_pos) * -1.0,
            0.0f, 0.0f, 0.0f, 1.0f;
        auto Extrinsics_eigen = Inv_Extrinsics_eigen.inverse();
        Extrinsics.reset(Extrinsics_eigen(0, 0), Extrinsics_eigen(0, 1), Extrinsics_eigen(0, 2), Extrinsics_eigen(0, 3),
                         Extrinsics_eigen(1, 0), Extrinsics_eigen(1, 1), Extrinsics_eigen(1, 2), Extrinsics_eigen(1, 3),
                         Extrinsics_eigen(2, 0), Extrinsics_eigen(2, 1), Extrinsics_eigen(2, 2), Extrinsics_eigen(2, 3),
                         Extrinsics_eigen(3, 0), Extrinsics_eigen(3, 1), Extrinsics_eigen(3, 2), Extrinsics_eigen(3, 3));
    }

    // store the image
    void storeImage(const std::string& filename,
                    const int width,
                    const int height,
                    const std::vector<V3f>& image) {
        printf("Camera::storeImage\n");
        // store the image
        cv::Mat img(height, width, CV_32FC3, (void*)image.data());
        img.convertTo(img, CV_8UC3, 255.0);
        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
        cv::imwrite(filename, img);
    }

    // generate rays in the world coordinate system
    void generateRay(const int width,
                     const int height,
                     std::vector<Ray>& rays);

    // render the scene
    void render_raytrace(const int width,
                         const int height,
                         const Mesh& meshes,
                         const std::vector<Ray>& rays,
                         std::vector<V3f>& image);
};

#endif  // CAMERA_CUH