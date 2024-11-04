#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "ray.hpp"
#include "rasterize.hpp"
#include "raytrace.hpp"

class Camera {
   public:
    // * data members
    M3f Intrinsics;
    M4f Extrinsics;
    M3f Inv_Intrinsics;
    M4f Inv_Extrinsics;

    // * constructors
    Camera();
    Camera(const M4f& intrinsics, const M4f& extrinsics) : Intrinsics(intrinsics), Extrinsics(extrinsics) {}
    ~Camera();

    // * member functions
    void setIntrinsics(const float f_sx = 1.0f, const float f_sy = 1.0f, const float c_x = 0.5f, const float c_y = 0.5f, const float s = 0.0f) {
        // * from camera coordinate system to image coordinate system
        // f_sx: focal length in x direction
        // f_sy: focal length in y direction
        // c_x: principal point in x direction
        // c_y: principal point in y direction
        Intrinsics = M4f(f_sx, s, c_x,
                         0.0f, f_sy, c_y,
                         0.0f, 0.0f, 1.0f);
    }

    void setExtrinsics(const V4f& cam_pos, const V4f& cam_lookat, const V4f& cam_up) {
        // * from world coordinate system to camera coordinate system
        // cam_pos: camera position
        // cam_lookat: camera lookat, the point camera is looking at
        // cam_up: camera up
        // camera coordinate system
        V4f z_cam = cam_pos - cam_lookat;  // lookat the negative z direction
        z_cam.normalized();
        V4f x_cam = Cross4(cam_up, z_cam);  // x_cam = cam_up(y) x z_cam
        x_cam.normalized();
        V4f y_cam = Cross4(z_cam, x_cam);  // y_cam = z_cam x x_cam
        y_cam.normalized();

        Extrinsics = M4f(x_cam[0], x_cam[1], x_cam[2], -x_cam.dot(cam_pos),
                         y_cam[0], y_cam[1], y_cam[2], -y_cam.dot(cam_pos),
                         z_cam[0], z_cam[1], z_cam[2], -z_cam.dot(cam_pos),
                         0.0f, 0.0f, 0.0f, 1.0f);
    }

    // generate rays in the world coordinate system
    void generateRay(const int x, const int y, const int width, const int height, Ray& ray) {
        // * generate ray from camera to the pixel (x, y)
        // x: x coordinate of the pixel
        // y: y coordinate of the pixel
        // width: width of the image
        // height: height of the image
        // ray: the generated ray
        V4f p_film((x + 0.5f) / width, (y + 0.5f) / height, 0.0f, 1.0f);  // film coordinate system
        V4f p_camera = Inv_Intrinsics * p_film;                           // camera coordinate system
        V4f p_world = Inv_Extrinsics * p_camera;                          // world coordinate system
        ray.orig = Inv_Extrinsics.col(3);
        ray.dir = p_world - ray.orig;
    }

    // render the scene
};

#endif  // CAMERA_HPP