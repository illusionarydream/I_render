#ifndef BBOX_CUH
#define BBOX_CUH

#include "math_materials.cuh"

class Bbox {
   public:
    // * data members
    V3f min;
    V3f max;

    // * constructors
    __host__ __device__ Bbox() {
        min = V3f(INFINITY, INFINITY, INFINITY);
        max = V3f(-INFINITY, -INFINITY, -INFINITY);
    }

    __host__ __device__ Bbox(const V3f& min, const V3f& max) : min(min), max(max) {}

    // * member functions
    // expand the box by a point
    __host__ __device__ void expand(const V3f& point) {
        min = V3f(fmin(min[0], point[0]), fmin(min[1], point[1]), fmin(min[2], point[2]));
        max = V3f(fmax(max[0], point[0]), fmax(max[1], point[1]), fmax(max[2], point[2]));
    }

    // expand the box by another box
    __host__ __device__ void expand(const Bbox& bbox) {
        min = V3f(fmin(min[0], bbox.min[0]), fmin(min[1], bbox.min[1]), fmin(min[2], bbox.min[2]));
        max = V3f(fmax(max[0], bbox.max[0]), fmax(max[1], bbox.max[1]), fmax(max[2], bbox.max[2]));
    }

    // get the centroid of the box
    __host__ __device__ V3f centroid() const {
        return 0.5f * (min + max);
    }

    // get the maximum extent of the box
    __host__ __device__ int maxExtent() const {
        V3f diag = max - min;
        if (diag[0] > diag[1] && diag[0] > diag[2]) {
            return 0;
        } else if (diag[1] > diag[2]) {
            return 1;
        } else {
            return 2;
        }
    }

    // intersect the box with a ray
    __host__ __device__ bool intersect(const Ray& ray, float& t_near, float& t_far) const {
        float t0 = 0.0f, t1 = MAX;
        for (int i = 0; i < 3; i++) {
            float inv_ray_dir = 1.0f / ray.dir[i];
            float t_near_i = (min[i] - ray.orig[i]) * inv_ray_dir;
            float t_far_i = (max[i] - ray.orig[i]) * inv_ray_dir;
            if (t_near_i > t_far_i) {
                float temp = t_near_i;
                t_near_i = t_far_i;
                t_far_i = temp;
            }
            t0 = t_near_i > t0 ? t_near_i : t0;
            t1 = t_far_i < t1 ? t_far_i : t1;
            if (t0 > t1) {
                return false;
            }
        }
        t_near = t0;
        t_far = t1;
        return t_far > 0.0f;
    }
};
#endif  // BBOX_CUH