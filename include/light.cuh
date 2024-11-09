#ifndef LIGHT_CUH
#define LIGHT_CUH

#include "math_materials.cuh"

// only support spot light
class Light {
   public:
    // * member variable
    V4f emission;
    float area;
    V4f position;
    V4f position_view;

    // * constructor
    __host__ __device__ Light() {
        emission = V4f(1.0f, 1.0f, 1.0f, 1.0f);
        position = V4f(0.0f, 0.0f, 0.0f, 1.0f);
        area = 1.0f;
    }

    __host__ __device__ Light(const Light &light)
        : emission(light.emission), position(light.position), area(light.area) {}

    __host__ __device__ Light(const V4f &e, const V4f &p, const float a)
        : emission(e), position(p), area(a) {}

    // * overload operator
};

#endif  // LIGHT_CUH