#ifndef MATERIALS_CUH
#define MATERIALS_CUH

#include "math_materials.cuh"

// the father class of all materials
// ! Note:
// ! color is (r, g, b, 1.0f)
// ! normal is (x, y, z, 0.0f)
// ! point is (x, y, z, 1.0f)
// * the different type
// 0: light
// 1: lambertian
// 2: metal
class Material {
   public:
    // * member variable
    int type;

    // for all materials
    V4f albedo;  // the absorption coefficient, (r, g, b, 1.0f)
    bool if_light;

    // for metal material
    float fuzz;  // the fuzziness of the metal

    // * constructor
    // default constructor: lambertian material
    __host__ __device__ Material() {
        albedo = V4f(0.5f, 0.5f, 0.5f, 1.0f);
        type = 1;
        if_light = false;
        fuzz = 0.0f;
    }

    __host__ __device__ Material(const Material &mat) {
        type = mat.type;
        albedo = mat.albedo;
        if_light = mat.if_light;
        fuzz = mat.fuzz;
    }

    // for light material and lambertian material
    __host__ __device__ Material(int type, const V4f &a) : type(type),
                                                           albedo(a) {
        if (type == 0) {
            if_light = true;
        } else {
            if_light = false;
        }
        fuzz = 0.0f;
    }
    // for metal material
    __host__ __device__ Material(int type, const V4f &a, const float f) : type(type),
                                                                          albedo(a),
                                                                          fuzz(f) {
        if_light = false;
    }

    // * member functions
    // for metal material
    // the v and output are both outgoing
    __host__ __device__ V4f reflect(const V4f &v, const V4f &n) const {
        V4f reflected = n * 2.0f - v;
        return reflected;
    }

    // * scatter function
    // input: the incident Ray, the normal, the collision point
    // output: albedo, scattered Ray
    // light material will return false
    // non-light material will return true
    __device__ virtual bool scatter(const Ray &wi,
                                    const V4f &collision,
                                    const V4f &normal,
                                    Ray &wo,
                                    V4f &albedo,
                                    curandState *state) const {
        // *0: light
        if (type == 0) {
            albedo = this->albedo;                             // the albedo is the color of the light
            wo = Ray(collision, V4f(0.0f, 0.0f, 0.0f, 0.0f));  // the scattered Ray is the light itself
            return false;
        }
        // *1: lambertian
        else if (type == 1) {
            albedo = this->albedo;  // the albedo is the color of the material
            // the scattered Ray is the Ray that starts from the collision point
            // and goes to a random point on the hemisphere whose normal is the normal
            V4f diro = normal + random_in_unit_sphere_V4f(state);

            wo = Ray(collision, normalize(diro));
            return true;
        }
        // *2: metal
        else if (type == 2) {
            albedo = this->albedo;  // the albedo is the color of the material
            // the scattered Ray is the Ray that starts from the collision point
            // and goes to the reflection point of the incident Ray
            V4f reflected = reflect(normalize(wi.dir * -1.0f), normal);
            V4f diro = reflected + random_in_unit_sphere_V4f(state) * fuzz;

            wo = Ray(collision, normalize(diro));
            return true;
        }
        return false;
    }
};

#endif  // MATERIALS_CUH