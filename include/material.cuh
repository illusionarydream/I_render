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

    // for refractive material
    float ref_idx;  // the refractive index

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
    // for metal material and refractive material
    __host__ __device__ Material(int type, const V4f &a, const float f) : type(type),
                                                                          albedo(a) {
        if (type == 2) {
            fuzz = f;
        } else if (type == 3) {
            ref_idx = f;
        } else {
            fuzz = 0.0f;
            ref_idx = 1.0f;
        }
        if_light = false;
    }

    // * member functions
    // for metal material
    // the v and output are both outgoing
    __host__ __device__ V4f reflect(const V4f &v, const V4f &n) const {
        V4f reflected = n * 2.0f - v;

        return reflected;
    }

    // for refractive material
    // the v and output are both outgoing
    __host__ __device__ bool refract(const V4f &v, const V4f &n, float ni_over_nt, V4f &refracted) const {
        V4f uv = normalize(v);
        float dt = dot(uv, n);
        float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
        if (discriminant > 0.0f) {
            refracted = (uv - n * dt) * ni_over_nt - n * sqrt(discriminant);
            return true;
        } else {
            return false;
        }
    }

    // shlick's approximation
    __host__ __device__ float schlick(float cosine, float ref_idx) const {
        float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
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
            albedo = this->albedo;  // the albedo is the color of the light
            // no reflection
            wo = Ray(collision, V4f(0.0f, 0.0f, 0.0f, 0.0f));

            return false;
        }
        // *1: lambertian
        else if (type == 1) {
            albedo = this->albedo;  // the albedo is the color of the material
            // diffuse reflection
            V4f diro = normal + random_in_unit_sphere_V4f(state);
            wo = Ray(collision, diro);
            wo.orig = wo(MIN_surface);
            return true;
        }
        // *2: metal
        else if (type == 2) {
            albedo = this->albedo;  // the albedo is the color of the material
            // specular reflection
            V4f reflected = reflect(normalize(wi.dir * -1.0f), normal);
            V4f diro = reflected + random_in_unit_sphere_V4f(state) * fuzz;
            wo = Ray(collision, diro);
            wo.orig = wo(MIN_surface);
            return true;
        }
        // *3. refractive
        else if (type == 3) {
            albedo = this->albedo;  // the albedo is the color of the material
            // refraction
            V4f outward_normal;
            float ni_over_nt;
            float cosine;
            float reflect_prob;
            float reflectance;

            V4f reflected = reflect(wi.dir * -1.0f, normal);
            if (dot(wi.dir, normal) > 0.0f) {
                outward_normal = normal * -1.0f;
                ni_over_nt = ref_idx;
                cosine = ref_idx * dot(wi.dir, normal);
            } else {
                outward_normal = normal;
                ni_over_nt = 1.0f / ref_idx;
                cosine = -dot(wi.dir, normal);
            }

            V4f refracted;
            if (refract(wi.dir, outward_normal, ni_over_nt, refracted)) {
                reflect_prob = schlick(cosine, ref_idx);
            } else {
                reflect_prob = 1.0f;
            }

            if (random_float(state) < reflect_prob) {
                wo = Ray(collision, reflected);
            } else {
                wo = Ray(collision, refracted);
            }
            wo.orig = wo(MIN_surface);

            return true;
        }
        return false;
    }
};

#endif  // MATERIALS_CUH