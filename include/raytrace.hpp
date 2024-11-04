#ifndef RAYTRACE_HPP
#define RAYTRACE_HPP

#include "ray.hpp"
#include "mesh.hpp"

class RayTrace {
   public:
    // * data members
    Mesh mesh;

    int max_depth = 5;  // maximum depth of the ray

    // * constructors
    RayTrace() {}
    RayTrace(const Mesh& mesh) : mesh(mesh) {}
    ~RayTrace() {}

    // * member functions
    V4f trace(const Ray& ray, int depth = 0) const {
        // * trace the ray
        // ray: the ray
        // depth: the bounce depth
        // return: the color of the ray
        if (depth > max_depth)
            return V4f(0.0f, 0.0f, 0.0f, 1.0f);
        float t, u, v;
        int tri_id;
        if (!mesh.intersect(ray, t, u, v, tri_id))
            return V4f(0.0f, 0.0f, 0.0f, 1.0f);
        V4f normal = mesh.interpolate(u, v, tri_id);
        Ray new_ray(ray(t) + 0.001f * normal, ray.dir - 2 * ray.dir.dot(normal) * normal);
        return trace(new_ray, depth + 1);
    }
};

#endif  // RAYTRACE_HPP