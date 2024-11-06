#ifndef MESH_CUH
#define MESH_CUH

#include "math_materials.cuh"

class Triangle {
   public:
    // * member variable
    V3f vertices[3];

    // * constructor
    __host__ __device__ Triangle() {}

    __host__ __device__ Triangle(const Triangle &triangle) {
        for (int i = 0; i < 3; i++) {
            vertices[i] = triangle.vertices[i];
        }
    }

    __host__ __device__ Triangle(V3f *vertices) {
        for (int i = 0; i < 3; i++) {
            this->vertices[i] = vertices[i];
        }
    }
    __host__ __device__ Triangle(V3f v0, V3f v1, V3f v2) {
        vertices[0] = v0;
        vertices[1] = v1;
        vertices[2] = v2;
    }

    // * member functions
    __device__ V3f get_vertex(int i) const {
        return vertices[i];
    }
    __device__ V3f get_normal() const {
        return normalize(cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
    }
    __device__ float intersect(const Ray &ray) const {
        auto ray_orig = ray.orig.toV3f() / ray.orig[3];
        auto ray_dir = ray.dir.toV3f();

        // get t
        V3f e1 = vertices[1] - vertices[0];
        V3f e2 = vertices[2] - vertices[0];
        V3f n = cross(e1, e2);

        float t = dot(n, vertices[0] - ray_orig) / dot(n, ray_dir);

        if (t < 0 || t > MAX_bound) {
            return -1;
        }

        // judge if the intersection is inside the triangle
        V3f p = ray(t).toV3f();

        auto pv1 = vertices[0] - p;
        auto pv2 = vertices[1] - p;
        auto pv3 = vertices[2] - p;

        auto c12 = cross(pv1, pv2);
        auto c23 = cross(pv2, pv3);
        auto c31 = cross(pv3, pv1);

        if (dot(c12, c23) < 0 || dot(c12, c31) < 0) {
            return -1;
        }

        return t;
    }
};

class Mesh {
   public:
    // * member variable
    Triangle triangles[MAX_mesh];
    int num_triangles;

    // * constructor
    __host__ __device__ Mesh(const Triangle *triangles, int num_triangles) {
        this->num_triangles = num_triangles;
        for (int i = 0; i < num_triangles; i++) {
            this->triangles[i] = triangles[i];
        }
    }

    // * member functions
    __device__ Triangle get_triangle(int i) const {
        return triangles[i];
    }
    __host__ __device__ int get_num_triangles() const {
        return num_triangles;
    }
    // get hitting triangle
    __device__ bool hitting(const Ray &ray, float &t, int &id) const {
        t = MAX;
        id = -1;

        // find the intersection with the mesh
        for (int i = 0; i < num_triangles; i++) {
            const Triangle &triangle = triangles[i];

            // find the intersection with the triangle
            float t_tri = triangle.intersect(ray);

            // update the intersection
            if (t_tri > 0 && t_tri < t) {
                t = t_tri;
                id = i;
            }
        }
        return t < MAX - 10;
    }
};

#endif  // MESH_CUH