#ifndef MESH_CUH
#define MESH_CUH

#include "math_materials.cuh"
#include "material.cuh"

class Triangle {
   public:
    // * member variable
    V3f vertices[3];
    V3f normals[3];

    Material mat;

    // * constructor
    __host__ __device__ Triangle() {}

    __host__ __device__ Triangle(const Triangle &triangle) {
        for (int i = 0; i < 3; i++) {
            vertices[i] = triangle.vertices[i];
            normals[i] = triangle.normals[i];
        }
        mat = triangle.mat;
    }

    __host__ __device__ Triangle(V3f *vertices, V3f *normals) {
        for (int i = 0; i < 3; i++) {
            this->vertices[i] = vertices[i];
            this->normals[i] = normals[i];
        }
    }
    __host__ __device__ Triangle(V3f v0, V3f v1, V3f v2, V3f n0, V3f n1, V3f n2) {
        vertices[0] = v0;
        vertices[1] = v1;
        vertices[2] = v2;
        normals[0] = n0;
        normals[1] = n1;
        normals[2] = n2;
    }

    // * member functions
    __host__ __device__ V3f get_vertex(int i) const {
        return vertices[i];
    }

    __host__ __device__ V3f get_normal(int i) const {
        return normals[i];
    }

    __host__ __device__ void set_material(const Material &mat) {
        this->mat = mat;
    }

    // get the normal of the triangle
    __device__ V3f interpolate_normal(const V3f &p) const {
        V3f e1 = vertices[1] - vertices[0];
        V3f e2 = vertices[2] - vertices[0];
        V3f n = cross(e1, e2);

        V3f pv1 = vertices[0] - p;
        V3f pv2 = vertices[1] - p;
        V3f pv3 = vertices[2] - p;

        float area = length(n);
        float w1 = length(cross(pv2, pv3)) / area;
        float w2 = length(cross(pv3, pv1)) / area;
        float w3 = length(cross(pv1, pv2)) / area;

        return w1 * normals[0] + w2 * normals[1] + w3 * normals[2];
    }

    // get the intersection of the ray and the triangle
    __device__ float intersect(const Ray &ray) const {
        auto ray_orig = ray.orig.toV3f() / ray.orig[3];
        auto ray_dir = ray.dir.toV3f();

        // the back face is not visible
        if (dot(ray_dir, normals[0]) > 0) {
            return -1;
        }

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

    __host__ __device__ void set_material(const Material &mat) {
        for (int i = 0; i < num_triangles; i++) {
            triangles[i].set_material(mat);
        }
    }

    __host__ __device__ void add_triangle(const Triangle &triangle) {
        triangles[num_triangles] = triangle;
        num_triangles++;
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