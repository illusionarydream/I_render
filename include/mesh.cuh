#ifndef MESH_CUH
#define MESH_CUH

#include "math_materials.cuh"
#include "material.cuh"
#include "bbox.cuh"

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

    __host__ __device__ Triangle(V3f *vertices, V3f *normals, const Material &mat) {
        for (int i = 0; i < 3; i++) {
            this->vertices[i] = vertices[i];
            this->normals[i] = normals[i];
        }
        this->mat = mat;
    }

    __host__ __device__ Triangle(V3f v0, V3f v1, V3f v2) {
        // add a triangle with the same normal(flat)
        vertices[0] = v0;
        vertices[1] = v1;
        vertices[2] = v2;
        normals[0] = cross(v1 - v0, v2 - v0);
        normals[1] = normals[0];
        normals[2] = normals[0];
    }

    __host__ __device__ Triangle(V3f v0, V3f v1, V3f v2, V3f n0, V3f n1, V3f n2) {
        vertices[0] = v0;
        vertices[1] = v1;
        vertices[2] = v2;
        normals[0] = n0;
        normals[1] = n1;
        normals[2] = n2;
    }

    __host__ __device__ Triangle(V3f v0, V3f v1, V3f v2, V3f n0, V3f n1, V3f n2, const Material &mat) {
        vertices[0] = v0;
        vertices[1] = v1;
        vertices[2] = v2;
        normals[0] = n0;
        normals[1] = n1;
        normals[2] = n2;
        this->mat = mat;
    }

    // * member functions
    __host__ __device__ V3f get_vertex(int i) const {
        return vertices[i];
    }

    __host__ __device__ V3f get_normal(int i) const {
        return normals[i];
    }

    __host__ __device__ Bbox get_bbox() const {
        Bbox bbox;
        bbox.expand(vertices[0]);
        bbox.expand(vertices[1]);
        bbox.expand(vertices[2]);

        bbox.max = bbox.max + V3f(0.001f, 0.001f, 0.001f);
        bbox.min = bbox.min - V3f(0.001f, 0.001f, 0.001f);
        return bbox;
    }

    __host__ __device__ void
    set_material(const Material &mat) {
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
        auto oriented_normals = (normals[0] + normals[1] + normals[2]) / 3.0f;
        if (dot(ray_dir, oriented_normals) > 0) {
            return -1;
        }

        // * Moller-Trumbore algorithm
        V3f e1 = vertices[1] - vertices[0];
        V3f e2 = vertices[2] - vertices[0];
        V3f s = ray_orig - vertices[0];
        V3f s1 = cross(ray_dir, e2);
        V3f s2 = cross(s, e1);

        float inv_d = 1.0f / dot(s1, e1);

        float t = dot(s2, e2) * inv_d;
        float u = dot(s1, s) * inv_d;
        float v = dot(s2, ray_dir) * inv_d;

        // if the intersection is inside the triangle
        if (t > 0 && u > 0 && v > 0 && u + v < 1) {
            return t;
        }

        // if the intersection is outside the triangle
        return -1;
    }

    // * override the operator
    __host__ __device__ Triangle &operator=(const Triangle &triangle) {
        for (int i = 0; i < 3; i++) {
            vertices[i] = triangle.vertices[i];
            normals[i] = triangle.normals[i];
        }
        mat = triangle.mat;
        return *this;
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
    __host__ __device__ Triangle get_triangle(int i) const {
        return triangles[i];
    }

    __host__ __device__ int get_num_triangles() const {
        return num_triangles;
    }

    __host__ __device__ void set_material(const Material &mat, int tri_idx = -1) {
        // if tri_idx == -1, set the material for all triangles
        if (tri_idx == -1) {
            for (int i = 0; i < num_triangles; i++) {
                triangles[i].set_material(mat);
            }
            return;
        }
        triangles[tri_idx].set_material(mat);
    }

    __host__ __device__ void add_triangle(const Triangle &triangle) {
        triangles[num_triangles] = triangle;
        num_triangles++;
    }

    __host__ __device__ void add_triangles(const Triangle *triangles, int num_triangles) {
        for (int i = 0; i < num_triangles; i++) {
            this->triangles[num_triangles + i] = triangles[i];
        }
        this->num_triangles += num_triangles;
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

    // * Adding BVH acceleration structure
   public:
    // * member variable
    Bbox box_tree[MAX_mesh * 2 - 1];                 // the bounding box of the tree
    std::pair<int, int> idx_tree[MAX_mesh * 2 - 1];  // the left and right child of the tree
    int tridx_left[MAX_mesh * 2 - 1];                // the start index of the triangles in the left child
    int tridx_right[MAX_mesh * 2 - 1];               // the start index of the triangles in the right child

    // * constructor
    __host__ void build_BVH() {
        // build Bbox for each triangle
        Bbox *bbox = new Bbox[num_triangles];  // only for temporary use
        for (int i = 0; i < num_triangles; i++)
            bbox[i] = triangles[i].get_bbox();

        // recursive build the BVH
        build_bvh_recursive(0, num_triangles, 0, bbox);
    }

    // * member functions
    __host__ __device__ void build_bvh_recursive(int l, int r, int idx, Bbox *bbox) {
        // build the Bbox for the current node
        Bbox box;
        for (int i = l; i < r; i++)
            box.expand(bbox[i]);
        box_tree[idx] = box;

        // if the number of triangles is less than 4, set the leaf node
        // l ~ r-1
        if (r - l <= 4) {
            idx_tree[idx] = std::make_pair(-1, -1);
            tridx_left[idx] = l;
            tridx_right[idx] = r;
            return;
        }

        // get the maximum extent of the box
        int max_extent = box.maxExtent();

        // sort the triangles
        std::sort(bbox + l, bbox + r, [max_extent](const Bbox &a, const Bbox &b) {
            return a.centroid()[max_extent] < b.centroid()[max_extent];
        });
        std::sort(triangles + l, triangles + r, [max_extent](const Triangle &a, const Triangle &b) {
            return a.get_bbox().centroid()[max_extent] < b.get_bbox().centroid()[max_extent];
        });

        // build the left and right child
        int mid = (l + r) / 2;
        build_bvh_recursive(l, mid, 2 * idx + 1, bbox);
        build_bvh_recursive(mid, r, 2 * idx + 2, bbox);

        // set the index of the left and right child
        idx_tree[idx] = std::make_pair(2 * idx + 1, 2 * idx + 2);
        tridx_left[idx] = -1;
        tridx_right[idx] = -1;
    }

// get the hitting triangle
#define MAX_ITR 50
    __device__ bool hitting_BVH(const Ray &ray, float &t, int &id) const {
        t = MAX;
        id = -1;
        bool if_hit = false;

        // the stack for the BVH
        int stack[MAX_ITR];
        int stack_top = 0;
        stack[stack_top++] = 0;

        // find the intersection with the mesh
        while (stack_top) {
            int idx = stack[--stack_top];
            float t_near, t_far;
            if (!box_tree[idx].intersect(ray, t_near, t_far))
                continue;

            // ! error handling
            if (MAX_ITR <= stack_top) {
                printf("Error: stack overflow\n");
                return false;
            }

            // if the bounding box is too far
            if (t_near > t)
                continue;

            // if the node is leaf node
            if (idx_tree[idx].first == -1) {
                for (int i = tridx_left[idx]; i < tridx_right[idx]; i++) {
                    const Triangle &triangle = triangles[i];

                    // find the intersection with the triangle
                    float t_tri = triangle.intersect(ray);

                    // update the intersection
                    if (t_tri > 0 && t_tri < t) {
                        t = t_tri;
                        id = i;
                        if_hit = true;
                    }
                }
            } else {
                stack[stack_top++] = idx_tree[idx].first;
                stack[stack_top++] = idx_tree[idx].second;
            }
        }
        return if_hit;
    }
};

#endif  // MESH_CUH