#ifndef MESHHPP
#define MESHHPP

#include "triangle.hpp"
#include <fstream>

class Mesh {
   public:
    // * data members
    std::vector<Triangle> triangles;

    // * constructors
    Mesh() {}
    Mesh(const std::vector<Triangle>& triangles) : triangles(triangles) {}
    Mesh(const std::string& filename) { readObj(filename); }
    ~Mesh() {}

    // * member functions
    bool intersect(const Ray& ray, float& t, float& u, float& v, int& tri_id) const {
        // * intersect the ray with the mesh
        // ray: the ray
        // t: the distance from the ray origin to the intersection point
        // u, v: the barycentric coordinates of the intersection point
        // tri_id: the id of the intersected triangle
        // return: true if the ray intersects with the mesh
        //         false otherwise
        bool hit = false;
        float t_min = INFINITY;
        for (int i = 0; i < triangles.size(); i++) {
            float t_temp, u_temp, v_temp;
            if (triangles[i].intersect(ray, t_temp, u_temp, v_temp) && t_temp < t_min) {
                hit = true;
                t_min = t_temp;
                u = u_temp;
                v = v_temp;
                tri_id = i;
            }
        }
        t = t_min;
        return hit;
    }

    V4f interpolate(const float u, const float v, const int tri_id, const int method = 0) const {
        // * interpolate the normal of the mesh
        // u, v: the barycentric coordinates of the intersection point
        // tri_id: the id of the intersected triangle
        // return: the interpolated normal
        return triangles[tri_id].interpolate(u, v, method);
    }

    void readObj(const std::string& filename) {
        // * read the mesh from the obj file
        printf("Reading obj file: %s\n", filename.c_str());
        // filename: the path of the obj file
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: cannot open the file " << filename << std::endl;
            return;
        }
        // temporary storage
        std::vector<V4f> vertices;
        std::vector<V4f> normals;
        std::vector<V4f> colors;
        std::vector<Triangle> tris;

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string type;
            iss >> type;
            if (type == "v") {
                float x, y, z;
                iss >> x >> y >> z;
                vertices.push_back(V4f(x, y, z, 1.0f));
            } else if (type == "vn") {
                float x, y, z;
                iss >> x >> y >> z;
                normals.push_back(V4f(x, y, z, 0.0f));
            } else if (type == "vc") {
                float r, g, b;
                iss >> r >> g >> b;
                colors.push_back(V4f(r, g, b, 1.0f));
            } else if (type == "f") {
                std::string vertex1, vertex2, vertex3;
                iss >> vertex1 >> vertex2 >> vertex3;

                int v0, v1, v2, n0, n1, n2, c0, c1, c2;
                // judge the format of the obj file
                // if the format is "a/b/c", sscanf will return 3
                // if the format is "a/c", sscanf will return 2
                // if the format is "a", sscanf will return 1
                std::string format;
                int format_count = std::count(vertex1.begin(), vertex1.end(), '/');
                if (format_count == 2) {
                    sscanf(vertex1.c_str(), "%d/%d/%d", &v0, &c0, &n0);
                    sscanf(vertex2.c_str(), "%d/%d/%d", &v1, &c1, &n1);
                    sscanf(vertex3.c_str(), "%d/%d/%d", &v2, &c2, &n2);

                    tris.push_back(Triangle(vertices[v0 - 1], vertices[v1 - 1], vertices[v2 - 1],
                                            normals[n0 - 1], normals[n1 - 1], normals[n2 - 1],
                                            colors[c0 - 1], colors[c1 - 1], colors[c2 - 1]));
                } else if (format_count == 1) {
                    sscanf(vertex1.c_str(), "%d/%d", &v0, &n0);
                    sscanf(vertex2.c_str(), "%d/%d", &v1, &n1);
                    sscanf(vertex3.c_str(), "%d/%d", &v2, &n2);

                    tris.push_back(Triangle(vertices[v0 - 1], vertices[v1 - 1], vertices[v2 - 1],
                                            normals[n0 - 1], normals[n1 - 1], normals[n2 - 1],
                                            V4f(1.0f, 1.0f, 1.0f, 1.0f), V4f(1.0f, 1.0f, 1.0f, 1.0f), V4f(1.0f, 1.0f, 1.0f, 1.0f)));
                } else {
                    sscanf(vertex1.c_str(), "%d", &v0);
                    sscanf(vertex2.c_str(), "%d", &v1);
                    sscanf(vertex3.c_str(), "%d", &v2);

                    tris.push_back(Triangle(vertices[v0 - 1], vertices[v1 - 1], vertices[v2 - 1],
                                            V4f(0.0f, 0.0f, 0.0f, 0.0f), V4f(0.0f, 0.0f, 0.0f, 0.0f), V4f(0.0f, 0.0f, 0.0f, 0.0f),
                                            V4f(1.0f, 1.0f, 1.0f, 1.0f), V4f(1.0f, 1.0f, 1.0f, 1.0f), V4f(1.0f, 1.0f, 1.0f, 1.0f)));
                }
            }
        }
        file.close();
        // update the mesh
        triangles = tris;
    }

    void print() const {
        // * print the mesh
        for (int i = 0; i < triangles.size(); i++) {
            std::cout << "Triangle " << i << std::endl;
            for (int j = 0; j < 3; j++) {
                std::cout << "  Vertex " << j << ": " << triangles[i].vertices[j] << std::endl;
                std::cout << "  Normal " << j << ": " << triangles[i].normals[j] << std::endl;
                std::cout << "  Color " << j << ": " << triangles[i].color[j] << std::endl;
            }
        }
    }
};

#endif  // MESHHPP