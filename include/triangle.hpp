#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include "ray.hpp"

class Triangle {
   public:
    // * data members
    V4f vertices[3];  // vertices
    V4f normals[3];   // normals
    V4f color[3];     // colors

    // * constructors
    Triangle() {}
    Triangle(const V4f& v0, const V4f& v1, const V4f& v2) : vertices{v0, v1, v2} {}
    Triangle(const V4f& v0, const V4f& v1, const V4f& v2, const V4f& n0, const V4f& n1, const V4f& n2, const V4f& c0, const V4f& c1, const V4f& c2)
        : vertices{v0, v1, v2},
          normals{n0, n1, n2},
          color{c0, c1, c2} {}
    ~Triangle() {}

    // * member functions
    bool intersect(const Ray& ray, float& t, float& u, float& v) const {
        // * intersect the ray with the triangle
        // ray: the ray
        // t: the distance from the ray origin to the intersection point
        // u, v: the barycentric coordinates of the intersection point
        // return: true if the ray intersects with the triangle
        //         false otherwise
        V4f e1 = vertices[1] - vertices[0];
        V4f e2 = vertices[2] - vertices[0];
        V4f s = ray.orig - vertices[0];
        V4f s1 = Cross4(ray.dir, e2);
        V4f s2 = Cross4(s, e1);
        float inv_denom = 1.0f / s1.dot(e1);
        t = s2.dot(e2) * inv_denom;
        u = s1.dot(s) * inv_denom;
        v = s2.dot(ray.dir) * inv_denom;
        return u >= 0.0f && v >= 0.0f && u + v <= 1.0f;
    }

    V4f interpolate(const float u, const float v, int method = 0) const {
        // * interpolate the normal of the triangle
        // u, v: the barycentric coordinates of the intersection point
        // return: the interpolated normal
        if (method == 0)
            return (1.0f - u - v) * normals[0] + u * normals[1] + v * normals[2];
        // * interpolate the color of the triangle
        // u, v: the barycentric coordinates of the intersection point
        // return: the interpolated color
        else
            return (1.0f - u - v) * color[0] + u * color[1] + v * color[2];
    }
};

#endif  // TRIANGLE_HPP