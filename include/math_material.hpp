#ifndef MATH_MATERIAL_HPP
#define MATH_MATERIAL_HPP

#include "error.hpp"
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>

using namespace Eigen;

#define MIN_FLOAT 1.0e-5

#define M4f Matrix4f
#define M3f Matrix3f
#define V4f Vector4f
#define V3f Vector3f

// judge if a float is zero
bool IsZero(float f) { return std::abs(f) < MIN_FLOAT; }

// judge if two floats are equal
bool IsEqual(float a, float b) { return IsZero(a - b); }

// ? not sure
// judge if a float is in the range [min, max]
bool IsBounds(float f, float min, float max) { return f >= min && f <= max; }

// judge min or max
template <typename T>
T Max(const T& a, const T& b) { return std::max(a, b); }

template <typename T>
T Min(const T& a, const T& b) { return std::min(a, b); }

template <typename T>
T Max(const T& a, const T& b, const T& c) { return std::max(std::max(a, b), c); }

template <typename T>
T Min(const T& a, const T& b, const T& c) { return std::min(std::min(a, b), c); }

V4f Max(const V4f& a, const V4f& b) {
    return V4f(Max(a[0], b[0]), Max(a[1], b[1]), Max(a[2], b[2]), 1.0f);
}

V4f Min(const V4f& a, const V4f& b) {
    return V4f(Min(a[0], b[0]), Min(a[1], b[1]), Min(a[2], b[2]), 1.0f);
}

// interpolate
template <typename T>
T Lerp(float t, const T& a, const T& b) { return (1 - t) * a + t * b; }

// get the max axis of a 4f vector, only count the first 3 elements
int MaxAxis4(const V4f& v) {
    if (v[0] > v[1] && v[0] > v[2])
        return 0;
    else if (v[1] > v[2])
        return 1;
    else
        return 2;
}

// cross for 4f vector, only vector can make cross product
V4f Cross4(const V4f& a, const V4f& b) {
    return V4f(a[1] * b[2] - a[2] * b[1],
               a[2] * b[0] - a[0] * b[2],
               a[0] * b[1] - a[1] * b[0],
               0.0f);
}

#endif