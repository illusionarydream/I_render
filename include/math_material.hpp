#ifndef MATH_MATERIAL_HPP
#define MATH_MATERIAL_HPP

#include "error.hpp"
#include <cmath>

#define MIN_FLOAT 1.0e-5

bool IsZero(float f) { return std::abs(f) < MIN_FLOAT; }

bool IsEqual(float a, float b) { return IsZero(a - b); }

// ? not sure
bool IsBounds(float f, float min, float max) { return f >= min && f <= max; }

int Max(int a, int b) { return (a > b) ? a : b; }
int Min(int a, int b) { return (a < b) ? a : b; }
float Max(float a, float b) { return (a > b) ? a : b; }
float Min(float a, float b) { return (a < b) ? a : b; }

template <typename T>
T Max(const T& a, const T& b, const T& c) { return std::max(std::max(a, b), c); }

template <typename T>
T Min(const T& a, const T& b, const T& c) { return std::min(std::min(a, b), c); }

template <typename T>
int MaxAxis3(const T& v) {
    return (v[0] > v[1]) ? ((v[0] > v[2]) ? 0 : 2) : ((v[1] > v[2]) ? 1 : 2);
}

template <typename T>
int MinAxis3(const T& v) {
    return (v[0] < v[1]) ? ((v[0] < v[2]) ? 0 : 2) : ((v[1] < v[2]) ? 1 : 2);
}

template <typename T>
T Lerp(float t, const T& a, const T& b) { return (1 - t) * a + t * b; }

#endif