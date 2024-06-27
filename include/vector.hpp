#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <cmath>
#include <iostream>
#include "math_material.hpp"

// Class Vector
class Vector {
   private:
    // NAN check
    bool HasNaNs() const { return std::isnan(x) || std::isnan(y) || std::isnan(z); }

   public:
    // data
    float x, y, z;
    // constructors
    Vector() {
        x = 0;
        y = 0;
        z = 0;
    }
    Vector(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {
        ASSERT(!HasNaNs(), "Vector has NaNs");
    }
    // methods
    float Length() const { return std::sqrt(x * x + y * y + z * z); }
    float LengthSquared() const { return x * x + y * y + z * z; }
    // - operator
    Vector operator-() const { return Vector(-x, -y, -z); }
    // + operator
    Vector operator+(const Vector& v) const {
        return Vector(x + v.x, y + v.y, z + v.z);
    }
    Vector& operator+=(const Vector& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    // - operator
    Vector operator-(const Vector& v) const {
        return Vector(x - v.x, y - v.y, z - v.z);
    }
    Vector& operator-=(const Vector& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    // _* operator
    Vector operator*(float f) const { return Vector(x * f, y * f, z * f); }
    Vector& operator*=(float f) {
        x *= f;
        y *= f;
        z *= f;
        return *this;
    }
    // _/ operator
    Vector operator/(float f) const {
        ASSERT(f != 0, "Division by zero");
        float inv = 1 / f;
        return Vector(x * inv, y * inv, z * inv);
    }
    Vector& operator/=(float f) {
        ASSERT(f != 0, "Division by zero");
        float inv = 1 / f;
        x *= inv;
        y *= inv;
        z *= inv;
        return *this;
    }
    // [] operator
    float operator[](int i) const {
        ASSERT(i >= 0 && i <= 2, "Index out of bounds");
        return (&x)[i];
    }
    float& operator[](int i) {
        ASSERT(i >= 0 && i <= 2, "Index out of bounds");
        return (&x)[i];
    }
    // == operator
    bool operator==(const Vector& v) const { return IsEqual(x, v.x) && IsEqual(y, v.y) && IsEqual(z, v.z); }
    bool operator!=(const Vector& v) const { return !(*this == v); }
    // > operator
    bool operator>(const Vector& v) const { return x > v.x && y > v.y && z > v.z; }
    // >= operator
    bool operator>=(const Vector& v) const { return x >= v.x && y >= v.y && z >= v.z; }
    // < operator
    bool operator<(const Vector& v) const { return x < v.x && y < v.y && z < v.z; }
    // <= operator
    bool operator<=(const Vector& v) const { return x <= v.x && y <= v.y && z <= v.z; }
    // Normalized
    Vector Normalized() const { return *this / Length(); }
};

// _* operator
inline Vector operator*(float f, const Vector& v) { return v * f; }
// dot product
inline float Dot(const Vector& v1, const Vector& v2) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }
inline float AbsDot(const Vector& v1, const Vector& v2) { return std::abs(Dot(v1, v2)); }
// cross product
inline Vector Cross(const Vector& v1, const Vector& v2) {
    return Vector(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}
// normalize
inline Vector Normalize(const Vector& v) { return v / v.Length(); }
// coordinate system
inline void CoordinateSystem(const Vector& v1, Vector* v2, Vector* v3) {
    if (std::abs(v1.x) > std::abs(v1.y)) {
        float invLen = 1 / std::sqrt(v1.x * v1.x + v1.z * v1.z);
        *v2 = Vector(-v1.z * invLen, 0, v1.x * invLen);
    } else {
        float invLen = 1 / std::sqrt(v1.y * v1.y + v1.z * v1.z);
        *v2 = Vector(0, v1.z * invLen, -v1.y * invLen);
    }
    *v3 = Cross(v1, *v2);
}
// Faceforward
inline Vector Faceforward(const Vector& v1, const Vector& v2) { return (Dot(v1, v2) < 0) ? -v1 : v1; }
// cout
inline std::ostream& operator<<(std::ostream& os, const Vector& v) {
    os << "Vector[" << v.x << ", " << v.y << ", " << v.z << "]";
    return os;
}

#define Normal Vector
#endif  // VECTOR_HPP