#ifndef POINT_HPP
#define POINT_HPP
#include "vector.hpp"
class Point {
   public:
    Vector p;
    Point() {
        p = Vector(0, 0, 0);
    }
    Point(float x, float y, float z) {
        p = Vector(x, y, z);
    }
    Point(const Vector& v) {
        p = v;
    }
    // + operator
    Point operator+(const Vector& v) const {
        return Point(p + v);
    }
    Point& operator+=(const Vector& v) {
        p += v;
        return *this;
    }
    // - operator
    Vector operator-(const Point& p2) const {
        return p - p2.p;
    }
    Point operator-(const Vector& v) const {
        return Point(p - v);
    }
    Point& operator-=(const Vector& v) {
        p -= v;
        return *this;
    }
    // _* operator
    Point operator*(float f) const {
        return Point(p * f);
    }
    Point& operator*=(float f) {
        p *= f;
        return *this;
    }
    // _/ operator
    Point operator/(float f) const {
        ASSERT(f != 0, "Division by zero");
        return Point(p / f);
    }
    Point& operator/=(float f) {
        ASSERT(f != 0, "Division by zero");
        p /= f;
        return *this;
    }
    // [] operator
    float operator[](int i) const {
        ASSERT(i >= 0 && i <= 2, "Index out of bounds");
        return (&p.x)[i];
    }
    float& operator[](int i) {
        ASSERT(i >= 0 && i <= 2, "Index out of bounds");
        return (&p.x)[i];
    }
    // == operator
    bool operator==(const Point& p2) const {
        return p == p2.p;
    }
    bool operator!=(const Point& p2) const {
        return p != p2.p;
    }
};

// _* operator
inline Point operator*(float f, const Point& p) {
    return p * f;
}
// Distance between two points
inline float Distance(const Point& p1, const Point& p2) {
    return (p1.p - p2.p).Length();
}
// Squared distance between two points
inline float DistanceSquared(const Point& p1, const Point& p2) {
    return (p1.p - p2.p).LengthSquared();
}
#endif