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
    // 1. Point + Vector = Point
    // 2, Point + Point = Point maybe useful in weighted sum
    Point operator+(const Vector& v) const {
        return Point(p + v);
    }
    Point operator+(const Point& p2) const {
        return Point(p + p2.p);
    }
    Point& operator+=(const Vector& v) {
        p += v;
        return *this;
    }
    Point& operator+=(const Point& p2) {
        p += p2.p;
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
    // > operator
    bool operator>(const Point& p2) const {
        return p > p2.p;
    }
    // >= operator
    bool operator>=(const Point& p2) const {
        return p >= p2.p;
    }
    // < operator
    bool operator<(const Point& p2) const {
        return p < p2.p;
    }
    // <= operator
    bool operator<=(const Point& p2) const {
        return p <= p2.p;
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
// PMin: select the minimum value of each component
inline Point Min(const Point& p1, const Point& p2) {
    return Point(Min(p1.p[0], p2.p[0]), Min(p1.p[1], p2.p[1]), Min(p1.p[2], p2.p[2]));
}
// PMax: select the maximum value of each component
inline Point Max(const Point& p1, const Point& p2) {
    return Point(Max(p1.p[0], p2.p[0]), Max(p1.p[1], p2.p[1]), Max(p1.p[2], p2.p[2]));
}
// cout
inline std::ostream& operator<<(std::ostream& os, const Point& p) {
    os << p.p;
    return os;
}

#endif