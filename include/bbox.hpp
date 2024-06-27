#ifndef BBOX_HPP
#define BBOX_HPP

#include "ray.hpp"

// Class Bbox
// 1. pMin must be the minimum point of the bounding box and pMax must be the maximum point of the bounding box
class Bbox {
   private:
    // * data
    Point pMin, pMax;

   public:
    // * constructors
    Bbox() {
        pMin = Point(INFINITY, INFINITY, INFINITY);
        pMax = Point(-INFINITY, -INFINITY, -INFINITY);
    }
    Bbox(const Point& p1, const Point& p2)
        : pMin(Min(p1, p2)), pMax(Max(p1, p2)) {}
    // * methods
    // Union
    Bbox Union(const Point& p) {
        Bbox ret;
        ret.pMin = Min(pMin, p);
        ret.pMax = Max(pMax, p);
        return ret;
    }
    Bbox Union(const Bbox& b) {
        Bbox ret;
        ret.pMin = Min(pMin, b.pMin);
        ret.pMax = Max(pMax, b.pMax);
        return ret;
    }
    // Overlaps
    bool Overlaps(const Bbox& b) const {
        for (int i = 0; i < 3; i++)
            if (!(pMax[i] >= b.pMin[i] && pMin[i] <= b.pMax[i]))
                return false;
        return true;
    }
    // Inside
    bool Inside(const Point& pt) const {
        for (int i = 0; i < 3; i++)
            if (!(pt[i] >= pMin[i] && pt[i] <= pMax[i]))
                return false;
        return true;
    }
    // Expand
    void Expand(float delta) {
        Point p1 = pMin - Vector(delta, delta, delta);
        Point p2 = pMax + Vector(delta, delta, delta);
        pMin = Min(p1, p2);
        pMax = Max(p1, p2);
    }
    // SurfaceArea
    float SurfaceArea() const {
        Vector d = pMax - pMin;
        return 2 * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2]);
    }
    // Volume
    float Volume() const {
        Vector d = pMax - pMin;
        return d[0] * d[1] * d[2];
    }
    // MaximumExtent: get the maximum extent of the bounding box
    int MaximumExtent() const {
        Vector diag = pMax - pMin;
        return MaxAxis3(diag);
    }
    // operator[]: 0 for pMin, 1 for pMax
    Point operator[](int i) const {
        return i == 0 ? pMin : pMax;
    }
    // SetpMin
    void SetpMin(const Point& p) {
        pMax = Max(p, pMax);
        pMin = Min(p, pMin);
    }
    // SetpMax
    void SetpMax(const Point& p) {
        pMax = Max(p, pMax);
        pMin = Min(p, pMin);
    }
    // Lerp: use offset to interpolate the bounding box
    Point Lerp(const Vector& t) const {
        return Point(::Lerp(t[0], pMin[0], pMax[0]), ::Lerp(t[1], pMin[1], pMax[1]), ::Lerp(t[2], pMin[2], pMax[2]));
    }
    // Offset
    Vector Offset(const Point& p) const {
        Vector o = p - pMin;
        for (int i = 0; i < 3; i++)
            if (pMax[i] > pMin[i])
                o[i] /= pMax[i] - pMin[i];
        return o;
    }
    // BoundingSphere
    void BoundingSphere(Point* c, float* rad) const {
        *c = (pMin + pMax) / 2;
        *rad = Inside(*c) ? Distance(*c, pMax) : 0;
    }
};
#endif