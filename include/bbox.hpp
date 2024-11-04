#ifndef BBOX_HPP
#define BBOX_HPP

#include "math_material.hpp"

// Class Bbox
// 1. pMin must be the minimum point of the bounding box and pMax must be the maximum point of the bounding box
// 2. all the points should be in the form of homogeneous coordinates: (a, b, c, 1)
class Bbox {
   private:
    // * data
    V4f pMin, pMax;

   public:
    // * constructors
    Bbox() {
        pMax = V4f(INFINITY, INFINITY, INFINITY, 1.0f);
        pMin = V4f(-INFINITY, -INFINITY, -INFINITY, 1.0f);
    }
    Bbox(const V4f& p1, const V4f& p2) {
        if (IsEqual(p1[3], 0.0f) || IsEqual(p2[3], 0.0f))
            pMax = V4f(INFINITY, INFINITY, INFINITY, 1.0f), pMin = V4f(-INFINITY, -INFINITY, -INFINITY, 1.0f);
        else
            pMin = Min(p1, p2),
            pMax = Max(p1, p2);
    }
    // * methods
    // Union
    Bbox Union(const V4f& p) {
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
    bool Inside(const V4f& pt) const {
        for (int i = 0; i < 3; i++)
            if (!(pt[i] >= pMin[i] && pt[i] <= pMax[i]))
                return false;
        return true;
    }
    // MaximumExtent: get the maximum extent of the bounding box
    int MaximumExtent() const {
        auto diag = pMax - pMin;
        return MaxAxis4(diag);
    }
    // operator[]: 0 for pMin, 1 for pMax
    V4f operator[](int i) const {
        return i == 0 ? pMin : pMax;
    }
};
#endif