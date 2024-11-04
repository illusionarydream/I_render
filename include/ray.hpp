#ifndef RAY_HPP
#define RAY_HPP

#include "math_material.hpp"

// feature:
// 1. direction vector is normalized
// 2. it should be intersection with origin point
class Ray {
   public:
    V4f orig;
    V4f dir;
    float tMax, tMin;
    float time;  // time
    int depth;   // bounce depth
    // constructors
    Ray() {
        orig = V4f(0.0f, 0.0f, 0.0f, 1.0f);  // Point
        dir = V4f(0.0f, 0.0f, -1.0f, 0.0f);  // Vector
        tMin = MIN_FLOAT;
        tMax = INFINITY;
        time = 0;
        depth = 0;
    }
    Ray(const V4f& o, const V4f& d, float tMin = MIN_FLOAT, float tMax = INFINITY, float time = 0, int depth = 0)
        : orig(o),
          dir(d),
          tMin(tMin),
          tMax(tMax),
          time(time),
          depth(depth) {
        if (tMin > tMax)
            std::swap(this->tMin, this->tMax);
        if (!IsEqual(dir[3], 0.0f))
            ASSERT(false, "The direction vector of the ray should be a vector, not a point.");
        if (IsEqual(orig[3], 0.0f))
            ASSERT(false, "The origin point of the ray should be a point, not a vector.");
        dir.normalize();
    }

    // * member functions
    V4f operator()(float t) const { return orig + t * dir; }
};

#endif  // RAY_HPP