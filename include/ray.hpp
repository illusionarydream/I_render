#ifndef RAY_HPP
#define RAY_HPP

#include "point.hpp"
#include "vector.hpp"

// feature:
// 1. direction vector is normalized
// 2. it should be intersection with origin point
class Ray {
   public:
    Point o;
    Vector d;
    float tMax, tMin;
    float time;  // time
    int depth;   // bounce depth
    // constructors
    Ray() {
        o = Point();
        d = Vector().Normalized();
        tMin = MIN_FLOAT;
        tMax = INFINITY;
        time = 0;
        depth = 0;
    }
    Ray(const Point& _o, const Vector& _d, float _tMin = MIN_FLOAT, float _tMax = INFINITY,
        float _time = 0, int _depth = 0)
        : o(_o), d(_d.Normalized()), tMax(_tMax), tMin(_tMin), time(_time), depth(_depth) {}
    // methods
    Point operator()(float t) const {
        if (IsBounds(t, tMin, tMax)) return o + d * t;
        return (t > tMax) ? o + d * tMax : o + d * tMin;
    }
};

#endif  // RAY_HPP