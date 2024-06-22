#ifndef MATH_MATERIAL_HPP
#define MATH_MATERIAL_HPP

#include "error.hpp"

#define MIN_FLOAT 1.0e-5

bool IsZero(float f) { return std::abs(f) < MIN_FLOAT; }

bool IsEqual(float a, float b) { return IsZero(a - b); }

#endif