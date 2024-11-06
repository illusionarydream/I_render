#ifndef MATH_MATERIALS_CUH
#define MATH_MATERIALS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <cmath>
#include "global.cuh"
#include "error.hpp"
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#define MAX 1e6
#define MAX_mesh 10000
#define MAX_bound 20
#define MIN_surface 1e-4

class V3f {
   public:
    // * member variable
    float data[3];

    // * constructor
    __host__ __device__ V3f() {};
    __host__ __device__ V3f(float x, float y, float z) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }
    __host__ __device__ V3f(const V3f &v) {
        data[0] = v[0];
        data[1] = v[1];
        data[2] = v[2];
    }

    // * operator overloading
    __host__ __device__ float &operator[](int i) { return data[i]; }
    __host__ __device__ float operator[](int i) const { return data[i]; }
};

class M3f {
   public:
    // * member variable
    float data[3][3];

    // * constructor
    __host__ __device__ M3f() {}
    __host__ __device__ M3f(const float &v00, const float &v01, const float &v02,
                            const float &v10, const float &v11, const float &v12,
                            const float &v20, const float &v21, const float &v22) {
        data[0][0] = v00;
        data[0][1] = v01;
        data[0][2] = v02;
        data[1][0] = v10;
        data[1][1] = v11;
        data[1][2] = v12;
        data[2][0] = v20;
        data[2][1] = v21;
        data[2][2] = v22;
    }
    __host__ __device__ M3f(float *d) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                data[i][j] = d[i * 3 + j];
            }
        }
    }
    __host__ __device__ M3f(const M3f &m) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                data[i][j] = m.data[i][j];
            }
        }
    }

    // * operator overloading
    __host__ __device__ M3f operator*(const M3f &m) const {
        M3f res;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                res.data[i][j] = 0;
                for (int k = 0; k < 3; k++) {
                    res.data[i][j] += data[i][k] * m.data[k][j];
                }
            }
        }
        return res;
    }
    __host__ __device__ V3f operator*(const V3f &v) const {
        V3f res;
        for (int i = 0; i < 3; i++) {
            res[i] = 0;
            for (int j = 0; j < 3; j++) {
                res[i] += data[i][j] * v[j];
            }
        }
        return res;
    }
    __host__ __device__ V3f col(int i) const {
        return V3f(data[0][i], data[1][i], data[2][i]);
    }

    // * member function
    __host__ __device__ void reset(const float &v00, const float &v01, const float &v02,
                                   const float &v10, const float &v11, const float &v12,
                                   const float &v20, const float &v21, const float &v22) {
        data[0][0] = v00;
        data[0][1] = v01;
        data[0][2] = v02;
        data[1][0] = v10;
        data[1][1] = v11;
        data[1][2] = v12;
        data[2][0] = v20;
        data[2][1] = v21;
        data[2][2] = v22;
    }
};

class V4f {
   public:
    // * member variable
    float data[4];

    // * constructor
    __host__ __device__ V4f() {}
    __host__ __device__ V4f(float x, float y, float z, float w) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
        data[3] = w;
    }
    __host__ __device__ V4f(const V4f &v) {
        data[0] = v[0];
        data[1] = v[1];
        data[2] = v[2];
        data[3] = v[3];
    }

    // * operator overloading
    __host__ __device__ float &operator[](int i) { return data[i]; }
    __host__ __device__ float operator[](int i) const { return data[i]; }

    // * member function
    __host__ __device__ V3f toV3f() const { return V3f(data[0], data[1], data[2]); }
};

class M4f {
   public:
    // * member variable
    float data[4][4];

    // * constructor
    __host__ __device__ M4f() {}
    __host__ __device__ M4f(const float &v00, const float &v01, const float &v02, const float &v03,
                            const float &v10, const float &v11, const float &v12, const float &v13,
                            const float &v20, const float &v21, const float &v22, const float &v23,
                            const float &v30, const float &v31, const float &v32, const float &v33) {
        data[0][0] = v00;
        data[0][1] = v01;
        data[0][2] = v02;
        data[0][3] = v03;
        data[1][0] = v10;
        data[1][1] = v11;
        data[1][2] = v12;
        data[1][3] = v13;
        data[2][0] = v20;
        data[2][1] = v21;
        data[2][2] = v22;
        data[2][3] = v23;
        data[3][0] = v30;
        data[3][1] = v31;
        data[3][2] = v32;
        data[3][3] = v33;
    }
    __host__ __device__ M4f(float *d) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                data[i][j] = d[i * 4 + j];
            }
        }
    }
    __host__ __device__ M4f(const M4f &m) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                data[i][j] = m.data[i][j];
            }
        }
    }

    // * operator overloading
    __host__ __device__ M4f operator*(const M4f &m) const {
        M4f res;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                res.data[i][j] = 0;
                for (int k = 0; k < 4; k++) {
                    res.data[i][j] += data[i][k] * m.data[k][j];
                }
            }
        }
        return res;
    }
    __host__ __device__ V4f operator*(const V4f &v) const {
        V4f res;
        for (int i = 0; i < 4; i++) {
            res[i] = 0;
            for (int j = 0; j < 4; j++) {
                res[i] += data[i][j] * v[j];
            }
        }
        return res;
    }
    __host__ __device__ V4f col(int i) const {
        return V4f(data[0][i], data[1][i], data[2][i], data[3][i]);
    }

    // * member function
    __host__ __device__ void reset(const float &v00, const float &v01, const float &v02, const float &v03,
                                   const float &v10, const float &v11, const float &v12, const float &v13,
                                   const float &v20, const float &v21, const float &v22, const float &v23,
                                   const float &v30, const float &v31, const float &v32, const float &v33) {
        data[0][0] = v00;
        data[0][1] = v01;
        data[0][2] = v02;
        data[0][3] = v03;
        data[1][0] = v10;
        data[1][1] = v11;
        data[1][2] = v12;
        data[1][3] = v13;
        data[2][0] = v20;
        data[2][1] = v21;
        data[2][2] = v22;
        data[2][3] = v23;
        data[3][0] = v30;
        data[3][1] = v31;
        data[3][2] = v32;
        data[3][3] = v33;
    }
};

// toV4f
__forceinline__ __host__ __device__ V4f toV4f(const V3f &v, float eps = 0.0f) {
    return V4f(v[0], v[1], v[2], eps);
}

// cross product
__forceinline__ __host__ __device__ V3f cross(const V3f &a, const V3f &b) {
    return V3f(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}
__forceinline__ __host__ __device__ V4f cross(const V4f &a, const V4f &b) {
    return V4f(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0], 0);
}

// element *
__forceinline__ __host__ __device__ V3f operator*(const V3f &a, const V3f &b) {
    return V3f(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}
__forceinline__ __host__ __device__ V4f operator*(const V4f &a, const V4f &b) {
    return V4f(a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]);
}

// dot product
__forceinline__ __host__ __device__ float dot(const V3f &a, const V3f &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
__forceinline__ __host__ __device__ float dot(const V4f &a, const V4f &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

// length
__forceinline__ __host__ __device__ float length(const V3f &v) {
    return sqrtf(dot(v, v));
}
__forceinline__ __host__ __device__ float length(const V4f &v) {
    return sqrtf(dot(v, v));
}

// operator /
__forceinline__ __host__ __device__ V3f operator/(const V3f &v, float s) {
    return V3f(v[0] / s, v[1] / s, v[2] / s);
}
__forceinline__ __host__ __device__ V4f operator/(const V4f &v, float s) {
    return V4f(v[0] / s, v[1] / s, v[2] / s, v[3] / s);
}

// operator *
__forceinline__ __host__ __device__ V3f operator*(const V3f &v, float s) {
    return V3f(v[0] * s, v[1] * s, v[2] * s);
}
__forceinline__ __host__ __device__ V4f operator*(const V4f &v, float s) {
    return V4f(v[0] * s, v[1] * s, v[2] * s, v[3] * s);
}
__forceinline__ __host__ __device__ V3f operator*(float s, const V3f &v) {
    return V3f(v[0] * s, v[1] * s, v[2] * s);
}
__forceinline__ __host__ __device__ V4f operator*(float s, const V4f &v) {
    return V4f(v[0] * s, v[1] * s, v[2] * s, v[3] * s);
}

// operator +
__forceinline__ __host__ __device__ V3f operator+(const V3f &a, const V3f &b) {
    return V3f(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
__forceinline__ __host__ __device__ V4f operator+(const V4f &a, const V4f &b) {
    return V4f(a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]);
}

// operator -
__forceinline__ __host__ __device__ V3f operator-(const V3f &a, const V3f &b) {
    return V3f(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
__forceinline__ __host__ __device__ V4f operator-(const V4f &a, const V4f &b) {
    return V4f(a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]);
}

// normalize
__forceinline__ __host__ __device__ V3f normalize(const V3f &v) {
    return v / length(v);
}
__forceinline__ __host__ __device__ V4f normalize(const V4f &v) {
    return v / length(v);
}

// random get 0~1 float
__forceinline__ __device__ float random_float(curandState *state) {
    return curand_uniform(state);  // 返回0到1之间的浮点数
}

// random_in_unit_sphere V3f
__forceinline__ __device__ V3f random_in_unit_sphere_V3f(curandState *state) {
    V3f p;
    do {
        p = V3f(random_float(state), random_float(state), random_float(state)) * 2.0f - V3f(1.0f, 1.0f, 1.0f);
    } while (dot(p, p) >= 1.0f);
    return p;
}

// ramdom_in_unit_sphere V4f
__forceinline__ __device__ V4f random_in_unit_sphere_V4f(curandState *state) {
    V4f p;
    do {
        p = V4f(random_float(state), random_float(state), random_float(state), 0.0f) * 2.0f - V4f(1.0f, 1.0f, 1.0f, 0.0f);
    } while (dot(p, p) >= 1.0f);
    return p;
}

class Ray {
   public:
    // * member variable
    V4f orig, dir;

    // * constructor
    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const V4f &o, const V4f &d) : orig(o), dir(d) {}

    // * operator overloading
    __host__ __device__ V4f operator()(float t) const { return orig + dir * t; }
};

__forceinline__ __host__ __device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
#endif  // MATERIALS_CUH