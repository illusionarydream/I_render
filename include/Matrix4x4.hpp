#ifndef MATRIX4X4_HPP
#define MATRIX4X4_HPP

#include <math_material.hpp>
#include <iostream>

// class Matrix4x4
// 1. Matrix4x4 initialization as default is identity matrix
// 2. Matrix4x4 only support multiplication with another Matrix4x4
class Matrix4x4 {
   private:
    // *data
    float m[4][4];

   public:
    // * constructor
    Matrix4x4() {
        m[0][0] = 1;
        m[0][1] = 0;
        m[0][2] = 0;
        m[0][3] = 0;
        m[1][0] = 0;
        m[1][1] = 1;
        m[1][2] = 0;
        m[1][3] = 0;
        m[2][0] = 0;
        m[2][1] = 0;
        m[2][2] = 1;
        m[2][3] = 0;
        m[3][0] = 0;
        m[3][1] = 0;
        m[3][2] = 0;
        m[3][3] = 1;
    }
    Matrix4x4(float mat[4][4]) {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                m[i][j] = mat[i][j];
    }
    Matrix4x4(float t00, float t01, float t02, float t03, float t10, float t11, float t12, float t13, float t20, float t21, float t22, float t23, float t30, float t31, float t32, float t33) {
        m[0][0] = t00;
        m[0][1] = t01;
        m[0][2] = t02;
        m[0][3] = t03;
        m[1][0] = t10;
        m[1][1] = t11;
        m[1][2] = t12;
        m[1][3] = t13;
        m[2][0] = t20;
        m[2][1] = t21;
        m[2][2] = t22;
        m[2][3] = t23;
        m[3][0] = t30;
        m[3][1] = t31;
        m[3][2] = t32;
        m[3][3] = t33;
    }
    // * operator
    // _* operator
    Matrix4x4 operator*(const Matrix4x4& mat) const {
        Matrix4x4 res;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                res.m[i][j] = m[i][0] * mat.m[0][j] + m[i][1] * mat.m[1][j] + m[i][2] * mat.m[2][j] + m[i][3] * mat.m[3][j];
        return res;
    }
    Matrix4x4& operator*=(const Matrix4x4& mat) {
        *this = *this * mat;
        return *this;
    }
    // index operator
    inline float operator()(int i, int j) const {
        return m[i][j];
    }
    inline float& operator()(int i, int j) {
        return m[i][j];
    }
    // == operator
    bool operator==(const Matrix4x4& mat) const {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                if (m[i][j] != mat.m[i][j])
                    return false;
        return true;
    }
    // _!= operator
    bool operator!=(const Matrix4x4& mat) const {
        return !(*this == mat);
    }
    // << operator
    friend std::ostream& operator<<(std::ostream& os, const Matrix4x4& mat) {
        os << "[";
        for (int i = 0; i < 4; i++) {
            os << "[";
            for (int j = 0; j < 4; j++) {
                os << mat.m[i][j];
                if (j != 3)
                    os << ", ";
            }
            os << "]";
            if (i != 3)
                os << ", ";
        }
        os << "]";
        return os;
    }
    // * methods
    // Transpose
    Matrix4x4 Transpose() const {
        Matrix4x4 res;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                res.m[i][j] = m[j][i];
        return res;
    }
    // Determinant
    float Determinant() const {
        double det = 0;
        for (int i = 0; i < 4; i++)
            det += m[0][i] * m[1][(i + 1) % 4] * m[2][(i + 2) % 4] * m[3][(i + 3) % 4];
        for (int i = 0; i < 4; i++)
            det -= m[0][i] * m[1][(i + 3) % 4] * m[2][(i + 2) % 4] * m[3][(i + 1) % 4];
        return det;
    }
    // Inverse: Gauss-Jordan Elimination
    Matrix4x4 Inverse() const {
        // If det is 0, return identity matrix
        if (IsZero(Determinant()))
            return Matrix4x4();
        Matrix4x4 res;
        float mat[4][8];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                mat[i][j] = m[i][j];
        for (int i = 0; i < 4; i++)
            for (int j = 4; j < 8; j++)
                mat[i][j] = (i == j - 4) ? 1 : 0;
        for (int i = 0; i < 4; i++) {
            int pivot = i;
            for (int j = i + 1; j < 4; j++)
                if (fabs(mat[j][i]) > fabs(mat[pivot][i]))
                    pivot = j;
            for (int j = 0; j < 8; j++)
                std::swap(mat[i][j], mat[pivot][j]);
            float div = mat[i][i];
            for (int j = 0; j < 8; j++)
                mat[i][j] /= div;
            for (int j = 0; j < 4; j++)
                if (i != j) {
                    float mul = mat[j][i];
                    for (int k = 0; k < 8; k++)
                        mat[j][k] -= mul * mat[i][k];
                }
        }
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                res.m[i][j] = mat[i][j + 4];
        return res;
    }
};

#endif