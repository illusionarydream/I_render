// main.cpp
#include <iostream>
#include "Matrix4x4.hpp"

int main() {
    // Test default constructor
    Matrix4x4 mat1;
    std::cout << "Default constructor: " << std::endl
              << mat1 << std::endl;

    // Test constructor with array
    float arr[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
    Matrix4x4 mat2(arr);
    std::cout << "Constructor with array: " << std::endl
              << mat2 << std::endl;

    // Test constructor with individual elements
    Matrix4x4 mat3(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    std::cout << "Constructor with individual elements: " << std::endl
              << mat3 << std::endl;

    // Test multiplication
    Matrix4x4 mat4 = mat2 * mat3;
    std::cout << "Multiplication: " << std::endl
              << mat4 << std::endl;

    // Test transpose
    Matrix4x4 mat5 = mat4.Transpose();
    std::cout << "Transpose: " << std::endl
              << mat5 << std::endl;

    // Test inverse
    mat5(0, 0) = 1, mat5(0, 1) = 2, mat5(0, 2) = 3, mat5(0, 3) = 4;
    mat5(1, 0) = 2, mat5(1, 1) = 3, mat5(1, 2) = 1, mat5(1, 3) = 2;
    mat5(2, 0) = 1, mat5(2, 1) = 1, mat5(2, 2) = 1, mat5(2, 3) = -1;
    mat5(3, 0) = 1, mat5(3, 1) = 0, mat5(3, 2) = -2, mat5(3, 3) = -6;
    Matrix4x4 mat6 = mat5.Inverse();
    std::cout << "Determinant: " << mat5.Determinant() << std::endl;
    std::cout << "Inverse: " << std::endl
              << mat6 << std::endl;

    return 0;
}