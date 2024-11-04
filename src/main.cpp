// main.cpp
#include <iostream>
#include "mesh.hpp"

int main() {
    Mesh mesh("/home/illusionary/文档/计算机图形学/I_render/datasets/stanford-bunny.obj");

    mesh.print();

    return 0;
}