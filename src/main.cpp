// main.cpp
#include <iostream>
#include "dataload.cuh"
#include "window.cuh"

int main() {
    Window win(800, 800, "/home/illusionary/文档/计算机图形学/I_render/datasets/bunny.obj");
    win.start();

    return 0;
}