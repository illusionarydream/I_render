// main.cpp
#include <iostream>
#include "dataload.cuh"
#include "window.cuh"

int main() {
    Window win(IMAGE_WIDTH, IMAGE_HEIGHT, 0, "/home/illusionary/文档/计算机图形学/I_render/datasets/tyra.obj");
    win.start();

    return 0;
}