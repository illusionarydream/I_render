// main.cpp
#include <iostream>
#include "dataload.cuh"
#include "window.cuh"

int main() {
    // Window win(IMAGE_WIDTH, IMAGE_HEIGHT, 1,
    //            "../datasets/zebra.obj",
    //            "../datasets/texture/zebra-atlas.jpg");
    Window win(IMAGE_WIDTH, IMAGE_HEIGHT, 1,
               "../datasets/bunny_little.obj");
    // Window win(IMAGE_WIDTH, IMAGE_HEIGHT, 0,
    //    "../datasets/bunny_little.obj");
    win.start();

    return 0;
}