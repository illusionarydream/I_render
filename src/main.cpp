// main.cpp
#include <iostream>
#include "dataload.cuh"
#include "window.cuh"

int main() {
    // Window win(IMAGE_WIDTH, IMAGE_HEIGHT, 0,
    //            "../datasets/zebra.obj",
    //            "../datasets/texture/zebra-atlas.jpg");
    Window win(IMAGE_WIDTH, IMAGE_HEIGHT, 1,
               "../datasets/bunny_little.obj",
               "../datasets/texture/bunny_little.jpg",
               "../log/dncnn7.pt");
    // Window win(IMAGE_WIDTH, IMAGE_HEIGHT, 1,
    //            "../datasets/dragon.obj");
    // Window win(IMAGE_WIDTH, IMAGE_HEIGHT, 1,
    //    "../datasets/armadillo.obj");
    // Window win(IMAGE_WIDTH, IMAGE_HEIGHT, 1,
    //            "../datasets/tyra.obj");
    // Window win(IMAGE_WIDTH, IMAGE_HEIGHT, 1,
    //            "../datasets/zebra.obj");
    win.start();
    // win.renderSingleFrame("/home/illusionary/文档/计算机图形学/Rendering/I_render/images/denoise.png");
    // win.renderMultipleFrame("/home/illusionary/文档/计算机图形学/Rendering/I_render/datasets/local/noise");
    // win.renderMultipleFrame("/home/illusionary/文档/计算机图形学/Rendering/I_render/datasets/local/clean");

    // win.renderMultipleFrame("/home/illusionary/文档/计算机图形学/Rendering/I_render/log/dae/denoise_twobunny");
    // win.renderMultipleFrame("/home/illusionary/文档/计算机图形学/Rendering/I_render/log/dncnn5/denoise_twobunny");
    // win.renderMultipleFrame("/home/illusionary/文档/计算机图形学/Rendering/I_render/log/dncnn7/denoise_twobunny");

    return 0;
}