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
               "../log/dae.pt");
    // Window win(IMAGE_WIDTH, IMAGE_HEIGHT, 1,
    //            "../datasets/dragon.obj");
    // Window win(IMAGE_WIDTH, IMAGE_HEIGHT, 1,
    //    "../datasets/armadillo.obj");
    // Window win(IMAGE_WIDTH, IMAGE_HEIGHT, 1,
    //            "../datasets/tyra.obj");
    // Window win(IMAGE_WIDTH, IMAGE_HEIGHT, 1,
    //            "../datasets/armadillo.obj",
    //            "",
    //            "../log/dncnn10.pt");
    win.start();
    // win.renderSingleFrame("/home/illusionary/文档/计算机图形学/Rendering/I_render/images/denoise.png");
    // win.renderMultipleFrame("/home/illusionary/文档/计算机图形学/Rendering/I_render/datasets/local/noise");
    // win.renderMultipleFrame("/home/illusionary/文档/计算机图形学/Rendering/I_render/datasets/local/clean");

    // win.renderMultipleFrame("/home/illusionary/文档/计算机图形学/Rendering/I_render/log/noise/armadillo");
    // win.renderMultipleFrame("/home/illusionary/文档/计算机图形学/Rendering/I_render/log/clean/armadillo");
    // win.renderMultipleFrame("/home/illusionary/文档/计算机图形学/Rendering/I_render/log/median/armadillo");
    // win.renderMultipleFrame("/home/illusionary/文档/计算机图形学/Rendering/I_render/log/gaussian/armadillo");
    // win.renderMultipleFrame("/home/illusionary/文档/计算机图形学/Rendering/I_render/log/biliteral/armadillo");
    // win.renderMultipleFrame("/home/illusionary/文档/计算机图形学/Rendering/I_render/log/dae/armadillo");
    // win.renderMultipleFrame("/home/illusionary/文档/计算机图形学/Rendering/I_render/log/dncnn5/armadillo");
    // win.renderMultipleFrame("/home/illusionary/文档/计算机图形学/Rendering/I_render/log/dncnn10/armadillo");

    return 0;
}