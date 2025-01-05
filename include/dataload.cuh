#ifndef DATA_LOAD_CUH
#define DATA_LOAD_CUH

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include "mesh.cuh"

// load_obj
std::vector<Triangle> load_obj(const std::string& filename, bool if_texture = false, V3f corner = V3f(0.0f, 0.0f, 0.0f), float scale = 1.0f);

// load_texture
std::vector<V3f> load_texture(const std::string& filename, int& width, int& height);
#endif  // DATA_LOAD_CUH