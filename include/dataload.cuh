#ifndef DATA_LOAD_CUH
#define DATA_LOAD_CUH

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include "mesh.cuh"

// load_obj
std::vector<Triangle> load_obj(const std::string& filename);

#endif  // DATA_LOAD_CUH