cmake_minimum_required(VERSION 3.10)

# task name
project(I_render)

# C++ standard
set(CMAKE_CXX_STANDARD 14)

# All src_code
file(GLOB SOURCES "src/*.cpp")

# All headers
include_directories("include")

# build executable
add_executable(I_render ${SOURCES})
