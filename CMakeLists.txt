cmake_minimum_required(VERSION 3.10)

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(MyProject ${CUDA_LIBRARIES})

# task name
project(MyProject)

# C++ standard
set(CMAKE_CXX_STANDARD 14)

# All src_code
file(GLOB SOURCES "src/*.cpp")

# All headers
include_directories("include")

# build executable
add_executable(MyProject ${SOURCES})
