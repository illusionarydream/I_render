cmake_minimum_required(VERSION 3.10)

project(I_render CXX CUDA)

enable_language(CUDA)

# Set C++ standard to C++17 (required by libtorch)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(GLFW3_DIR "/usr/lib/cmake/glfw3")
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(OpenCV REQUIRED)

# Torch paths
set(Torch_DIR "/home/illusionary/软件/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# Include dirs
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    "include"
    "eigen-3.4.0"
    "cuda"
    /opt/cuda/targets/x86_64-linux/include
)

include_directories(
    "/home/illusionary/软件/libtorch/include"
    "/home/illusionary/软件/libtorch/include/torch/csrc/api/include"
)

# Sources
file(GLOB SOURCES "src/*.cpp" "src/*.cu")

# Executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Compile options for CUDA
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--gpu-code=sm_86>
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=compute_86>
)

# Link libs
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${OPENGL_LIBRARIES}
    GLEW::GLEW
    glfw
    glm::glm
    "${TORCH_LIBRARIES}"
)

# Prevent static linking error (optional)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)