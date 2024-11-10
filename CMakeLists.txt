cmake_minimum_required(VERSION 3.10)

# task name
project(I_render CXX CUDA)

enable_language(CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

# All src_code
# Add the source files
file(GLOB SOURCES "src/*.cpp" "src/*.cu")



# All headers
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("include")
include_directories("eigen-3.4.0")
include_directories("cuda")
include_directories(/opt/cuda/targets/x86_64-linux/include)
link_directories(/opt/cuda/targets/x86_64-linux/lib)

# Add the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Add compile options for CUDA
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--gpu-code=sm_86>
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=compute_86>)
    

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${OPENGL_LIBRARIES} GLEW::GLEW glfw glm::glm)