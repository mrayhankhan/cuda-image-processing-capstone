cmake_minimum_required(VERSION 3.10)
project(cuda_image_processor LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architectures (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/main.cpp
    src/image_loader.cpp
    src/utils.cpp
    src/image_processor.cu
)

# Create executable
add_executable(cuda_image_processor ${SOURCES})

# Set CUDA properties
set_property(TARGET cuda_image_processor PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cuda_image_processor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Link libraries
target_link_libraries(cuda_image_processor 
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)

# Compiler-specific options
target_compile_options(cuda_image_processor PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_80,code=sm_80>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_86,code=sm_86>
)

# Set output directory
set_target_properties(cuda_image_processor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Installation
install(TARGETS cuda_image_processor DESTINATION bin)
