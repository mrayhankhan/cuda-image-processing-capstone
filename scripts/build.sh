#!/bin/bash

# CUDA Image Processor Build Script
# This script builds the CUDA image processing application

set -e

echo "=========================================="
echo "CUDA Image Processor Build Script"
echo "=========================================="

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA compiler (nvcc) not found!"
    echo "Please install NVIDIA CUDA Toolkit"
    exit 1
fi

# Check if CMake is available
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found!"
    echo "Please install CMake version 3.10 or higher"
    exit 1
fi

# Display CUDA version
echo "CUDA Compiler Version:"
nvcc --version | grep "release"
echo

# Display CMake version
echo "CMake Version:"
cmake --version | head -1
echo

# Check for OpenCV
if pkg-config --exists opencv4; then
    echo "OpenCV Version:"
    pkg-config --modversion opencv4
elif pkg-config --exists opencv; then
    echo "OpenCV Version:"
    pkg-config --modversion opencv
else
    echo "Warning: OpenCV not found via pkg-config"
    echo "Make sure OpenCV is properly installed"
fi
echo

# Create build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake ..

# Build the project
echo "Building project..."
make -j$(nproc)

# Check if build was successful
if [ -f "bin/cuda_image_processor" ]; then
    echo
    echo "=========================================="
    echo "Build completed successfully!"
    echo "Executable: $(pwd)/bin/cuda_image_processor"
    echo "File size: $(du -h bin/cuda_image_processor | cut -f1)"
    echo "=========================================="
    
    # Try to get GPU info if available
    if command -v nvidia-smi &> /dev/null; then
        echo
        echo "Available GPU(s):"
        nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
    else
        echo
        echo "Note: nvidia-smi not available - GPU runtime detection will be done at execution time"
    fi
else
    echo
    echo "=========================================="
    echo "Build failed! Executable not found."
    echo "=========================================="
    exit 1
fi
