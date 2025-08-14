# CUDA Image Processing at Scale

## Project Overview

This project implements a high-performance CUDA-based image processing pipeline that can handle hundreds of images simultaneously. The application applies various image filters including Gaussian blur, edge detection (Sobel), brightness adjustment, and contrast enhancement using GPU acceleration.

## Features

- **Batch Processing**: Process multiple images in parallel using CUDA
- **Multiple Filters**: Gaussian blur, Sobel edge detection, brightness/contrast adjustment
- **Command Line Interface**: Flexible CLI for different processing modes
- **Performance Metrics**: Execution time measurement and throughput reporting
- **Multiple Image Formats**: Support for common image formats (JPEG, PNG, BMP)

## Requirements

- NVIDIA GPU with CUDA capability 3.0 or higher
- CUDA Toolkit 11.0 or later
- OpenCV 4.0 or later
- CMake 3.10 or later
- C++14 compatible compiler

## Building the Project

### Using CMake
```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Using Makefile (alternative)
```bash
make clean
make
```

## Usage

### Basic Usage
```bash
# Process images with Gaussian blur
./cuda_image_processor --input_dir ./sample_images --output_dir ./output --filter gaussian --kernel_size 15

# Apply Sobel edge detection
./cuda_image_processor --input_dir ./sample_images --output_dir ./output --filter sobel

# Adjust brightness and contrast
./cuda_image_processor --input_dir ./sample_images --output_dir ./output --filter brightness --brightness 50 --contrast 1.5

# Process with multiple filters
./cuda_image_processor --input_dir ./sample_images --output_dir ./output --filter all
```

### Command Line Arguments

- `--input_dir` : Directory containing input images
- `--output_dir` : Directory for processed images
- `--filter` : Filter type (gaussian, sobel, brightness, all)
- `--kernel_size` : Size of filter kernel (for Gaussian blur, default: 15)
- `--brightness` : Brightness adjustment value (-100 to 100, default: 0)
- `--contrast` : Contrast multiplier (0.5 to 3.0, default: 1.0)
- `--batch_size` : Number of images to process simultaneously (default: 32)
- `--help` : Display help information

### Sample Data

The project includes sample images in the `sample_images/` directory for testing. Additional test images can be downloaded from:
- USC SIPI Image Database
- Creative Commons images
- Any JPEG/PNG images

## Performance

The application is designed to process:
- **Small Images**: 100+ images (256x256 pixels) simultaneously
- **Large Images**: 10+ high-resolution images (1920x1080+) with efficient memory management
- **Throughput**: Typically 50-200 images/second depending on GPU and image size

## Implementation Details

### CUDA Kernels
- **Gaussian Blur**: Separable 2D convolution with shared memory optimization
- **Sobel Edge Detection**: Combined horizontal and vertical gradient computation
- **Brightness/Contrast**: Point-wise pixel operations with saturation handling

### Memory Management
- Efficient GPU memory allocation and deallocation
- Batch processing to maximize GPU utilization
- Asynchronous memory transfers using CUDA streams

### Error Handling
- Comprehensive CUDA error checking
- Input validation and graceful failure handling
- Memory leak prevention

## Project Structure

```
├── src/
│   ├── main.cpp              # Main application entry point
│   ├── image_processor.cu    # CUDA kernels and GPU processing
│   ├── image_loader.cpp      # Image I/O operations
│   └── utils.cpp             # Utility functions
├── include/
│   ├── image_processor.h     # Header for CUDA functions
│   ├── image_loader.h        # Header for image operations
│   └── utils.h               # Utility function headers
├── sample_images/            # Test images
├── CMakeLists.txt           # CMake build configuration
├── Makefile                 # Alternative build system
├── run.sh                   # Execution script
└── README.md               # This file
```

## Results and Performance Analysis

The application demonstrates significant speedup compared to CPU-only processing:
- **GPU Processing**: ~10-50x faster than CPU depending on image size and filter complexity
- **Memory Bandwidth**: Efficient utilization of GPU memory bandwidth
- **Scalability**: Linear performance scaling with number of images up to GPU memory limits

## License

This project is released under the MIT License.
