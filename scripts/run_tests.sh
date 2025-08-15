#!/bin/bash

# CUDA Image Processor Test Script
# This script downloads sample images and runs comprehensive tests

set -e

echo "=========================================="
echo "CUDA Image Processor Test Suite"
echo "=========================================="

# Create data directory
mkdir -p ../data/sample_images
cd ../data/sample_images

# Download sample images from USC SIPI database
echo "Downloading sample test images..."

# Download Lena image (classic computer vision test image)
if [ ! -f "lena.jpg" ]; then
    echo "Downloading Lena image..."
    curl -o lena.tiff "https://sipi.usc.edu/database/download.php?vol=misc&img=4.2.04"
    # Convert to JPG using ImageMagick if available, otherwise use OpenCV tools
    if command -v convert &> /dev/null; then
        convert lena.tiff lena.jpg
        rm lena.tiff
    else
        mv lena.tiff lena.jpg
    fi
fi

# Download Baboon image
if [ ! -f "baboon.jpg" ]; then
    echo "Downloading Baboon image..."
    curl -o baboon.tiff "https://sipi.usc.edu/database/download.php?vol=misc&img=4.2.03"
    if command -v convert &> /dev/null; then
        convert baboon.tiff baboon.jpg
        rm baboon.tiff
    else
        mv baboon.tiff baboon.jpg
    fi
fi

# Create a simple test pattern if downloads fail
if [ ! -f "lena.jpg" ] && [ ! -f "baboon.jpg" ]; then
    echo "Warning: Could not download images. Creating test pattern..."
    python3 -c "
import cv2
import numpy as np

# Create a test pattern
img = np.zeros((512, 512, 3), dtype=np.uint8)
for i in range(0, 512, 64):
    for j in range(0, 512, 64):
        if (i//64 + j//64) % 2 == 0:
            img[i:i+64, j:j+64] = [255, 255, 255]
        else:
            img[i:i+64, j:j+64] = [0, 0, 0]

cv2.imwrite('test_pattern.jpg', img)
print('Created test pattern image')
" 2>/dev/null || echo "Could not create test pattern - Python/OpenCV not available"
fi

# Go back to build directory
cd ../../build

# Check if executable exists
if [ ! -f "bin/cuda_image_processor" ]; then
    echo "Error: Executable not found! Please run build.sh first."
    exit 1
fi

# Create results directory
mkdir -p ../results

# Run tests with available images
echo
echo "Running image processing tests..."

# Test with each available image
for img in ../data/sample_images/*.jpg; do
    if [ -f "$img" ]; then
        filename=$(basename "$img" .jpg)
        echo
        echo "Processing: $filename"
        echo "----------------------------------------"
        
        # Run with all filters and benchmarking
        ./bin/cuda_image_processor "$img" ../results/ \
            --blur --edge --emboss \
            --kernel-size 9 --sigma 2.0 \
            --benchmark --iterations 10 \
            --verbose
        
        echo "Results saved to ../results/${filename}_*"
    fi
done

# Summary
echo
echo "=========================================="
echo "Test Suite Completed!"
echo "=========================================="
echo "Input images processed:"
ls -la ../data/sample_images/
echo
echo "Output images generated:"
ls -la ../results/
echo
echo "Test completed successfully!"
