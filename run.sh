#!/bin/bash

# CUDA Image Processor Execution Script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}CUDA Image Processing at Scale - Execution Script${NC}"
echo "=================================================="

# Check if executable exists
EXECUTABLE="./bin/cuda_image_processor"
if [ ! -f "$EXECUTABLE" ]; then
    if [ -f "./build/bin/cuda_image_processor" ]; then
        EXECUTABLE="./build/bin/cuda_image_processor"
    elif [ -f "./cuda_image_processor" ]; then
        EXECUTABLE="./cuda_image_processor"
    else
        echo -e "${RED}Error: Executable not found. Please build the project first.${NC}"
        echo "Run: make clean && make"
        echo "Or:  mkdir build && cd build && cmake .. && make"
        exit 1
    fi
fi

# Create sample images directory if it doesn't exist
if [ ! -d "sample_images" ]; then
    echo -e "${YELLOW}Creating sample_images directory...${NC}"
    mkdir -p sample_images
fi

# Check if sample images exist
IMAGE_COUNT=$(find sample_images -name "*.jpg" -o -name "*.png" -o -name "*.bmp" | wc -l)
if [ $IMAGE_COUNT -eq 0 ]; then
    echo -e "${YELLOW}No sample images found. Generating test images...${NC}"
    python3 -c "
import cv2
import numpy as np
import os

# Create sample_images directory
os.makedirs('sample_images', exist_ok=True)

# Generate test images
for i in range(20):
    # Create random noise image
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    
    # Add some patterns
    cv2.circle(img, (256, 256), 100, (255, 255, 255), -1)
    cv2.rectangle(img, (100, 100), (400, 400), (0, 255, 0), 5)
    
    # Add text
    cv2.putText(img, f'Test {i+1}', (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    
    # Save image
    cv2.imwrite(f'sample_images/test_{i+1:03d}.jpg', img)

print(f'Generated 20 test images in sample_images/')
" 2>/dev/null || echo -e "${RED}Python3/OpenCV not available for generating test images. Please add images manually to sample_images/ directory.${NC}"
fi

# Create output directory
mkdir -p output

echo -e "${GREEN}Starting image processing demonstrations...${NC}"
echo ""

# Test 1: Gaussian Blur
echo -e "${BLUE}Test 1: Gaussian Blur Filter${NC}"
echo "Command: $EXECUTABLE --filter gaussian --kernel_size 15 --batch_size 16"
$EXECUTABLE --filter gaussian --kernel_size 15 --batch_size 16
echo ""

# Test 2: Sobel Edge Detection
echo -e "${BLUE}Test 2: Sobel Edge Detection${NC}"
echo "Command: $EXECUTABLE --filter sobel --batch_size 16"
$EXECUTABLE --filter sobel --batch_size 16
echo ""

# Test 3: Brightness and Contrast
echo -e "${BLUE}Test 3: Brightness and Contrast Adjustment${NC}"
echo "Command: $EXECUTABLE --filter brightness --brightness 30 --contrast 1.5 --batch_size 16"
$EXECUTABLE --filter brightness --brightness 30 --contrast 1.5 --batch_size 16
echo ""

# Test 4: All filters
echo -e "${BLUE}Test 4: All Filters${NC}"
echo "Command: $EXECUTABLE --filter all --batch_size 8"
$EXECUTABLE --filter all --batch_size 8
echo ""

# Performance test with larger batch
if [ $IMAGE_COUNT -gt 10 ]; then
    echo -e "${BLUE}Performance Test: Large Batch Processing${NC}"
    echo "Command: $EXECUTABLE --filter gaussian --batch_size 32"
    $EXECUTABLE --filter gaussian --batch_size 32
    echo ""
fi

echo -e "${GREEN}All tests completed successfully!${NC}"
echo "Check the 'output' directory for processed images."
echo ""

# Display results summary
echo -e "${YELLOW}Results Summary:${NC}"
OUTPUT_COUNT=$(find output -name "*.jpg" -o -name "*.png" -o -name "*.bmp" | wc -l)
echo "Input images: $IMAGE_COUNT"
echo "Output images: $OUTPUT_COUNT"
echo "Output directory: ./output/"

# List some output files
echo ""
echo "Sample output files:"
ls -la output/ | head -10
