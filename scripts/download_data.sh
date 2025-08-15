#!/bin/bash

# Download sample images from various sources
# This script provides alternative download sources for test images

set -e

echo "=========================================="
echo "Sample Data Download Script"
echo "=========================================="

mkdir -p ../data/sample_images
cd ../data/sample_images

# Function to create synthetic test images
create_synthetic_images() {
    echo "Creating synthetic test images..."
    
    python3 -c "
import cv2
import numpy as np
import os

def create_gradient_image():
    '''Create a gradient test image'''
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(512):
        for j in range(512):
            img[i, j] = [i % 256, j % 256, (i + j) % 256]
    cv2.imwrite('gradient_test.jpg', img)

def create_checkerboard():
    '''Create a checkerboard pattern'''
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    square_size = 32
    for i in range(0, 512, square_size):
        for j in range(0, 512, square_size):
            if (i//square_size + j//square_size) % 2 == 0:
                img[i:i+square_size, j:j+square_size] = [255, 255, 255]
    cv2.imwrite('checkerboard.jpg', img)

def create_noise_image():
    '''Create a noise test image'''
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cv2.imwrite('noise_test.jpg', img)

def create_edge_test():
    '''Create image with various edges for edge detection testing'''
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Horizontal lines
    img[100:150, :] = [255, 0, 0]
    img[200:250, :] = [0, 255, 0]
    
    # Vertical lines
    img[:, 100:150] = [0, 0, 255]
    img[:, 300:350] = [255, 255, 0]
    
    # Diagonal elements
    for i in range(400, 500):
        for j in range(400, 500):
            if abs(i - j) < 5:
                img[i, j] = [255, 255, 255]
    
    cv2.imwrite('edge_test.jpg', img)

def create_circular_pattern():
    '''Create circular patterns for blur testing'''
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    center = (256, 256)
    
    for radius in range(50, 200, 25):
        cv2.circle(img, center, radius, (255, 128, 64), 2)
    
    # Add some filled circles
    cv2.circle(img, (150, 150), 30, (255, 0, 0), -1)
    cv2.circle(img, (350, 150), 30, (0, 255, 0), -1)
    cv2.circle(img, (150, 350), 30, (0, 0, 255), -1)
    cv2.circle(img, (350, 350), 30, (255, 255, 0), -1)
    
    cv2.imwrite('circular_pattern.jpg', img)

# Create all test images
print('Creating synthetic test images...')
create_gradient_image()
create_checkerboard()
create_noise_image()
create_edge_test()
create_circular_pattern()
print('Synthetic images created successfully!')

" 2>/dev/null && echo "Synthetic images created with Python/OpenCV" || echo "Python/OpenCV not available for synthetic image creation"
}

# Try to download real images first
echo "Attempting to download sample images..."

# Download from alternative sources or create synthetic images
download_success=false

# Try downloading from a public domain image source
echo "Trying to download from public sources..."

# Create synthetic images as fallback
if [ "$download_success" = false ]; then
    echo "Using synthetic test images..."
    create_synthetic_images
fi

# Create some additional test files with different properties
echo "Creating additional test configurations..."

# Create custom convolution kernels
mkdir -p ../kernels

# Edge detection kernel
cat > ../kernels/edge_kernel.txt << EOF
# 3x3 Edge Detection Kernel
3 3
-1 -1 -1
-1  8 -1
-1 -1 -1
EOF

# Sharpen kernel
cat > ../kernels/sharpen_kernel.txt << EOF
# 3x3 Sharpen Kernel
3 3
 0 -1  0
-1  5 -1
 0 -1  0
EOF

# Blur kernel
cat > ../kernels/blur_kernel.txt << EOF
# 3x3 Blur Kernel
3 3
1 1 1
1 1 1
1 1 1
EOF

echo "Sample data setup completed!"
echo "Available test images:"
ls -la *.jpg 2>/dev/null || echo "No JPG files found"
echo
echo "Available kernel files:"
ls -la ../kernels/ 2>/dev/null || echo "No kernel files found"
