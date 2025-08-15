#!/bin/bash

# Simple test image creation using ImageMagick (if available)
# This creates basic test images for the CUDA image processor

echo "Creating simple test images..."

cd ../data/sample_images

# Try to create images with ImageMagick
if command -v convert &> /dev/null; then
    echo "Using ImageMagick to create test images..."
    
    # Create a gradient image
    convert -size 512x512 gradient:black-white gradient_test.jpg
    
    # Create a checkerboard pattern
    convert -size 512x512 pattern:checkerboard checkerboard.jpg
    
    # Create a simple noise pattern
    convert -size 512x512 xc: +noise Random noise_test.jpg
    
    echo "Test images created with ImageMagick"
    
elif command -v wget &> /dev/null; then
    echo "Trying to download a sample image..."
    # Try to download a Creative Commons image
    wget -O sample_image.jpg "https://picsum.photos/512/512" 2>/dev/null || echo "Download failed"
    
else
    echo "Creating a simple PPM test image manually..."
    
    # Create a simple PPM file and convert it
    cat > test_pattern.ppm << EOF
P3
512 512
255
EOF
    
    # Generate simple pattern data
    for ((i=0; i<512; i++)); do
        for ((j=0; j<512; j++)); do
            r=$(((i + j) % 256))
            g=$(((i * 2) % 256))
            b=$(((j * 2) % 256))
            echo "$r $g $b" >> test_pattern.ppm
        done
    done
    
    # Try to convert PPM to JPG
    if command -v convert &> /dev/null; then
        convert test_pattern.ppm test_pattern.jpg
        rm test_pattern.ppm
    else
        echo "Created PPM file: test_pattern.ppm"
    fi
fi

echo "Available test images:"
ls -la *.jpg *.ppm 2>/dev/null || echo "No image files created"
