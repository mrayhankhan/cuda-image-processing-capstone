#!/usr/bin/env python3

import sys
import os

try:
    import numpy as np
    # Try to create a simple RGB image as raw data, then we'll convert it
    
    # Create a 256x256 RGB test pattern
    width, height = 256, 256
    
    # Create image data
    image_data = []
    
    print("Creating test image data...")
    
    for y in range(height):
        for x in range(width):
            # Create a simple pattern
            r = (x * 255) // width
            g = (y * 255) // height  
            b = ((x + y) * 255) // (width + height)
            
            image_data.extend([r, g, b])
    
    # Write as PPM (simple format)
    with open('../data/sample_images/test_pattern.ppm', 'w') as f:
        f.write(f'P3\n{width} {height}\n255\n')
        
        for i in range(0, len(image_data), 3):
            f.write(f'{image_data[i]} {image_data[i+1]} {image_data[i+2]}\n')
    
    print("Created test_pattern.ppm")
    
    # Try to create a simple checkerboard pattern
    with open('../data/sample_images/checkerboard.ppm', 'w') as f:
        f.write(f'P3\n{width} {height}\n255\n')
        
        for y in range(height):
            for x in range(width):
                if (x // 32 + y // 32) % 2:
                    f.write('255 255 255\n')  # White
                else:
                    f.write('0 0 0\n')        # Black
    
    print("Created checkerboard.ppm")
    
except ImportError:
    print("NumPy not available, creating minimal test image")
    
    # Create a very simple PPM without NumPy
    width, height = 64, 64
    
    with open('../data/sample_images/simple_test.ppm', 'w') as f:
        f.write(f'P3\n{width} {height}\n255\n')
        
        for y in range(height):
            for x in range(width):
                r = (x * 255) // width
                g = (y * 255) // height
                b = 128
                f.write(f'{r} {g} {b}\n')
    
    print("Created simple_test.ppm")

print("Test image creation completed!")
