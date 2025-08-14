# Sample Images Directory

This directory should contain sample images for testing the CUDA image processor.

For testing purposes, you can add:
- JPEG files (.jpg, .jpeg)
- PNG files (.png) 
- BMP files (.bmp)

You can download sample images from:
- USC SIPI Image Database: https://sipi.usc.edu/database/database.php
- Creative Commons: https://search.creativecommons.org
- Any stock photo website with appropriate licenses

Example test setup:
1. Download 20-50 images of various sizes
2. Place them in this directory
3. Run the image processor: ./bin/cuda_image_processor --filter gaussian

The application will process all images in this directory.
