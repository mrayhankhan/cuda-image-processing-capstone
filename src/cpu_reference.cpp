#include "cpu_reference.h"
#include <algorithm>
#include <cmath>

void CpuImageProcessor::gaussian_blur(const cv::Mat& input, cv::Mat& output, 
                                     int kernel_size, float sigma) {
    output.create(input.size(), input.type());
    
    // Generate Gaussian kernel
    std::vector<float> kernel(kernel_size * kernel_size);
    generate_gaussian_kernel(kernel, kernel_size, sigma);
    
    // Apply convolution
    apply_convolution(input, output, kernel, kernel_size);
}

void CpuImageProcessor::sobel_edge_detection(const cv::Mat& input, cv::Mat& output) {
    output.create(input.size(), input.type());
    
    // Sobel operators
    const float sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const float sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    
    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                float gx = 0.0f, gy = 0.0f;
                
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int nx = std::max(0, std::min(width - 1, x + kx));
                        int ny = std::max(0, std::min(height - 1, y + ky));
                        
                        uchar pixel_val = input.at<cv::Vec3b>(ny, nx)[c];
                        int kernel_idx = (ky + 1) * 3 + (kx + 1);
                        
                        gx += pixel_val * sobel_x[kernel_idx];
                        gy += pixel_val * sobel_y[kernel_idx];
                    }
                }
                
                float magnitude = std::sqrt(gx * gx + gy * gy);
                output.at<cv::Vec3b>(y, x)[c] = (uchar)std::min(255.0f, magnitude);
            }
        }
    }
}

void CpuImageProcessor::emboss_filter(const cv::Mat& input, cv::Mat& output) {
    output.create(input.size(), input.type());
    
    // Emboss kernel
    const float emboss_kernel[9] = {
        -2, -1,  0,
        -1,  1,  1,
         0,  1,  2
    };
    
    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;
                
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int nx = std::max(0, std::min(width - 1, x + kx));
                        int ny = std::max(0, std::min(height - 1, y + ky));
                        
                        uchar pixel_val = input.at<cv::Vec3b>(ny, nx)[c];
                        int kernel_idx = (ky + 1) * 3 + (kx + 1);
                        
                        sum += pixel_val * emboss_kernel[kernel_idx];
                    }
                }
                
                // Add 128 for proper emboss effect and clamp to [0, 255]
                output.at<cv::Vec3b>(y, x)[c] = (uchar)std::max(0.0f, std::min(255.0f, sum + 128.0f));
            }
        }
    }
}

void CpuImageProcessor::custom_convolution(const cv::Mat& input, cv::Mat& output,
                                          const std::vector<float>& kernel,
                                          int kernel_size) {
    output.create(input.size(), input.type());
    apply_convolution(input, output, kernel, kernel_size);
}

void CpuImageProcessor::apply_convolution(const cv::Mat& input, cv::Mat& output,
                                         const std::vector<float>& kernel,
                                         int kernel_size) {
    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();
    int half_kernel = kernel_size / 2;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;
                float weight_sum = 0.0f;
                
                for (int ky = -half_kernel; ky <= half_kernel; ky++) {
                    for (int kx = -half_kernel; kx <= half_kernel; kx++) {
                        int nx = std::max(0, std::min(width - 1, x + kx));
                        int ny = std::max(0, std::min(height - 1, y + ky));
                        
                        uchar pixel_val = input.at<cv::Vec3b>(ny, nx)[c];
                        int kernel_idx = (ky + half_kernel) * kernel_size + (kx + half_kernel);
                        
                        float weight = kernel[kernel_idx];
                        sum += pixel_val * weight;
                        weight_sum += weight;
                    }
                }
                
                if (weight_sum > 0) {
                    sum /= weight_sum;
                }
                
                output.at<cv::Vec3b>(y, x)[c] = (uchar)std::max(0.0f, std::min(255.0f, sum));
            }
        }
    }
}

void CpuImageProcessor::generate_gaussian_kernel(std::vector<float>& kernel,
                                                int size, float sigma) {
    int half_size = size / 2;
    float sum = 0.0f;
    
    // Generate 2D Gaussian kernel
    for (int y = -half_size; y <= half_size; y++) {
        for (int x = -half_size; x <= half_size; x++) {
            float value = std::exp(-(x*x + y*y) / (2.0f * sigma * sigma));
            int idx = (y + half_size) * size + (x + half_size);
            kernel[idx] = value;
            sum += value;
        }
    }
    
    // Normalize kernel
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

// CpuTimer implementation
void CpuTimer::start() {
    start_time = std::chrono::high_resolution_clock::now();
}

float CpuTimer::stop() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    return duration.count() / 1000.0f; // Convert to milliseconds
}
