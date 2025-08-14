#include "image_processor.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

// CUDA kernel for Gaussian blur using separable convolution
__global__ void gaussian_blur_kernel(float* input, float* output, int width, int height, 
                                    int channels, float* kernel, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int half_kernel = kernel_size / 2;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        // Horizontal convolution
        for (int k = -half_kernel; k <= half_kernel; k++) {
            int sample_x = min(max(x + k, 0), width - 1);
            int idx = (y * width + sample_x) * channels + c;
            sum += input[idx] * kernel[k + half_kernel];
        }
        
        int output_idx = (y * width + x) * channels + c;
        output[output_idx] = sum;
    }
}

// CUDA kernel for Sobel edge detection
__global__ void sobel_edge_kernel(float* input, float* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height || x == 0 || y == 0 || x == width-1 || y == height-1) {
        if (x < width && y < height) {
            for (int c = 0; c < channels; c++) {
                output[(y * width + x) * channels + c] = 0.0f;
            }
        }
        return;
    }
    
    // Sobel X kernel: [-1, 0, 1; -2, 0, 2; -1, 0, 1]
    // Sobel Y kernel: [-1, -2, -1; 0, 0, 0; 1, 2, 1]
    
    for (int c = 0; c < channels; c++) {
        float gx = 0.0f, gy = 0.0f;
        
        // Calculate gradients
        gx += -1.0f * input[((y-1) * width + (x-1)) * channels + c];
        gx += -2.0f * input[((y) * width + (x-1)) * channels + c];
        gx += -1.0f * input[((y+1) * width + (x-1)) * channels + c];
        gx += 1.0f * input[((y-1) * width + (x+1)) * channels + c];
        gx += 2.0f * input[((y) * width + (x+1)) * channels + c];
        gx += 1.0f * input[((y+1) * width + (x+1)) * channels + c];
        
        gy += -1.0f * input[((y-1) * width + (x-1)) * channels + c];
        gy += -2.0f * input[((y-1) * width + (x)) * channels + c];
        gy += -1.0f * input[((y-1) * width + (x+1)) * channels + c];
        gy += 1.0f * input[((y+1) * width + (x-1)) * channels + c];
        gy += 2.0f * input[((y+1) * width + (x)) * channels + c];
        gy += 1.0f * input[((y+1) * width + (x+1)) * channels + c];
        
        float magnitude = sqrtf(gx * gx + gy * gy);
        magnitude = fminf(magnitude, 255.0f);
        
        output[(y * width + x) * channels + c] = magnitude;
    }
}

// CUDA kernel for brightness and contrast adjustment
__global__ void brightness_contrast_kernel(float* input, float* output, int width, int height, 
                                          int channels, float brightness, float contrast) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * channels;
    
    for (int c = 0; c < channels; c++) {
        float pixel = input[idx + c];
        pixel = pixel * contrast + brightness;
        pixel = fmaxf(0.0f, fminf(255.0f, pixel));
        output[idx + c] = pixel;
    }
}

// Host functions to launch kernels
extern "C" void launch_gaussian_blur_kernel(float* input, float* output, int width, int height, 
                                           int channels, float* kernel, int kernel_size, 
                                           cudaStream_t stream) {
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                   (height + block_size.y - 1) / block_size.y);
    
    gaussian_blur_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, width, height, channels, kernel, kernel_size);
}

extern "C" void launch_sobel_edge_kernel(float* input, float* output, int width, int height, 
                                        int channels, cudaStream_t stream) {
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                   (height + block_size.y - 1) / block_size.y);
    
    sobel_edge_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, width, height, channels);
}

extern "C" void launch_brightness_contrast_kernel(float* input, float* output, int width, int height, 
                                                 int channels, float brightness, float contrast, 
                                                 cudaStream_t stream) {
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                   (height + block_size.y - 1) / block_size.y);
    
    brightness_contrast_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, width, height, channels, brightness, contrast);
}

// GPU memory management functions
void allocate_gpu_memory(ImageData* gpu_images, const std::vector<cv::Mat>& cpu_images, int batch_size) {
    for (int i = 0; i < batch_size && i < cpu_images.size(); i++) {
        const cv::Mat& img = cpu_images[i];
        size_t img_size = img.total() * img.elemSize();
        
        gpu_images[i].width = img.cols;
        gpu_images[i].height = img.rows;
        gpu_images[i].channels = img.channels();
        
        CUDA_CHECK(cudaMalloc(&gpu_images[i].data, img_size));
    }
}

void free_gpu_memory(ImageData* gpu_images, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        if (gpu_images[i].data) {
            CUDA_CHECK(cudaFree(gpu_images[i].data));
            gpu_images[i].data = nullptr;
        }
    }
}

// Main image processing function
std::vector<cv::Mat> process_images_gpu(const std::vector<cv::Mat>& images, 
                                       const ProcessingParams& params) {
    std::vector<cv::Mat> processed_images;
    processed_images.reserve(images.size());
    
    // Create CUDA streams for asynchronous processing
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Process images in batches
    int num_batches = (images.size() + params.batch_size - 1) / params.batch_size;
    
    for (int batch = 0; batch < num_batches; batch++) {
        int batch_start = batch * params.batch_size;
        int current_batch_size = std::min(params.batch_size, 
                                         static_cast<int>(images.size()) - batch_start);
        
        std::cout << "Processing batch " << (batch + 1) << "/" << num_batches 
                  << " (" << current_batch_size << " images)" << std::endl;
        
        // Allocate GPU memory for current batch
        ImageData* gpu_input = new ImageData[current_batch_size];
        ImageData* gpu_output = new ImageData[current_batch_size];
        
        // Process each image in the batch
        for (int i = 0; i < current_batch_size; i++) {
            const cv::Mat& input_img = images[batch_start + i];
            
            // Convert image to float and normalize
            cv::Mat float_img;
            input_img.convertTo(float_img, CV_32F);
            
            // Allocate GPU memory
            size_t img_size = float_img.total() * float_img.elemSize();
            
            gpu_input[i].width = float_img.cols;
            gpu_input[i].height = float_img.rows;
            gpu_input[i].channels = float_img.channels();
            
            CUDA_CHECK(cudaMalloc(&gpu_input[i].data, img_size));
            CUDA_CHECK(cudaMalloc(&gpu_output[i].data, img_size));
            
            // Copy image to GPU
            CUDA_CHECK(cudaMemcpyAsync(gpu_input[i].data, float_img.ptr<float>(), 
                                     img_size, cudaMemcpyHostToDevice, stream));
        }
        
        // Apply the specified filter
        for (int i = 0; i < current_batch_size; i++) {
            if (params.filter_type == "gaussian") {
                // Create Gaussian kernel
                float* d_kernel;
                float* h_kernel = create_gaussian_kernel(params.kernel_size, 
                                                       params.kernel_size / 6.0f);
                size_t kernel_size_bytes = params.kernel_size * sizeof(float);
                
                CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size_bytes));
                CUDA_CHECK(cudaMemcpyAsync(d_kernel, h_kernel, kernel_size_bytes, 
                                         cudaMemcpyHostToDevice, stream));
                
                launch_gaussian_blur_kernel((float*)gpu_input[i].data, (float*)gpu_output[i].data,
                                          gpu_input[i].width, gpu_input[i].height, 
                                          gpu_input[i].channels, d_kernel, 
                                          params.kernel_size, stream);
                
                CUDA_CHECK(cudaFree(d_kernel));
                delete[] h_kernel;
            }
            else if (params.filter_type == "sobel") {
                launch_sobel_edge_kernel((float*)gpu_input[i].data, (float*)gpu_output[i].data,
                                       gpu_input[i].width, gpu_input[i].height, 
                                       gpu_input[i].channels, stream);
            }
            else if (params.filter_type == "brightness") {
                launch_brightness_contrast_kernel((float*)gpu_input[i].data, (float*)gpu_output[i].data,
                                                gpu_input[i].width, gpu_input[i].height, 
                                                gpu_input[i].channels, 
                                                static_cast<float>(params.brightness), 
                                                params.contrast, stream);
            }
        }
        
        // Copy results back to CPU
        for (int i = 0; i < current_batch_size; i++) {
            const cv::Mat& original = images[batch_start + i];
            cv::Mat result(original.rows, original.cols, CV_32FC(original.channels()));
            
            size_t img_size = result.total() * result.elemSize();
            CUDA_CHECK(cudaMemcpyAsync(result.ptr<float>(), gpu_output[i].data, 
                                     img_size, cudaMemcpyDeviceToHost, stream));
            
            // Convert back to 8-bit
            cv::Mat final_result;
            result.convertTo(final_result, CV_8U);
            
            processed_images.push_back(final_result);
        }
        
        // Wait for all operations to complete
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Free GPU memory for this batch
        for (int i = 0; i < current_batch_size; i++) {
            CUDA_CHECK(cudaFree(gpu_input[i].data));
            CUDA_CHECK(cudaFree(gpu_output[i].data));
        }
        
        delete[] gpu_input;
        delete[] gpu_output;
    }
    
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    return processed_images;
}

// Utility functions
float* create_gaussian_kernel(int size, float sigma) {
    float* kernel = new float[size];
    int half = size / 2;
    float sum = 0.0f;
    
    for (int i = 0; i < size; i++) {
        int x = i - half;
        kernel[i] = expf(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel[i];
    }
    
    // Normalize
    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }
    
    return kernel;
}

void print_performance_stats(double processing_time, int num_images, const std::string& filter_type) {
    std::cout << "\n=== Performance Statistics ===" << std::endl;
    std::cout << "Filter Type: " << filter_type << std::endl;
    std::cout << "Total Images: " << num_images << std::endl;
    std::cout << "Processing Time: " << processing_time << " seconds" << std::endl;
    std::cout << "Throughput: " << (num_images / processing_time) << " images/second" << std::endl;
    std::cout << "Average Time per Image: " << (processing_time / num_images * 1000.0) << " ms" << std::endl;
}
