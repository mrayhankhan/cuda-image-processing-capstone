#include "image_processor.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>

// CUDA kernel for Gaussian blur
__global__ void gaussian_blur_kernel(
    unsigned char* input, 
    unsigned char* output,
    int width, 
    int height, 
    int channels,
    float* kernel, 
    int kernel_size) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int half_kernel = kernel_size / 2;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        float weight_sum = 0.0f;
        
        for (int ky = -half_kernel; ky <= half_kernel; ky++) {
            for (int kx = -half_kernel; kx <= half_kernel; kx++) {
                int nx = x + kx;
                int ny = y + ky;
                
                // Handle boundary conditions with clamping
                nx = max(0, min(width - 1, nx));
                ny = max(0, min(height - 1, ny));
                
                int input_idx = (ny * width + nx) * channels + c;
                int kernel_idx = (ky + half_kernel) * kernel_size + (kx + half_kernel);
                
                float weight = kernel[kernel_idx];
                sum += input[input_idx] * weight;
                weight_sum += weight;
            }
        }
        
        int output_idx = (y * width + x) * channels + c;
        output[output_idx] = (unsigned char)(sum / weight_sum);
    }
}

// CUDA kernel for Sobel edge detection
__global__ void sobel_edge_kernel(
    unsigned char* input, 
    unsigned char* output,
    int width, 
    int height, 
    int channels) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Sobel operators
    const int sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const int sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    
    for (int c = 0; c < channels; c++) {
        float gx = 0.0f, gy = 0.0f;
        
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int nx = max(0, min(width - 1, x + kx));
                int ny = max(0, min(height - 1, y + ky));
                
                int input_idx = (ny * width + nx) * channels + c;
                int kernel_idx = (ky + 1) * 3 + (kx + 1);
                
                float pixel_val = input[input_idx];
                gx += pixel_val * sobel_x[kernel_idx];
                gy += pixel_val * sobel_y[kernel_idx];
            }
        }
        
        float magnitude = sqrtf(gx * gx + gy * gy);
        int output_idx = (y * width + x) * channels + c;
        output[output_idx] = (unsigned char)min(255.0f, magnitude);
    }
}

// CUDA kernel for emboss filter
__global__ void emboss_kernel(
    unsigned char* input, 
    unsigned char* output,
    int width, 
    int height, 
    int channels) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Emboss kernel
    const float emboss_kernel[9] = {
        -2, -1,  0,
        -1,  1,  1,
         0,  1,  2
    };
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int nx = max(0, min(width - 1, x + kx));
                int ny = max(0, min(height - 1, y + ky));
                
                int input_idx = (ny * width + nx) * channels + c;
                int kernel_idx = (ky + 1) * 3 + (kx + 1);
                
                sum += input[input_idx] * emboss_kernel[kernel_idx];
            }
        }
        
        // Add 128 for proper emboss effect and clamp to [0, 255]
        int output_idx = (y * width + x) * channels + c;
        output[output_idx] = (unsigned char)max(0.0f, min(255.0f, sum + 128.0f));
    }
}

// CUDA kernel for custom convolution
__global__ void convolution_kernel(
    unsigned char* input, 
    unsigned char* output,
    int width, 
    int height, 
    int channels,
    float* kernel, 
    int kernel_size) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int half_kernel = kernel_size / 2;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int ky = -half_kernel; ky <= half_kernel; ky++) {
            for (int kx = -half_kernel; kx <= half_kernel; kx++) {
                int nx = max(0, min(width - 1, x + kx));
                int ny = max(0, min(height - 1, y + ky));
                
                int input_idx = (ny * width + nx) * channels + c;
                int kernel_idx = (ky + half_kernel) * kernel_size + (kx + half_kernel);
                
                sum += input[input_idx] * kernel[kernel_idx];
            }
        }
        
        int output_idx = (y * width + x) * channels + c;
        output[output_idx] = (unsigned char)max(0.0f, min(255.0f, sum));
    }
}

// Host function implementations
extern "C" {

void launch_gaussian_blur_kernel(
    unsigned char* d_input, 
    unsigned char* d_output,
    int width, 
    int height, 
    int channels,
    float* d_kernel, 
    int kernel_size, 
    float sigma,
    TimingInfo* timing) {
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Calculate grid and block dimensions
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    cudaEventRecord(start);
    
    // Launch kernel
    gaussian_blur_kernel<<<grid_size, block_size>>>(
        d_input, d_output, width, height, channels, d_kernel, kernel_size);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    if (timing) {
        cudaEventElapsedTime(&timing->kernel_time, start, stop);
        timing->total_time = timing->kernel_time;
    }
    
    CUDA_CHECK(cudaGetLastError());
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void launch_sobel_edge_kernel(
    unsigned char* d_input, 
    unsigned char* d_output,
    int width, 
    int height, 
    int channels,
    TimingInfo* timing) {
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    cudaEventRecord(start);
    
    sobel_edge_kernel<<<grid_size, block_size>>>(
        d_input, d_output, width, height, channels);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    if (timing) {
        cudaEventElapsedTime(&timing->kernel_time, start, stop);
        timing->total_time = timing->kernel_time;
    }
    
    CUDA_CHECK(cudaGetLastError());
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void launch_emboss_kernel(
    unsigned char* d_input, 
    unsigned char* d_output,
    int width, 
    int height, 
    int channels,
    TimingInfo* timing) {
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    cudaEventRecord(start);
    
    emboss_kernel<<<grid_size, block_size>>>(
        d_input, d_output, width, height, channels);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    if (timing) {
        cudaEventElapsedTime(&timing->kernel_time, start, stop);
        timing->total_time = timing->kernel_time;
    }
    
    CUDA_CHECK(cudaGetLastError());
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void launch_convolution_kernel(
    unsigned char* d_input, 
    unsigned char* d_output,
    int width, 
    int height, 
    int channels,
    float* d_kernel, 
    int kernel_size,
    TimingInfo* timing) {
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    cudaEventRecord(start);
    
    convolution_kernel<<<grid_size, block_size>>>(
        d_input, d_output, width, height, channels, d_kernel, kernel_size);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    if (timing) {
        cudaEventElapsedTime(&timing->kernel_time, start, stop);
        timing->total_time = timing->kernel_time;
    }
    
    CUDA_CHECK(cudaGetLastError());
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void generate_gaussian_kernel(float* kernel, int size, float sigma) {
    int half_size = size / 2;
    float sum = 0.0f;
    
    // Generate 2D Gaussian kernel
    for (int y = -half_size; y <= half_size; y++) {
        for (int x = -half_size; x <= half_size; x++) {
            float value = expf(-(x*x + y*y) / (2.0f * sigma * sigma));
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

void allocate_gpu_memory(unsigned char** d_ptr, size_t size) {
    CUDA_CHECK(cudaMalloc(d_ptr, size));
}

void free_gpu_memory(unsigned char* d_ptr) {
    if (d_ptr) {
        CUDA_CHECK(cudaFree(d_ptr));
    }
}

void copy_to_gpu(unsigned char* d_dst, unsigned char* h_src, size_t size) {
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice));
}

void copy_from_gpu(unsigned char* h_dst, unsigned char* d_src, size_t size) {
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost));
}

} // extern "C"

// CudaImageProcessor class implementation
CudaImageProcessor::CudaImageProcessor() 
    : d_input(nullptr), d_output(nullptr), d_kernel(nullptr),
      width(0), height(0), channels(0), image_size(0), memory_allocated(false) {
}

CudaImageProcessor::~CudaImageProcessor() {
    cleanup();
}

bool CudaImageProcessor::initialize(int w, int h, int c) {
    width = w;
    height = h;
    channels = c;
    image_size = width * height * channels * sizeof(unsigned char);
    
    try {
        // Allocate GPU memory for input and output images
        CUDA_CHECK(cudaMalloc(&d_input, image_size));
        CUDA_CHECK(cudaMalloc(&d_output, image_size));
        
        // Allocate memory for kernels (max size assumption)
        CUDA_CHECK(cudaMalloc(&d_kernel, 25 * 25 * sizeof(float))); // Max 25x25 kernel
        
        memory_allocated = true;
        return true;
    } catch (...) {
        cleanup();
        return false;
    }
}

void CudaImageProcessor::cleanup() {
    if (d_input) {
        cudaFree(d_input);
        d_input = nullptr;
    }
    if (d_output) {
        cudaFree(d_output);
        d_output = nullptr;
    }
    if (d_kernel) {
        cudaFree(d_kernel);
        d_kernel = nullptr;
    }
    memory_allocated = false;
}

bool CudaImageProcessor::process_gaussian_blur(const cv::Mat& input, cv::Mat& output, 
                                              int kernel_size, float sigma,
                                              TimingInfo* timing) {
    if (!memory_allocated) return false;
    
    // Generate Gaussian kernel
    std::vector<float> h_kernel(kernel_size * kernel_size);
    generate_gaussian_kernel(h_kernel.data(), kernel_size, sigma);
    
    // Copy kernel to GPU
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(), 
                         kernel_size * kernel_size * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Copy input image to GPU
    CUDA_CHECK(cudaMemcpy(d_input, input.data, image_size, cudaMemcpyHostToDevice));
    
    // Launch kernel
    launch_gaussian_blur_kernel(d_input, d_output, width, height, channels,
                               d_kernel, kernel_size, sigma, timing);
    
    // Copy result back to host
    output.create(height, width, input.type());
    CUDA_CHECK(cudaMemcpy(output.data, d_output, image_size, cudaMemcpyDeviceToHost));
    
    return true;
}

bool CudaImageProcessor::process_sobel_edge(const cv::Mat& input, cv::Mat& output,
                                           TimingInfo* timing) {
    if (!memory_allocated) return false;
    
    // Copy input image to GPU
    CUDA_CHECK(cudaMemcpy(d_input, input.data, image_size, cudaMemcpyHostToDevice));
    
    // Launch kernel
    launch_sobel_edge_kernel(d_input, d_output, width, height, channels, timing);
    
    // Copy result back to host
    output.create(height, width, input.type());
    CUDA_CHECK(cudaMemcpy(output.data, d_output, image_size, cudaMemcpyDeviceToHost));
    
    return true;
}

bool CudaImageProcessor::process_emboss(const cv::Mat& input, cv::Mat& output,
                                       TimingInfo* timing) {
    if (!memory_allocated) return false;
    
    // Copy input image to GPU
    CUDA_CHECK(cudaMemcpy(d_input, input.data, image_size, cudaMemcpyHostToDevice));
    
    // Launch kernel
    launch_emboss_kernel(d_input, d_output, width, height, channels, timing);
    
    // Copy result back to host
    output.create(height, width, input.type());
    CUDA_CHECK(cudaMemcpy(output.data, d_output, image_size, cudaMemcpyDeviceToHost));
    
    return true;
}

bool CudaImageProcessor::process_custom_convolution(const cv::Mat& input, cv::Mat& output,
                                                   const std::vector<float>& kernel,
                                                   int kernel_size,
                                                   TimingInfo* timing) {
    if (!memory_allocated) return false;
    
    // Copy kernel to GPU
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data(), 
                         kernel.size() * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Copy input image to GPU
    CUDA_CHECK(cudaMemcpy(d_input, input.data, image_size, cudaMemcpyHostToDevice));
    
    // Launch kernel
    launch_convolution_kernel(d_input, d_output, width, height, channels,
                             d_kernel, kernel_size, timing);
    
    // Copy result back to host
    output.create(height, width, input.type());
    CUDA_CHECK(cudaMemcpy(output.data, d_output, image_size, cudaMemcpyDeviceToHost));
    
    return true;
}

void CudaImageProcessor::print_device_info() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::cout << "GPU Device Information:" << std::endl;
    std::cout << "  Name: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
}

size_t CudaImageProcessor::get_memory_usage() const {
    return image_size * 2 + 25 * 25 * sizeof(float); // Input + Output + Max kernel
}
