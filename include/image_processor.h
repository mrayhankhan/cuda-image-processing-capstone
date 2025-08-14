#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Structure to hold processing parameters
struct ProcessingParams {
    int brightness;
    float contrast;
    int kernel_size;
    int batch_size;
    std::string filter_type;
};

// Structure to hold image data
struct ImageData {
    unsigned char* data;
    int width;
    int height;
    int channels;
    size_t pitch;
};

// CUDA kernel declarations
extern "C" {
    void launch_gaussian_blur_kernel(float* input, float* output, int width, int height, 
                                   int channels, float* kernel, int kernel_size, 
                                   cudaStream_t stream);
    
    void launch_sobel_edge_kernel(float* input, float* output, int width, int height, 
                                int channels, cudaStream_t stream);
    
    void launch_brightness_contrast_kernel(float* input, float* output, int width, int height, 
                                         int channels, float brightness, float contrast, 
                                         cudaStream_t stream);
}

// GPU memory management functions
void allocate_gpu_memory(ImageData* gpu_images, const std::vector<cv::Mat>& cpu_images, 
                        int batch_size);
void free_gpu_memory(ImageData* gpu_images, int batch_size);

// Image processing functions
std::vector<cv::Mat> process_images_gpu(const std::vector<cv::Mat>& images, 
                                       const ProcessingParams& params);

// Utility functions
float* create_gaussian_kernel(int size, float sigma);
void print_performance_stats(double processing_time, int num_images, 
                           const std::string& filter_type);

#endif // IMAGE_PROCESSOR_H
