#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Structure to hold timing information
struct TimingInfo {
    float kernel_time;
    float memory_time;
    float total_time;
};

// CUDA kernel function declarations
extern "C" {
    // Gaussian blur kernel
    void launch_gaussian_blur_kernel(
        unsigned char* d_input, 
        unsigned char* d_output,
        int width, 
        int height, 
        int channels,
        float* d_kernel, 
        int kernel_size, 
        float sigma,
        TimingInfo* timing
    );

    // Sobel edge detection kernel
    void launch_sobel_edge_kernel(
        unsigned char* d_input, 
        unsigned char* d_output,
        int width, 
        int height, 
        int channels,
        TimingInfo* timing
    );

    // Emboss filter kernel
    void launch_emboss_kernel(
        unsigned char* d_input, 
        unsigned char* d_output,
        int width, 
        int height, 
        int channels,
        TimingInfo* timing
    );

    // Custom convolution kernel
    void launch_convolution_kernel(
        unsigned char* d_input, 
        unsigned char* d_output,
        int width, 
        int height, 
        int channels,
        float* d_kernel, 
        int kernel_size,
        TimingInfo* timing
    );

    // Utility functions
    void generate_gaussian_kernel(float* kernel, int size, float sigma);
    void allocate_gpu_memory(unsigned char** d_ptr, size_t size);
    void free_gpu_memory(unsigned char* d_ptr);
    void copy_to_gpu(unsigned char* d_dst, unsigned char* h_src, size_t size);
    void copy_from_gpu(unsigned char* h_dst, unsigned char* d_src, size_t size);
}

// C++ wrapper class for easier usage
class CudaImageProcessor {
private:
    unsigned char* d_input;
    unsigned char* d_output;
    float* d_kernel;
    int width, height, channels;
    size_t image_size;
    bool memory_allocated;

public:
    CudaImageProcessor();
    ~CudaImageProcessor();
    
    bool initialize(int w, int h, int c);
    void cleanup();
    
    bool process_gaussian_blur(const cv::Mat& input, cv::Mat& output, 
                              int kernel_size = 9, float sigma = 2.0,
                              TimingInfo* timing = nullptr);
    
    bool process_sobel_edge(const cv::Mat& input, cv::Mat& output,
                           TimingInfo* timing = nullptr);
    
    bool process_emboss(const cv::Mat& input, cv::Mat& output,
                       TimingInfo* timing = nullptr);
    
    bool process_custom_convolution(const cv::Mat& input, cv::Mat& output,
                                   const std::vector<float>& kernel,
                                   int kernel_size,
                                   TimingInfo* timing = nullptr);
    
    void print_device_info();
    size_t get_memory_usage() const;
};

#endif // IMAGE_PROCESSOR_H
