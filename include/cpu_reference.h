#ifndef CPU_REFERENCE_H
#define CPU_REFERENCE_H

#include <opencv2/opencv.hpp>
#include <chrono>

// CPU reference implementations for performance comparison
class CpuImageProcessor {
public:
    static void gaussian_blur(const cv::Mat& input, cv::Mat& output, 
                             int kernel_size = 9, float sigma = 2.0);
    
    static void sobel_edge_detection(const cv::Mat& input, cv::Mat& output);
    
    static void emboss_filter(const cv::Mat& input, cv::Mat& output);
    
    static void custom_convolution(const cv::Mat& input, cv::Mat& output,
                                  const std::vector<float>& kernel,
                                  int kernel_size);

private:
    static void apply_convolution(const cv::Mat& input, cv::Mat& output,
                                 const std::vector<float>& kernel,
                                 int kernel_size);
    
    static void generate_gaussian_kernel(std::vector<float>& kernel,
                                       int size, float sigma);
};

// Timer utility class for CPU benchmarking
class CpuTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start();
    float stop(); // Returns elapsed time in milliseconds
};

#endif // CPU_REFERENCE_H
