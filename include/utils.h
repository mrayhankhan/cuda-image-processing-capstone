#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// Command line argument structure
struct ProcessingOptions {
    std::string input_path;
    std::string output_dir;
    bool apply_blur = false;
    bool apply_edge = false;
    bool apply_emboss = false;
    bool apply_custom = false;
    std::string custom_kernel_file;
    int kernel_size = 9;
    float sigma = 2.0;
    bool benchmark = false;
    int iterations = 10;
    bool cpu_compare = false;
    bool verbose = false;
};

// Utility functions
namespace Utils {
    // Command line parsing
    bool parse_arguments(int argc, char* argv[], ProcessingOptions& options);
    void print_usage(const char* program_name);
    
    // File operations
    bool file_exists(const std::string& path);
    bool create_directory(const std::string& path);
    std::string get_filename_without_extension(const std::string& path);
    std::string get_file_extension(const std::string& path);
    
    // Image operations
    bool load_image(const std::string& path, cv::Mat& image);
    bool save_image(const std::string& path, const cv::Mat& image);
    void print_image_info(const cv::Mat& image, const std::string& name);
    
    // Kernel operations
    bool load_custom_kernel(const std::string& path, std::vector<float>& kernel, int& size);
    void save_kernel_to_file(const std::string& path, const std::vector<float>& kernel, int size);
    
    // Performance utilities
    void print_performance_summary(const std::string& algorithm,
                                 float gpu_time, float cpu_time = -1.0f);
    
    void print_memory_usage(size_t gpu_memory, size_t cpu_memory);
    
    // Device information
    void print_cuda_device_info();
    bool check_cuda_capability();
    
    // Validation utilities
    bool validate_output_directory(const std::string& path);
    bool validate_image_format(const std::string& path);
    
    // Math utilities
    float calculate_psnr(const cv::Mat& original, const cv::Mat& processed);
    float calculate_mse(const cv::Mat& img1, const cv::Mat& img2);
    
    // String utilities
    std::vector<std::string> split_string(const std::string& str, char delimiter);
    std::string trim_string(const std::string& str);
    std::string to_lower(const std::string& str);
}

#endif // UTILS_H
