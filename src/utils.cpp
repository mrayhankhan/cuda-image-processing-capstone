#include "utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <sys/stat.h>
#include <cuda_runtime.h>

bool Utils::parse_arguments(int argc, char* argv[], ProcessingOptions& options) {
    if (argc < 3) {
        return false;
    }
    
    options.input_path = argv[1];
    options.output_dir = argv[2];
    
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--blur") {
            options.apply_blur = true;
        } else if (arg == "--edge") {
            options.apply_edge = true;
        } else if (arg == "--emboss") {
            options.apply_emboss = true;
        } else if (arg == "--custom" && i + 1 < argc) {
            options.apply_custom = true;
            options.custom_kernel_file = argv[++i];
        } else if (arg == "--kernel-size" && i + 1 < argc) {
            options.kernel_size = std::stoi(argv[++i]);
        } else if (arg == "--sigma" && i + 1 < argc) {
            options.sigma = std::stof(argv[++i]);
        } else if (arg == "--benchmark") {
            options.benchmark = true;
        } else if (arg == "--iterations" && i + 1 < argc) {
            options.iterations = std::stoi(argv[++i]);
        } else if (arg == "--cpu-compare") {
            options.cpu_compare = true;
        } else if (arg == "--verbose") {
            options.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            return false;
        }
    }
    
    return true;
}

void Utils::print_usage(const char* program_name) {
    std::cout << "CUDA Image Processing Pipeline\n\n";
    std::cout << "Usage: " << program_name << " <input_image> <output_directory> [options]\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  input_image        Path to input image file (JPG, PNG, BMP)\n";
    std::cout << "  output_directory   Directory to save processed images\n\n";
    std::cout << "Options:\n";
    std::cout << "  --blur             Apply Gaussian blur filter\n";
    std::cout << "  --edge             Apply Sobel edge detection\n";
    std::cout << "  --emboss           Apply emboss filter\n";
    std::cout << "  --custom <file>    Apply custom convolution kernel from file\n";
    std::cout << "  --kernel-size <n>  Blur kernel size (default: 9)\n";
    std::cout << "  --sigma <value>    Gaussian sigma value (default: 2.0)\n";
    std::cout << "  --benchmark        Enable performance benchmarking\n";
    std::cout << "  --iterations <n>   Number of benchmark iterations (default: 10)\n";
    std::cout << "  --cpu-compare      Include CPU implementation comparison\n";
    std::cout << "  --verbose          Enable detailed output\n";
    std::cout << "  --help, -h         Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " image.jpg results/ --blur --edge\n";
    std::cout << "  " << program_name << " photo.png output/ --benchmark --cpu-compare\n";
    std::cout << "  " << program_name << " test.bmp results/ --custom kernel.txt --verbose\n";
}

bool Utils::file_exists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

bool Utils::create_directory(const std::string& path) {
    struct stat st = {0};
    if (stat(path.c_str(), &st) == -1) {
        return mkdir(path.c_str(), 0755) == 0;
    }
    return true;
}

std::string Utils::get_filename_without_extension(const std::string& path) {
    size_t last_slash = path.find_last_of("/\\");
    size_t last_dot = path.find_last_of(".");
    
    std::string filename = (last_slash == std::string::npos) ? path : path.substr(last_slash + 1);
    
    if (last_dot != std::string::npos && last_dot > last_slash) {
        filename = filename.substr(0, last_dot - (last_slash == std::string::npos ? 0 : last_slash + 1));
    }
    
    return filename;
}

std::string Utils::get_file_extension(const std::string& path) {
    size_t last_dot = path.find_last_of(".");
    if (last_dot != std::string::npos) {
        return path.substr(last_dot);
    }
    return "";
}

bool Utils::load_image(const std::string& path, cv::Mat& image) {
    image = cv::imread(path, cv::IMREAD_COLOR);
    return !image.empty();
}

bool Utils::save_image(const std::string& path, const cv::Mat& image) {
    return cv::imwrite(path, image);
}

void Utils::print_image_info(const cv::Mat& image, const std::string& name) {
    std::cout << "Image '" << name << "': " 
              << image.cols << "x" << image.rows 
              << " (" << image.channels() << " channels, "
              << "Type: " << image.type() << ")" << std::endl;
}

bool Utils::load_custom_kernel(const std::string& path, std::vector<float>& kernel, int& size) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    std::vector<std::vector<float>> rows;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<float> row;
        float value;
        
        while (iss >> value) {
            row.push_back(value);
        }
        
        if (!row.empty()) {
            rows.push_back(row);
        }
    }
    
    if (rows.empty() || rows.size() != rows[0].size()) {
        return false; // Must be square
    }
    
    size = rows.size();
    kernel.clear();
    kernel.reserve(size * size);
    
    for (const auto& row : rows) {
        for (float val : row) {
            kernel.push_back(val);
        }
    }
    
    return true;
}

void Utils::save_kernel_to_file(const std::string& path, const std::vector<float>& kernel, int size) {
    std::ofstream file(path);
    if (file.is_open()) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                file << kernel[i * size + j];
                if (j < size - 1) file << " ";
            }
            file << "\n";
        }
    }
}

void Utils::print_performance_summary(const std::string& algorithm,
                                     float gpu_time, float cpu_time) {
    std::cout << "Performance - " << algorithm << ":\n";
    std::cout << "  GPU Time: " << gpu_time << " ms\n";
    
    if (cpu_time > 0) {
        std::cout << "  CPU Time: " << cpu_time << " ms\n";
        std::cout << "  Speedup: " << (cpu_time / gpu_time) << "x\n";
    }
    std::cout << std::endl;
}

void Utils::print_memory_usage(size_t gpu_memory, size_t cpu_memory) {
    std::cout << "Memory Usage:\n";
    std::cout << "  GPU: " << (gpu_memory / (1024 * 1024)) << " MB\n";
    std::cout << "  CPU: " << (cpu_memory / (1024 * 1024)) << " MB\n";
}

void Utils::print_cuda_device_info() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    std::cout << "CUDA Device Information:\n";
    std::cout << "Number of devices: " << device_count << "\n\n";
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Max Grid Size: " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << "\n";
        std::cout << "  Memory Clock Rate: " << (prop.memoryClockRate / 1000) << " MHz\n";
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits\n\n";
    }
}

bool Utils::check_cuda_capability() {
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA-capable devices found!\n";
        return false;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    if (prop.major < 3) {
        std::cerr << "GPU compute capability too low (requires 3.0+)\n";
        return false;
    }
    
    return true;
}

bool Utils::validate_output_directory(const std::string& path) {
    if (!file_exists(path)) {
        return create_directory(path);
    }
    return true;
}

bool Utils::validate_image_format(const std::string& path) {
    std::string ext = to_lower(get_file_extension(path));
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

float Utils::calculate_psnr(const cv::Mat& original, const cv::Mat& processed) {
    cv::Mat diff;
    cv::absdiff(original, processed, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);
    
    cv::Scalar mse = cv::mean(diff);
    double mse_avg = (mse[0] + mse[1] + mse[2]) / 3.0;
    
    if (mse_avg == 0) return INFINITY;
    
    return 10.0 * log10(255.0 * 255.0 / mse_avg);
}

float Utils::calculate_mse(const cv::Mat& img1, const cv::Mat& img2) {
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);
    
    cv::Scalar mse = cv::mean(diff);
    return (mse[0] + mse[1] + mse[2]) / 3.0;
}

std::vector<std::string> Utils::split_string(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(trim_string(token));
    }
    
    return tokens;
}

std::string Utils::trim_string(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    
    size_t end = str.find_last_not_of(" \t\n\r");
    return str.substr(start, end - start + 1);
}

std::string Utils::to_lower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}
