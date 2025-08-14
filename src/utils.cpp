#include "utils.h"
#include <iostream>
#include <filesystem>
#include <getopt.h>
#include <cstring>
#include <vector>
#include <algorithm>

CommandLineArgs parse_command_line(int argc, char* argv[]) {
    CommandLineArgs args;
    
    // Set default values
    args.input_dir = "./sample_images";
    args.output_dir = "./output";
    args.filter_type = "gaussian";
    args.kernel_size = 15;
    args.brightness = 0;
    args.contrast = 1.0f;
    args.batch_size = 32;
    args.help = false;
    
    // Define long options
    static struct option long_options[] = {
        {"input_dir",    required_argument, 0, 'i'},
        {"output_dir",   required_argument, 0, 'o'},
        {"filter",       required_argument, 0, 'f'},
        {"kernel_size",  required_argument, 0, 'k'},
        {"brightness",   required_argument, 0, 'b'},
        {"contrast",     required_argument, 0, 'c'},
        {"batch_size",   required_argument, 0, 's'},
        {"help",         no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    int option_index = 0;
    int c;
    
    while ((c = getopt_long(argc, argv, "i:o:f:k:b:c:s:h", long_options, &option_index)) != -1) {
        switch (c) {
            case 'i':
                args.input_dir = optarg;
                break;
            case 'o':
                args.output_dir = optarg;
                break;
            case 'f':
                args.filter_type = optarg;
                break;
            case 'k':
                args.kernel_size = std::atoi(optarg);
                break;
            case 'b':
                args.brightness = std::atoi(optarg);
                break;
            case 'c':
                args.contrast = std::atof(optarg);
                break;
            case 's':
                args.batch_size = std::atoi(optarg);
                break;
            case 'h':
                args.help = true;
                break;
            case '?':
                // getopt_long already printed an error message
                args.help = true;
                break;
            default:
                abort();
        }
    }
    
    return args;
}

void print_usage(const char* program_name) {
    std::cout << "CUDA Image Processing at Scale\n" << std::endl;
    std::cout << "Usage: " << program_name << " [OPTIONS]\n" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -i, --input_dir DIR     Input directory containing images (default: ./sample_images)" << std::endl;
    std::cout << "  -o, --output_dir DIR    Output directory for processed images (default: ./output)" << std::endl;
    std::cout << "  -f, --filter TYPE       Filter type: gaussian, sobel, brightness, all (default: gaussian)" << std::endl;
    std::cout << "  -k, --kernel_size SIZE  Gaussian kernel size (default: 15)" << std::endl;
    std::cout << "  -b, --brightness VAL    Brightness adjustment -100 to 100 (default: 0)" << std::endl;
    std::cout << "  -c, --contrast VAL      Contrast multiplier 0.5 to 3.0 (default: 1.0)" << std::endl;
    std::cout << "  -s, --batch_size SIZE   Batch size for processing (default: 32)" << std::endl;
    std::cout << "  -h, --help              Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " --filter gaussian --kernel_size 21" << std::endl;
    std::cout << "  " << program_name << " --filter sobel --input_dir ./my_images" << std::endl;
    std::cout << "  " << program_name << " --filter brightness --brightness 30 --contrast 1.5" << std::endl;
    std::cout << "  " << program_name << " --filter all --batch_size 16" << std::endl;
}

bool validate_arguments(const CommandLineArgs& args) {
    // Check if input directory exists
    if (!std::filesystem::exists(args.input_dir)) {
        std::cerr << "Error: Input directory does not exist: " << args.input_dir << std::endl;
        return false;
    }
    
    // Validate filter type
    std::vector<std::string> valid_filters = {"gaussian", "sobel", "brightness", "all"};
    if (std::find(valid_filters.begin(), valid_filters.end(), args.filter_type) == valid_filters.end()) {
        std::cerr << "Error: Invalid filter type. Valid options: gaussian, sobel, brightness, all" << std::endl;
        return false;
    }
    
    // Validate kernel size (should be odd and positive)
    if (args.kernel_size <= 0 || args.kernel_size % 2 == 0) {
        std::cerr << "Error: Kernel size must be positive and odd" << std::endl;
        return false;
    }
    
    // Validate brightness range
    if (args.brightness < -100 || args.brightness > 100) {
        std::cerr << "Error: Brightness must be between -100 and 100" << std::endl;
        return false;
    }
    
    // Validate contrast range
    if (args.contrast < 0.1f || args.contrast > 5.0f) {
        std::cerr << "Error: Contrast must be between 0.1 and 5.0" << std::endl;
        return false;
    }
    
    // Validate batch size
    if (args.batch_size <= 0 || args.batch_size > 1000) {
        std::cerr << "Error: Batch size must be between 1 and 1000" << std::endl;
        return false;
    }
    
    return true;
}

void create_directory_if_not_exists(const std::string& path) {
    try {
        std::filesystem::create_directories(path);
    } catch (const std::filesystem::filesystem_error& ex) {
        std::cerr << "Error creating directory " << path << ": " << ex.what() << std::endl;
    }
}

// Timer implementation
void Timer::start() {
    start_time = std::chrono::high_resolution_clock::now();
}

double Timer::elapsed_seconds() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    return duration.count() / 1000000.0;
}

double Timer::elapsed_milliseconds() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    return duration.count() / 1000.0;
}
