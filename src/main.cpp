#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "image_processor.h"
#include "image_loader.h"
#include "utils.h"

int main(int argc, char* argv[]) {
    std::cout << "CUDA Image Processing at Scale" << std::endl;
    std::cout << "==============================" << std::endl;
    
    // Parse command line arguments
    CommandLineArgs args = parse_command_line(argc, argv);
    
    if (args.help) {
        print_usage(argv[0]);
        return 0;
    }
    
    if (!validate_arguments(args)) {
        std::cerr << "Invalid arguments. Use --help for usage information." << std::endl;
        return 1;
    }
    
    // Create output directory if it doesn't exist
    create_directory_if_not_exists(args.output_dir);
    
    // Load images from input directory
    std::cout << "Loading images from: " << args.input_dir << std::endl;
    std::vector<cv::Mat> images = load_images_from_directory(args.input_dir);
    
    if (images.empty()) {
        std::cerr << "No images found in directory: " << args.input_dir << std::endl;
        return 1;
    }
    
    std::cout << "Loaded " << images.size() << " images" << std::endl;
    
    // Setup processing parameters
    ProcessingParams params;
    params.brightness = args.brightness;
    params.contrast = args.contrast;
    params.kernel_size = args.kernel_size;
    params.batch_size = args.batch_size;
    params.filter_type = args.filter_type;
    
    std::cout << "Processing images with filter: " << params.filter_type << std::endl;
    std::cout << "Batch size: " << params.batch_size << std::endl;
    
    // Initialize CUDA
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, 0));
    std::cout << "Using GPU: " << device_prop.name << std::endl;
    std::cout << "GPU Memory: " << device_prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    
    // Start timing
    Timer timer;
    timer.start();
    
    // Process images
    std::vector<cv::Mat> processed_images;
    
    if (params.filter_type == "all") {
        // Apply all filters sequentially
        std::vector<std::string> filters = {"gaussian", "sobel", "brightness"};
        
        for (const auto& filter : filters) {
            ProcessingParams filter_params = params;
            filter_params.filter_type = filter;
            
            std::cout << "Applying " << filter << " filter..." << std::endl;
            std::vector<cv::Mat> filtered_images = process_images_gpu(images, filter_params);
            
            // Save images with filter-specific suffix
            std::vector<std::string> filenames = get_image_filenames(args.input_dir);
            save_images_to_directory(filtered_images, filenames, args.output_dir, "_" + filter);
        }
    } else {
        // Apply single filter
        processed_images = process_images_gpu(images, params);
        
        // Save processed images
        std::vector<std::string> filenames = get_image_filenames(args.input_dir);
        save_images_to_directory(processed_images, filenames, args.output_dir, "_" + params.filter_type);
    }
    
    // Report performance
    double total_time = timer.elapsed_seconds();
    print_performance_stats(total_time, images.size(), params.filter_type);
    
    std::cout << "Processing complete! Results saved to: " << args.output_dir << std::endl;
    
    return 0;
}
