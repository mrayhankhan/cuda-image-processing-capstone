#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>

#include "image_processor.h"
#include "cpu_reference.h"
#include "utils.h"

int main(int argc, char* argv[]) {
    // Parse command line arguments
    ProcessingOptions options;
    if (!Utils::parse_arguments(argc, argv, options)) {
        Utils::print_usage(argv[0]);
        return 1;
    }
    
    // Check CUDA availability
    if (!Utils::check_cuda_capability()) {
        std::cerr << "CUDA not available or insufficient capability!" << std::endl;
        return 1;
    }
    
    // Validate input file
    if (!Utils::file_exists(options.input_path)) {
        std::cerr << "Error: Input file '" << options.input_path << "' not found!" << std::endl;
        return 1;
    }
    
    if (!Utils::validate_image_format(options.input_path)) {
        std::cerr << "Error: Unsupported image format!" << std::endl;
        return 1;
    }
    
    // Validate and create output directory
    if (!Utils::validate_output_directory(options.output_dir)) {
        std::cerr << "Error: Cannot create output directory '" << options.output_dir << "'!" << std::endl;
        return 1;
    }
    
    // Load input image
    cv::Mat input_image;
    if (!Utils::load_image(options.input_path, input_image)) {
        std::cerr << "Error: Failed to load image '" << options.input_path << "'!" << std::endl;
        return 1;
    }
    
    if (options.verbose) {
        Utils::print_image_info(input_image, "Input");
        Utils::print_cuda_device_info();
    }
    
    // Initialize CUDA image processor
    CudaImageProcessor gpu_processor;
    if (!gpu_processor.initialize(input_image.cols, input_image.rows, input_image.channels())) {
        std::cerr << "Error: Failed to initialize GPU processor!" << std::endl;
        return 1;
    }
    
    if (options.verbose) {
        gpu_processor.print_device_info();
        std::cout << "GPU Memory Usage: " << (gpu_processor.get_memory_usage() / (1024 * 1024)) << " MB" << std::endl;
    }
    
    // Get base filename for output files
    std::string base_filename = Utils::get_filename_without_extension(options.input_path);
    std::string output_extension = ".jpg";
    
    // Process images based on options
    std::cout << "Processing image: " << options.input_path 
              << " (" << input_image.cols << "x" << input_image.rows << ")" << std::endl;
    
    // Gaussian Blur Processing
    if (options.apply_blur) {
        std::cout << "\nApplying Gaussian Blur..." << std::endl;
        
        cv::Mat gpu_result, cpu_result;
        TimingInfo gpu_timing = {0};
        CpuTimer cpu_timer;
        
        // GPU processing
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < options.iterations; i++) {
            if (!gpu_processor.process_gaussian_blur(input_image, gpu_result, 
                                                    options.kernel_size, options.sigma, 
                                                    options.benchmark ? &gpu_timing : nullptr)) {
                std::cerr << "Error: GPU Gaussian blur failed!" << std::endl;
                continue;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        float avg_gpu_time = (duration.count() / 1000.0f) / options.iterations;
        
        // CPU processing (if requested)
        float avg_cpu_time = -1.0f;
        if (options.cpu_compare) {
            cpu_timer.start();
            for (int i = 0; i < options.iterations; i++) {
                CpuImageProcessor::gaussian_blur(input_image, cpu_result, 
                                               options.kernel_size, options.sigma);
            }
            avg_cpu_time = cpu_timer.stop() / options.iterations;
        }
        
        // Save result
        std::string output_path = options.output_dir + "/" + base_filename + "_blur" + output_extension;
        if (Utils::save_image(output_path, gpu_result)) {
            std::cout << "Saved: " << output_path << std::endl;
        }
        
        if (options.benchmark) {
            Utils::print_performance_summary("Gaussian Blur", avg_gpu_time, avg_cpu_time);
        }
        
        if (options.verbose && options.cpu_compare) {
            float psnr = Utils::calculate_psnr(cpu_result, gpu_result);
            std::cout << "PSNR (CPU vs GPU): " << psnr << " dB" << std::endl;
        }
    }
    
    // Sobel Edge Detection Processing
    if (options.apply_edge) {
        std::cout << "\nApplying Sobel Edge Detection..." << std::endl;
        
        cv::Mat gpu_result, cpu_result;
        TimingInfo gpu_timing = {0};
        CpuTimer cpu_timer;
        
        // GPU processing
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < options.iterations; i++) {
            if (!gpu_processor.process_sobel_edge(input_image, gpu_result, 
                                                 options.benchmark ? &gpu_timing : nullptr)) {
                std::cerr << "Error: GPU Sobel edge detection failed!" << std::endl;
                continue;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        float avg_gpu_time = (duration.count() / 1000.0f) / options.iterations;
        
        // CPU processing (if requested)
        float avg_cpu_time = -1.0f;
        if (options.cpu_compare) {
            cpu_timer.start();
            for (int i = 0; i < options.iterations; i++) {
                CpuImageProcessor::sobel_edge_detection(input_image, cpu_result);
            }
            avg_cpu_time = cpu_timer.stop() / options.iterations;
        }
        
        // Save result
        std::string output_path = options.output_dir + "/" + base_filename + "_edges" + output_extension;
        if (Utils::save_image(output_path, gpu_result)) {
            std::cout << "Saved: " << output_path << std::endl;
        }
        
        if (options.benchmark) {
            Utils::print_performance_summary("Sobel Edge Detection", avg_gpu_time, avg_cpu_time);
        }
        
        if (options.verbose && options.cpu_compare) {
            float psnr = Utils::calculate_psnr(cpu_result, gpu_result);
            std::cout << "PSNR (CPU vs GPU): " << psnr << " dB" << std::endl;
        }
    }
    
    // Emboss Filter Processing
    if (options.apply_emboss) {
        std::cout << "\nApplying Emboss Filter..." << std::endl;
        
        cv::Mat gpu_result, cpu_result;
        TimingInfo gpu_timing = {0};
        CpuTimer cpu_timer;
        
        // GPU processing
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < options.iterations; i++) {
            if (!gpu_processor.process_emboss(input_image, gpu_result, 
                                             options.benchmark ? &gpu_timing : nullptr)) {
                std::cerr << "Error: GPU emboss filter failed!" << std::endl;
                continue;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        float avg_gpu_time = (duration.count() / 1000.0f) / options.iterations;
        
        // CPU processing (if requested)
        float avg_cpu_time = -1.0f;
        if (options.cpu_compare) {
            cpu_timer.start();
            for (int i = 0; i < options.iterations; i++) {
                CpuImageProcessor::emboss_filter(input_image, cpu_result);
            }
            avg_cpu_time = cpu_timer.stop() / options.iterations;
        }
        
        // Save result
        std::string output_path = options.output_dir + "/" + base_filename + "_emboss" + output_extension;
        if (Utils::save_image(output_path, gpu_result)) {
            std::cout << "Saved: " << output_path << std::endl;
        }
        
        if (options.benchmark) {
            Utils::print_performance_summary("Emboss Filter", avg_gpu_time, avg_cpu_time);
        }
        
        if (options.verbose && options.cpu_compare) {
            float psnr = Utils::calculate_psnr(cpu_result, gpu_result);
            std::cout << "PSNR (CPU vs GPU): " << psnr << " dB" << std::endl;
        }
    }
    
    // Custom Convolution Processing
    if (options.apply_custom && !options.custom_kernel_file.empty()) {
        std::cout << "\nApplying Custom Convolution..." << std::endl;
        
        std::vector<float> kernel;
        int kernel_size;
        
        if (!Utils::load_custom_kernel(options.custom_kernel_file, kernel, kernel_size)) {
            std::cerr << "Error: Failed to load custom kernel from '" 
                      << options.custom_kernel_file << "'!" << std::endl;
        } else {
            cv::Mat gpu_result, cpu_result;
            TimingInfo gpu_timing = {0};
            CpuTimer cpu_timer;
            
            // GPU processing
            auto start_time = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < options.iterations; i++) {
                if (!gpu_processor.process_custom_convolution(input_image, gpu_result, 
                                                             kernel, kernel_size,
                                                             options.benchmark ? &gpu_timing : nullptr)) {
                    std::cerr << "Error: GPU custom convolution failed!" << std::endl;
                    continue;
                }
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            float avg_gpu_time = (duration.count() / 1000.0f) / options.iterations;
            
            // CPU processing (if requested)
            float avg_cpu_time = -1.0f;
            if (options.cpu_compare) {
                cpu_timer.start();
                for (int i = 0; i < options.iterations; i++) {
                    CpuImageProcessor::custom_convolution(input_image, cpu_result, kernel, kernel_size);
                }
                avg_cpu_time = cpu_timer.stop() / options.iterations;
            }
            
            // Save result
            std::string output_path = options.output_dir + "/" + base_filename + "_custom" + output_extension;
            if (Utils::save_image(output_path, gpu_result)) {
                std::cout << "Saved: " << output_path << std::endl;
            }
            
            if (options.benchmark) {
                Utils::print_performance_summary("Custom Convolution", avg_gpu_time, avg_cpu_time);
            }
            
            if (options.verbose && options.cpu_compare) {
                float psnr = Utils::calculate_psnr(cpu_result, gpu_result);
                std::cout << "PSNR (CPU vs GPU): " << psnr << " dB" << std::endl;
            }
        }
    }
    
    // Summary
    if (!options.apply_blur && !options.apply_edge && !options.apply_emboss && !options.apply_custom) {
        std::cout << "\nNo processing filters specified. Use --help for options." << std::endl;
        return 1;
    }
    
    std::cout << "\nProcessing completed successfully!" << std::endl;
    
    if (options.verbose) {
        std::cout << "Total GPU memory used: " << (gpu_processor.get_memory_usage() / (1024 * 1024)) << " MB" << std::endl;
    }
    
    return 0;
}
