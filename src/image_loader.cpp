#include "image_loader.h"
#include <filesystem>
#include <iostream>
#include <algorithm>

std::vector<cv::Mat> load_images_from_directory(const std::string& directory_path) {
    std::vector<cv::Mat> images;
    
    if (!std::filesystem::exists(directory_path)) {
        std::cerr << "Directory does not exist: " << directory_path << std::endl;
        return images;
    }
    
    std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
    
    for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            std::string extension = entry.path().extension().string();
            
            // Convert extension to lowercase
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
            // Check if file has valid image extension
            if (std::find(extensions.begin(), extensions.end(), extension) != extensions.end()) {
                cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
                
                if (!image.empty()) {
                    images.push_back(image);
                    std::cout << "Loaded: " << filename << " (" << image.cols << "x" 
                              << image.rows << ")" << std::endl;
                } else {
                    std::cerr << "Failed to load: " << filename << std::endl;
                }
            }
        }
    }
    
    std::cout << "Total images loaded: " << images.size() << std::endl;
    return images;
}

void save_images_to_directory(const std::vector<cv::Mat>& images, 
                             const std::vector<std::string>& filenames,
                             const std::string& output_directory,
                             const std::string& suffix) {
    if (images.size() != filenames.size()) {
        std::cerr << "Error: Number of images and filenames don't match!" << std::endl;
        return;
    }
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_directory);
    
    for (size_t i = 0; i < images.size(); i++) {
        std::string original_filename = filenames[i];
        
        // Extract file name and extension
        size_t dot_pos = original_filename.find_last_of('.');
        std::string name = original_filename.substr(0, dot_pos);
        std::string extension = original_filename.substr(dot_pos);
        
        // Create output filename with suffix
        std::string output_filename = name + suffix + extension;
        std::string output_path = output_directory + "/" + output_filename;
        
        // Save the image
        if (cv::imwrite(output_path, images[i])) {
            std::cout << "Saved: " << output_filename << std::endl;
        } else {
            std::cerr << "Failed to save: " << output_filename << std::endl;
        }
    }
}

bool is_valid_image_file(const std::string& filename) {
    std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
    
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos == std::string::npos) {
        return false;
    }
    
    std::string extension = filename.substr(dot_pos);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    return std::find(extensions.begin(), extensions.end(), extension) != extensions.end();
}

std::vector<std::string> get_image_filenames(const std::string& directory_path) {
    std::vector<std::string> filenames;
    
    if (!std::filesystem::exists(directory_path)) {
        std::cerr << "Directory does not exist: " << directory_path << std::endl;
        return filenames;
    }
    
    for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            
            if (is_valid_image_file(filename)) {
                filenames.push_back(filename);
            }
        }
    }
    
    std::sort(filenames.begin(), filenames.end());
    return filenames;
}
