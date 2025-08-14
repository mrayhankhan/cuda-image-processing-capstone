#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Image loading and saving functions
std::vector<cv::Mat> load_images_from_directory(const std::string& directory_path);
void save_images_to_directory(const std::vector<cv::Mat>& images, 
                             const std::vector<std::string>& filenames,
                             const std::string& output_directory,
                             const std::string& suffix = "");

// Utility functions for image validation
bool is_valid_image_file(const std::string& filename);
std::vector<std::string> get_image_filenames(const std::string& directory_path);

#endif // IMAGE_LOADER_H
