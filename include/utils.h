#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <chrono>

// Command line argument parsing
struct CommandLineArgs {
    std::string input_dir;
    std::string output_dir;
    std::string filter_type;
    int kernel_size;
    int brightness;
    float contrast;
    int batch_size;
    bool help;
};

// Function declarations
CommandLineArgs parse_command_line(int argc, char* argv[]);
void print_usage(const char* program_name);
bool validate_arguments(const CommandLineArgs& args);
void create_directory_if_not_exists(const std::string& path);

// Timer class for performance measurement
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start();
    double elapsed_seconds();
    double elapsed_milliseconds();
};

#endif // UTILS_H
