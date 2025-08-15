# CUDA Image Processor Makefile

# Compiler and flags
NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
NVCCFLAGS = -std=c++17 -O3 --use_fast_math
CUDAFLAGS = -gencode arch=compute_30,code=sm_30 \
            -gencode arch=compute_35,code=sm_35 \
            -gencode arch=compute_50,code=sm_50 \
            -gencode arch=compute_52,code=sm_52 \
            -gencode arch=compute_60,code=sm_60 \
            -gencode arch=compute_61,code=sm_61 \
            -gencode arch=compute_70,code=sm_70 \
            -gencode arch=compute_75,code=sm_75

# Directories
SRCDIR = src
INCDIR = include
OBJDIR = build/obj
BINDIR = build/bin

# OpenCV flags (auto-detect)
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
OPENCV_LIBS = $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)

# Include paths
INCLUDES = -I$(INCDIR) $(OPENCV_CFLAGS)

# Libraries
LIBS = $(OPENCV_LIBS) -lcuda -lcudart

# Source files
CPP_SOURCES = $(SRCDIR)/main.cpp $(SRCDIR)/cpu_reference.cpp $(SRCDIR)/utils.cpp
CU_SOURCES = $(SRCDIR)/image_processor.cu

# Object files
CPP_OBJECTS = $(CPP_SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
CU_OBJECTS = $(CU_SOURCES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)
OBJECTS = $(CPP_OBJECTS) $(CU_OBJECTS)

# Target executable
TARGET = $(BINDIR)/cuda_image_processor

# Default target
all: directories $(TARGET)

# Create directories
directories:
	@mkdir -p $(OBJDIR) $(BINDIR)

# Link executable
$(TARGET): $(OBJECTS)
	@echo "Linking $@..."
	$(NVCC) $(NVCCFLAGS) $(CUDAFLAGS) -o $@ $^ $(LIBS)
	@echo "Build completed successfully!"

# Compile C++ source files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile CUDA source files
$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@echo "Compiling $<..."
	$(NVCC) $(NVCCFLAGS) $(CUDAFLAGS) $(INCLUDES) -c $< -o $@

# Clean build files
clean:
	@echo "Cleaning build files..."
	rm -rf build/

# Install (copy to system location)
install: $(TARGET)
	@echo "Installing to /usr/local/bin..."
	sudo cp $(TARGET) /usr/local/bin/

# Uninstall
uninstall:
	@echo "Removing from /usr/local/bin..."
	sudo rm -f /usr/local/bin/cuda_image_processor

# Run basic test
test: $(TARGET)
	@echo "Running basic test..."
	./$(TARGET) --help

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: NVCCFLAGS += -g -G -DDEBUG
debug: $(TARGET)

# Create sample data directory and download test images
setup-data:
	@echo "Setting up sample data..."
	mkdir -p data/sample_images
	mkdir -p data/kernels
	@echo "Creating sample convolution kernels..."
	@echo "0 -1 0\n-1 5 -1\n0 -1 0" > data/kernels/sharpen.txt
	@echo "-1 -1 -1\n-1 8 -1\n-1 -1 -1" > data/kernels/edge.txt
	@echo "0.0625 0.125 0.0625\n0.125 0.25 0.125\n0.0625 0.125 0.0625" > data/kernels/blur.txt

# Help target
help:
	@echo "CUDA Image Processor Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all         - Build the project (default)"
	@echo "  clean       - Remove build files"
	@echo "  debug       - Build with debug symbols"
	@echo "  install     - Install to system location"
	@echo "  uninstall   - Remove from system location"
	@echo "  test        - Run basic functionality test"
	@echo "  setup-data  - Create sample data directory"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Build requirements:"
	@echo "  - NVIDIA CUDA Toolkit (version 10.0+)"
	@echo "  - OpenCV development libraries"
	@echo "  - GCC/G++ compiler with C++17 support"
	@echo ""
	@echo "Example usage:"
	@echo "  make                    # Build the project"
	@echo "  make setup-data         # Setup sample data"
	@echo "  make install            # Install system-wide"

# Dependency information
deps:
	@echo "Checking dependencies..."
	@which nvcc > /dev/null || (echo "ERROR: nvcc not found! Install CUDA Toolkit."; exit 1)
	@which g++ > /dev/null || (echo "ERROR: g++ not found! Install GCC."; exit 1)
	@pkg-config --exists opencv4 || pkg-config --exists opencv || (echo "ERROR: OpenCV not found! Install OpenCV development libraries."; exit 1)
	@echo "All dependencies satisfied!"

# Show compiler and library versions
info:
	@echo "Build Environment Information:"
	@echo "------------------------------"
	@echo "NVCC Version:"
	@nvcc --version 2>/dev/null || echo "NVCC not available"
	@echo ""
	@echo "GCC Version:"
	@g++ --version | head -1 2>/dev/null || echo "GCC not available"
	@echo ""
	@echo "OpenCV Version:"
	@pkg-config --modversion opencv4 2>/dev/null || pkg-config --modversion opencv 2>/dev/null || echo "OpenCV not available"
	@echo ""
	@echo "CUDA Runtime Version:"
	@cat /usr/local/cuda/version.txt 2>/dev/null || echo "CUDA version file not found"

.PHONY: all clean debug install uninstall test setup-data help deps info directories
