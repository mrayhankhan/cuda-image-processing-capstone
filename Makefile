# CUDA Image Processor Makefile

# Compiler settings
NVCC = nvcc
CXX = g++
TARGET = cuda_image_processor

# Directories
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin

# Source files
CXX_SOURCES = $(SRCDIR)/main.cpp $(SRCDIR)/image_loader.cpp $(SRCDIR)/utils.cpp
CUDA_SOURCES = $(SRCDIR)/image_processor.cu

# Object files
CXX_OBJECTS = $(CXX_SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
CUDA_OBJECTS = $(CUDA_SOURCES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)

# Compiler flags
CXXFLAGS = -std=c++17 -O3 -I$(INCDIR)
NVCCFLAGS = -std=c++17 -O3 -I$(INCDIR) -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86

# OpenCV settings
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

# CUDA settings
CUDA_LIBS = -lcudart

# All flags
ALL_CXXFLAGS = $(CXXFLAGS) $(OPENCV_CFLAGS)
ALL_NVCCFLAGS = $(NVCCFLAGS) $(OPENCV_CFLAGS)
ALL_LIBS = $(OPENCV_LIBS) $(CUDA_LIBS)

# Default target
all: $(BINDIR)/$(TARGET)

# Create directories
$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

# Build executable
$(BINDIR)/$(TARGET): $(CXX_OBJECTS) $(CUDA_OBJECTS) | $(BINDIR)
	$(NVCC) $(CXX_OBJECTS) $(CUDA_OBJECTS) -o $@ $(ALL_LIBS)

# Compile C++ files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(ALL_CXXFLAGS) -c $< -o $@

# Compile CUDA files
$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(NVCC) $(ALL_NVCCFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Install target
install: $(BINDIR)/$(TARGET)
	cp $(BINDIR)/$(TARGET) /usr/local/bin/

# Uninstall target
uninstall:
	rm -f /usr/local/bin/$(TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  all       - Build the application (default)"
	@echo "  clean     - Remove build files"
	@echo "  install   - Install to /usr/local/bin/"
	@echo "  uninstall - Remove from /usr/local/bin/"
	@echo "  help      - Show this help message"

.PHONY: all clean install uninstall help
