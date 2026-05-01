#include "image_io.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>

// Allocates memory aligned to 64 bytes for RISC-V Vector compatibility
Image allocate_image(int width, int height) {
    Image img;
    img.width = width;
    img.height = height;
    // aligned_alloc requires the size to be a multiple of the alignment
    size_t size = (static_cast<size_t>(width * height) + 63) & ~63;
    img.data = static_cast<uint8_t*>(aligned_alloc(64, size));
    
    if (img.data == nullptr) {
        std::cerr << "Error: Memory allocation failed." << std::endl;
    }
    return img;
}

void free_image(Image& img) {
    if (img.data) {
        free(img.data);
        img.data = nullptr;
    }
}

bool load_raw_image(const std::string& filename, Image& img) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }
    file.read(reinterpret_cast<char*>(img.data), img.width * img.height);
    return file.gcount() == (img.width * img.height);
}

bool save_raw_image(const std::string& filename, const Image& img) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return false;
    }
    file.write(reinterpret_cast<const char*>(img.data), img.width * img.height);
    return true;
}

// Generates a 255-value (white) square on a 0-value (black) background
void generate_test_image(Image& img) {
    std::memset(img.data, 0, img.width * img.height);
    int square_size = img.width / 4;
    int start_x = img.width / 2 - square_size / 2;
    int start_y = img.height / 2 - square_size / 2;

    for (int y = start_y; y < start_y + square_size; ++y) {
        for (int x = start_x; x < start_x + square_size; ++x) {
            img.data[y * img.width + x] = 255;
        }
    }
}
