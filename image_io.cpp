#include "image_io.h"
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>

Image allocate_image(int width, int height) {
    Image img = {nullptr, width, height};
    size_t size = (static_cast<size_t>(width * height) + 63) & ~63;
    img.data = static_cast<uint8_t*>(aligned_alloc(64, size));
    if (!img.data) std::cerr << "Memory allocation failed.\n";
    return img;
}

void free_image(Image& img) {
    if (img.data) {
        free(img.data);
        img.data = nullptr;
    }
}

// Fixed: Added load_raw_image back in
bool load_raw_image(const std::string& filename, Image& img) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;
    file.read(reinterpret_cast<char*>(img.data), img.width * img.height);
    return file.gcount() == (img.width * img.height);
}

bool save_raw_image(const std::string& filename, const Image& img) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;
    file.write(reinterpret_cast<const char*>(img.data), img.width * img.height);
    return true;
}

void generate_rect(Image& img) {
    std::memset(img.data, 0, img.width * img.height);
    int sq_size = img.width / 4;
    int start_x = img.width / 2 - sq_size / 2;
    int start_y = img.height / 2 - sq_size / 2;
    for (int y = start_y; y < start_y + sq_size; ++y) {
        for (int x = start_x; x < start_x + sq_size; ++x) {
            img.data[y * img.width + x] = 255;
        }
    }
}

void generate_circle(Image& img) {
    std::memset(img.data, 0, img.width * img.height);
    int cx = img.width / 2, cy = img.height / 2, r = img.width / 4;
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            if (std::sqrt(std::pow(x - cx, 2) + std::pow(y - cy, 2)) < r)
                img.data[y * img.width + x] = 255;
        }
    }
}

void generate_diagonal(Image& img) {
    std::memset(img.data, 0, img.width * img.height);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            if (x > y - 10 && x < y + 10) img.data[y * img.width + x] = 255;
        }
    }
}