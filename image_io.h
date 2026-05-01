#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <cstdint>
#include <string>

struct Image {
    uint8_t* data;
    int width;
    int height;
};

// Task 1: Image I/O Functions
Image allocate_image(int width, int height);
void free_image(Image& img);
bool load_raw_image(const std::string& filename, Image& img);
bool save_raw_image(const std::string& filename, const Image& img);

// Task 1: Test Image Generator
void generate_test_image(Image& img);

#endif
