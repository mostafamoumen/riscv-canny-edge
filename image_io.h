#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <cstdint>
#include <string>

struct Image {
    uint8_t* data;
    int width;
    int height;
};

// Memory & File I/O
Image allocate_image(int width, int height);
void free_image(Image& img);
bool load_raw_image(const std::string& filename, Image& img); // Added Load function
bool save_raw_image(const std::string& filename, const Image& img);

// Test Pattern Generators
void generate_rect(Image& img);
void generate_circle(Image& img);
void generate_diagonal(Image& img);

#endif