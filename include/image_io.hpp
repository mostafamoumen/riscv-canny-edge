#pragma once

#include <cstddef>
#include <cstdint>

namespace imgio {

constexpr std::size_t kAlignment = 64;

std::size_t image_bytes(int width, int height);
std::size_t aligned_size(std::size_t n, std::size_t alignment = kAlignment);

std::uint8_t* allocate_image(int width, int height);
void free_image(std::uint8_t* ptr);

bool load_raw_grayscale(const char* path, std::uint8_t* dst, std::size_t bytes);
bool save_raw_grayscale(const char* path, const std::uint8_t* src, std::size_t bytes);

void fill_rectangle(std::uint8_t* img, int width, int height);
void fill_circle(std::uint8_t* img, int width, int height);
void fill_diagonal_edge(std::uint8_t* img, int width, int height);

}
