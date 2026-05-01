#include "magnitude.h"
#include <cmath>     // For std::abs and std::sqrt
#include <algorithm> // For std::max

void compute_magnitude_l1(const int16_t* gx, const int16_t* gy, int width, int height, uint8_t* magnitude_output) {
    int total_pixels = width * height;
    int max_mag = 0;

    // PASS 1: Find the maximum magnitude in the entire image
    for (int i = 0; i < total_pixels; i++) {
        int mag = std::abs(gx[i]) + std::abs(gy[i]);
        if (mag > max_mag) {
            max_mag = mag;
        }
    }

    // Prevent division by zero if the image is completely blank/black
    if (max_mag == 0) max_mag = 1;

    // PASS 2: Normalize to [0, 255]
    for (int i = 0; i < total_pixels; i++) {
        int mag = std::abs(gx[i]) + std::abs(gy[i]);
        // Scale the value relative to the maximum, then store as 8-bit integer
        magnitude_output[i] = static_cast<uint8_t>((mag * 255) / max_mag);
    }
}

void compute_magnitude_l2(const int16_t* gx, const int16_t* gy, int width, int height, uint8_t* magnitude_output) {
    int total_pixels = width * height;
    float max_mag = 0.0f;

    // PASS 1: Find the maximum magnitude using the Pythagorean theorem
    for (int i = 0; i < total_pixels; i++) {
        // We cast to float before multiplying to prevent integer overflow
        float mag = std::sqrt(static_cast<float>(gx[i]*gx[i] + gy[i]*gy[i]));
        if (mag > max_mag) {
            max_mag = mag;
        }
    }

    if (max_mag == 0.0f) max_mag = 1.0f;

    // PASS 2: Normalize to [0, 255]
    for (int i = 0; i < total_pixels; i++) {
        float mag = std::sqrt(static_cast<float>(gx[i]*gx[i] + gy[i]*gy[i]));
        magnitude_output[i] = static_cast<uint8_t>((mag * 255.0f) / max_mag);
    }
}