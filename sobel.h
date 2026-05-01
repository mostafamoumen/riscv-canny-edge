#ifndef SOBEL_H
#define SOBEL_H

#include <cstdint>

// Computes the X and Y Sobel gradients for a given 8-bit grayscale image.
// Uses Structure of Arrays (SoA) layout for the outputs.
void compute_sobel_gradients(const uint8_t* input_image, int width, int height, int16_t* gx_output, int16_t* gy_output);

#endif