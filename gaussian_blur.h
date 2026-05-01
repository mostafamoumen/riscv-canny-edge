#ifndef GAUSSIAN_BLUR_H
#define GAUSSIAN_BLUR_H

#include "image_io.h"

/**
 * Task 2: Gaussian Blur
 * Applies a 5x5 Gaussian convolution to the input image.
 * Uses zero-padding for boundary handling.
 */
void apply_gaussian_blur(const Image& input, Image& output);

#endif
