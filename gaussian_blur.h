#ifndef GAUSSIAN_BLUR_H
#define GAUSSIAN_BLUR_H

#include "image_io.h"

// Task 2: Standard 2D Convolution (Template)
template <typename TPixel, typename TAcc, typename TKernel>
void apply_gaussian_2d(const Image& input, Image& output);

// Task 2: Separable Gaussian (Template)
template <typename TPixel, typename TAcc, typename TKernel>
void apply_gaussian_separable(const Image& input, Image& output);

// C++ Templates must have their implementation visible to the compiler at instantiation.
#include "gaussian_blur.cpp" 

#endif