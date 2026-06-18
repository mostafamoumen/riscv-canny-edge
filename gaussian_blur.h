#ifndef GAUSSIAN_BLUR_H
#define GAUSSIAN_BLUR_H

#include <stdint.h>
#include "image_io.h"

// 1. SCALAR BASES (Template declarations)
template <typename PixelType, typename AccType, typename KernelType>
void apply_gaussian_separable(const Image& input, Image& output);

template <typename PixelType, typename AccType, typename KernelType>
void apply_gaussian_separable_padded(const Image& input, Image& output);

// 2. RVV INTRINSIC SYMBOLS
extern "C" {
    void apply_gaussian_separable_rvv(const uint8_t* input, uint8_t* output, int width, int height);
}

// 3. LMUL Sweep Symbols
void gaussian_blur_rvv_m1(const uint8_t* src, uint8_t* dst, int width, int height);
void gaussian_blur_rvv_m2(const uint8_t* src, uint8_t* dst, int width, int height);
void gaussian_blur_rvv_m4(const uint8_t* src, uint8_t* dst, int width, int height);

#endif // GAUSSIAN_BLUR_H