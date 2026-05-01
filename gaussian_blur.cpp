#include "gaussian_blur.h"
#include <algorithm>

void apply_gaussian_blur(const Image& input, Image& output) {
    // 5x5 Gaussian Kernel (approx sigma = 1.0, sum = 273)
    const int kernel[5][5] = {
        {2,  4,  5,  4, 2},
        {4,  9, 12,  9, 4},
        {5, 12, 15, 12, 5},
        {4,  9, 12,  9, 4},
        {2,  4,  5,  4, 2}
    };
    const int kernel_sum = 273;

    int width = input.width;
    int height = input.height;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int32_t accumulator = 0;

            // 5x5 Convolution window
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    int iy = y + ky;
                    int ix = x + kx;

                    // Zero-padding boundary handling
                    uint8_t pixel_val = 0;
                    if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                        pixel_val = input.data[iy * width + ix];
                    }

                    accumulator += pixel_val * kernel[ky + 2][kx + 2];
                }
            }

            // Normalize and clamp to [0, 255]
            output.data[y * width + x] = static_cast<uint8_t>(accumulator / kernel_sum);
        }
    }
}
