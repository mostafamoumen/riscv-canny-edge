#ifndef GAUSSIAN_BLUR_CPP
#define GAUSSIAN_BLUR_CPP

#include "image_io.h"
#include <algorithm>

template <typename TPixel, typename TAcc, typename TKernel>
void apply_gaussian_2d(const Image& input, Image& output) {
    const TKernel kernel[5][5] = {
        {2,  4,  5,  4, 2},
        {4,  9, 12,  9, 4},
        {5, 12, 15, 12, 5},
        {4,  9, 12,  9, 4},
        {2,  4,  5,  4, 2}
    };
    const TKernel kernel_sum = 273;
    int w = input.width;
    int h = input.height;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            TAcc acc = 0;

            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    int iy = y + ky;
                    int ix = x + kx;

                    // Fixed: True Zero-Padding for boundary handling
                    TPixel pixel_val = 0; 
                    if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
                        pixel_val = input.data[iy * w + ix];
                    }
                    
                    acc += pixel_val * kernel[ky + 2][kx + 2];
                }
            }
            // Clamp final division to prevent integer overflow edge-cases
            output.data[y * w + x] = static_cast<TPixel>(std::clamp(acc / kernel_sum, (TAcc)0, (TAcc)255));
        }
    }
}

template <typename TPixel, typename TAcc, typename TKernel>
void apply_gaussian_separable(const Image& input, Image& output) {
    const TKernel kernel1D[5] = {2, 4, 5, 4, 2};
    const TKernel kernel_sum = 17; // 1D sum
    
    int w = input.width;
    int h = input.height;
    TAcc* temp = new TAcc[w * h];

    // Pass 1: Horizontal Convolution
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            TAcc acc = 0;
            for (int k = -2; k <= 2; ++k) {
                int ix = x + k;
                
                // Fixed: True Zero-Padding
                TPixel pixel_val = 0; 
                if (ix >= 0 && ix < w) {
                    pixel_val = input.data[y * w + ix];
                }
                acc += pixel_val * kernel1D[k + 2];
            }
            temp[y * w + x] = acc / kernel_sum;
        }
    }

    // Pass 2: Vertical Convolution
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            TAcc acc = 0;
            for (int k = -2; k <= 2; ++k) {
                int iy = y + k;
                
                // Fixed: True Zero-Padding
                TAcc temp_val = 0; 
                if (iy >= 0 && iy < h) {
                    temp_val = temp[iy * w + x];
                }
                acc += temp_val * kernel1D[k + 2];
            }
            // Final Clamp
            output.data[y * w + x] = static_cast<TPixel>(std::clamp(acc / kernel_sum, (TAcc)0, (TAcc)255));
        }
    }
    delete[] temp;
}

template <typename TPixel, typename TAcc, typename TKernel>
void apply_gaussian_separable_padded(const Image& input, Image& output) {
    const TKernel kernel1D[5] = {2, 4, 5, 4, 2};
    const TKernel kernel_sum = 17;
    
    int w = input.width;
    int h = input.height;
    
    // Allocate a larger temporary canvas buffer area (+4 width, +4 height)
    int pw = w + 4;
    int ph = h + 4;
    
    // Value initialization syntax () automatically fills the entire buffer allocations with zeros
    TPixel* padded = new TPixel[pw * ph](); 
    TAcc* temp = new TAcc[pw * ph]();

    // Copy original image canvas boundaries into the center frame of your zero-padded arena
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            padded[(y + 2) * pw + (x + 2)] = input.data[y * w + x];
        }
    }

    // Pass 1: Horizontal Convolution (NO INTERNAL JUMP BRANCHES OR IF STATEMENTS)
    for (int y = 2; y < h + 2; ++y) {
        for (int x = 2; x < w + 2; ++x) {
            TAcc acc = 0;
            for (int k = -2; k <= 2; ++k) {
                acc += padded[y * pw + (x + k)] * kernel1D[k + 2];
            }
            temp[y * pw + x] = acc / kernel_sum;
        }
    }

    // Pass 2: Vertical Convolution (NO INTERNAL JUMP BRANCHES OR IF STATEMENTS)
    for (int y = 2; y < h + 2; ++y) {
        for (int x = 2; x < w + 2; ++x) {
            TAcc acc = 0;
            for (int k = -2; k <= 2; ++k) {
                acc += temp[(y + k) * pw + x] * kernel1D[k + 2];
            }
            // Direct write out mapping back into the un-padded target array sizing
            output.data[(y - 2) * w + (x - 2)] = static_cast<TPixel>(std::clamp(acc / kernel_sum, (TAcc)0, (TAcc)255));
        }
    }

    delete[] padded;
    delete[] temp;
}

#endif