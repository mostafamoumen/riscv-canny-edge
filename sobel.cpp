#include "sobel.h"
#include <cstring> // Required for std::memset

void compute_sobel_gradients(const uint8_t* input_image, int width, int height, int16_t* gx_output, int16_t* gy_output) {
    // Zero out the entire output buffers to ensure borders are clean zeros
    std::memset(gx_output, 0, width * height * sizeof(int16_t));
    std::memset(gy_output, 0, width * height * sizeof(int16_t));

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int i = y * width + x;

            int16_t gx = 
                (-1 * input_image[(y - 1) * width + (x - 1)]) + (1 * input_image[(y - 1) * width + (x + 1)]) +
                (-2 * input_image[y * width + (x - 1)])       + (2 * input_image[y * width + (x + 1)]) +
                (-1 * input_image[(y + 1) * width + (x - 1)]) + (1 * input_image[(y + 1) * width + (x + 1)]);

            int16_t gy = 
                (-1 * input_image[(y - 1) * width + (x - 1)]) + (-2 * input_image[(y - 1) * width + x]) + (-1 * input_image[(y - 1) * width + (x + 1)]) +
                ( 1 * input_image[(y + 1) * width + (x - 1)]) + ( 2 * input_image[(y + 1) * width + x]) + ( 1 * input_image[(y + 1) * width + (x + 1)]);

            gx_output[i] = gx;
            gy_output[i] = gy;
        }
    }
}



extern "C" void compute_sobel_gradients_rvv(const uint8_t* input, int width, int height, int16_t* gx, int16_t* gy) {
    // Temporary placeholder: Run scalar code so the test framework can compile
    compute_sobel_gradients(input, width, height, gx, gy);
}