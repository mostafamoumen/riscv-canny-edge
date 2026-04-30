#include "sobel.h"

void compute_sobel_gradients(const uint8_t* input_image, int width, int height, int16_t* gx_output, int16_t* gy_output) {
    // We start at 1 and end at width-1/height-1 to avoid going out of bounds on the edges
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            
            // Calculate the 1D index of the current center pixel
            int i = y * width + x;

            // Apply the 3x3 Gx Kernel (Vertical Edges)
            // Left column is negative, right column is positive. Middle column is 0.
            int16_t gx = 
                (-1 * input_image[(y - 1) * width + (x - 1)]) + (1 * input_image[(y - 1) * width + (x + 1)]) +
                (-2 * input_image[y * width + (x - 1)])       + (2 * input_image[y * width + (x + 1)]) +
                (-1 * input_image[(y + 1) * width + (x - 1)]) + (1 * input_image[(y + 1) * width + (x + 1)]);

            // Apply the 3x3 Gy Kernel (Horizontal Edges)
            // Top row is negative, bottom row is positive. Middle row is 0.
            int16_t gy = 
                (-1 * input_image[(y - 1) * width + (x - 1)]) + (-2 * input_image[(y - 1) * width + x]) + (-1 * input_image[(y - 1) * width + (x + 1)]) +
                ( 1 * input_image[(y + 1) * width + (x - 1)]) + ( 2 * input_image[(y + 1) * width + x]) + ( 1 * input_image[(y + 1) * width + (x + 1)]);

            // Save the results to the two separate output arrays
            gx_output[i] = gx;
            gy_output[i] = gy;
        }
    }
}