#include "direction.h"
#include <cmath> // We still need this for std::abs()

void compute_direction(const int16_t* gx, const int16_t* gy, int width, int height, uint8_t* direction_output) {
    int total_pixels = width * height;

    for (int i = 0; i < total_pixels; i++) {
        int16_t x = gx[i];
        int16_t y = gy[i];
        
        // 1. Get the absolute values
        int ax = std::abs(x);
        int ay = std::abs(y);

        // 2. Use integer cross-multiplication to find the angle bucket
        if (5 * ay < 2 * ax) {
            // Angle is near 0 degrees
            direction_output[i] = 0;
            
        } else if (5 * ay > 12 * ax) {
            // Angle is near 90 degrees
            direction_output[i] = 90;
            
        } else {
            // Angle is in the middle (a diagonal). 
            // We check if the signs of x and y are opposite to each other.
            if ((x < 0) != (y < 0)) {
                direction_output[i] = 135; 
            } else {
                direction_output[i] = 45;
            }
        }
    }
}