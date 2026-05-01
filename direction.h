#ifndef DIRECTION_H
#define DIRECTION_H

#include <cstdint>

// Computes the gradient direction and quantizes it to 0, 45, 90, or 135 degrees.
void compute_direction(const int16_t* gx, const int16_t* gy, int width, int height, uint8_t* direction_output);

#endif