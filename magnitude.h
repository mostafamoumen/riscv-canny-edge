#ifndef MAGNITUDE_H
#define MAGNITUDE_H

#include <cstdint>

// L1 Norm: Fast, absolute values
void compute_magnitude_l1(const int16_t* gx, const int16_t* gy, int width, int height, uint8_t* magnitude_output);

// L2 Norm: Mathematically correct, uses square roots
void compute_magnitude_l2(const int16_t* gx, const int16_t* gy, int width, int height, uint8_t* magnitude_output);

#endif