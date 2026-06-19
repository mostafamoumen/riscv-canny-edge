#include "image_io.h"
#include "gaussian_blur.h"
#include "sobel.h"
#include "magnitude.h"
#include "direction.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono> // Standard C++ high-resolution timing library

#define ITERATIONS 100

// Helper to get time in milliseconds using standard C++ chrono
double get_time_ms() {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    return duration.count() / 1000.0;
}

int main() {
    // Setting up a stable non-power-of-two resolution to verify tail-handling stability
    const int width = 100;
    const int height = 75;
    const int total_pixels = width * height;

    // Allocate aligned memory to fulfill vector boundary rules from Phase 2 guidelines
    Image input = allocate_image(width, height);
    Image blur_out = allocate_image(width, height);
    
    int16_t* gx_buffer = static_cast<int16_t*>(aligned_alloc(64, total_pixels * sizeof(int16_t)));
    int16_t* gy_buffer = static_cast<int16_t*>(aligned_alloc(64, total_pixels * sizeof(int16_t)));
    uint8_t* mag_l1 = static_cast<uint8_t*>(aligned_alloc(64, total_pixels * sizeof(uint8_t)));
    uint8_t* mag_l2 = static_cast<uint8_t*>(aligned_alloc(64, total_pixels * sizeof(uint8_t)));
    uint8_t* dir_map = static_cast<uint8_t*>(aligned_alloc(64, total_pixels * sizeof(uint8_t)));

    // Synthesize a basic geometric pattern for the baseline input
    for (int i = 0; i < total_pixels; ++i) {
        input.data[i] = static_cast<uint8_t>((i % width > width / 2) ? 255 : 0);
    }

    double start, end;

    // 1. Measure Gaussian Blur (Separable implementation used as baseline)
    start = get_time_ms();
    for (int i = 0; i < ITERATIONS; ++i) {
        apply_gaussian_separable<uint8_t, int32_t, int16_t>(input, blur_out);
    }
    end = get_time_ms();
    double gaussian_time = (end - start) / ITERATIONS;

    // 2. Measure Sobel Gradient Computation
    start = get_time_ms();
    for (int i = 0; i < ITERATIONS; ++i) {
        compute_sobel_gradients(blur_out.data, width, height, gx_buffer, gy_buffer);
    }
    end = get_time_ms();
    double sobel_time = (end - start) / ITERATIONS;

    // 3. Measure Magnitude L1 Computation
    start = get_time_ms();
    for (int i = 0; i < ITERATIONS; ++i) {
        compute_magnitude_l1(gx_buffer, gy_buffer, width, height, mag_l1);
    }
    end = get_time_ms();
    double mag_l1_time = (end - start) / ITERATIONS;

    // 4. Measure Magnitude L2 Computation
    start = get_time_ms();
    for (int i = 0; i < ITERATIONS; ++i) {
        compute_magnitude_l2(gx_buffer, gy_buffer, width, height, mag_l2);
    }
    end = get_time_ms();
    double mag_l2_time = (end - start) / ITERATIONS;

    // 5. Measure Direction Quantization
    start = get_time_ms();
    for (int i = 0; i < ITERATIONS; ++i) {
        compute_direction(gx_buffer, gy_buffer, width, height, dir_map);
    }
    end = get_time_ms();
    double direction_time = (end - start) / ITERATIONS;

    // Print results out in a clean, parseable format for the sweep script
    std::cout << "GAUSSIAN_TIME:" << gaussian_time << " ms" << std::endl;
    std::cout << "SOBEL_TIME:" << sobel_time << " ms" << std::endl;
    std::cout << "MAG_L1_TIME:" << mag_l1_time << " ms" << std::endl;
    std::cout << "MAG_L2_TIME:" << mag_l2_time << " ms" << std::endl;
    std::cout << "DIRECTION_TIME:" << direction_time << " ms" << std::endl;

    // Free all allocated memory blocks
    free(input.data);
    free(blur_out.data);
    free(gx_buffer);
    free(gy_buffer);
    free(mag_l1);
    free(mag_l2);
    free(dir_map);

    return 0;
}
