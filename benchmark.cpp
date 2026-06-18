#include "image_io.h"
#include "gaussian_blur.h"
#include "sobel.h"
#include "magnitude.h"
#include "direction.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <chrono>  // Modern C++ timing library (replaces clock_gettime)

const int ITERATIONS = 100;

// Dummy volatile variable to prevent aggressive compilers (-O3) from deleting our loops
volatile int dummy_sink = 0;

int main() {
    int width = 100;
    int height = 75;
    int total_pixels = width * height;

    Image input = allocate_image(width, height);
    generate_circle(input);

    Image blur_out = allocate_image(width, height);
    int16_t* gx = static_cast<int16_t*>(aligned_alloc(64, total_pixels * sizeof(int16_t)));
    int16_t* gy = static_cast<int16_t*>(aligned_alloc(64, total_pixels * sizeof(int16_t)));
    uint8_t* mag = static_cast<uint8_t*>(aligned_alloc(64, total_pixels * sizeof(uint8_t)));
    uint8_t* dir = static_cast<uint8_t*>(aligned_alloc(64, total_pixels * sizeof(uint8_t)));

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "--- Performance Benchmark (" << ITERATIONS << " Iterations) ---\n";

    // 1a. Original Gaussian Separable Baseline
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        apply_gaussian_separable<uint8_t, int32_t, int16_t>(input, blur_out);
        dummy_sink += blur_out.data[0]; // Prevent Dead Code Elimination
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "Gaussian Separable (Base): " << ms.count() / ITERATIONS << " ms\n";

    // 1b. New Restructured Padded Experiment Loop
    std::cout << "\n--- Running Restructured Padded Experiment Loop ---\n";
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        apply_gaussian_separable_padded<uint8_t, int32_t, int16_t>(input, blur_out);
        dummy_sink += blur_out.data[0]; 
    }
    end = std::chrono::high_resolution_clock::now();
    ms = end - start;
    std::cout << "Gaussian Padded Separable: " << ms.count() / ITERATIONS << " ms\n\n";

    // 2. Sobel
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        compute_sobel_gradients(blur_out.data, width, height, gx, gy);
        dummy_sink += gx[0]; 
    }
    end = std::chrono::high_resolution_clock::now();
    ms = end - start;
    std::cout << "Sobel Gradients:    " << ms.count() / ITERATIONS << " ms\n";

    // 3. Magnitude L1
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        compute_magnitude_l1(gx, gy, width, height, mag);
        dummy_sink += mag[0];
    }
    end = std::chrono::high_resolution_clock::now();
    ms = end - start;
    std::cout << "Magnitude (L1):     " << ms.count() / ITERATIONS << " ms\n";

    // 4. Direction
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        compute_direction(gx, gy, width, height, dir);
        dummy_sink += dir[0];
    }
    end = std::chrono::high_resolution_clock::now();
    ms = end - start;
    std::cout << "Direction:          " << ms.count() / ITERATIONS << " ms\n";

    // Free memory
    free_image(input); free_image(blur_out);
    free(gx); free(gy); free(mag); free(dir);
    
    return 0;
}