#include "image_io.h"
#include "gaussian_blur.h"
#include "sobel.h"
#include "magnitude.h"
#include "direction.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <chrono>

const int ITERATIONS = 100;
volatile int dummy_sink = 0;

int main() {
    // Utilizing the mandated 100x75 non-power-of-two frame size to exercise tail cases
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
    std::cout << "==================================================\n";
    std::cout << "PHASE 5: INITIALIZING SCALAR PROFILING HARNESS\n";
    std::cout << "==================================================\n";

    // Accumulators for per-stage tracking
    double accum_gaussian = 0.0;
    double accum_sobel = 0.0;
    double accum_magnitude = 0.0;
    double accum_direction = 0.0;

    // 1. Profiling Gaussian Padded Separable Stage
    for (int i = 0; i < ITERATIONS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        apply_gaussian_separable_padded<uint8_t, int32_t, int16_t>(input, blur_out);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = end - start;
        accum_gaussian += ms.count();
        dummy_sink += blur_out.data[0]; 
    }

    // 2. Profiling Sobel Gradients Stage
    for (int i = 0; i < ITERATIONS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        compute_sobel_gradients(blur_out.data, width, height, gx, gy);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = end - start;
        accum_sobel += ms.count();
        dummy_sink += gx[0]; 
    }

    // 3. Profiling Magnitude L1 Stage
    for (int i = 0; i < ITERATIONS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        compute_magnitude_l1(gx, gy, width, height, mag);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = end - start;
        accum_magnitude += ms.count();
        dummy_sink += mag[0];
    }

    // 4. Profiling Direction Quantization Stage
    for (int i = 0; i < ITERATIONS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        compute_direction(gx, gy, width, height, dir);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = end - start;
        accum_direction += ms.count();
        dummy_sink += dir[0];
    }

    // Compute System Metrics
    double total_pipeline_time = accum_gaussian + accum_sobel + accum_magnitude + accum_direction;

    std::cout << "\n--------------------------------------------------\n";
    std::cout << "PERFORMANCE BREAKDOWN OVER " << ITERATIONS << " ITERATIONS\n";
    std::cout << "--------------------------------------------------\n";
    std::cout << "Gaussian Blur : " << std::setw(8) << accum_gaussian << " ms | Percentage: " << (accum_gaussian / total_pipeline_time) * 100.0 << "%\n";
    std::cout << "Sobel Filter  : " << std::setw(8) << accum_sobel    << " ms | Percentage: " << (accum_sobel / total_pipeline_time) * 100.0 << "%\n";
    std::cout << "Magnitude (L1): " << std::setw(8) << accum_magnitude<< " ms | Percentage: " << (accum_magnitude / total_pipeline_time) * 100.0 << "%\n";
    std::cout << "Direction     : " << std::setw(8) << accum_direction<< " ms | Percentage: " << (accum_direction / total_pipeline_time) * 100.0 << "%\n";
    std::cout << "--------------------------------------------------\n";
    std::cout << "Total Pure Computational Pipeline Time: " << total_pipeline_time << " ms\n";
    std::cout << "==================================================\n";

    free_image(input); 
    free_image(blur_out);
    free(gx); 
    free(gy); 
    free(mag); 
    free(dir);
    
    return 0;
}