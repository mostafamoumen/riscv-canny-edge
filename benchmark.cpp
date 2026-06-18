#include "image_io.h"
#include "gaussian_blur.h"
#include "sobel.h"
#include "magnitude.h"
#include "direction.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <algorithm> // Needed for std::min

const int ITERATIONS = 100;
volatile int dummy_sink = 0;

// Forward declarations for our RVV multi-LMUL implementations 
// so the compiler knows they exist!
void gaussian_blur_rvv_m1(const uint8_t* src, uint8_t* dst, int width, int height);
void gaussian_blur_rvv_m2(const uint8_t* src, uint8_t* dst, int width, int height);
void gaussian_blur_rvv_m4(const uint8_t* src, uint8_t* dst, int width, int height);
void compute_magnitude_l1_rvv(const int16_t* gx, const int16_t* gy, int width, int height, uint8_t* mag);
void compute_sobel_gradients_rvv(const uint8_t* src, int width, int height, int16_t* gx, int16_t* gy);

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

// 1. Profiling Gaussian Separable Stage
    auto start_gaussian = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        apply_gaussian_separable_padded<uint8_t, int32_t, int16_t>(input, blur_out);
        dummy_sink += blur_out.data[0]; 
    }
    auto end_gaussian = std::chrono::high_resolution_clock::now();
    accum_gaussian = std::chrono::duration<double, std::milli>(end_gaussian - start_gaussian).count();

    // 2. Profiling Sobel Gradients Stage
    auto start_sobel = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        compute_sobel_gradients(blur_out.data, width, height, gx, gy);
        dummy_sink += gx[0]; 
    }
    auto end_sobel = std::chrono::high_resolution_clock::now();
    accum_sobel = std::chrono::duration<double, std::milli>(end_sobel - start_sobel).count();

    // 3. Profiling Magnitude L1 Stage
    auto start_magnitude = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        compute_magnitude_l1(gx, gy, width, height, mag);
        dummy_sink += mag[0];
    }
    auto end_magnitude = std::chrono::high_resolution_clock::now();
    accum_magnitude = std::chrono::duration<double, std::milli>(end_magnitude - start_magnitude).count();

    // 4. Profiling Direction Quantization Stage
    auto start_direction = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        compute_direction(gx, gy, width, height, dir);
        dummy_sink += dir[0];
    }
    auto end_direction = std::chrono::high_resolution_clock::now();
    accum_direction = std::chrono::duration<double, std::milli>(end_direction - start_direction).count();

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


    std::cout << "\n==================================================\n";
    std::cout << "PHASE 6.2: RVV LMUL SWEEP (GAUSSIAN BLUR OVER " << ITERATIONS << " ITERATIONS)\n";
    std::cout << "==================================================\n";

    // Setup pointers and dimensions
    const uint8_t* src_buf = input.data;  
    uint8_t* rvv_dst_buf = blur_out.data; 
    int w = input.width;
    int h = input.height;

    // --------------------------------------------------
    // Time LMUL = 1
    // --------------------------------------------------
    auto start_m1 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < ITERATIONS; i++) {
        gaussian_blur_rvv_m1(src_buf, rvv_dst_buf, w, h);
        dummy_sink += rvv_dst_buf[0];
    }
    auto end_m1 = std::chrono::high_resolution_clock::now();
    double time_m1 = std::chrono::duration<double, std::milli>(end_m1 - start_m1).count();

    // --------------------------------------------------
    // Time LMUL = 2
    // --------------------------------------------------
    auto start_m2 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < ITERATIONS; i++) {
        gaussian_blur_rvv_m2(src_buf, rvv_dst_buf, w, h);
        dummy_sink += rvv_dst_buf[0];
    }
    auto end_m2 = std::chrono::high_resolution_clock::now();
    double time_m2 = std::chrono::duration<double, std::milli>(end_m2 - start_m2).count();

    // --------------------------------------------------
    // Time LMUL = 4
    // --------------------------------------------------
    auto start_m4 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < ITERATIONS; i++) {
        gaussian_blur_rvv_m4(src_buf, rvv_dst_buf, w, h);
        dummy_sink += rvv_dst_buf[0];
    }
    auto end_m4 = std::chrono::high_resolution_clock::now();
    double time_m4 = std::chrono::duration<double, std::milli>(end_m4 - start_m4).count();

    // Print the results
    std::cout << "LMUL = 1 Version :  " << time_m1 << " ms\n";
    std::cout << "LMUL = 2 Version :  " << time_m2 << " ms\n";
    std::cout << "LMUL = 4 Version :  " << time_m4 << " ms\n";
    
    // Compare best RVV to your Scalar baseline
    double best_rvv = std::min({time_m1, time_m2, time_m4});
    double speedup = accum_gaussian / best_rvv;
    std::cout << "--------------------------------------------------\n";
    std::cout << "Maximum Speedup vs Scalar : " << speedup << "x faster!\n";
    std::cout << "==================================================\n";




std::cout << "\n==================================================\n";
std::cout << "PHASE 6.5: RVV MAGNITUDE PROFILING\n";
std::cout << "==================================================\n";

auto start_mag_rvv = std::chrono::high_resolution_clock::now();
for (int i = 0; i < ITERATIONS; i++) {
    compute_magnitude_l1_rvv(gx, gy, width, height, mag);
    dummy_sink += mag[0];
}
auto end_mag_rvv = std::chrono::high_resolution_clock::now();
double time_mag_rvv = std::chrono::duration<double, std::milli>(end_mag_rvv - start_mag_rvv).count();

std::cout << "Scalar Magnitude Time : " << accum_magnitude << " ms\n";
std::cout << "RVV L1 Magnitude Time : " << time_mag_rvv << " ms\n";
std::cout << "Magnitude Speedup     : " << (accum_magnitude / time_mag_rvv) << "x faster!\n";
std::cout << "==================================================\n";





std::cout << "\n==================================================\n";
std::cout << "PHASE 6.X: RVV SOBEL PROFILING\n";
std::cout << "==================================================\n";

auto start_sobel_rvv = std::chrono::high_resolution_clock::now();
for (int i = 0; i < ITERATIONS; i++) {
    compute_sobel_gradients_rvv(blur_out.data, width, height, gx, gy);
    dummy_sink += gx[0];
}
auto end_sobel_rvv = std::chrono::high_resolution_clock::now();
double time_sobel_rvv = std::chrono::duration<double, std::milli>(end_sobel_rvv - start_sobel_rvv).count();

std::cout << "Scalar Sobel Time : " << accum_sobel << " ms\n";
std::cout << "RVV Sobel Time    : " << time_sobel_rvv << " ms\n";
std::cout << "Sobel Speedup     : " << (accum_sobel / time_sobel_rvv) << "x faster!\n";
std::cout << "==================================================\n";


    // Clean up memory safely at the very end
    free_image(input); 
    free_image(blur_out);
    free(gx); 
    free(gy); 
    free(mag); 
    free(dir);

    return 0;
}