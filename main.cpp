#include "image_io.h"
#include "gaussian_blur.h"
#include "sobel.h"
#include "magnitude.h"
#include "direction.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstring>

// Ensure C++ prototypes are visible if they aren't in your headers
void apply_gaussian_separable_rvv(const uint8_t* input, uint8_t* output, int width, int height);
void compute_sobel_gradients_rvv(const uint8_t* src, int width, int height, int16_t* gx, int16_t* gy);
void compute_magnitude_l1_rvv(const int16_t* gx, const int16_t* gy, int width, int height, uint8_t* mag);

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <width> <height> [input_raw_file]" << std::endl;
        return 1;
    }

    int width = std::stoi(argv[1]);
    int height = std::stoi(argv[2]);
    int total_pixels = width * height;

    Image input = allocate_image(width, height);
    Image output_sep = allocate_image(width, height);
    Image output_rvv_blur = allocate_image(width, height);

    int16_t* gx_buffer = static_cast<int16_t*>(aligned_alloc(64, total_pixels * sizeof(int16_t)));
    int16_t* gy_buffer = static_cast<int16_t*>(aligned_alloc(64, total_pixels * sizeof(int16_t)));
    uint8_t* magnitude_l1 = static_cast<uint8_t*>(aligned_alloc(64, total_pixels * sizeof(uint8_t)));
    
    // RVV Buffers
    int16_t* gx_rvv_buffer = static_cast<int16_t*>(aligned_alloc(64, total_pixels * sizeof(int16_t)));
    int16_t* gy_rvv_buffer = static_cast<int16_t*>(aligned_alloc(64, total_pixels * sizeof(int16_t)));
    uint8_t* magnitude_l1_rvv = static_cast<uint8_t*>(aligned_alloc(64, total_pixels * sizeof(uint8_t)));

    if (!gx_buffer || !gy_buffer || !magnitude_l1 || !gx_rvv_buffer || !gy_rvv_buffer || !magnitude_l1_rvv) {
        std::cerr << "Buffer memory allocation failed." << std::endl;
        return 1;
    }

    if (argc >= 4) {
        if (!load_raw_image(argv[3], input)) {
            std::cerr << "Failed to load " << argv[3] << std::endl;
            return 1;
        }
        std::cout << "Loaded real image: " << argv[3] << " (" << width << "x" << height << ")" << std::endl;
    } else {
        generate_circle(input);
        std::cout << "Generated circle test pattern." << std::endl;
    }

    // =========================================================================
    // SCALAR PIPELINE (Baseline)
    // =========================================================================
    std::cout << "\nExecuting Scalar Pipeline..." << std::endl;
    apply_gaussian_separable<uint8_t, int32_t, int16_t>(input, output_sep);
    compute_sobel_gradients(output_sep.data, width, height, gx_buffer, gy_buffer);
    compute_magnitude_l1(gx_buffer, gy_buffer, width, height, magnitude_l1);
    save_raw_image("output_scalar.raw", Image{magnitude_l1, width, height});
    std::cout << "-> Saved scalar output to output_scalar.raw" << std::endl;

    // =========================================================================
    // RVV PIPELINE (Your Optimized Vector Code!)
    // =========================================================================
    std::cout << "\nExecuting RVV Accelerated Pipeline..." << std::endl;
    
    // Initialize RVV gradient buffers to zero to clear borders cleanly
    std::memset(gx_rvv_buffer, 0, total_pixels * sizeof(int16_t));
    std::memset(gy_rvv_buffer, 0, total_pixels * sizeof(int16_t));

    // Run your vector code functions back-to-back
    apply_gaussian_separable_rvv(input.data, output_rvv_blur.data, width, height);
    compute_sobel_gradients_rvv(output_rvv_blur.data, width, height, gx_rvv_buffer, gy_rvv_buffer);
    compute_magnitude_l1_rvv(gx_rvv_buffer, gy_rvv_buffer, width, height, magnitude_l1_rvv);
    
    save_raw_image("output_rvv.raw", Image{magnitude_l1_rvv, width, height});
    std::cout << "-> Saved RVV accelerated output to output_rvv.raw" << std::endl;

    // Clean up
    free_image(input);
    free_image(output_sep);
    free_image(output_rvv_blur);
    free(gx_buffer); free(gy_buffer); free(magnitude_l1);
    free(gx_rvv_buffer); free(gy_rvv_buffer); free(magnitude_l1_rvv);

    std::cout << "\nExecution Complete! Compare output_scalar.raw and output_rvv.raw" << std::endl;
    return 0;
}