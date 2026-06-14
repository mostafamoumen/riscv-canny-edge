#include "image_io.h"
#include "gaussian_blur.h"
#include "sobel.h"
#include "magnitude.h"
#include "direction.h"
#include <iostream>
#include <string>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <width> <height> [input_raw_file]" << std::endl;
        return 1;
    }

    int width = std::stoi(argv[1]);
    int height = std::stoi(argv[2]);
    int total_pixels = width * height;

    Image input = allocate_image(width, height);
    Image output_2d = allocate_image(width, height);
    Image output_sep = allocate_image(width, height);

    int16_t* gx_buffer = static_cast<int16_t*>(aligned_alloc(64, total_pixels * sizeof(int16_t)));
    int16_t* gy_buffer = static_cast<int16_t*>(aligned_alloc(64, total_pixels * sizeof(int16_t)));
    uint8_t* magnitude_l1 = static_cast<uint8_t*>(aligned_alloc(64, total_pixels * sizeof(uint8_t)));
    uint8_t* magnitude_l2 = static_cast<uint8_t*>(aligned_alloc(64, total_pixels * sizeof(uint8_t)));
    uint8_t* direction_map = static_cast<uint8_t*>(aligned_alloc(64, total_pixels * sizeof(uint8_t)));

    if (!gx_buffer || !gy_buffer || !magnitude_l1 || !magnitude_l2 || !direction_map) {
        std::cerr << "Phase 2 buffer memory allocation failed." << std::endl;
        return 1;
    }

    if (argc >= 4) {
        if (!load_raw_image(argv[3], input)) {
            std::cerr << "Failed to load " << argv[3] << std::endl;
            return 1;
        }
        std::cout << "Loaded image: " << argv[3] << std::endl;
    } else {
        generate_circle(input);
        std::cout << "Generated circle test pattern." << std::endl;
    }

    // STAGE 1: Gaussian Blur Execution
    apply_gaussian_2d<uint8_t, int32_t, int16_t>(input, output_2d);
    save_raw_image("blur_2d.raw", output_2d);
    std::cout << "Saved 2D blur to blur_2d.raw" << std::endl;

    apply_gaussian_separable<uint8_t, int32_t, int16_t>(input, output_sep);
    save_raw_image("blur_separable.raw", output_sep);
    std::cout << "Saved Separable blur to blur_separable.raw" << std::endl;

    // STAGE 2: Sobel, Magnitude, and Direction Operators
    std::cout << "\nExecuting Full Stage 2 Scalar Pipeline..." << std::endl;
    compute_sobel_gradients(output_sep.data, width, height, gx_buffer, gy_buffer);
    
    compute_magnitude_l1(gx_buffer, gy_buffer, width, height, magnitude_l1);
    save_raw_image("magnitude_l1.raw", Image{magnitude_l1, width, height});
    std::cout << "-> Saved L1 Magnitude to magnitude_l1.raw" << std::endl;

    compute_magnitude_l2(gx_buffer, gy_buffer, width, height, magnitude_l2);
    save_raw_image("magnitude_l2.raw", Image{magnitude_l2, width, height});
    std::cout << "-> Saved L2 Magnitude to magnitude_l2.raw" << std::endl;

    compute_direction(gx_buffer, gy_buffer, width, height, direction_map);
    save_raw_image("direction.raw", Image{direction_map, width, height});
    std::cout << "-> Saved Quantized Directions to direction.raw" << std::endl;

    // Clean up
    free_image(input);
    free_image(output_2d);
    free_image(output_sep);
    free(gx_buffer);
    free(gy_buffer);
    free(magnitude_l1);
    free(magnitude_l2);
    free(direction_map);

    std::cout << "\nPhase 2 Baseline Execution Complete Summary Successfully Generated." << std::endl;
    return 0;
}