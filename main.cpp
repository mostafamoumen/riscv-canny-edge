#include "image_io.h"
#include "gaussian_blur.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <width> <height> [input_raw_file]" << std::endl;
        return 1;
    }

    int width = std::stoi(argv[1]);
    int height = std::stoi(argv[2]);

    Image input = allocate_image(width, height);
    Image output_2d = allocate_image(width, height);
    Image output_sep = allocate_image(width, height);

    // If a file is provided, load it. Otherwise, generate a circle.
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

    // Test 1: Full 2D Convolution
    apply_gaussian_2d<uint8_t, int32_t, int16_t>(input, output_2d);
    save_raw_image("blur_2d.raw", output_2d);
    std::cout << "Saved 2D blur to blur_2d.raw" << std::endl;

    // Test 2: Separable Convolution
    apply_gaussian_separable<uint8_t, int32_t, int16_t>(input, output_sep);
    save_raw_image("blur_separable.raw", output_sep);
    std::cout << "Saved Separable blur to blur_separable.raw" << std::endl;

    free_image(input);
    free_image(output_2d);
    free_image(output_sep);
    return 0;
}