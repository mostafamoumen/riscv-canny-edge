#include "image_io.h"
#include "gaussian_blur.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <width> <height>" << std::endl;
        return 1;
    }
    int w = std::stoi(argv[1]);
    int h = std::stoi(argv[2]);

    Image input = allocate_image(w, h);
    Image output = allocate_image(w, h);

    generate_test_image(input);
    apply_gaussian_blur(input, output); // Apply Task 2

    save_raw_image("blurred_output.raw", output);
    std::cout << "Blurred image saved to blurred_output.raw" << std::endl;

    free_image(input);
    free_image(output);
    return 0;
}
