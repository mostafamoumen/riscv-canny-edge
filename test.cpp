#include <gtest/gtest.h>
#include <vector>
#include <fstream>
#include <cstdint>
#include <string>

// Include your actual headers
#include "gaussian_blur.h"
#include "image_io.h"
#include "direction.h"

const int WIDTH = 100;
const int HEIGHT = 75;

// Helper to load raw images directly into your Image struct
#include <algorithm> // Add this near the top!

// Helper to load raw images directly into your Image struct
Image load_raw_to_image(const std::string& filename) {
    Image img;
    img.width = WIDTH;   
    img.height = HEIGHT;
    
    // Allocate memory for the raw pointer
    img.data = new uint8_t[WIDTH * HEIGHT]; 

    std::ifstream file(filename, std::ios::binary);
    if (file) {
        // Read directly into the raw pointer
        file.read(reinterpret_cast<char*>(img.data), WIDTH * HEIGHT);
    }
    return img;
}

// Helper to create an empty output Image
Image create_empty_image(uint8_t fill_value = 0) {
    Image img;
    img.width = WIDTH;
    img.height = HEIGHT;
    
    // Allocate memory and fill it with the starting value
    img.data = new uint8_t[WIDTH * HEIGHT];
    std::fill(img.data, img.data + (WIDTH * HEIGHT), fill_value);
    
    return img;
}

// 1. Uniform Image Test
TEST(GaussianBlurTest, UniformImage) {
    Image input = load_raw_to_image("test_uniform_128.raw");
    Image output = create_empty_image(0);

    // EXACT FUNCTION CALL:
    // Calling your separable gaussian template. 
    // Assuming Pixel=uint8_t, Accumulator=int, Kernel=int. Change if you used floats!
    apply_gaussian_separable<uint8_t, int, int>(input, output);

    // Check interior pixels (skip the 2-pixel border of a 5x5 kernel)
    for (int y = 2; y < HEIGHT - 2; ++y) {
        for (int x = 2; x < WIDTH - 2; ++x) {
            int val = output.data[y * WIDTH + x];
            // Allow +/- 1 for integer division rounding differences
            EXPECT_NEAR(val, 128, 1); 
        }
    }
}

// 2. All-Black Image Test
TEST(GaussianBlurTest, AllBlackImage) {
    Image input = load_raw_to_image("test_black.raw");
    // Initialize output to 255 to guarantee the function actually modifies it
    Image output = create_empty_image(255); 

    // EXACT FUNCTION CALL
    apply_gaussian_separable<uint8_t, int, int>(input, output);

    for (int y = 2; y < HEIGHT - 2; ++y) {
        for (int x = 2; x < WIDTH - 2; ++x) {
            EXPECT_EQ(output.data[y * WIDTH + x], 0);
        }
    }
}

// 3. Impulse Symmetry Test
TEST(GaussianBlurTest, ImpulseSymmetry) {
    Image input = load_raw_to_image("test_impulse.raw");
    Image output = create_empty_image(0);

    // EXACT FUNCTION CALL
    apply_gaussian_separable<uint8_t, int, int>(input, output);

    int cy = HEIGHT / 2;
    int cx = WIDTH / 2;

    // Check symmetry around the center pixel
    EXPECT_EQ(output.data[(cy - 1) * WIDTH + cx], output.data[(cy + 1) * WIDTH + cx]); // Top vs Bottom
    EXPECT_EQ(output.data[cy * WIDTH + (cx - 1)], output.data[cy * WIDTH + (cx + 1)]); // Left vs Right
}


// =========================================================================
// 3. GRADIENT DIRECTION TESTS (Full Image Simulation)
// =========================================================================

TEST(GradientDirectionTest, AssignmentRequirements) {
    // Allocate image-sized arrays for Gx, Gy, and the output directions
    std::vector<int16_t> gx(WIDTH * HEIGHT, 0);
    std::vector<int16_t> gy(WIDTH * HEIGHT, 0);
    std::vector<uint8_t> dir_output(WIDTH * HEIGHT, 255); 

    int center_idx = (HEIGHT / 2) * WIDTH + (WIDTH / 2);

    // -----------------------------------------------------------------
    // Scenario A: Vertical Edge (Produces a Horizontal Gradient)
    // Requirement: Direction should be 0
    // -----------------------------------------------------------------
    std::fill(gx.begin(), gx.end(), 500); // Strong horizontal change everywhere
    std::fill(gy.begin(), gy.end(), 0);   // No vertical change
    
    compute_direction(gx.data(), gy.data(), WIDTH, HEIGHT, dir_output.data());
    
    EXPECT_EQ(dir_output[center_idx], 0) 
        << "Failed: Vertical edge image did not produce direction 0!";


    // -----------------------------------------------------------------
    // Scenario B: Horizontal Edge (Produces a Vertical Gradient)
    // Requirement: Direction should be 2
    // -----------------------------------------------------------------
    std::fill(gx.begin(), gx.end(), 0);   // No horizontal change
    std::fill(gy.begin(), gy.end(), 500); // Strong vertical change everywhere
    
    compute_direction(gx.data(), gy.data(), WIDTH, HEIGHT, dir_output.data());
    
    EXPECT_EQ(dir_output[center_idx], 2) 
        << "Failed: Horizontal edge image did not produce direction 2!";


    // -----------------------------------------------------------------
    // Scenario C: Diagonal Edge (Produces Significant Gx and Gy)
    // Requirement: Direction should be 1 or 3
    // -----------------------------------------------------------------
    std::fill(gx.begin(), gx.end(), 400); // Equal changes in both axes (45 degrees)
    std::fill(gy.begin(), gy.end(), 400); 
    
    compute_direction(gx.data(), gy.data(), WIDTH, HEIGHT, dir_output.data());
    
    uint8_t diag_dir = dir_output[center_idx];
    EXPECT_TRUE(diag_dir == 1 || diag_dir == 3) 
        << "Failed: Diagonal edge image did not produce direction 1 or 3! Got: " << (int)diag_dir;
}