#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

// Configuration
const int WIDTH = 100;
const int HEIGHT = 75; // Using non-power-of-two as recommended 
const int SIZE = WIDTH * HEIGHT;

// Helper function to save the raw buffer to a file
void save_raw(const std::string& filename, const std::vector<uint8_t>& buffer) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
        file.close();
        std::cout << "Generated: " << filename << "\n";
    } else {
        std::cerr << "Failed to open " << filename << " for writing.\n";
    }
}

int main() {
    std::vector<uint8_t> img(SIZE, 0);

    // 1. All-Black Image (Gaussian test: should produce all-black) 
    std::fill(img.begin(), img.end(), 0);
    save_raw("test_black.raw", img);

    // 2. Uniform 128 Image (Gaussian test: should produce 128, Sobel test: zero gradient) 
    std::fill(img.begin(), img.end(), 128);
    save_raw("test_uniform_128.raw", img);

    // 3. Impulse Image (Gaussian test: should spread symmetrically) 
    std::fill(img.begin(), img.end(), 0);
    img[(HEIGHT / 2) * WIDTH + (WIDTH / 2)] = 255; // Center pixel is bright white
    save_raw("test_impulse.raw", img);

    // 4. Vertical Edge (Sobel test: left=black, right=white -> large |Gx|, near-zero |Gy|) 
    std::fill(img.begin(), img.end(), 0);
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = WIDTH / 2; x < WIDTH; ++x) {
            img[y * WIDTH + x] = 255;
        }
    }
    save_raw("test_edge_vertical.raw", img);

    // 5. Horizontal Edge (Sobel test: top=black, bottom=white -> large |Gy|, near-zero |Gx|) 
    std::fill(img.begin(), img.end(), 0);
    for (int y = HEIGHT / 2; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            img[y * WIDTH + x] = 255;
        }
    }
    save_raw("test_edge_horizontal.raw", img);

    // 6. Diagonal Edge (Sobel test: significant values in both Gx and Gy) 
    std::fill(img.begin(), img.end(), 0);
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            // Simple diagonal line equation
            if (x > y * (WIDTH / (float)HEIGHT)) {
                img[y * WIDTH + x] = 255;
            }
        }
    }
    save_raw("test_edge_diagonal.raw", img);

    return 0;
}