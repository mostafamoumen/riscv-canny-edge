#include "sobel.h"
#include <cstring> // Required for std::memset
#include <riscv_vector.h>
#include <stdint.h>

void compute_sobel_gradients(const uint8_t* input_image, int width, int height, int16_t* gx_output, int16_t* gy_output) {
    // Zero out the entire output buffers to ensure borders are clean zeros
    std::memset(gx_output, 0, width * height * sizeof(int16_t));
    std::memset(gy_output, 0, width * height * sizeof(int16_t));

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int i = y * width + x;

            int16_t gx = 
                (-1 * input_image[(y - 1) * width + (x - 1)]) + (1 * input_image[(y - 1) * width + (x + 1)]) +
                (-2 * input_image[y * width + (x - 1)])       + (2 * input_image[y * width + (x + 1)]) +
                (-1 * input_image[(y + 1) * width + (x - 1)]) + (1 * input_image[(y + 1) * width + (x + 1)]);

            int16_t gy = 
                (-1 * input_image[(y - 1) * width + (x - 1)]) + (-2 * input_image[(y - 1) * width + x]) + (-1 * input_image[(y - 1) * width + (x + 1)]) +
                ( 1 * input_image[(y + 1) * width + (x - 1)]) + ( 2 * input_image[(y + 1) * width + x]) + ( 1 * input_image[(y + 1) * width + (x + 1)]);

            gx_output[i] = gx;
            gy_output[i] = gy;
        }
    }
}

void compute_sobel_gradients_rvv(const uint8_t* src, int width, int height, int16_t* gx, int16_t* gy) {
    // Process the image avoiding the 1-pixel outer border
    for (int y = 1; y < height - 1; y++) {
        int x = 1;
        int remaining = width - 2;
        
        while (remaining > 0) {
            // 1. Operation: Sets vector length for 16-bit processing.
            // 2. LMUL: Chose LMUL=4 because we need multiple vectors loaded at once (3 rows) and LMUL=8 would cause register spilling.
            // 3. VLEN Agnosticism: Dynamically sets 'vl' so the loop strip-mines safely across any hardware width.
            size_t vl = __riscv_vsetvl_e16m4(remaining);
            
            // Set up pointers for the 3 rows of our 3x3 window
            const uint8_t* row0 = src + (y - 1) * width + x;
            const uint8_t* row1 = src + y * width + x;
            const uint8_t* row2 = src + (y + 1) * width + x;
            
            // --- LOAD 8-BIT VECTORS (LMUL=2) ---
            // Top row
            vuint8m2_t p00_8 = __riscv_vle8_v_u8m2(row0 - 1, vl);
            vuint8m2_t p01_8 = __riscv_vle8_v_u8m2(row0, vl);
            vuint8m2_t p02_8 = __riscv_vle8_v_u8m2(row0 + 1, vl);
            
            // Middle row (p11 is the center pixel, which is multiplied by 0 in Sobel, so we skip loading it!)
            vuint8m2_t p10_8 = __riscv_vle8_v_u8m2(row1 - 1, vl);
            vuint8m2_t p12_8 = __riscv_vle8_v_u8m2(row1 + 1, vl);
            
            // Bottom row
            vuint8m2_t p20_8 = __riscv_vle8_v_u8m2(row2 - 1, vl);
            vuint8m2_t p21_8 = __riscv_vle8_v_u8m2(row2, vl);
            vuint8m2_t p22_8 = __riscv_vle8_v_u8m2(row2 + 1, vl);
            
            // --- WIDEN TO 16-BIT SIGNED VECTORS (LMUL=4) ---
            vint16m4_t p00 = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(p00_8, vl));
            vint16m4_t p01 = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(p01_8, vl));
            vint16m4_t p02 = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(p02_8, vl));
            
            vint16m4_t p10 = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(p10_8, vl));
            vint16m4_t p12 = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(p12_8, vl));
            
            vint16m4_t p20 = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(p20_8, vl));
            vint16m4_t p21 = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(p21_8, vl));
            vint16m4_t p22 = __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(p22_8, vl));
            
            // --- COMPUTE Gx ---
            // Gx = (p02 - p00) + 2*(p12 - p10) + (p22 - p20)
            
            // 1. Operation: Vector addition/subtraction to calculate the Sobel gradient differences.
            // 2. LMUL: Kept at LMUL=4 to maximize throughput for 16-bit signed gradient data.
            // 3. VLEN Agnosticism: Will compute parallel additions/subtractions proportional to the hardware VLEN.
            vint16m4_t gx0 = __riscv_vsub_vv_i16m4(p02, p00, vl);
            vint16m4_t gx1 = __riscv_vsub_vv_i16m4(p12, p10, vl);
            vint16m4_t gx2 = __riscv_vsub_vv_i16m4(p22, p20, vl);
            
            // 1. Operation: Vector shift-left by a scalar. Used as a fast substitute for multiplying by 2.
            // 2. LMUL: LMUL=4 matches the rest of the 16-bit data path.
            // 3. VLEN Agnosticism: Will shift 'vl' elements simultaneously regardless of underlying physical VLEN.
            gx1 = __riscv_vsll_vx_i16m4(gx1, 1, vl); 
            vint16m4_t v_gx = __riscv_vadd_vv_i16m4(__riscv_vadd_vv_i16m4(gx0, gx1, vl), gx2, vl);
            
            // --- COMPUTE Gy ---
            // Gy = (p20 - p00) + 2*(p21 - p01) + (p22 - p02)
            vint16m4_t gy0 = __riscv_vsub_vv_i16m4(p20, p00, vl);
            vint16m4_t gy1 = __riscv_vsub_vv_i16m4(p21, p01, vl);
            vint16m4_t gy2 = __riscv_vsub_vv_i16m4(p22, p02, vl);
            
            gy1 = __riscv_vsll_vx_i16m4(gy1, 1, vl);
            vint16m4_t v_gy = __riscv_vadd_vv_i16m4(__riscv_vadd_vv_i16m4(gy0, gy1, vl), gy2, vl);
            
            // --- STORE RESULTS ---
            __riscv_vse16_v_i16m4(gx + y * width + x, v_gx, vl);
            __riscv_vse16_v_i16m4(gy + y * width + x, v_gy, vl);
            
            x += vl;
            remaining -= vl;
        }
    }
}