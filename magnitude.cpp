#include "magnitude.h"
#include <cmath>     // For std::abs and std::sqrt
#include <algorithm> // For std::max
#include <riscv_vector.h>
#include <stdint.h>

void compute_magnitude_l1(const int16_t* gx, const int16_t* gy, int width, int height, uint8_t* magnitude_output) {
    int total_pixels = width * height;
    int max_mag = 0;

    // PASS 1: Find the maximum magnitude in the entire image
    for (int i = 0; i < total_pixels; i++) {
        int mag = std::abs(gx[i]) + std::abs(gy[i]);
        if (mag > max_mag) {
            max_mag = mag;
        }
    }

    // Prevent division by zero if the image is completely blank/black
    if (max_mag == 0) max_mag = 1;

    // PASS 2: Normalize to [0, 255]
    for (int i = 0; i < total_pixels; i++) {
        int mag = std::abs(gx[i]) + std::abs(gy[i]);
        // Scale the value relative to the maximum, then store as 8-bit integer
        magnitude_output[i] = static_cast<uint8_t>((mag * 255) / max_mag);
    }
}

void compute_magnitude_l2(const int16_t* gx, const int16_t* gy, int width, int height, uint8_t* magnitude_output) {
    int total_pixels = width * height;
    float max_mag = 0.0f;

    // PASS 1: Find the maximum magnitude using the Pythagorean theorem
    for (int i = 0; i < total_pixels; i++) {
        // We cast to float before multiplying to prevent integer overflow
        float mag = std::sqrt(static_cast<float>(gx[i]*gx[i] + gy[i]*gy[i]));
        if (mag > max_mag) {
            max_mag = mag;
        }
    }

    if (max_mag == 0.0f) max_mag = 1.0f;

    // PASS 2: Normalize to [0, 255]
    for (int i = 0; i < total_pixels; i++) {
        float mag = std::sqrt(static_cast<float>(gx[i]*gx[i] + gy[i]*gy[i]));
        magnitude_output[i] = static_cast<uint8_t>((mag * 255.0f) / max_mag);
    }
}

void compute_magnitude_l1_rvv(const int16_t* gx, const int16_t* gy, int width, int height, uint8_t* mag) {
    int total_pixels = width * height;
    
    // =========================================================================
    // PASS 1: Find the Global Maximum using a Vector Reduction
    // =========================================================================
    int remaining = total_pixels;
    const int16_t* ptr_gx = gx;
    const int16_t* ptr_gy = gy;
    
    // We need a scalar register to hold our running maximum. 
    // Reductions always store their result in element 0 of an m1 register.
    vint16m1_t vmax_vec = __riscv_vmv_s_x_i16m1(0, __riscv_vsetvlmax_e16m1()); 
    
    while (remaining > 0) {
        // We use LMUL=4 to process large chunks of memory at once
        size_t vl = __riscv_vsetvl_e16m4(remaining);
        
        // Load Gx and Gy
        vint16m4_t v_gx = __riscv_vle16_v_i16m4(ptr_gx, vl);
        vint16m4_t v_gy = __riscv_vle16_v_i16m4(ptr_gy, vl);
        
        // Compute absolute value: |Gx| = max(Gx, -Gx)
        vint16m4_t neg_gx = __riscv_vrsub_vx_i16m4(v_gx, 0, vl); // 0 - Gx
        
        // 1. Operation: Computes absolute value by taking the maximum of a vector and its negated self.
        // 2. LMUL: LMUL=4 maximizes elements processed per instruction while leaving room for the two input vectors (Gx, Gy).
        // 3. VLEN Agnosticism: Correctly computes the absolute value of exactly 'vl' elements without out-of-bounds errors on any VLEN.
        vint16m4_t abs_gx = __riscv_vmax_vv_i16m4(v_gx, neg_gx, vl);
        
        // Compute absolute value: |Gy| = max(Gy, -Gy)
        vint16m4_t neg_gy = __riscv_vrsub_vx_i16m4(v_gy, 0, vl); // 0 - Gy
        vint16m4_t abs_gy = __riscv_vmax_vv_i16m4(v_gy, neg_gy, vl);
        
        // Sum = |Gx| + |Gy|
        vint16m4_t v_sum = __riscv_vadd_vv_i16m4(abs_gx, abs_gy, vl);
        
        // 1. Operation: Vector reduction to find the global maximum magnitude across the active vector.
        // 2. LMUL: Input is LMUL=4 (the magnitude data), but the output is ALWAYS LMUL=1 scalar destination register per RVV spec.
        // 3. VLEN Agnosticism: The reduction instruction inherently collapses however many elements fit in the VLEN down to a single element.
        vmax_vec = __riscv_vredmax_vs_i16m4_i16m1(v_sum, vmax_vec, vl);
        
        ptr_gx += vl;
        ptr_gy += vl;
        remaining -= vl;
    }
    
    // 1. Operation: Moves the scalar result from element 0 of the vector register into a standard CPU register.
    // 2. LMUL: Operates on an LMUL=1 register because reduction outputs are stored in m1.
    // 3. VLEN Agnosticism: Independent of VLEN, as it always extracts specifically element 0.
    int16_t global_max = __riscv_vmv_x_s_i16m1_i16(vmax_vec);
    if (global_max == 0) global_max = 1; // Prevent division by zero on an all-black image
    
// =========================================================================
    // PASS 2: Normalize and Store (High-Precision Fixed-Point Optimization)
    // =========================================================================
    remaining = total_pixels;
    ptr_gx = gx;
    ptr_gy = gy;
    uint8_t* ptr_mag = mag;
    
    // UPGRADE: Changed from (255 * 256) to (255 * 65536) for 256x more fractional precision
    int32_t multiplier = (255 * 65536) / global_max; 
    
    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e16m4(remaining);
        
        vint16m4_t v_gx = __riscv_vle16_v_i16m4(ptr_gx, vl);
        vint16m4_t v_gy = __riscv_vle16_v_i16m4(ptr_gy, vl);
        
        vint16m4_t abs_gx = __riscv_vmax_vv_i16m4(v_gx, __riscv_vrsub_vx_i16m4(v_gx, 0, vl), vl);
        vint16m4_t abs_gy = __riscv_vmax_vv_i16m4(v_gy, __riscv_vrsub_vx_i16m4(v_gy, 0, vl), vl);
        vint16m4_t v_sum = __riscv_vadd_vv_i16m4(abs_gx, abs_gy, vl);
        
        // Widens the math to 32-bit (vint32m8_t) so the large multiplication won't overflow
        vint32m8_t v_mul = __riscv_vwmul_vx_i32m8(v_sum, multiplier, vl); 
        
        // UPGRADE: Shift right by 16 instead of 8 to extract the final scaled pixel value
        vint32m8_t v_shifted = __riscv_vsra_vx_i32m8(v_mul, 16, vl);
        
        // Narrow from 32-bit back down to 16-bit
        vint16m4_t v_norm16 = __riscv_vncvt_x_x_w_i16m4(v_shifted, vl);
        
        // Clip, convert to unsigned 8-bit, and store
        vuint8m2_t v_out = __riscv_vnclipu_wx_u8m2(
            __riscv_vreinterpret_v_i16m4_u16m4(v_norm16), 0, __RISCV_VXRM_RNU, vl
        );
        
        __riscv_vse8_v_u8m2(ptr_mag, v_out, vl);
        
        ptr_gx += vl;
        ptr_gy += vl;
        ptr_mag += vl;
        remaining -= vl;
    }
}

extern "C" void compute_magnitude_l2_rvv(const int16_t* gx, const int16_t* gy, int width, int height, uint8_t* output) {
    compute_magnitude_l2(gx, gy, width, height, output);
}