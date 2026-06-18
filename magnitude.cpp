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
        vint16m4_t abs_gx = __riscv_vmax_vv_i16m4(v_gx, neg_gx, vl);
        
        // Compute absolute value: |Gy| = max(Gy, -Gy)
        vint16m4_t neg_gy = __riscv_vrsub_vx_i16m4(v_gy, 0, vl); // 0 - Gy
        vint16m4_t abs_gy = __riscv_vmax_vv_i16m4(v_gy, neg_gy, vl);
        
        // Sum = |Gx| + |Gy|
        vint16m4_t v_sum = __riscv_vadd_vv_i16m4(abs_gx, abs_gy, vl);
        
        // REDUCTION: Compares all elements in v_sum to vmax_vec[0] 
        // and stores the highest value back into vmax_vec[0]
        vmax_vec = __riscv_vredmax_vs_i16m4_i16m1(v_sum, vmax_vec, vl);
        
        ptr_gx += vl;
        ptr_gy += vl;
        remaining -= vl;
    }
    
    // Extract the final global maximum from the vector register back to standard C++ scalar
    int16_t global_max = __riscv_vmv_x_s_i16m1_i16(vmax_vec);
    if (global_max == 0) global_max = 1; // Prevent division by zero on an all-black image
    
    // =========================================================================
    // PASS 2: Normalize and Store
    // =========================================================================
    remaining = total_pixels;
    ptr_gx = gx;
    ptr_gy = gy;
    uint8_t* ptr_mag = mag;
    
    // Fixed-point division trick to avoid hardware division!
    // Instead of: result = (sum * 255) / max
    // We do:      result = (sum * multiplier) >> 8
    int32_t multiplier = (255 * 256) / global_max; 
    
    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e16m4(remaining);
        
        // Re-load and re-compute sums (often faster than writing intermediate buffers to memory)
        vint16m4_t v_gx = __riscv_vle16_v_i16m4(ptr_gx, vl);
        vint16m4_t v_gy = __riscv_vle16_v_i16m4(ptr_gy, vl);
        
        vint16m4_t abs_gx = __riscv_vmax_vv_i16m4(v_gx, __riscv_vrsub_vx_i16m4(v_gx, 0, vl), vl);
        vint16m4_t abs_gy = __riscv_vmax_vv_i16m4(v_gy, __riscv_vrsub_vx_i16m4(v_gy, 0, vl), vl);
        vint16m4_t v_sum = __riscv_vadd_vv_i16m4(abs_gx, abs_gy, vl);
        
        // Multiply by fixed-point multiplier (widens to 32-bit LMUL=8 to prevent overflow)
        vint32m8_t v_mul = __riscv_vwmul_vx_i32m8(v_sum, multiplier, vl); 
        
        // Shift right by 8 to finalize the fixed-point math
        vint32m8_t v_shifted = __riscv_vsra_vx_i32m8(v_mul, 8, vl);
        
        // Narrow from 32-bit back down to 16-bit
        vint16m4_t v_norm16 = __riscv_vncvt_x_x_w_i16m4(v_shifted, vl);
        
        // Narrow from 16-bit down to 8-bit unsigned (using vnclipu handles clamping automatically)
        vuint8m2_t v_out = __riscv_vnclipu_wx_u8m2(
            __riscv_vreinterpret_v_i16m4_u16m4(v_norm16), 0, __RISCV_VXRM_RNU, vl
        );
        
        // Store the final 8-bit magnitudes
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