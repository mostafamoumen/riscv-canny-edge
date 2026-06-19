#include "image_io.h"
#include <algorithm>
#include <cstring>
#include <riscv_vector.h>
#include "gaussian_blur.h"

// ============================================================================
// PHASE 6.2: RVV TYPE & INTRINSIC TRAITS CONFIGURATION
// ============================================================================

template <int LMUL> struct RvvTraits;

template <> struct RvvTraits<1> {
    using pixel_v  = vuint8mf2_t;  
    using kernel_v = vint16m1_t;   
    using accum_v  = vint32m2_t;   
    
    static size_t setvl(size_t n) { return __riscv_vsetvl_e16m1(n); }
    static accum_v zero_accum(size_t vl) { return __riscv_vmv_v_x_i32m2(0, vl); }
    static pixel_v load_pixel(const uint8_t* ptr, size_t vl) { return __riscv_vle8_v_u8mf2(ptr, vl); }
    
    static kernel_v widen_pixel(pixel_v v, size_t vl) { 
        vuint16m1_t ext = __riscv_vzext_vf2_u16m1(v, vl);
        return __riscv_vreinterpret_v_u16m1_i16m1(ext); 
    }
    
    static accum_v macc(accum_v sum, int16_t coeff, kernel_v pix16, size_t vl) { 
        return __riscv_vwmacc_vx_i32m2(sum, coeff, pix16, vl); 
    }
    static accum_v mul(accum_v v, int32_t factor, size_t vl) { return __riscv_vmul_vx_i32m2(v, factor, vl); }
    static accum_v sra(accum_v v, unsigned int shift, size_t vl) { return __riscv_vsra_vx_i32m2(v, shift, vl); }
    
    static void store_pixel(uint8_t* ptr, accum_v v_final32, size_t vl) {
        vint16m1_t v_16 = __riscv_vncvt_x_x_w_i16m1(v_final32, vl);
        vuint16m1_t vu_16 = __riscv_vreinterpret_v_i16m1_u16m1(v_16);
        vuint8mf2_t v_8 = __riscv_vncvt_x_x_w_u8mf2(vu_16, vl);
        __riscv_vse8_v_u8mf2(ptr, v_8, vl);
    }
};

template <> struct RvvTraits<2> {
    using pixel_v  = vuint8m1_t;   
    using kernel_v = vint16m2_t;   
    using accum_v  = vint32m4_t;   
    
    static size_t setvl(size_t n) { return __riscv_vsetvl_e16m2(n); }
    static accum_v zero_accum(size_t vl) { return __riscv_vmv_v_x_i32m4(0, vl); }
    static pixel_v load_pixel(const uint8_t* ptr, size_t vl) { return __riscv_vle8_v_u8m1(ptr, vl); }
    
    static kernel_v widen_pixel(pixel_v v, size_t vl) { 
        vuint16m2_t ext = __riscv_vzext_vf2_u16m2(v, vl);
        return __riscv_vreinterpret_v_u16m2_i16m2(ext); 
    }
    
    static accum_v macc(accum_v sum, int16_t coeff, kernel_v pix16, size_t vl) { 
        return __riscv_vwmacc_vx_i32m4(sum, coeff, pix16, vl); 
    }
    static accum_v mul(accum_v v, int32_t factor, size_t vl) { return __riscv_vmul_vx_i32m4(v, factor, vl); }
    static accum_v sra(accum_v v, unsigned int shift, size_t vl) { return __riscv_vsra_vx_i32m4(v, shift, vl); }
    
    static void store_pixel(uint8_t* ptr, accum_v v_final32, size_t vl) {
        vint16m2_t v_16 = __riscv_vncvt_x_x_w_i16m2(v_final32, vl);
        vuint16m2_t vu_16 = __riscv_vreinterpret_v_i16m2_u16m2(v_16);
        vuint8m1_t v_8 = __riscv_vncvt_x_x_w_u8m1(vu_16, vl);
        __riscv_vse8_v_u8m1(ptr, v_8, vl);
    }
};

template <> struct RvvTraits<4> {
    using pixel_v  = vuint8m2_t;   
    using kernel_v = vint16m4_t;   
    using accum_v  = vint32m8_t;   
    
    // 1. Operation: Sets the vector length (vl) for the strip-mining loop.
    // 2. LMUL: Matches the target template LMUL (4) to balance register pressure and throughput.
    // 3. VLEN Agnosticism: If VLEN changes, this dynamically returns a larger/smaller 'vl', meaning the exact same loop code will automatically process more/fewer pixels per iteration without recompiling.
    static size_t setvl(size_t n) { return __riscv_vsetvl_e16m4(n); }
    
    static accum_v zero_accum(size_t vl) { return __riscv_vmv_v_x_i32m8(0, vl); }
    
    // 1. Operation: Vector load to fetch 8-bit grayscale pixels from memory.
    // 2. LMUL: Uses half the target LMUL (m2 for m4 processing) because the 8-bit data will be widened to 16-bit for math.
    // 3. VLEN Agnosticism: Automatically loads exactly 'vl' elements from memory, regardless of hardware vector register size.
    static pixel_v load_pixel(const uint8_t* ptr, size_t vl) { return __riscv_vle8_v_u8m2(ptr, vl); }
    
    // 1. Operation: Zero-extends an 8-bit unsigned vector into a 16-bit vector to prevent overflow during multiplication.
    // 2. LMUL: This explicitly doubles the LMUL (from m2 to m4) because 16-bit elements take twice the register space of 8-bit elements.
    // 3. VLEN Agnosticism: Works flawlessly across VLEN sizes because 'vl' guarantees we only process the active elements.
    static kernel_v widen_pixel(pixel_v v, size_t vl) { 
        vuint16m4_t ext = __riscv_vzext_vf2_u16m4(v, vl);
        return __riscv_vreinterpret_v_u16m4_i16m4(ext); 
    }
    
    // 1. Operation: Widening multiply-accumulate. Multiplies 16-bit pixel by 16-bit scalar coeff and accumulates into a 32-bit vector.
    // 2. LMUL: Doubles the LMUL again (from m4 to m8) to hold the 32-bit accumulated sum.
    // 3. VLEN Agnosticism: Hardware automatically scales the number of parallel multiply-accumulates based on VLEN.
    static accum_v macc(accum_v sum, int16_t coeff, kernel_v pix16, size_t vl) { 
        return __riscv_vwmacc_vx_i32m8(sum, coeff, pix16, vl); 
    }
    
    static accum_v mul(accum_v v, int32_t factor, size_t vl) { return __riscv_vmul_vx_i32m8(v, factor, vl); }
    
    // 1. Operation: Arithmetic shift right. Used alongside vector multiply to simulate fast fixed-point division.
    // 2. LMUL: Matches the m8 accumulator LMUL.
    // 3. VLEN Agnosticism: Shifts exactly 'vl' elements in parallel across any hardware width.
    static accum_v sra(accum_v v, unsigned int shift, size_t vl) { return __riscv_vsra_vx_i32m8(v, shift, vl); }
    
    // 1. Operation: Narrows the 32-bit accumulator back down to an 8-bit pixel vector and stores it to memory.
    // 2. LMUL: Steps down from m8 -> m4 -> m2 to accurately map the data sizes back to the original pixel format.
    // 3. VLEN Agnosticism: The hardware applies the exact same narrowing conversion and store to 'vl' elements dynamically.
    static void store_pixel(uint8_t* ptr, accum_v v_final32, size_t vl) {
        vint16m4_t v_16 = __riscv_vncvt_x_x_w_i16m4(v_final32, vl);
        vuint16m4_t vu_16 = __riscv_vreinterpret_v_i16m4_u16m4(v_16);
        vuint8m2_t v_8 = __riscv_vncvt_x_x_w_u8m2(vu_16, vl);
        __riscv_vse8_v_u8m2(ptr, v_8, vl);
    }
};

// ============================================================================
// CORE RVV IMPLEMENTATION (SEPARABLE O(K+K))
// ============================================================================

// ============================================================================
// CORE RVV IMPLEMENTATION (SEPARABLE O(K+K)) - BORDER PADDED
// ============================================================================

template <int LMUL>
void gaussian_blur_rvv_core(const uint8_t* src, uint8_t* dst, int width, int height) {
    using traits = RvvTraits<LMUL>;
    
    const int16_t kernel1D[5] = {2, 4, 5, 4, 2};

    // 1. Create dimensions that include a 2-pixel border on all sides
    int pw = width + 4;
    int ph = height + 4;

    // 2. Allocate padded buffers (zero-initialized)
    uint8_t* padded_src = new uint8_t[pw * ph]();
    int16_t* temp = new int16_t[pw * ph]();

    // 3. Copy the original image into the center of the padded buffer
    for (int y = 0; y < height; ++y) {
        std::memcpy(&padded_src[(y + 2) * pw + 2], &src[y * width], width);
    }

    // ------------------------------------------------------------------------
    // PASS 1: Horizontal Convolution (Process all padded rows)
    // ------------------------------------------------------------------------
    for (int y = 0; y < ph; ++y) {
        int x = 2; 
        while (x < pw - 2) {
            // Process 'width' elements per row
            size_t vl = traits::setvl(pw - 2 - x);
            typename traits::accum_v v_sum = traits::zero_accum(vl);
            
            for (int kx = -2; kx <= 2; ++kx) {
                int16_t coeff = kernel1D[kx + 2];
                const uint8_t* pixel_ptr = &padded_src[y * pw + (x + kx)];
                typename traits::pixel_v v_pixel = traits::load_pixel(pixel_ptr, vl); 
                typename traits::kernel_v v_pixel16 = traits::widen_pixel(v_pixel, vl);
                v_sum = traits::macc(v_sum, coeff, v_pixel16, vl);
            }
            
            // Divide by 17 (Multiply by 3856, shift right by 16)
            typename traits::accum_v v_scaled = traits::mul(v_sum, 3856, vl);
            typename traits::accum_v v_final32 = traits::sra(v_scaled, 16, vl);
            
            if constexpr (LMUL == 1) {
                __riscv_vse16_v_i16m1(&temp[y * pw + x], __riscv_vncvt_x_x_w_i16m1(v_final32, vl), vl);
            } else if constexpr (LMUL == 2) {
                __riscv_vse16_v_i16m2(&temp[y * pw + x], __riscv_vncvt_x_x_w_i16m2(v_final32, vl), vl);
            } else if constexpr (LMUL == 4) {
                __riscv_vse16_v_i16m4(&temp[y * pw + x], __riscv_vncvt_x_x_w_i16m4(v_final32, vl), vl);
            }

            x += vl; 
        }
    }

    // ------------------------------------------------------------------------
    // PASS 2: Vertical Convolution (Process only the valid central rows)
    // ------------------------------------------------------------------------
    for (int y = 2; y < ph - 2; ++y) {
        int x = 2; 
        while (x < pw - 2) {
            size_t vl = traits::setvl(pw - 2 - x);
            typename traits::accum_v v_sum = traits::zero_accum(vl);
            
            for (int ky = -2; ky <= 2; ++ky) {
                int16_t coeff = kernel1D[ky + 2];
                const int16_t* pixel_ptr = &temp[(y + ky) * pw + x];
                
                typename traits::kernel_v v_pixel16;
                if constexpr (LMUL == 1) {
                    v_pixel16 = __riscv_vle16_v_i16m1(pixel_ptr, vl);
                } else if constexpr (LMUL == 2) {
                    v_pixel16 = __riscv_vle16_v_i16m2(pixel_ptr, vl);
                } else if constexpr (LMUL == 4) {
                    v_pixel16 = __riscv_vle16_v_i16m4(pixel_ptr, vl);
                }

                v_sum = traits::macc(v_sum, coeff, v_pixel16, vl);
            }
            
            typename traits::accum_v v_scaled = traits::mul(v_sum, 3856, vl);
            typename traits::accum_v v_final32 = traits::sra(v_scaled, 16, vl);
            
            // Write directly to the original destination (shifting coordinates back)
            uint8_t* dst_ptr = &dst[(y - 2) * width + (x - 2)];
            traits::store_pixel(dst_ptr, v_final32, vl);
            
            x += vl; 
        }
    }

    delete[] padded_src;
    delete[] temp;
}

// ============================================================================
// SCALAR REFERENCE PIPELINES
// ============================================================================

template <typename TPixel, typename TAcc, typename TKernel>
void apply_gaussian_2d(const Image& input, Image& output) {
    const TKernel kernel[5][5] = {
        {2,  4,  5,  4, 2},
        {4,  9, 12,  9, 4},
        {5, 12, 15, 12, 5},
        {4,  9, 12,  9, 4},
        {2,  4,  5,  4, 2}
    };
    const TKernel kernel_sum = 273;
    int w = input.width;
    int h = input.height;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            TAcc acc = 0;
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    int iy = y + ky;
                    int ix = x + kx;
                    TPixel pixel_val = 0; 
                    if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
                        pixel_val = input.data[iy * w + ix];
                    }
                    acc += pixel_val * kernel[ky + 2][kx + 2];
                }
            }
            output.data[y * w + x] = static_cast<TPixel>(std::clamp(acc / kernel_sum, (TAcc)0, (TAcc)255));
        }
    }
}

template <typename TPixel, typename TAcc, typename TKernel>
void apply_gaussian_separable(const Image& input, Image& output) {
    const TKernel kernel1D[5] = {2, 4, 5, 4, 2};
    const TKernel kernel_sum = 17;
    int w = input.width;
    int h = input.height;
    TAcc* temp = new TAcc[w * h]();

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            TAcc acc = 0;
            for (int k = -2; k <= 2; ++k) {
                int ix = x + k;
                TPixel pixel_val = 0; 
                if (ix >= 0 && ix < w) {
                    pixel_val = input.data[y * w + ix];
                }
                acc += pixel_val * kernel1D[k + 2];
            }
            temp[y * w + x] = acc / kernel_sum;
        }
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            TAcc acc = 0;
            for (int k = -2; k <= 2; ++k) {
                int iy = y + k;
                TAcc temp_val = 0; 
                if (iy >= 0 && iy < h) {
                    temp_val = temp[iy * w + x];
                }
                acc += temp_val * kernel1D[k + 2];
            }
            output.data[y * w + x] = static_cast<TPixel>(std::clamp(acc / kernel_sum, (TAcc)0, (TAcc)255));
        }
    }
    delete[] temp;
}

template <typename TPixel, typename TAcc, typename TKernel>
void apply_gaussian_separable_padded(const Image& input, Image& output) {
    const TKernel kernel1D[5] = {2, 4, 5, 4, 2};
    const TKernel kernel_sum = 17;
    int w = input.width;
    int h = input.height;
    int pw = w + 4;
    int ph = h + 4;
    
    TPixel* padded = new TPixel[pw * ph](); 
    TAcc* temp = new TAcc[pw * ph]();

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            padded[(y + 2) * pw + (x + 2)] = input.data[y * w + x];
        }
    }

    for (int y = 2; y < h + 2; ++y) {
        for (int x = 2; x < w + 2; ++x) {
            TAcc acc = 0;
            for (int k = -2; k <= 2; ++k) {
                acc += padded[y * pw + (x + k)] * kernel1D[k + 2];
            }
            temp[y * pw + x] = acc / kernel_sum;
        }
    }

    for (int y = 2; y < h + 2; ++y) {
        for (int x = 2; x < w + 2; ++x) {
            TAcc acc = 0;
            for (int k = -2; k <= 2; ++k) {
                acc += temp[(y + k) * pw + x] * kernel1D[k + 2];
            }
            output.data[(y - 2) * w + (x - 2)] = static_cast<TPixel>(std::clamp(acc / kernel_sum, (TAcc)0, (TAcc)255));
        }
    }
    delete[] padded;
    delete[] temp;
}

// ============================================================================
// EXPLICIT INSTANTIATIONS
// ============================================================================

void gaussian_blur_rvv_m1(const uint8_t* src, uint8_t* dst, int width, int height) {
    gaussian_blur_rvv_core<1>(src, dst, width, height);
}
void gaussian_blur_rvv_m2(const uint8_t* src, uint8_t* dst, int width, int height) {
    gaussian_blur_rvv_core<2>(src, dst, width, height);
}
void gaussian_blur_rvv_m4(const uint8_t* src, uint8_t* dst, int width, int height) {
    gaussian_blur_rvv_core<4>(src, dst, width, height);
}

void gaussian_blur_scalar_2d(const Image& input, Image& output) {
    apply_gaussian_2d<uint8_t, int32_t, int16_t>(input, output);
}
void gaussian_blur_scalar_separable(const Image& input, Image& output) {
    apply_gaussian_separable<uint8_t, int32_t, int16_t>(input, output);
}

extern "C" void apply_gaussian_separable_rvv(const uint8_t* input, uint8_t* output, int width, int height) {
    gaussian_blur_rvv_core<1>(input, output, width, height);
}

template void apply_gaussian_separable<uint8_t, int32_t, int16_t>(const Image& input, Image& output);
template void apply_gaussian_separable_padded<uint8_t, int32_t, int16_t>(const Image& input, Image& output);