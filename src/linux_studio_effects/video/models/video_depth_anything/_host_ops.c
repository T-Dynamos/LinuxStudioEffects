#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * Fast single-threaded AVX-512 / AVX-512-BF16 host kernels for Video Depth Anything.
 * Compiled with:
 *   gcc -O3 -mavx512f -mavx512bf16 -mavx512bw -mavx512vl -mfma -shared -fPIC
 */

static inline __m512 exp_ps_avx512(__m512 v) {
    /* Clamp to prevent overflow/underflow traps; <= -104 underflows cleanly to 0. */
    const __m512 min_val = _mm512_set1_ps(-104.0f);
    const __m512 max_val = _mm512_set1_ps(88.3762626647949f);
    v = _mm512_max_ps(v, min_val);
    v = _mm512_min_ps(v, max_val);

    const __m512 log2e = _mm512_set1_ps(1.44269504088896341f);
    const __m512 ln2_hi = _mm512_set1_ps(0.693145751953125f);
    const __m512 ln2_lo = _mm512_set1_ps(1.428606765330187045e-6f);

    const __m512 c1 = _mm512_set1_ps(1.0f);
    const __m512 c2 = _mm512_set1_ps(0.5f);
    const __m512 c3 = _mm512_set1_ps(0.1666666716337204f);
    const __m512 c4 = _mm512_set1_ps(0.0416666679084301f);
    const __m512 c5 = _mm512_set1_ps(0.008333333767950535f);
    const __m512 c6 = _mm512_set1_ps(0.0013888889225199819f);

    __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(v, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, ln2_hi, v);
    r = _mm512_fnmadd_ps(n, ln2_lo, r);

    __m512 p = _mm512_fmadd_ps(c6, r, c5);
    p = _mm512_fmadd_ps(p, r, c4);
    p = _mm512_fmadd_ps(p, r, c3);
    p = _mm512_fmadd_ps(p, r, c2);
    p = _mm512_fmadd_ps(p, r, c1);
    p = _mm512_fmadd_ps(p, r, c1);

    return _mm512_scalef_ps(p, n);
}

void f32_to_bf16_pad(
    const float *src,
    int64_t src_stride_elems,
    uint16_t *dst,
    int64_t dst_stride_elems,
    int64_t dst_rows,
    int64_t dst_cols,
    int64_t rows,
    int64_t cols,
    int zero_tail
) {
    for (int64_t r = 0; r < rows; ++r) {
        const float *srow = src + r * src_stride_elems;
        uint16_t *drow = dst + r * dst_stride_elems;
        int64_t c = 0;
        for (; c + 16 <= cols; c += 16) {
            __m512 vf = _mm512_loadu_ps(srow + c);
            __m256bh vb = _mm512_cvtneps_pbh(vf);
            _mm256_storeu_si256((__m256i *)(drow + c), (__m256i)vb);
        }
        if (c < cols) {
            __mmask16 km = (__mmask16)((1u << (cols - c)) - 1u);
            __m512 vf = _mm512_maskz_loadu_ps(km, srow + c);
            __m256bh vb = _mm512_cvtneps_pbh(vf);
            _mm256_mask_storeu_epi16(drow + c, km, (__m256i)vb);
        }
        if (dst_cols > cols) {
            memset(drow + cols, 0, (size_t)(dst_cols - cols) * sizeof(uint16_t));
        }
    }
    if (zero_tail && dst_rows > rows) {
        if (dst_stride_elems == dst_cols) {
            memset(
                dst + rows * dst_stride_elems,
                0,
                (size_t)(dst_rows - rows) * (size_t)dst_cols * sizeof(uint16_t)
            );
        } else {
            for (int64_t r = rows; r < dst_rows; ++r) {
                memset(dst + r * dst_stride_elems, 0, (size_t)dst_cols * sizeof(uint16_t));
            }
        }
    }
}

void bf16_to_f32_unpad(
    const uint16_t *src,
    int64_t src_stride_elems,
    float *dst,
    int64_t dst_stride_elems,
    int64_t rows,
    int64_t cols
) {
    for (int64_t r = 0; r < rows; ++r) {
        const uint16_t *srow = src + r * src_stride_elems;
        float *drow = dst + r * dst_stride_elems;
        int64_t c = 0;
        for (; c + 16 <= cols; c += 16) {
            __m256bh vb = (__m256bh)_mm256_loadu_si256((const __m256i *)(srow + c));
            __m512 vf = _mm512_cvtpbh_ps(vb);
            _mm512_storeu_ps(drow + c, vf);
        }
        if (c < cols) {
            __mmask16 km = (__mmask16)((1u << (cols - c)) - 1u);
            __m256bh vb = (__m256bh)_mm256_maskz_loadu_epi16(km, srow + c);
            __m512 vf = _mm512_cvtpbh_ps(vb);
            _mm512_mask_storeu_ps(drow + c, km, vf);
        }
    }
}

static inline void interp_row_x_f32(
    const float *in_row,
    float *dst,
    int out_w,
    const int32_t *x0,
    const int32_t *x1,
    const float *wx
) {
    int ox = 0;
    for (; ox + 16 <= out_w; ox += 16) {
        __m512i idx0 = _mm512_loadu_si512((const __m512i *)(x0 + ox));
        __m512i idx1 = _mm512_loadu_si512((const __m512i *)(x1 + ox));
        __m512 v0 = _mm512_i32gather_ps(idx0, in_row, 4);
        __m512 v1 = _mm512_i32gather_ps(idx1, in_row, 4);
        __m512 vw = _mm512_loadu_ps(wx + ox);
        __m512 res = _mm512_fmadd_ps(vw, _mm512_sub_ps(v1, v0), v0);
        _mm512_storeu_ps(dst + ox, res);
    }
    for (; ox < out_w; ++ox) {
        float v0 = in_row[x0[ox]];
        float v1 = in_row[x1[ox]];
        dst[ox] = v0 + wx[ox] * (v1 - v0);
    }
}

int bilinear_resize_nchw_f32(
    const float *x,
    float *out,
    int nc,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int align_corners
) {
    if (nc <= 0 || in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0) {
        return -1;
    }
    if (in_h == out_h && in_w == out_w) {
        memcpy(out, x, (size_t)nc * (size_t)in_h * (size_t)in_w * sizeof(float));
        return 0;
    }

    size_t int_count = (size_t)out_w * 2 + (size_t)out_h * 2;
    size_t flt_count = (size_t)out_w * 3 + (size_t)out_h;
    int32_t *ibuf = (int32_t *)malloc(int_count * sizeof(int32_t));
    float *fbuf = (float *)malloc(flt_count * sizeof(float));
    if (!ibuf || !fbuf) {
        free(ibuf);
        free(fbuf);
        return -1;
    }

    int32_t *x0 = ibuf;
    int32_t *x1 = x0 + out_w;
    int32_t *y0 = x1 + out_w;
    int32_t *y1 = y0 + out_h;

    float *wx = fbuf;
    float *wy = wx + out_w;
    float *row_a = wy + out_h;
    float *row_b = row_a + out_w;

    if (out_w == 1) {
        x0[0] = 0;
        x1[0] = 0;
        wx[0] = 0.0f;
    } else if (align_corners) {
        double step = (double)(in_w - 1) / (double)(out_w - 1);
        for (int i = 0; i < out_w; ++i) {
            float xc = (i == out_w - 1) ? (float)(in_w - 1) : (float)((double)i * step);
            int i0 = (int)floorf(xc);
            if (i0 < 0) i0 = 0;
            if (i0 >= in_w) i0 = in_w - 1;
            int i1 = (i0 + 1 < in_w) ? i0 + 1 : in_w - 1;
            x0[i] = i0;
            x1[i] = i1;
            wx[i] = xc - (float)i0;
        }
    } else {
        float scale = (float)((double)in_w / (double)out_w);
        for (int i = 0; i < out_w; ++i) {
            float prod = ((float)i + 0.5f) * scale;
            __asm__("" : "+x"(prod));
            float xc = prod - 0.5f;
            if (xc < 0.0f) xc = 0.0f;
            if (xc > (float)(in_w - 1)) xc = (float)(in_w - 1);
            int i0 = (int)floorf(xc);
            int i1 = (i0 + 1 < in_w) ? i0 + 1 : in_w - 1;
            x0[i] = i0;
            x1[i] = i1;
            wx[i] = xc - (float)i0;
        }
    }

    if (out_h == 1) {
        y0[0] = 0;
        y1[0] = 0;
        wy[0] = 0.0f;
    } else if (align_corners) {
        double step = (double)(in_h - 1) / (double)(out_h - 1);
        for (int i = 0; i < out_h; ++i) {
            float yc = (i == out_h - 1) ? (float)(in_h - 1) : (float)((double)i * step);
            int i0 = (int)floorf(yc);
            if (i0 < 0) i0 = 0;
            if (i0 >= in_h) i0 = in_h - 1;
            int i1 = (i0 + 1 < in_h) ? i0 + 1 : in_h - 1;
            y0[i] = i0;
            y1[i] = i1;
            wy[i] = yc - (float)i0;
        }
    } else {
        float scale = (float)((double)in_h / (double)out_h);
        for (int i = 0; i < out_h; ++i) {
            float prod = ((float)i + 0.5f) * scale;
            __asm__("" : "+x"(prod));
            float yc = prod - 0.5f;
            if (yc < 0.0f) yc = 0.0f;
            if (yc > (float)(in_h - 1)) yc = (float)(in_h - 1);
            int i0 = (int)floorf(yc);
            int i1 = (i0 + 1 < in_h) ? i0 + 1 : in_h - 1;
            y0[i] = i0;
            y1[i] = i1;
            wy[i] = yc - (float)i0;
        }
    }

    const size_t in_plane = (size_t)in_h * (size_t)in_w;
    const size_t out_plane = (size_t)out_h * (size_t)out_w;

    for (int c = 0; c < nc; ++c) {
        const float *in_ch = x + (size_t)c * in_plane;
        float *out_ch = out + (size_t)c * out_plane;

        int cached_id[2] = {-1, -1};
        float *slots[2] = {row_a, row_b};

        for (int oy = 0; oy < out_h; ++oy) {
            int r0 = y0[oy];
            int r1 = y1[oy];
            float w = wy[oy];

            float *p0;
            if (r0 == cached_id[0]) {
                p0 = slots[0];
            } else if (r0 == cached_id[1]) {
                p0 = slots[1];
            } else {
                int slot = (cached_id[0] == r1) ? 1 : 0;
                interp_row_x_f32(in_ch + (size_t)r0 * in_w, slots[slot], out_w, x0, x1, wx);
                cached_id[slot] = r0;
                p0 = slots[slot];
            }

            float *out_row = out_ch + (size_t)oy * out_w;
            if (r1 == r0 || w == 0.0f) {
                memcpy(out_row, p0, (size_t)out_w * sizeof(float));
                continue;
            }

            float *p1;
            if (r1 == cached_id[0]) {
                p1 = slots[0];
            } else if (r1 == cached_id[1]) {
                p1 = slots[1];
            } else {
                int slot = (p0 == slots[0]) ? 1 : 0;
                interp_row_x_f32(in_ch + (size_t)r1 * in_w, slots[slot], out_w, x0, x1, wx);
                cached_id[slot] = r1;
                p1 = slots[slot];
            }

            __m512 vwy = _mm512_set1_ps(w);
            int ox = 0;
            for (; ox + 16 <= out_w; ox += 16) {
                __m512 va = _mm512_loadu_ps(p0 + ox);
                __m512 vb = _mm512_loadu_ps(p1 + ox);
                _mm512_storeu_ps(out_row + ox, _mm512_fmadd_ps(vwy, _mm512_sub_ps(vb, va), va));
            }
            if (ox < out_w) {
                __mmask16 km = (__mmask16)((1u << (out_w - ox)) - 1u);
                __m512 va = _mm512_maskz_loadu_ps(km, p0 + ox);
                __m512 vb = _mm512_maskz_loadu_ps(km, p1 + ox);
                _mm512_mask_storeu_ps(
                    out_row + ox, km, _mm512_fmadd_ps(vwy, _mm512_sub_ps(vb, va), va)
                );
            }
        }
    }

    free(ibuf);
    free(fbuf);
    return 0;
}

void layer_norm_f32(
    const float *x,
    const float *weight,
    const float *bias,
    float *out,
    int64_t M,
    int64_t D,
    float eps
) {
    if (M <= 0 || D <= 0) {
        return;
    }
    const float inv_d = 1.0f / (float)D;
    for (int64_t r = 0; r < M; ++r) {
        const float *xrow = x + r * D;
        float *orow = out + r * D;

        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        __m512 s3 = _mm512_setzero_ps();
        int64_t j = 0;
        for (; j + 64 <= D; j += 64) {
            s0 = _mm512_add_ps(s0, _mm512_loadu_ps(xrow + j));
            s1 = _mm512_add_ps(s1, _mm512_loadu_ps(xrow + j + 16));
            s2 = _mm512_add_ps(s2, _mm512_loadu_ps(xrow + j + 32));
            s3 = _mm512_add_ps(s3, _mm512_loadu_ps(xrow + j + 48));
        }
        s0 = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
        for (; j + 16 <= D; j += 16) {
            s0 = _mm512_add_ps(s0, _mm512_loadu_ps(xrow + j));
        }
        if (j < D) {
            __mmask16 km = (__mmask16)((1u << (D - j)) - 1u);
            s0 = _mm512_add_ps(s0, _mm512_maskz_loadu_ps(km, xrow + j));
        }
        float mean = _mm512_reduce_add_ps(s0) * inv_d;
        __m512 vmean = _mm512_set1_ps(mean);

        __m512 v0 = _mm512_setzero_ps();
        __m512 v1 = _mm512_setzero_ps();
        __m512 v2 = _mm512_setzero_ps();
        __m512 v3 = _mm512_setzero_ps();
        j = 0;
        for (; j + 64 <= D; j += 64) {
            __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(xrow + j), vmean);
            __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(xrow + j + 16), vmean);
            __m512 d2 = _mm512_sub_ps(_mm512_loadu_ps(xrow + j + 32), vmean);
            __m512 d3 = _mm512_sub_ps(_mm512_loadu_ps(xrow + j + 48), vmean);
            v0 = _mm512_fmadd_ps(d0, d0, v0);
            v1 = _mm512_fmadd_ps(d1, d1, v1);
            v2 = _mm512_fmadd_ps(d2, d2, v2);
            v3 = _mm512_fmadd_ps(d3, d3, v3);
        }
        v0 = _mm512_add_ps(_mm512_add_ps(v0, v1), _mm512_add_ps(v2, v3));
        for (; j + 16 <= D; j += 16) {
            __m512 d = _mm512_sub_ps(_mm512_loadu_ps(xrow + j), vmean);
            v0 = _mm512_fmadd_ps(d, d, v0);
        }
        if (j < D) {
            __mmask16 km = (__mmask16)((1u << (D - j)) - 1u);
            __m512 d = _mm512_maskz_sub_ps(km, _mm512_maskz_loadu_ps(km, xrow + j), vmean);
            v0 = _mm512_fmadd_ps(d, d, v0);
        }
        float var = _mm512_reduce_add_ps(v0) * inv_d;
        float inv_std = 1.0f / sqrtf(var + eps);
        __m512 vinv = _mm512_set1_ps(inv_std);

        j = 0;
        for (; j + 16 <= D; j += 16) {
            __m512 d = _mm512_sub_ps(_mm512_loadu_ps(xrow + j), vmean);
            __m512 normed = _mm512_mul_ps(d, vinv);
            __m512 w = _mm512_loadu_ps(weight + j);
            __m512 b = _mm512_loadu_ps(bias + j);
            _mm512_storeu_ps(orow + j, _mm512_fmadd_ps(normed, w, b));
        }
        if (j < D) {
            __mmask16 km = (__mmask16)((1u << (D - j)) - 1u);
            __m512 d = _mm512_sub_ps(_mm512_maskz_loadu_ps(km, xrow + j), vmean);
            __m512 normed = _mm512_mul_ps(d, vinv);
            __m512 w = _mm512_maskz_loadu_ps(km, weight + j);
            __m512 b = _mm512_maskz_loadu_ps(km, bias + j);
            _mm512_mask_storeu_ps(orow + j, km, _mm512_fmadd_ps(normed, w, b));
        }
    }
}

void softmax_rows_f32(const float *x, float *out, int64_t M, int64_t D) {
    if (M <= 0 || D <= 0) {
        return;
    }
    if (D == 1) {
        for (int64_t r = 0; r < M; ++r) {
            out[r] = 1.0f;
        }
        return;
    }
    for (int64_t r = 0; r < M; ++r) {
        const float *xrow = x + r * D;
        float *orow = out + r * D;

        __m512 vmax = _mm512_set1_ps(-INFINITY);
        int64_t j = 0;
        for (; j + 16 <= D; j += 16) {
            vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(xrow + j));
        }
        if (j < D) {
            __mmask16 km = (__mmask16)((1u << (D - j)) - 1u);
            __m512 vtail = _mm512_mask_loadu_ps(vmax, km, xrow + j);
            vmax = _mm512_max_ps(vmax, vtail);
        }
        float max_val = _mm512_reduce_max_ps(vmax);
        vmax = _mm512_set1_ps(max_val);

        __m512 vsum = _mm512_setzero_ps();
        j = 0;
        for (; j + 16 <= D; j += 16) {
            __m512 e = exp_ps_avx512(_mm512_sub_ps(_mm512_loadu_ps(xrow + j), vmax));
            _mm512_storeu_ps(orow + j, e);
            vsum = _mm512_add_ps(vsum, e);
        }
        if (j < D) {
            __mmask16 km = (__mmask16)((1u << (D - j)) - 1u);
            __m512 vdiff = _mm512_sub_ps(_mm512_maskz_loadu_ps(km, xrow + j), vmax);
            __m512 e = exp_ps_avx512(vdiff);
            _mm512_mask_storeu_ps(orow + j, km, e);
            vsum = _mm512_mask_add_ps(vsum, km, vsum, e);
        }
        float inv_sum = 1.0f / _mm512_reduce_add_ps(vsum);
        __m512 vinv = _mm512_set1_ps(inv_sum);

        j = 0;
        for (; j + 16 <= D; j += 16) {
            _mm512_storeu_ps(orow + j, _mm512_mul_ps(_mm512_loadu_ps(orow + j), vinv));
        }
        if (j < D) {
            __mmask16 km = (__mmask16)((1u << (D - j)) - 1u);
            __m512 v = _mm512_maskz_loadu_ps(km, orow + j);
            _mm512_mask_storeu_ps(orow + j, km, _mm512_mul_ps(v, vinv));
        }
    }
}

static inline __m512 gelu_vec_avx512(__m512 vx) {
    const __m512 c_044715 = _mm512_set1_ps(0.044715f);
    const __m512 c_sqrt_2_pi = _mm512_set1_ps(0.7978845608028654f);
    const __m512 c_neg2 = _mm512_set1_ps(-2.0f);
    const __m512 c_one = _mm512_set1_ps(1.0f);
    const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);

    __m512 vx2 = _mm512_mul_ps(vx, vx);
    __m512 vx3 = _mm512_mul_ps(vx2, vx);
    __m512 inner = _mm512_fmadd_ps(c_044715, vx3, vx);
    __m512 z = _mm512_mul_ps(c_sqrt_2_pi, inner);

    __m512 az = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(z), abs_mask));
    __m512 e = exp_ps_avx512(_mm512_mul_ps(az, c_neg2));
    __m512 denom = _mm512_add_ps(c_one, e);
    __mmask16 is_neg = _mm512_cmp_ps_mask(z, _mm512_setzero_ps(), _CMP_LT_OQ);
    __m512 num = _mm512_mask_mov_ps(c_one, is_neg, e);
    __m512 sig = _mm512_div_ps(num, denom);
    return _mm512_mul_ps(vx, sig);
}

void gelu_inplace_f32(float *x, int64_t rows, int64_t cols, int64_t row_stride_elems) {
    for (int64_t r = 0; r < rows; ++r) {
        float *row = x + r * row_stride_elems;
        int64_t c = 0;
        for (; c + 16 <= cols; c += 16) {
            __m512 vx = _mm512_loadu_ps(row + c);
            _mm512_storeu_ps(row + c, gelu_vec_avx512(vx));
        }
        if (c < cols) {
            __mmask16 km = (__mmask16)((1u << (cols - c)) - 1u);
            __m512 vx = _mm512_maskz_loadu_ps(km, row + c);
            _mm512_mask_storeu_ps(row + c, km, gelu_vec_avx512(vx));
        }
    }
}

void group_norm_nchw_f32(
    const float *x,
    const float *weight,
    const float *bias,
    float *out,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t num_groups,
    float eps
) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || num_groups <= 0) {
        return;
    }
    const int64_t c_per_g = C / num_groups;
    const int64_t hw = H * W;
    const int64_t g_size = c_per_g * hw;
    const double inv_g = 1.0 / (double)g_size;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < num_groups; ++g) {
            const float *gx = x + (n * C + g * c_per_g) * hw;
            float *go = out + (n * C + g * c_per_g) * hw;

            __m512 s0 = _mm512_setzero_ps();
            __m512 s1 = _mm512_setzero_ps();
            __m512 s2 = _mm512_setzero_ps();
            __m512 s3 = _mm512_setzero_ps();
            int64_t i = 0;
            for (; i + 64 <= g_size; i += 64) {
                s0 = _mm512_add_ps(s0, _mm512_loadu_ps(gx + i));
                s1 = _mm512_add_ps(s1, _mm512_loadu_ps(gx + i + 16));
                s2 = _mm512_add_ps(s2, _mm512_loadu_ps(gx + i + 32));
                s3 = _mm512_add_ps(s3, _mm512_loadu_ps(gx + i + 48));
            }
            s0 = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
            for (; i + 16 <= g_size; i += 16) {
                s0 = _mm512_add_ps(s0, _mm512_loadu_ps(gx + i));
            }
            if (i < g_size) {
                __mmask16 km = (__mmask16)((1u << (g_size - i)) - 1u);
                s0 = _mm512_add_ps(s0, _mm512_maskz_loadu_ps(km, gx + i));
            }
            float mean = (float)((double)_mm512_reduce_add_ps(s0) * inv_g);
            __m512 vmean = _mm512_set1_ps(mean);

            __m512 v0 = _mm512_setzero_ps();
            __m512 v1 = _mm512_setzero_ps();
            __m512 v2 = _mm512_setzero_ps();
            __m512 v3 = _mm512_setzero_ps();
            i = 0;
            for (; i + 64 <= g_size; i += 64) {
                __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(gx + i), vmean);
                __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(gx + i + 16), vmean);
                __m512 d2 = _mm512_sub_ps(_mm512_loadu_ps(gx + i + 32), vmean);
                __m512 d3 = _mm512_sub_ps(_mm512_loadu_ps(gx + i + 48), vmean);
                v0 = _mm512_fmadd_ps(d0, d0, v0);
                v1 = _mm512_fmadd_ps(d1, d1, v1);
                v2 = _mm512_fmadd_ps(d2, d2, v2);
                v3 = _mm512_fmadd_ps(d3, d3, v3);
            }
            v0 = _mm512_add_ps(_mm512_add_ps(v0, v1), _mm512_add_ps(v2, v3));
            for (; i + 16 <= g_size; i += 16) {
                __m512 d = _mm512_sub_ps(_mm512_loadu_ps(gx + i), vmean);
                v0 = _mm512_fmadd_ps(d, d, v0);
            }
            if (i < g_size) {
                __mmask16 km = (__mmask16)((1u << (g_size - i)) - 1u);
                __m512 d = _mm512_maskz_sub_ps(km, _mm512_maskz_loadu_ps(km, gx + i), vmean);
                v0 = _mm512_fmadd_ps(d, d, v0);
            }
            double var = (double)_mm512_reduce_add_ps(v0) * inv_g;
            float inv_std = (float)(1.0 / sqrt(var + (double)eps));

            for (int64_t cl = 0; cl < c_per_g; ++cl) {
                int64_t c = g * c_per_g + cl;
                const float *cx = gx + cl * hw;
                float *co = go + cl * hw;
                __m512 vscale = _mm512_set1_ps(inv_std * weight[c]);
                __m512 vbias = _mm512_set1_ps(bias[c]);
                int64_t j = 0;
                for (; j + 16 <= hw; j += 16) {
                    __m512 d = _mm512_sub_ps(_mm512_loadu_ps(cx + j), vmean);
                    _mm512_storeu_ps(co + j, _mm512_fmadd_ps(d, vscale, vbias));
                }
                if (j < hw) {
                    __mmask16 km = (__mmask16)((1u << (hw - j)) - 1u);
                    __m512 d = _mm512_sub_ps(_mm512_maskz_loadu_ps(km, cx + j), vmean);
                    _mm512_mask_storeu_ps(co + j, km, _mm512_fmadd_ps(d, vscale, vbias));
                }
            }
        }
    }
}

void im2col_3x3_s1_p1_f32(
    const float *x,
    float *out_cols,
    int64_t C,
    int64_t H,
    int64_t W
) {
    const int64_t hw = H * W;
    const size_t w_bytes = (size_t)W * sizeof(float);
    const size_t w_minus_1_bytes = (W > 1) ? (size_t)(W - 1) * sizeof(float) : 0;

    for (int64_t c = 0; c < C; ++c) {
        const float *in_c = x + c * hw;
        float *base = out_cols + c * 9 * hw;

        /* k=0: ky=-1, kx=-1 */
        float *d0 = base;
        memset(d0, 0, w_bytes);
        for (int64_t y = 1; y < H; ++y) {
            const float *srow = in_c + (y - 1) * W;
            float *drow = d0 + y * W;
            drow[0] = 0.0f;
            if (w_minus_1_bytes) memcpy(drow + 1, srow, w_minus_1_bytes);
        }

        /* k=1: ky=-1, kx=0 */
        float *d1 = base + hw;
        memset(d1, 0, w_bytes);
        if (H > 1) memcpy(d1 + W, in_c, (size_t)(H - 1) * w_bytes);

        /* k=2: ky=-1, kx=+1 */
        float *d2 = base + 2 * hw;
        memset(d2, 0, w_bytes);
        for (int64_t y = 1; y < H; ++y) {
            const float *srow = in_c + (y - 1) * W;
            float *drow = d2 + y * W;
            if (w_minus_1_bytes) memcpy(drow, srow + 1, w_minus_1_bytes);
            drow[W - 1] = 0.0f;
        }

        /* k=3: ky=0, kx=-1 */
        float *d3 = base + 3 * hw;
        for (int64_t y = 0; y < H; ++y) {
            const float *srow = in_c + y * W;
            float *drow = d3 + y * W;
            drow[0] = 0.0f;
            if (w_minus_1_bytes) memcpy(drow + 1, srow, w_minus_1_bytes);
        }

        /* k=4: ky=0, kx=0 */
        memcpy(base + 4 * hw, in_c, (size_t)hw * sizeof(float));

        /* k=5: ky=0, kx=+1 */
        float *d5 = base + 5 * hw;
        for (int64_t y = 0; y < H; ++y) {
            const float *srow = in_c + y * W;
            float *drow = d5 + y * W;
            if (w_minus_1_bytes) memcpy(drow, srow + 1, w_minus_1_bytes);
            drow[W - 1] = 0.0f;
        }

        /* k=6: ky=+1, kx=-1 */
        float *d6 = base + 6 * hw;
        for (int64_t y = 0; y < H - 1; ++y) {
            const float *srow = in_c + (y + 1) * W;
            float *drow = d6 + y * W;
            drow[0] = 0.0f;
            if (w_minus_1_bytes) memcpy(drow + 1, srow, w_minus_1_bytes);
        }
        memset(d6 + (H - 1) * W, 0, w_bytes);

        /* k=7: ky=+1, kx=0 */
        float *d7 = base + 7 * hw;
        if (H > 1) memcpy(d7, in_c + W, (size_t)(H - 1) * w_bytes);
        memset(d7 + (H - 1) * W, 0, w_bytes);

        /* k=8: ky=+1, kx=+1 */
        float *d8 = base + 8 * hw;
        for (int64_t y = 0; y < H - 1; ++y) {
            const float *srow = in_c + (y + 1) * W;
            float *drow = d8 + y * W;
            if (w_minus_1_bytes) memcpy(drow, srow + 1, w_minus_1_bytes);
            drow[W - 1] = 0.0f;
        }
        memset(d8 + (H - 1) * W, 0, w_bytes);
    }
}
