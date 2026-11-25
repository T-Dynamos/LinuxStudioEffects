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

void downsample2x_bg_hw4_f32(
    const uint8_t *frame_u8,
    const float *w0,
    float *out_hw4,
    int in_h,
    int in_w
) {
    const int out_h = (in_h + 1) >> 1;
    const int out_w = (in_w + 1) >> 1;
    for (int oy = 0; oy < out_h; ++oy) {
        const int y0 = oy * 2;
        const int y1 = (y0 + 1 < in_h) ? (y0 + 1) : y0;
        const uint8_t *f_row0 = frame_u8 + (size_t)y0 * (size_t)in_w * 3;
        const uint8_t *f_row1 = frame_u8 + (size_t)y1 * (size_t)in_w * 3;
        const float *w_row0 = w0 + (size_t)y0 * (size_t)in_w;
        const float *w_row1 = w0 + (size_t)y1 * (size_t)in_w;
        float *dst = out_hw4 + (size_t)oy * (size_t)out_w * 4;

        for (int ox = 0; ox < out_w; ++ox) {
            const int x0 = ox * 2;
            const int x1 = (x0 + 1 < in_w) ? (x0 + 1) : x0;
            const float wa = w_row0[x0];
            const float wb = w_row0[x1];
            const float wc = w_row1[x0];
            const float wd = w_row1[x1];

            const uint8_t *pa = f_row0 + (size_t)x0 * 3;
            const uint8_t *pb = f_row0 + (size_t)x1 * 3;
            const uint8_t *pc = f_row1 + (size_t)x0 * 3;
            const uint8_t *pd = f_row1 + (size_t)x1 * 3;

            dst[ox * 4 + 0] = 0.25f * ((float)pa[0] * wa + (float)pb[0] * wb + (float)pc[0] * wc + (float)pd[0] * wd);
            dst[ox * 4 + 1] = 0.25f * ((float)pa[1] * wa + (float)pb[1] * wb + (float)pc[1] * wc + (float)pd[1] * wd);
            dst[ox * 4 + 2] = 0.25f * ((float)pa[2] * wa + (float)pb[2] * wb + (float)pc[2] * wc + (float)pd[2] * wd);
            dst[ox * 4 + 3] = 0.25f * (wa + wb + wc + wd);
        }
    }
}

int bilinear_resize_hwc_f32_to_u8(
    const float *x,
    uint8_t *out,
    int in_h,
    int in_w,
    int out_h,
    int out_w
) {
    if (in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0) {
        return -1;
    }

    size_t int_count = (size_t)out_w * 2 + (size_t)out_h * 2;
    size_t flt_count = (size_t)out_w * 7 + (size_t)out_h;
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
    float *row_b = row_a + (size_t)out_w * 3;

    if (in_w == 1 || out_w == 1) {
        for (int i = 0; i < out_w; ++i) {
            x0[i] = 0;
            x1[i] = 0;
            wx[i] = 0.0f;
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

    if (in_h == 1 || out_h == 1) {
        for (int i = 0; i < out_h; ++i) {
            y0[i] = 0;
            y1[i] = 0;
            wy[i] = 0.0f;
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

    const int out_w3 = out_w * 3;
    const size_t in_stride = (size_t)in_w * 3;
    int cached_id[2] = {-1, -1};
    float *slots[2] = {row_a, row_b};
    const __m512 vzero = _mm512_setzero_ps();
    const __m512 v255 = _mm512_set1_ps(255.0f);

    for (int oy = 0; oy < out_h; ++oy) {
        const int r0 = y0[oy];
        const int r1 = y1[oy];
        const float wyv = wy[oy];

        float *p0;
        if (r0 == cached_id[0]) {
            p0 = slots[0];
        } else if (r0 == cached_id[1]) {
            p0 = slots[1];
        } else {
            int slot = (cached_id[0] == r1) ? 1 : 0;
            const float *in_row = x + (size_t)r0 * in_stride;
            float *dst = slots[slot];
            for (int ox = 0; ox < out_w; ++ox) {
                const float *s0 = in_row + (size_t)x0[ox] * 3;
                const float *s1 = in_row + (size_t)x1[ox] * 3;
                const float w = wx[ox];
                dst[ox * 3 + 0] = s0[0] + w * (s1[0] - s0[0]);
                dst[ox * 3 + 1] = s0[1] + w * (s1[1] - s0[1]);
                dst[ox * 3 + 2] = s0[2] + w * (s1[2] - s0[2]);
            }
            cached_id[slot] = r0;
            p0 = slots[slot];
        }

        uint8_t *out_row = out + (size_t)oy * (size_t)out_w3;
        if (r1 == r0 || wyv == 0.0f) {
            int k = 0;
            for (; k + 16 <= out_w3; k += 16) {
                __m512 vf = _mm512_loadu_ps(p0 + k);
                vf = _mm512_min_ps(_mm512_max_ps(vf, vzero), v255);
                __m512i vi = _mm512_cvtps_epi32(
                    _mm512_roundscale_ps(vf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
                );
                _mm_storeu_si128((__m128i *)(out_row + k), _mm512_cvtusepi32_epi8(vi));
            }
            for (; k < out_w3; ++k) {
                float v = p0[k];
                if (v < 0.0f) v = 0.0f;
                if (v > 255.0f) v = 255.0f;
                out_row[k] = (uint8_t)lrintf(v);
            }
            continue;
        }

        float *p1;
        if (r1 == cached_id[0]) {
            p1 = slots[0];
        } else if (r1 == cached_id[1]) {
            p1 = slots[1];
        } else {
            int slot = (p0 == slots[0]) ? 1 : 0;
            const float *in_row = x + (size_t)r1 * in_stride;
            float *dst = slots[slot];
            for (int ox = 0; ox < out_w; ++ox) {
                const float *s0 = in_row + (size_t)x0[ox] * 3;
                const float *s1 = in_row + (size_t)x1[ox] * 3;
                const float w = wx[ox];
                dst[ox * 3 + 0] = s0[0] + w * (s1[0] - s0[0]);
                dst[ox * 3 + 1] = s0[1] + w * (s1[1] - s0[1]);
                dst[ox * 3 + 2] = s0[2] + w * (s1[2] - s0[2]);
            }
            cached_id[slot] = r1;
            p1 = slots[slot];
        }

        const __m512 vwy = _mm512_set1_ps(wyv);
        int k = 0;
        for (; k + 16 <= out_w3; k += 16) {
            __m512 va = _mm512_loadu_ps(p0 + k);
            __m512 vb = _mm512_loadu_ps(p1 + k);
            __m512 vf = _mm512_fmadd_ps(vwy, _mm512_sub_ps(vb, va), va);
            vf = _mm512_min_ps(_mm512_max_ps(vf, vzero), v255);
            __m512i vi = _mm512_cvtps_epi32(
                _mm512_roundscale_ps(vf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
            );
            _mm_storeu_si128((__m128i *)(out_row + k), _mm512_cvtusepi32_epi8(vi));
        }
        for (; k < out_w3; ++k) {
            float v = p0[k] + wyv * (p1[k] - p0[k]);
            if (v < 0.0f) v = 0.0f;
            if (v > 255.0f) v = 255.0f;
            out_row[k] = (uint8_t)lrintf(v);
        }
    }

    free(ibuf);
    free(fbuf);
    return 0;
}

void bokeh_composite_u8(
    const uint8_t *l0,
    const uint8_t *l1,
    const uint8_t *l2,
    const uint8_t *l3,
    const uint8_t *coc_u8,
    uint8_t *out,
    int64_t num_pixels
) {
    for (int64_t i = 0; i < num_pixels; ++i) {
        int32_t c = (int32_t)coc_u8[i];
        int32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;
        if (c < 85) {
            w1 = c * 3;
            w0 = 256 - w1;
        } else if (c < 170) {
            w2 = (c - 85) * 3;
            w1 = 256 - w2;
        } else {
            int32_t d = c - 170;
            w3 = d * 3 + (d >> 6);
            w2 = 256 - w3;
        }
        const int64_t idx = i * 3;
        out[idx + 0] = (uint8_t)((128 + l0[idx + 0] * w0 + l1[idx + 0] * w1 + l2[idx + 0] * w2 + l3[idx + 0] * w3) >> 8);
        out[idx + 1] = (uint8_t)((128 + l0[idx + 1] * w0 + l1[idx + 1] * w1 + l2[idx + 1] * w2 + l3[idx + 1] * w3) >> 8);
        out[idx + 2] = (uint8_t)((128 + l0[idx + 2] * w0 + l1[idx + 2] * w1 + l2[idx + 2] * w2 + l3[idx + 2] * w3) >> 8);
    }
}

void f32_unpad_bias_gelu(
    const float *src,
    int64_t src_stride_elems,
    float *dst,
    int64_t dst_stride_elems,
    const float *bias,
    int64_t rows,
    int64_t cols,
    int apply_gelu
) {
    for (int64_t r = 0; r < rows; ++r) {
        const float *srow = src + r * src_stride_elems;
        float *drow = dst + r * dst_stride_elems;
        int64_t c = 0;
        for (; c + 16 <= cols; c += 16) {
            __m512 v = _mm512_loadu_ps(srow + c);
            if (bias) {
                v = _mm512_add_ps(v, _mm512_loadu_ps(bias + c));
            }
            if (apply_gelu) {
                v = gelu_vec_avx512(v);
            }
            _mm512_storeu_ps(drow + c, v);
        }
        if (c < cols) {
            __mmask16 km = (__mmask16)((1u << (cols - c)) - 1u);
            __m512 v = _mm512_maskz_loadu_ps(km, srow + c);
            if (bias) {
                v = _mm512_add_ps(v, _mm512_maskz_loadu_ps(km, bias + c));
            }
            if (apply_gelu) {
                v = gelu_vec_avx512(v);
            }
            _mm512_mask_storeu_ps(drow + c, km, v);
        }
    }
}

static inline void transpose_8x16_f32_to_bf16_strided(
    const float *src,
    int64_t src_stride_elems,
    uint16_t *dst,
    int64_t dst_stride_elems
) {
    /*
     * Reads 8 rows (r=0..7) x 16 cols (c=0..15) of float32, converts to bf16, and stores
     * transposed 16 rows x 8 cols of uint16 into dst[c * dst_stride_elems + 0..7].
     */
    __m256i v0 = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src + 0 * src_stride_elems));
    __m256i v1 = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src + 1 * src_stride_elems));
    __m256i v2 = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src + 2 * src_stride_elems));
    __m256i v3 = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src + 3 * src_stride_elems));
    __m256i v4 = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src + 4 * src_stride_elems));
    __m256i v5 = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src + 5 * src_stride_elems));
    __m256i v6 = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src + 6 * src_stride_elems));
    __m256i v7 = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src + 7 * src_stride_elems));

    __m256i t0 = _mm256_unpacklo_epi16(v0, v1);
    __m256i t1 = _mm256_unpackhi_epi16(v0, v1);
    __m256i t2 = _mm256_unpacklo_epi16(v2, v3);
    __m256i t3 = _mm256_unpackhi_epi16(v2, v3);
    __m256i t4 = _mm256_unpacklo_epi16(v4, v5);
    __m256i t5 = _mm256_unpackhi_epi16(v4, v5);
    __m256i t6 = _mm256_unpacklo_epi16(v6, v7);
    __m256i t7 = _mm256_unpackhi_epi16(v6, v7);

    __m256i u0 = _mm256_unpacklo_epi32(t0, t2);
    __m256i u1 = _mm256_unpackhi_epi32(t0, t2);
    __m256i u2 = _mm256_unpacklo_epi32(t1, t3);
    __m256i u3 = _mm256_unpackhi_epi32(t1, t3);
    __m256i u4 = _mm256_unpacklo_epi32(t4, t6);
    __m256i u5 = _mm256_unpackhi_epi32(t4, t6);
    __m256i u6 = _mm256_unpacklo_epi32(t5, t7);
    __m256i u7 = _mm256_unpackhi_epi32(t5, t7);

    __m256i w0 = _mm256_unpacklo_epi64(u0, u4);
    __m256i w1 = _mm256_unpackhi_epi64(u0, u4);
    __m256i w2 = _mm256_unpacklo_epi64(u1, u5);
    __m256i w3 = _mm256_unpackhi_epi64(u1, u5);
    __m256i w4 = _mm256_unpacklo_epi64(u2, u6);
    __m256i w5 = _mm256_unpackhi_epi64(u2, u6);
    __m256i w6 = _mm256_unpacklo_epi64(u3, u7);
    __m256i w7 = _mm256_unpackhi_epi64(u3, u7);

    _mm_storeu_si128((__m128i *)(dst + 0 * dst_stride_elems), _mm256_castsi256_si128(w0));
    _mm_storeu_si128((__m128i *)(dst + 1 * dst_stride_elems), _mm256_castsi256_si128(w1));
    _mm_storeu_si128((__m128i *)(dst + 2 * dst_stride_elems), _mm256_castsi256_si128(w2));
    _mm_storeu_si128((__m128i *)(dst + 3 * dst_stride_elems), _mm256_castsi256_si128(w3));
    _mm_storeu_si128((__m128i *)(dst + 4 * dst_stride_elems), _mm256_castsi256_si128(w4));
    _mm_storeu_si128((__m128i *)(dst + 5 * dst_stride_elems), _mm256_castsi256_si128(w5));
    _mm_storeu_si128((__m128i *)(dst + 6 * dst_stride_elems), _mm256_castsi256_si128(w6));
    _mm_storeu_si128((__m128i *)(dst + 7 * dst_stride_elems), _mm256_castsi256_si128(w7));
    _mm_storeu_si128((__m128i *)(dst + 8 * dst_stride_elems), _mm256_extracti128_si256(w0, 1));
    _mm_storeu_si128((__m128i *)(dst + 9 * dst_stride_elems), _mm256_extracti128_si256(w1, 1));
    _mm_storeu_si128((__m128i *)(dst + 10 * dst_stride_elems), _mm256_extracti128_si256(w2, 1));
    _mm_storeu_si128((__m128i *)(dst + 11 * dst_stride_elems), _mm256_extracti128_si256(w3, 1));
    _mm_storeu_si128((__m128i *)(dst + 12 * dst_stride_elems), _mm256_extracti128_si256(w4, 1));
    _mm_storeu_si128((__m128i *)(dst + 13 * dst_stride_elems), _mm256_extracti128_si256(w5, 1));
    _mm_storeu_si128((__m128i *)(dst + 14 * dst_stride_elems), _mm256_extracti128_si256(w6, 1));
    _mm_storeu_si128((__m128i *)(dst + 15 * dst_stride_elems), _mm256_extracti128_si256(w7, 1));
}

void f32_transpose_to_bf16_pad(
    const float *src,
    int64_t src_stride_elems,
    uint16_t *dst,
    int64_t dst_stride_elems,
    int64_t dst_rows,
    int64_t dst_cols,
    int64_t src_rows,
    int64_t src_cols,
    int zero_tail
) {
    /*
     * src has shape (src_rows=K, src_cols=M); dst has shape (dst_rows=M_pad, dst_cols=K_pad).
     * Writes dst[m, k] = bf16(src[k, m]) using AVX-512 8x16 register transposes.
     */
    const int64_t out_rows = src_cols;
    const int64_t out_cols = src_rows;
    const size_t tail_bytes = (dst_cols > out_cols) ? (size_t)(dst_cols - out_cols) * sizeof(uint16_t) : 0;

    int64_t m0 = 0;
    for (; m0 + 16 <= out_rows; m0 += 16) {
        int64_t k0 = 0;
        for (; k0 + 8 <= out_cols; k0 += 8) {
            transpose_8x16_f32_to_bf16_strided(
                src + k0 * src_stride_elems + m0,
                src_stride_elems,
                dst + m0 * dst_stride_elems + k0,
                dst_stride_elems
            );
        }
        if (k0 < out_cols) {
            const int64_t bk = out_cols - k0;
            uint16_t tile[8][16];
            for (int64_t kk = 0; kk < bk; ++kk) {
                __m512 vf = _mm512_loadu_ps(src + (k0 + kk) * src_stride_elems + m0);
                __m256bh vb = _mm512_cvtneps_pbh(vf);
                _mm256_storeu_si256((__m256i *)tile[kk], (__m256i)vb);
            }
            for (int64_t mm = 0; mm < 16; ++mm) {
                uint16_t *drow = dst + (m0 + mm) * dst_stride_elems + k0;
                for (int64_t kk = 0; kk < bk; ++kk) {
                    drow[kk] = tile[kk][mm];
                }
            }
        }
        if (tail_bytes) {
            for (int64_t mm = 0; mm < 16; ++mm) {
                memset(dst + (m0 + mm) * dst_stride_elems + out_cols, 0, tail_bytes);
            }
        }
    }

    if (m0 < out_rows) {
        const int64_t bm = out_rows - m0;
        uint16_t tile[16][16];
        for (int64_t k0 = 0; k0 < out_cols; k0 += 16) {
            const int64_t bk = (out_cols - k0 < 16) ? (out_cols - k0) : 16;
            const __mmask16 km = (__mmask16)((1u << bm) - 1u);
            for (int64_t kk = 0; kk < bk; ++kk) {
                __m512 vf = _mm512_maskz_loadu_ps(km, src + (k0 + kk) * src_stride_elems + m0);
                __m256bh vb = _mm512_cvtneps_pbh(vf);
                _mm256_mask_storeu_epi16(tile[kk], km, (__m256i)vb);
            }
            for (int64_t mm = 0; mm < bm; ++mm) {
                uint16_t *drow = dst + (m0 + mm) * dst_stride_elems + k0;
                for (int64_t kk = 0; kk < bk; ++kk) {
                    drow[kk] = tile[kk][mm];
                }
            }
        }
        if (tail_bytes) {
            for (int64_t mm = 0; mm < bm; ++mm) {
                memset(dst + (m0 + mm) * dst_stride_elems + out_cols, 0, tail_bytes);
            }
        }
    }

    if (zero_tail && dst_rows > out_rows) {
        if (dst_stride_elems == dst_cols) {
            memset(
                dst + out_rows * dst_stride_elems,
                0,
                (size_t)(dst_rows - out_rows) * (size_t)dst_cols * sizeof(uint16_t)
            );
        } else {
            for (int64_t r = out_rows; r < dst_rows; ++r) {
                memset(dst + r * dst_stride_elems, 0, (size_t)dst_cols * sizeof(uint16_t));
            }
        }
    }
}

int im2col_3x3_s1_p1_kykxc_bf16_pad(
    const float *x_nchw,
    uint16_t *dst_mk,
    int64_t dst_stride_elems,
    int64_t dst_rows,
    int64_t dst_cols,
    int64_t c_in,
    int64_t h,
    int64_t w,
    int zero_tail
) {
    /*
     * Converts (1, c_in, h, w) float32 -> temporary (h, w, c_in) bf16, then streams 3x3 s=1 p=1
     * patches directly into dst_mk[(h*w)_pad, (9*c_in)_pad] with K ordered as (ky, kx, c_in).
     */
    const int64_t hw = h * w;
    const int64_t k_elems = 9 * c_in;
    uint16_t *hwc = (uint16_t *)malloc((size_t)hw * (size_t)c_in * sizeof(uint16_t));
    if (!hwc) {
        return -1;
    }

    f32_transpose_to_bf16_pad(x_nchw, hw, hwc, c_in, hw, c_in, c_in, hw, 0);

    const size_t c_bytes = (size_t)c_in * sizeof(uint16_t);
    const size_t span3_bytes = 3 * c_bytes;
    const size_t k_tail_bytes = (dst_cols > k_elems) ? (size_t)(dst_cols - k_elems) * sizeof(uint16_t) : 0;

    for (int64_t y = 0; y < h; ++y) {
        for (int64_t x = 0; x < w; ++x) {
            uint16_t *drow = dst_mk + (y * w + x) * dst_stride_elems;
            for (int64_t ky = 0; ky < 3; ++ky) {
                const int64_t sy = y + ky - 1;
                uint16_t *dk = drow + ky * 3 * c_in;
                if (sy < 0 || sy >= h) {
                    memset(dk, 0, span3_bytes);
                } else {
                    const uint16_t *s_row = hwc + sy * w * c_in;
                    if (x > 0 && x + 1 < w) {
                        memcpy(dk, s_row + (x - 1) * c_in, span3_bytes);
                    } else {
                        if (x > 0) {
                            memcpy(dk, s_row + (x - 1) * c_in, c_bytes);
                        } else {
                            memset(dk, 0, c_bytes);
                        }
                        memcpy(dk + c_in, s_row + x * c_in, c_bytes);
                        if (x + 1 < w) {
                            memcpy(dk + 2 * c_in, s_row + (x + 1) * c_in, c_bytes);
                        } else {
                            memset(dk + 2 * c_in, 0, c_bytes);
                        }
                    }
                }
            }
            if (k_tail_bytes) {
                memset(drow + k_elems, 0, k_tail_bytes);
            }
        }
    }

    if (zero_tail && dst_rows > hw) {
        if (dst_stride_elems == dst_cols) {
            memset(
                dst_mk + hw * dst_stride_elems,
                0,
                (size_t)(dst_rows - hw) * (size_t)dst_cols * sizeof(uint16_t)
            );
        } else {
            for (int64_t r = hw; r < dst_rows; ++r) {
                memset(dst_mk + r * dst_stride_elems, 0, (size_t)dst_cols * sizeof(uint16_t));
            }
        }
    }

    free(hwc);
    return 0;
}

void f32_unpad_transpose_bias_relu(
    const float *src,
    int64_t src_stride_elems,
    float *dst_nchw,
    const float *bias,
    int64_t m_hw,
    int64_t c_out,
    int apply_relu
) {
    /*
     * Transposes (M, C_out) -> (C_out, M) float32 in 64x16 cache blocks with optional
     * vector bias addition and ReLU before storing contiguous channel rows.
     */
    const __m512 vzero = _mm512_setzero_ps();
    float tile[64][16];

    for (int64_t c0 = 0; c0 < c_out; c0 += 16) {
        const int64_t bc = (c_out - c0 < 16) ? (c_out - c0) : 16;
        __mmask16 km = (bc == 16) ? (__mmask16)0xFFFFu : (__mmask16)((1u << bc) - 1u);
        __m512 vbias = bias ? ((bc == 16) ? _mm512_loadu_ps(bias + c0) : _mm512_maskz_loadu_ps(km, bias + c0)) : vzero;

        for (int64_t m0 = 0; m0 < m_hw; m0 += 64) {
            const int64_t bm = (m_hw - m0 < 64) ? (m_hw - m0) : 64;
            for (int64_t mm = 0; mm < bm; ++mm) {
                const float *srow = src + (m0 + mm) * src_stride_elems + c0;
                __m512 v = (bc == 16) ? _mm512_loadu_ps(srow) : _mm512_maskz_loadu_ps(km, srow);
                if (bias) {
                    v = _mm512_add_ps(v, vbias);
                }
                if (apply_relu) {
                    v = _mm512_max_ps(v, vzero);
                }
                _mm512_storeu_ps(tile[mm], v);
            }
            for (int64_t cc = 0; cc < bc; ++cc) {
                float *drow = dst_nchw + (c0 + cc) * m_hw + m0;
                for (int64_t mm = 0; mm < bm; ++mm) {
                    drow[mm] = tile[mm][cc];
                }
            }
        }
    }
}

void temporal_attn_roll_f32(
    const float *q,
    const float *k_cur,
    const float *v_cur,
    float *k_buf,
    float *v_buf,
    const float *k_pe,
    const float *v_pe,
    float *out,
    int64_t d,
    int64_t t_len,
    int64_t n_heads,
    int64_t dh,
    float scale,
    int has_history
) {
    const int64_t ch = n_heads * dh;

    if (!has_history) {
        const size_t ch_bytes = (size_t)ch * sizeof(float);
        for (int64_t tok = 0; tok < d; ++tok) {
            const float *kc = k_cur + tok * ch;
            const float *vc = v_cur + tok * ch;
            float *out_tok = out + tok * ch;
            int64_t c = 0;
            for (; c + 16 <= ch; c += 16) {
                __m512 vv = _mm512_add_ps(_mm512_loadu_ps(vc + c), _mm512_loadu_ps(v_pe + c));
                _mm512_storeu_ps(out_tok + c, vv);
            }
            if (c < ch) {
                __mmask16 km = (__mmask16)((1u << (ch - c)) - 1u);
                __m512 vv = _mm512_add_ps(_mm512_maskz_loadu_ps(km, vc + c), _mm512_maskz_loadu_ps(km, v_pe + c));
                _mm512_mask_storeu_ps(out_tok + c, km, vv);
            }
            float *kb = k_buf + tok * t_len * ch;
            float *vb = v_buf + tok * t_len * ch;
            for (int64_t t = 0; t < t_len - 1; ++t) {
                memcpy(kb + t * ch, kc, ch_bytes);
                memcpy(vb + t * ch, vc, ch_bytes);
            }
        }
        return;
    }

    float *scores = (t_len <= 128) ? NULL : (float *)malloc((size_t)t_len * sizeof(float));
    float stack_scores[128];
    float *sc = scores ? scores : stack_scores;

    const size_t ch_bytes = (size_t)ch * sizeof(float);
    const size_t shift_bytes = (t_len > 4) ? (size_t)(t_len - 4) * ch_bytes : 0;

    for (int64_t tok = 0; tok < d; ++tok) {
        const float *q_tok = q + tok * ch;
        const float *kc = k_cur + tok * ch;
        const float *vc = v_cur + tok * ch;
        float *kb = k_buf + tok * t_len * ch;
        float *vb = v_buf + tok * t_len * ch;
        float *out_tok = out + tok * ch;

        memcpy(kb + (t_len - 1) * ch, kc, ch_bytes);
        memcpy(vb + (t_len - 1) * ch, vc, ch_bytes);

        for (int64_t h = 0; h < n_heads; ++h) {
            const int64_t h_off = h * dh;
            const float *qh = q_tok + h_off;
            float *oh = out_tok + h_off;

            float max_sc = -3.402823466e+38f;
            for (int64_t t = 0; t < t_len; ++t) {
                const float *kt = kb + t * ch + h_off;
                const float *kpet = k_pe + t * ch + h_off;
                __m512 vacc = _mm512_setzero_ps();
                int64_t i = 0;
                for (; i + 16 <= dh; i += 16) {
                    __m512 vq = _mm512_loadu_ps(qh + i);
                    __m512 vk = _mm512_add_ps(_mm512_loadu_ps(kt + i), _mm512_loadu_ps(kpet + i));
                    vacc = _mm512_fmadd_ps(vq, vk, vacc);
                }
                if (i < dh) {
                    __mmask16 km = (__mmask16)((1u << (dh - i)) - 1u);
                    __m512 vq = _mm512_maskz_loadu_ps(km, qh + i);
                    __m512 vk = _mm512_add_ps(_mm512_maskz_loadu_ps(km, kt + i), _mm512_maskz_loadu_ps(km, kpet + i));
                    vacc = _mm512_fmadd_ps(vq, vk, vacc);
                }
                float s = _mm512_reduce_add_ps(vacc) * scale;
                sc[t] = s;
                if (s > max_sc) max_sc = s;
            }

            const __m512 vmax = _mm512_set1_ps(max_sc);
            __m512 vsum = _mm512_setzero_ps();
            int64_t t = 0;
            for (; t + 16 <= t_len; t += 16) {
                __m512 ve = exp_ps_avx512(_mm512_sub_ps(_mm512_loadu_ps(sc + t), vmax));
                _mm512_storeu_ps(sc + t, ve);
                vsum = _mm512_add_ps(vsum, ve);
            }
            if (t < t_len) {
                __mmask16 km = (__mmask16)((1u << (t_len - t)) - 1u);
                __m512 ve = exp_ps_avx512(_mm512_sub_ps(_mm512_maskz_loadu_ps(km, sc + t), vmax));
                _mm512_mask_storeu_ps(sc + t, km, ve);
                vsum = _mm512_mask_add_ps(vsum, km, vsum, ve);
            }
            float inv_sum = 1.0f / _mm512_reduce_add_ps(vsum);
            const __m512 vinv = _mm512_set1_ps(inv_sum);
            for (t = 0; t + 16 <= t_len; t += 16) {
                _mm512_storeu_ps(sc + t, _mm512_mul_ps(_mm512_loadu_ps(sc + t), vinv));
            }
            if (t < t_len) {
                __mmask16 km = (__mmask16)((1u << (t_len - t)) - 1u);
                _mm512_mask_storeu_ps(sc + t, km, _mm512_mul_ps(_mm512_maskz_loadu_ps(km, sc + t), vinv));
            }

            int64_t i = 0;
            for (; i + 16 <= dh; i += 16) {
                __m512 oacc = _mm512_setzero_ps();
                for (int64_t tt = 0; tt < t_len; ++tt) {
                    const float *vt = vb + tt * ch + h_off + i;
                    const float *vpet = v_pe + tt * ch + h_off + i;
                    __m512 vw = _mm512_set1_ps(sc[tt]);
                    __m512 vv = _mm512_add_ps(_mm512_loadu_ps(vt), _mm512_loadu_ps(vpet));
                    oacc = _mm512_fmadd_ps(vw, vv, oacc);
                }
                _mm512_storeu_ps(oh + i, oacc);
            }
            if (i < dh) {
                __mmask16 km = (__mmask16)((1u << (dh - i)) - 1u);
                __m512 oacc = _mm512_setzero_ps();
                for (int64_t tt = 0; tt < t_len; ++tt) {
                    const float *vt = vb + tt * ch + h_off + i;
                    const float *vpet = v_pe + tt * ch + h_off + i;
                    __m512 vw = _mm512_set1_ps(sc[tt]);
                    __m512 vv = _mm512_add_ps(_mm512_maskz_loadu_ps(km, vt), _mm512_maskz_loadu_ps(km, vpet));
                    oacc = _mm512_fmadd_ps(vw, vv, oacc);
                }
                _mm512_mask_storeu_ps(oh + i, km, oacc);
            }
        }

        if (t_len >= 4) {
            if (shift_bytes) {
                memmove(kb + 2 * ch, kb + 3 * ch, shift_bytes);
                memmove(vb + 2 * ch, vb + 3 * ch, shift_bytes);
            }
            memcpy(kb + (t_len - 2) * ch, kc, ch_bytes);
            memcpy(vb + (t_len - 2) * ch, vc, ch_bytes);
        }
    }

    if (scores) {
        free(scores);
    }
}
