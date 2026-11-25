#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * Fast AVX-512 C host kernels for MODNet inference and portrait alpha compositing.
 */

void dwconv_3x3_p1_f32(
    const float *x,
    const float *w,
    const float *bias,
    float *out,
    int64_t c,
    int64_t in_h,
    int64_t in_w,
    int64_t stride,
    int apply_relu6
) {
    const int64_t out_h = (in_h + 2 - 3) / stride + 1;
    const int64_t out_w = (in_w + 2 - 3) / stride + 1;
    const int64_t in_hw = in_h * in_w;
    const int64_t out_hw = out_h * out_w;
    const __m512 vzero = _mm512_setzero_ps();
    const __m512 vsix = _mm512_set1_ps(6.0f);

    for (int64_t ch = 0; ch < c; ++ch) {
        const float *xc = x + ch * in_hw;
        const float *wc = w + ch * 9;
        float *oc = out + ch * out_hw;
        const float b = bias ? bias[ch] : 0.0f;

        const float w00 = wc[0], w01 = wc[1], w02 = wc[2];
        const float w10 = wc[3], w11 = wc[4], w12 = wc[5];
        const float w20 = wc[6], w21 = wc[7], w22 = wc[8];

        const __m512 vw00 = _mm512_set1_ps(w00), vw01 = _mm512_set1_ps(w01), vw02 = _mm512_set1_ps(w02);
        const __m512 vw10 = _mm512_set1_ps(w10), vw11 = _mm512_set1_ps(w11), vw12 = _mm512_set1_ps(w12);
        const __m512 vw20 = _mm512_set1_ps(w20), vw21 = _mm512_set1_ps(w21), vw22 = _mm512_set1_ps(w22);
        const __m512 vb = _mm512_set1_ps(b);

        for (int64_t oy = 0; oy < out_h; ++oy) {
            const int64_t sy = oy * stride;
            const float *r0 = (sy - 1 >= 0) ? (xc + (sy - 1) * in_w) : NULL;
            const float *r1 = xc + sy * in_w;
            const float *r2 = (sy + 1 < in_h) ? (xc + (sy + 1) * in_w) : NULL;
            float *drow = oc + oy * out_w;

            if (stride == 1 && out_w >= 3) {
                /* Left edge ox = 0 */
                {
                    float acc = b;
                    if (r0) acc += r0[0] * w01 + r0[1] * w02;
                    acc += r1[0] * w11 + r1[1] * w12;
                    if (r2) acc += r2[0] * w21 + r2[1] * w22;
                    if (apply_relu6) {
                        if (acc < 0.0f) acc = 0.0f;
                        if (acc > 6.0f) acc = 6.0f;
                    }
                    drow[0] = acc;
                }

                /* Interior 1 .. out_w - 2 vectorized along ox */
                int64_t ox = 1;
                for (; ox + 16 <= out_w - 1; ox += 16) {
                    __m512 acc = vb;
                    if (r0) {
                        acc = _mm512_fmadd_ps(_mm512_loadu_ps(r0 + ox - 1), vw00, acc);
                        acc = _mm512_fmadd_ps(_mm512_loadu_ps(r0 + ox), vw01, acc);
                        acc = _mm512_fmadd_ps(_mm512_loadu_ps(r0 + ox + 1), vw02, acc);
                    }
                    acc = _mm512_fmadd_ps(_mm512_loadu_ps(r1 + ox - 1), vw10, acc);
                    acc = _mm512_fmadd_ps(_mm512_loadu_ps(r1 + ox), vw11, acc);
                    acc = _mm512_fmadd_ps(_mm512_loadu_ps(r1 + ox + 1), vw12, acc);
                    if (r2) {
                        acc = _mm512_fmadd_ps(_mm512_loadu_ps(r2 + ox - 1), vw20, acc);
                        acc = _mm512_fmadd_ps(_mm512_loadu_ps(r2 + ox), vw21, acc);
                        acc = _mm512_fmadd_ps(_mm512_loadu_ps(r2 + ox + 1), vw22, acc);
                    }
                    if (apply_relu6) {
                        acc = _mm512_min_ps(_mm512_max_ps(acc, vzero), vsix);
                    }
                    _mm512_storeu_ps(drow + ox, acc);
                }
                for (; ox < out_w - 1; ++ox) {
                    float acc = b;
                    if (r0) acc += r0[ox - 1] * w00 + r0[ox] * w01 + r0[ox + 1] * w02;
                    acc += r1[ox - 1] * w10 + r1[ox] * w11 + r1[ox + 1] * w12;
                    if (r2) acc += r2[ox - 1] * w20 + r2[ox] * w21 + r2[ox + 1] * w22;
                    if (apply_relu6) {
                        if (acc < 0.0f) acc = 0.0f;
                        if (acc > 6.0f) acc = 6.0f;
                    }
                    drow[ox] = acc;
                }

                /* Right edge ox = out_w - 1 */
                {
                    const int64_t rx = out_w - 1;
                    float acc = b;
                    if (r0) acc += r0[rx - 1] * w00 + r0[rx] * w01;
                    acc += r1[rx - 1] * w10 + r1[rx] * w11;
                    if (r2) acc += r2[rx - 1] * w20 + r2[rx] * w21;
                    if (apply_relu6) {
                        if (acc < 0.0f) acc = 0.0f;
                        if (acc > 6.0f) acc = 6.0f;
                    }
                    drow[rx] = acc;
                }
            } else {
                for (int64_t ox = 0; ox < out_w; ++ox) {
                    const int64_t sx = ox * stride;
                    float acc = b;
                    if (r0) {
                        if (sx - 1 >= 0) acc += r0[sx - 1] * w00;
                        acc += r0[sx] * w01;
                        if (sx + 1 < in_w) acc += r0[sx + 1] * w02;
                    }
                    if (sx - 1 >= 0) acc += r1[sx - 1] * w10;
                    acc += r1[sx] * w11;
                    if (sx + 1 < in_w) acc += r1[sx + 1] * w12;
                    if (r2) {
                        if (sx - 1 >= 0) acc += r2[sx - 1] * w20;
                        acc += r2[sx] * w21;
                        if (sx + 1 < in_w) acc += r2[sx + 1] * w22;
                    }
                    if (apply_relu6) {
                        if (acc < 0.0f) acc = 0.0f;
                        if (acc > 6.0f) acc = 6.0f;
                    }
                    drow[ox] = acc;
                }
            }
        }
    }
}

void im2col_kxk_p_f32(
    const float *x,
    float *cols,
    int64_t c_in,
    int64_t h,
    int64_t w,
    int64_t k,
    int64_t stride,
    int64_t pad
) {
    const int64_t oh = (h + 2 * pad - k) / stride + 1;
    const int64_t ow = (w + 2 * pad - k) / stride + 1;
    const int64_t hw = h * w;
    const int64_t ohw = oh * ow;

    for (int64_t c = 0; c < c_in; ++c) {
        const float *xc = x + c * hw;
        float *c_base = cols + c * k * k * ohw;
        for (int64_t ky = 0; ky < k; ++ky) {
            for (int64_t kx = 0; kx < k; ++kx) {
                float *dst = c_base + (ky * k + kx) * ohw;
                if (stride == 1) {
                    const int64_t x_shift = kx - pad;
                    const int64_t valid_x0 = (x_shift < 0) ? -x_shift : 0;
                    const int64_t valid_x1 = (ow + x_shift > w) ? (w - x_shift) : ow;
                    const int64_t copy_len = (valid_x1 > valid_x0) ? (valid_x1 - valid_x0) : 0;
                    for (int64_t oy = 0; oy < oh; ++oy) {
                        const int64_t iy = oy + ky - pad;
                        float *drow = dst + oy * ow;
                        if (iy < 0 || iy >= h || copy_len <= 0) {
                            memset(drow, 0, (size_t)ow * sizeof(float));
                        } else {
                            if (valid_x0 > 0) {
                                memset(drow, 0, (size_t)valid_x0 * sizeof(float));
                            }
                            memcpy(drow + valid_x0, xc + iy * w + valid_x0 + x_shift, (size_t)copy_len * sizeof(float));
                            if (valid_x1 < ow) {
                                memset(drow + valid_x1, 0, (size_t)(ow - valid_x1) * sizeof(float));
                            }
                        }
                    }
                } else {
                    for (int64_t oy = 0; oy < oh; ++oy) {
                        const int64_t iy = oy * stride + ky - pad;
                        float *drow = dst + oy * ow;
                        if (iy < 0 || iy >= h) {
                            memset(drow, 0, (size_t)ow * sizeof(float));
                        } else {
                            const float *srow = xc + iy * w;
                            for (int64_t ox = 0; ox < ow; ++ox) {
                                const int64_t ix = ox * stride + kx - pad;
                                drow[ox] = (ix >= 0 && ix < w) ? srow[ix] : 0.0f;
                            }
                        }
                    }
                }
            }
        }
    }
}

void conv2d_3x3_s1_p1_direct_f32(
    const float *x,
    const float *w_4d,
    const float *bias,
    float *out,
    int64_t c_in,
    int64_t c_out,
    int64_t h,
    int64_t w,
    int act_mode
) {
    /*
     * Zero-im2col direct AVX-512 3x3 stride=1 pad=1 Conv2d for small-channel HR/Fusion layers.
     * Processes 4 output channels per pass across 16 horizontal pixels in zmm registers.
     */
    const int64_t hw = h * w;
    const int64_t k_stride = c_in * 9;
    const __m512 vzero = _mm512_setzero_ps();
    const __m512 vsix = _mm512_set1_ps(6.0f);

    int64_t oc = 0;
    for (; oc + 4 <= c_out; oc += 4) {
        const float *w0 = w_4d + (oc + 0) * k_stride;
        const float *w1 = w_4d + (oc + 1) * k_stride;
        const float *w2 = w_4d + (oc + 2) * k_stride;
        const float *w3 = w_4d + (oc + 3) * k_stride;
        const float b0 = bias ? bias[oc + 0] : 0.0f;
        const float b1 = bias ? bias[oc + 1] : 0.0f;
        const float b2 = bias ? bias[oc + 2] : 0.0f;
        const float b3 = bias ? bias[oc + 3] : 0.0f;

        for (int64_t oy = 0; oy < h; ++oy) {
            float *d0 = out + (oc + 0) * hw + oy * w;
            float *d1 = out + (oc + 1) * hw + oy * w;
            float *d2 = out + (oc + 2) * hw + oy * w;
            float *d3 = out + (oc + 3) * hw + oy * w;

            /* Initialize row with bias */
            const __m512 vb0 = _mm512_set1_ps(b0), vb1 = _mm512_set1_ps(b1);
            const __m512 vb2 = _mm512_set1_ps(b2), vb3 = _mm512_set1_ps(b3);
            int64_t ox = 0;
            for (; ox + 16 <= w; ox += 16) {
                _mm512_storeu_ps(d0 + ox, vb0);
                _mm512_storeu_ps(d1 + ox, vb1);
                _mm512_storeu_ps(d2 + ox, vb2);
                _mm512_storeu_ps(d3 + ox, vb3);
            }
            for (; ox < w; ++ox) {
                d0[ox] = b0; d1[ox] = b1; d2[ox] = b2; d3[ox] = b3;
            }

            const int has_top = (oy > 0);
            const int has_bot = (oy + 1 < h);
            for (int64_t ic = 0; ic < c_in; ++ic) {
                const float *xc = x + ic * hw;
                const float *r0 = has_top ? (xc + (oy - 1) * w) : NULL;
                const float *r1 = xc + oy * w;
                const float *r2 = has_bot ? (xc + (oy + 1) * w) : NULL;

                const float *wk0 = w0 + ic * 9;
                const float *wk1 = w1 + ic * 9;
                const float *wk2 = w2 + ic * 9;
                const float *wk3 = w3 + ic * 9;

                /* ox = 0 boundary */
                if (w >= 2) {
                    float a0 = d0[0], a1 = d1[0], a2 = d2[0], a3 = d3[0];
                    if (r0) {
                        float v1 = r0[0], v2 = r0[1];
                        a0 += v1 * wk0[1] + v2 * wk0[2];
                        a1 += v1 * wk1[1] + v2 * wk1[2];
                        a2 += v1 * wk2[1] + v2 * wk2[2];
                        a3 += v1 * wk3[1] + v2 * wk3[2];
                    }
                    {
                        float v1 = r1[0], v2 = r1[1];
                        a0 += v1 * wk0[4] + v2 * wk0[5];
                        a1 += v1 * wk1[4] + v2 * wk1[5];
                        a2 += v1 * wk2[4] + v2 * wk2[5];
                        a3 += v1 * wk3[4] + v2 * wk3[5];
                    }
                    if (r2) {
                        float v1 = r2[0], v2 = r2[1];
                        a0 += v1 * wk0[7] + v2 * wk0[8];
                        a1 += v1 * wk1[7] + v2 * wk1[8];
                        a2 += v1 * wk2[7] + v2 * wk2[8];
                        a3 += v1 * wk3[7] + v2 * wk3[8];
                    }
                    d0[0] = a0; d1[0] = a1; d2[0] = a2; d3[0] = a3;
                }

                /* Vectorized interior ox in [1, w - 1) */
                int64_t ix = 1;
                for (; ix + 16 <= w - 1; ix += 16) {
                    __m512 a0 = _mm512_loadu_ps(d0 + ix);
                    __m512 a1 = _mm512_loadu_ps(d1 + ix);
                    __m512 a2 = _mm512_loadu_ps(d2 + ix);
                    __m512 a3 = _mm512_loadu_ps(d3 + ix);

                    if (r0) {
                        __m512 x0 = _mm512_loadu_ps(r0 + ix - 1);
                        __m512 x1 = _mm512_loadu_ps(r0 + ix);
                        __m512 x2 = _mm512_loadu_ps(r0 + ix + 1);
                        a0 = _mm512_fmadd_ps(x0, _mm512_set1_ps(wk0[0]), a0);
                        a0 = _mm512_fmadd_ps(x1, _mm512_set1_ps(wk0[1]), a0);
                        a0 = _mm512_fmadd_ps(x2, _mm512_set1_ps(wk0[2]), a0);
                        a1 = _mm512_fmadd_ps(x0, _mm512_set1_ps(wk1[0]), a1);
                        a1 = _mm512_fmadd_ps(x1, _mm512_set1_ps(wk1[1]), a1);
                        a1 = _mm512_fmadd_ps(x2, _mm512_set1_ps(wk1[2]), a1);
                        a2 = _mm512_fmadd_ps(x0, _mm512_set1_ps(wk2[0]), a2);
                        a2 = _mm512_fmadd_ps(x1, _mm512_set1_ps(wk2[1]), a2);
                        a2 = _mm512_fmadd_ps(x2, _mm512_set1_ps(wk2[2]), a2);
                        a3 = _mm512_fmadd_ps(x0, _mm512_set1_ps(wk3[0]), a3);
                        a3 = _mm512_fmadd_ps(x1, _mm512_set1_ps(wk3[1]), a3);
                        a3 = _mm512_fmadd_ps(x2, _mm512_set1_ps(wk3[2]), a3);
                    }
                    {
                        __m512 x0 = _mm512_loadu_ps(r1 + ix - 1);
                        __m512 x1 = _mm512_loadu_ps(r1 + ix);
                        __m512 x2 = _mm512_loadu_ps(r1 + ix + 1);
                        a0 = _mm512_fmadd_ps(x0, _mm512_set1_ps(wk0[3]), a0);
                        a0 = _mm512_fmadd_ps(x1, _mm512_set1_ps(wk0[4]), a0);
                        a0 = _mm512_fmadd_ps(x2, _mm512_set1_ps(wk0[5]), a0);
                        a1 = _mm512_fmadd_ps(x0, _mm512_set1_ps(wk1[3]), a1);
                        a1 = _mm512_fmadd_ps(x1, _mm512_set1_ps(wk1[4]), a1);
                        a1 = _mm512_fmadd_ps(x2, _mm512_set1_ps(wk1[5]), a1);
                        a2 = _mm512_fmadd_ps(x0, _mm512_set1_ps(wk2[3]), a2);
                        a2 = _mm512_fmadd_ps(x1, _mm512_set1_ps(wk2[4]), a2);
                        a2 = _mm512_fmadd_ps(x2, _mm512_set1_ps(wk2[5]), a2);
                        a3 = _mm512_fmadd_ps(x0, _mm512_set1_ps(wk3[3]), a3);
                        a3 = _mm512_fmadd_ps(x1, _mm512_set1_ps(wk3[4]), a3);
                        a3 = _mm512_fmadd_ps(x2, _mm512_set1_ps(wk3[5]), a3);
                    }
                    if (r2) {
                        __m512 x0 = _mm512_loadu_ps(r2 + ix - 1);
                        __m512 x1 = _mm512_loadu_ps(r2 + ix);
                        __m512 x2 = _mm512_loadu_ps(r2 + ix + 1);
                        a0 = _mm512_fmadd_ps(x0, _mm512_set1_ps(wk0[6]), a0);
                        a0 = _mm512_fmadd_ps(x1, _mm512_set1_ps(wk0[7]), a0);
                        a0 = _mm512_fmadd_ps(x2, _mm512_set1_ps(wk0[8]), a0);
                        a1 = _mm512_fmadd_ps(x0, _mm512_set1_ps(wk1[6]), a1);
                        a1 = _mm512_fmadd_ps(x1, _mm512_set1_ps(wk1[7]), a1);
                        a1 = _mm512_fmadd_ps(x2, _mm512_set1_ps(wk1[8]), a1);
                        a2 = _mm512_fmadd_ps(x0, _mm512_set1_ps(wk2[6]), a2);
                        a2 = _mm512_fmadd_ps(x1, _mm512_set1_ps(wk2[7]), a2);
                        a2 = _mm512_fmadd_ps(x2, _mm512_set1_ps(wk2[8]), a2);
                        a3 = _mm512_fmadd_ps(x0, _mm512_set1_ps(wk3[6]), a3);
                        a3 = _mm512_fmadd_ps(x1, _mm512_set1_ps(wk3[7]), a3);
                        a3 = _mm512_fmadd_ps(x2, _mm512_set1_ps(wk3[8]), a3);
                    }
                    _mm512_storeu_ps(d0 + ix, a0);
                    _mm512_storeu_ps(d1 + ix, a1);
                    _mm512_storeu_ps(d2 + ix, a2);
                    _mm512_storeu_ps(d3 + ix, a3);
                }
                for (; ix < w; ++ix) {
                    float a0 = d0[ix], a1 = d1[ix], a2 = d2[ix], a3 = d3[ix];
                    for (int kx = -1; kx <= 1; ++kx) {
                        const int64_t sx = ix + kx;
                        if (sx < 0 || sx >= w) continue;
                        const int kcol = kx + 1;
                        if (r0) {
                            float v = r0[sx];
                            a0 += v * wk0[kcol]; a1 += v * wk1[kcol]; a2 += v * wk2[kcol]; a3 += v * wk3[kcol];
                        }
                        {
                            float v = r1[sx];
                            a0 += v * wk0[3 + kcol]; a1 += v * wk1[3 + kcol]; a2 += v * wk2[3 + kcol]; a3 += v * wk3[3 + kcol];
                        }
                        if (r2) {
                            float v = r2[sx];
                            a0 += v * wk0[6 + kcol]; a1 += v * wk1[6 + kcol]; a2 += v * wk2[6 + kcol]; a3 += v * wk3[6 + kcol];
                        }
                    }
                    d0[ix] = a0; d1[ix] = a1; d2[ix] = a2; d3[ix] = a3;
                }
            }

            if (act_mode != 0) {
                float *ds[4] = {d0, d1, d2, d3};
                for (int p = 0; p < 4; ++p) {
                    float *dp = ds[p];
                    int64_t j = 0;
                    for (; j + 16 <= w; j += 16) {
                        __m512 v = _mm512_loadu_ps(dp + j);
                        v = (act_mode == 1) ? _mm512_max_ps(v, vzero) : _mm512_min_ps(_mm512_max_ps(v, vzero), vsix);
                        _mm512_storeu_ps(dp + j, v);
                    }
                    for (; j < w; ++j) {
                        float v = dp[j];
                        if (act_mode == 1) { if (v < 0.0f) v = 0.0f; }
                        else { if (v < 0.0f) v = 0.0f; if (v > 6.0f) v = 6.0f; }
                        dp[j] = v;
                    }
                }
            }
        }
    }

    for (; oc < c_out; ++oc) {
        const float *wk = w_4d + oc * k_stride;
        const float b = bias ? bias[oc] : 0.0f;
        for (int64_t oy = 0; oy < h; ++oy) {
            float *dp = out + oc * hw + oy * w;
            for (int64_t ox = 0; ox < w; ++ox) dp[ox] = b;
            for (int64_t ic = 0; ic < c_in; ++ic) {
                const float *xc = x + ic * hw;
                const float *wc = wk + ic * 9;
                for (int ky = -1; ky <= 1; ++ky) {
                    const int64_t sy = oy + ky;
                    if (sy < 0 || sy >= h) continue;
                    const float *sr = xc + sy * w;
                    for (int kx = -1; kx <= 1; ++kx) {
                        const float wt = wc[(ky + 1) * 3 + (kx + 1)];
                        for (int64_t ox = 0; ox < w; ++ox) {
                            const int64_t sx = ox + kx;
                            if (sx >= 0 && sx < w) dp[ox] += sr[sx] * wt;
                        }
                    }
                }
            }
            if (act_mode != 0) {
                for (int64_t ox = 0; ox < w; ++ox) {
                    float v = dp[ox];
                    if (act_mode == 1) { if (v < 0.0f) v = 0.0f; }
                    else { if (v < 0.0f) v = 0.0f; if (v > 6.0f) v = 6.0f; }
                    dp[ox] = v;
                }
            }
        }
    }
}

void add_bias_act_f32(
    float *x,
    const float *bias,
    int64_t c,
    int64_t hw,
    int act_mode
) {
    /* act_mode: 0=none, 1=relu, 2=relu6 */
    const __m512 vzero = _mm512_setzero_ps();
    const __m512 vsix = _mm512_set1_ps(6.0f);

    for (int64_t ch = 0; ch < c; ++ch) {
        float *row = x + ch * hw;
        const float b = bias ? bias[ch] : 0.0f;
        const __m512 vb = _mm512_set1_ps(b);
        int64_t i = 0;
        for (; i + 16 <= hw; i += 16) {
            __m512 v = _mm512_add_ps(_mm512_loadu_ps(row + i), vb);
            if (act_mode == 1) {
                v = _mm512_max_ps(v, vzero);
            } else if (act_mode == 2) {
                v = _mm512_min_ps(_mm512_max_ps(v, vzero), vsix);
            }
            _mm512_storeu_ps(row + i, v);
        }
        for (; i < hw; ++i) {
            float v = row[i] + b;
            if (act_mode == 1) {
                if (v < 0.0f) v = 0.0f;
            } else if (act_mode == 2) {
                if (v < 0.0f) v = 0.0f;
                if (v > 6.0f) v = 6.0f;
            }
            row[i] = v;
        }
    }
}

void ibnorm_tail_relu_f32(
    float *x,
    int64_t c,
    int64_t hw,
    float eps,
    int apply_relu
) {
    /*
     * First c_bn = c // 2 channels already have BatchNorm2d folded into conv weight/bias.
     * Remaining c - c_bn channels apply InstanceNorm2d(affine=False, eps).
     * Finally applies ReLU across all channels if apply_relu != 0.
     */
    const int64_t c_bn = c / 2;
    const __m512 vzero = _mm512_setzero_ps();

    if (apply_relu) {
        int64_t n_bn = c_bn * hw;
        int64_t i = 0;
        for (; i + 16 <= n_bn; i += 16) {
            _mm512_storeu_ps(x + i, _mm512_max_ps(_mm512_loadu_ps(x + i), vzero));
        }
        for (; i < n_bn; ++i) {
            if (x[i] < 0.0f) x[i] = 0.0f;
        }
    }

    for (int64_t ch = c_bn; ch < c; ++ch) {
        float *row = x + ch * hw;
        __m512 vsum = _mm512_setzero_ps();
        int64_t i = 0;
        for (; i + 16 <= hw; i += 16) {
            vsum = _mm512_add_ps(vsum, _mm512_loadu_ps(row + i));
        }
        double mean_d = (double)_mm512_reduce_add_ps(vsum);
        for (; i < hw; ++i) {
            mean_d += (double)row[i];
        }
        const float mean = (float)(mean_d / (double)hw);
        const __m512 vmean = _mm512_set1_ps(mean);

        __m512 vvsum = _mm512_setzero_ps();
        for (i = 0; i + 16 <= hw; i += 16) {
            __m512 d = _mm512_sub_ps(_mm512_loadu_ps(row + i), vmean);
            vvsum = _mm512_fmadd_ps(d, d, vvsum);
        }
        double var_d = (double)_mm512_reduce_add_ps(vvsum);
        for (; i < hw; ++i) {
            double d = (double)row[i] - (double)mean;
            var_d += d * d;
        }
        const float inv_std = 1.0f / sqrtf((float)(var_d / (double)hw) + eps);
        const __m512 vinv = _mm512_set1_ps(inv_std);

        for (i = 0; i + 16 <= hw; i += 16) {
            __m512 v = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(row + i), vmean), vinv);
            if (apply_relu) {
                v = _mm512_max_ps(v, vzero);
            }
            _mm512_storeu_ps(row + i, v);
        }
        for (; i < hw; ++i) {
            float v = (row[i] - mean) * inv_std;
            if (apply_relu && v < 0.0f) v = 0.0f;
            row[i] = v;
        }
    }
}

void alpha_composite_bg_u8(
    const uint8_t *fg,
    const uint8_t *bg,
    const float *alpha_hw,
    uint8_t *out,
    int64_t num_pixels
) {
    for (int64_t i = 0; i < num_pixels; ++i) {
        float a = alpha_hw[i];
        if (a < 0.0f) a = 0.0f;
        if (a > 1.0f) a = 1.0f;
        const float inv = 1.0f - a;
        const int64_t idx = i * 3;
        out[idx + 0] = (uint8_t)(lrintf((float)fg[idx + 0] * a + (float)bg[idx + 0] * inv));
        out[idx + 1] = (uint8_t)(lrintf((float)fg[idx + 1] * a + (float)bg[idx + 1] * inv));
        out[idx + 2] = (uint8_t)(lrintf((float)fg[idx + 2] * a + (float)bg[idx + 2] * inv));
    }
}
