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
    const __m512 vzero = _mm512_setzero_ps();
    const __m512 vone = _mm512_set1_ps(1.0f);
    const __m512 v256 = _mm512_set1_ps(256.0f);
    int64_t i = 0;

    for (; i + 16 <= num_pixels; i += 16) {
        __m512 va = _mm512_loadu_ps(alpha_hw + i);
        va = _mm512_min_ps(_mm512_max_ps(va, vzero), vone);
        __m512i va_q8 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(va, v256), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        int32_t aq[16];
        _mm512_storeu_si512(aq, va_q8);
        const uint8_t *fp = fg + i * 3;
        const uint8_t *bp = bg + i * 3;
        uint8_t *op = out + i * 3;
        for (int p = 0; p < 16; ++p) {
            const int32_t a = aq[p];
            const int32_t inv = 256 - a;
            const int o = p * 3;
            op[o + 0] = (uint8_t)((fp[o + 0] * a + bp[o + 0] * inv + 128) >> 8);
            op[o + 1] = (uint8_t)((fp[o + 1] * a + bp[o + 1] * inv + 128) >> 8);
            op[o + 2] = (uint8_t)((fp[o + 2] * a + bp[o + 2] * inv + 128) >> 8);
        }
    }
    for (; i < num_pixels; ++i) {
        float af = alpha_hw[i];
        if (af < 0.0f) af = 0.0f;
        if (af > 1.0f) af = 1.0f;
        const int32_t a = (int32_t)(af * 256.0f + 0.5f);
        const int32_t inv = 256 - a;
        const int64_t idx = i * 3;
        out[idx + 0] = (uint8_t)((fg[idx + 0] * a + bg[idx + 0] * inv + 128) >> 8);
        out[idx + 1] = (uint8_t)((fg[idx + 1] * a + bg[idx + 1] * inv + 128) >> 8);
        out[idx + 2] = (uint8_t)((fg[idx + 2] * a + bg[idx + 2] * inv + 128) >> 8);
    }
}

static inline float _bilinear_scale_coord(int i, int in_len, float scale) {
    float prod = ((float)i + 0.5f) * scale;
    __asm__("" : "+x"(prod));
    float c = prod - 0.5f;
    if (c < 0.0f) c = 0.0f;
    if (c > (float)(in_len - 1)) c = (float)(in_len - 1);
    return c;
}

static void _build_bilinear_lut_f32(
    int in_len,
    int out_len,
    int32_t *i0_out,
    int32_t *i1_out,
    float *w_out
) {
    if (in_len == 1 || out_len == 1) {
        for (int i = 0; i < out_len; ++i) {
            i0_out[i] = 0;
            i1_out[i] = 0;
            w_out[i] = 0.0f;
        }
        return;
    }
    const float scale = (float)((double)in_len / (double)out_len);
    for (int i = 0; i < out_len; ++i) {
        float c = _bilinear_scale_coord(i, in_len, scale);
        int i0 = (int)floorf(c);
        i0_out[i] = i0;
        i1_out[i] = (i0 + 1 < in_len) ? (i0 + 1) : (in_len - 1);
        w_out[i] = c - (float)i0;
    }
}

/* fmt_code: 0=RGB3, 1=YUYV, 2=UYVY, 3=BGR3 */
int extract_luma_low_f32(
    const uint8_t *src,
    int fmt_code,
    int in_h,
    int in_w,
    float *out_f32,
    int out_h,
    int out_w
) {
    if (in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0) {
        return -1;
    }
    int32_t *ibuf = (int32_t *)malloc(((size_t)out_w * 2 + (size_t)out_h * 2) * sizeof(int32_t));
    float *fbuf = (float *)malloc(((size_t)out_w + (size_t)out_h + (size_t)out_w * 2) * sizeof(float));
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
    float *r0 = wy + out_h;
    float *r1 = r0 + out_w;
    const float inv255 = 1.0f / 255.0f;

    _build_bilinear_lut_f32(in_w, out_w, x0, x1, wx);
    _build_bilinear_lut_f32(in_h, out_h, y0, y1, wy);

    const float rw = (fmt_code == 3) ? (0.114f * inv255) : (0.299f * inv255);
    const float gw = 0.587f * inv255;
    const float bw = (fmt_code == 3) ? (0.299f * inv255) : (0.114f * inv255);
    const int yuyv_off = (fmt_code == 2) ? 1 : 0;

    for (int oy = 0; oy < out_h; ++oy) {
        const int sy0 = y0[oy], sy1 = y1[oy];
        const float fwy = wy[oy];
        if (fmt_code == 1 || fmt_code == 2) {
            const uint8_t *sr0 = src + (size_t)sy0 * (size_t)in_w * 2 + yuyv_off;
            const uint8_t *sr1 = src + (size_t)sy1 * (size_t)in_w * 2 + yuyv_off;
            for (int ox = 0; ox < out_w; ++ox) {
                const int ix0 = x0[ox] * 2, ix1 = x1[ox] * 2;
                const float fwx = wx[ox];
                r0[ox] = ((float)sr0[ix0] + fwx * ((float)sr0[ix1] - (float)sr0[ix0])) * inv255;
                r1[ox] = ((float)sr1[ix0] + fwx * ((float)sr1[ix1] - (float)sr1[ix0])) * inv255;
            }
        } else {
            const uint8_t *sr0 = src + (size_t)sy0 * (size_t)in_w * 3;
            const uint8_t *sr1 = src + (size_t)sy1 * (size_t)in_w * 3;
            for (int ox = 0; ox < out_w; ++ox) {
                const int ix0 = x0[ox] * 3, ix1 = x1[ox] * 3;
                const float fwx = wx[ox];
                float l00 = (float)sr0[ix0 + 0] * rw + (float)sr0[ix0 + 1] * gw + (float)sr0[ix0 + 2] * bw;
                float l01 = (float)sr0[ix1 + 0] * rw + (float)sr0[ix1 + 1] * gw + (float)sr0[ix1 + 2] * bw;
                float l10 = (float)sr1[ix0 + 0] * rw + (float)sr1[ix0 + 1] * gw + (float)sr1[ix0 + 2] * bw;
                float l11 = (float)sr1[ix1 + 0] * rw + (float)sr1[ix1 + 1] * gw + (float)sr1[ix1 + 2] * bw;
                r0[ox] = l00 + fwx * (l01 - l00);
                r1[ox] = l10 + fwx * (l11 - l10);
            }
        }
        float *drow = out_f32 + (size_t)oy * (size_t)out_w;
        for (int ox = 0; ox < out_w; ++ox) {
            drow[ox] = r0[ox] + fwy * (r1[ox] - r0[ox]);
        }
    }

    free(ibuf);
    free(fbuf);
    return 0;
}

int bilinear_resize_clamp01_f32(
    const float *src,
    float *dst,
    int in_h,
    int in_w,
    int out_h,
    int out_w
) {
    if (in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0) {
        return -1;
    }
    const __m512 vzero = _mm512_setzero_ps();
    const __m512 vone = _mm512_set1_ps(1.0f);
    if (in_h == out_h && in_w == out_w) {
        const int64_t n = (int64_t)out_h * out_w;
        int64_t i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 v = _mm512_min_ps(_mm512_max_ps(_mm512_loadu_ps(src + i), vzero), vone);
            _mm512_storeu_ps(dst + i, v);
        }
        for (; i < n; ++i) {
            float v = src[i];
            dst[i] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        }
        return 0;
    }

    int32_t *ibuf = (int32_t *)malloc(((size_t)out_w * 2 + (size_t)out_h * 2) * sizeof(int32_t));
    float *fbuf = (float *)malloc(((size_t)out_w + (size_t)out_h + (size_t)out_w * 2) * sizeof(float));
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
    float *slots[2] = {wy + out_h, wy + out_h + out_w};
    int cached_id[2] = {-1, -1};

    _build_bilinear_lut_f32(in_w, out_w, x0, x1, wx);
    _build_bilinear_lut_f32(in_h, out_h, y0, y1, wy);

    for (int oy = 0; oy < out_h; ++oy) {
        const int sy0 = y0[oy], sy1 = y1[oy];
        float *p0 = NULL;
        for (int s = 0; s < 2; ++s) {
            if (cached_id[s] == sy0) { p0 = slots[s]; break; }
        }
        if (!p0) {
            int slot = (cached_id[0] == sy1) ? 1 : 0;
            p0 = slots[slot];
            cached_id[slot] = sy0;
            const float *sr = src + (size_t)sy0 * (size_t)in_w;
            for (int ox = 0; ox < out_w; ++ox) {
                float a = sr[x0[ox]];
                p0[ox] = a + wx[ox] * (sr[x1[ox]] - a);
            }
        }
        float *p1 = NULL;
        if (sy1 == sy0 || wy[oy] == 0.0f) {
            p1 = p0;
        } else {
            for (int s = 0; s < 2; ++s) {
                if (cached_id[s] == sy1) { p1 = slots[s]; break; }
            }
            if (!p1) {
                int slot = (p0 == slots[0]) ? 1 : 0;
                p1 = slots[slot];
                cached_id[slot] = sy1;
                const float *sr = src + (size_t)sy1 * (size_t)in_w;
                for (int ox = 0; ox < out_w; ++ox) {
                    float a = sr[x0[ox]];
                    p1[ox] = a + wx[ox] * (sr[x1[ox]] - a);
                }
            }
        }
        float *drow = dst + (size_t)oy * (size_t)out_w;
        const __m512 vwy = _mm512_set1_ps(wy[oy]);
        int ox = 0;
        for (; ox + 16 <= out_w; ox += 16) {
            __m512 va = _mm512_loadu_ps(p0 + ox);
            __m512 vb = _mm512_loadu_ps(p1 + ox);
            __m512 vf = _mm512_fmadd_ps(vwy, _mm512_sub_ps(vb, va), va);
            vf = _mm512_min_ps(_mm512_max_ps(vf, vzero), vone);
            _mm512_storeu_ps(drow + ox, vf);
        }
        for (; ox < out_w; ++ox) {
            float v = p0[ox] + wy[oy] * (p1[ox] - p0[ox]);
            drow[ox] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        }
    }

    free(ibuf);
    free(fbuf);
    return 0;
}

/* fmt_code: 0=RGB3/BGR3, 1=YUYV, 2=UYVY */
void downsample2x_bg_hw4_alpha_f32(
    const uint8_t *frame_u8,
    int fmt_code,
    const float *alpha_hw,
    float *out_hw4,
    int in_h,
    int in_w
) {
    const int out_h = (in_h + 1) >> 1;
    const int out_w = (in_w + 1) >> 1;
    const __m512 vone = _mm512_set1_ps(1.0f);
    const __m512 vmin = _mm512_set1_ps(1e-3f);
    const __m512 vscale = _mm512_set1_ps(1.0f - 1e-3f);
    const __m512 v025 = _mm512_set1_ps(0.25f);
    const __m512i vperm_even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i vperm_odd  = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);

    if ((fmt_code == 1 || fmt_code == 2) && (in_w & 1) == 0) {
        const size_t row_bytes = (size_t)in_w * 2;
        const int y0_off = (fmt_code == 2) ? 1 : 0;
        const int u_off = (fmt_code == 2) ? 0 : 1;
        const int y1_off = (fmt_code == 2) ? 3 : 2;
        const int v_off = (fmt_code == 2) ? 2 : 3;

        for (int oy = 0; oy < out_h; ++oy) {
            const int sy0 = oy * 2;
            const int sy1 = (sy0 + 1 < in_h) ? (sy0 + 1) : sy0;
            const uint8_t *fr0 = frame_u8 + (size_t)sy0 * row_bytes;
            const uint8_t *fr1 = frame_u8 + (size_t)sy1 * row_bytes;
            const float *ar0 = alpha_hw + (size_t)sy0 * (size_t)in_w;
            const float *ar1 = alpha_hw + (size_t)sy1 * (size_t)in_w;
            float *dst = out_hw4 + (size_t)oy * (size_t)out_w * 4;

            int ox = 0;
            for (; ox + 16 <= out_w; ox += 16) {
                const int x0 = ox * 2;
                __m512 a0_lo = _mm512_loadu_ps(ar0 + x0);
                __m512 a0_hi = _mm512_loadu_ps(ar0 + x0 + 16);
                __m512 a1_lo = _mm512_loadu_ps(ar1 + x0);
                __m512 a1_hi = _mm512_loadu_ps(ar1 + x0 + 16);

                __m512 wa = _mm512_min_ps(_mm512_max_ps(_mm512_fmadd_ps(_mm512_sub_ps(vone, _mm512_permutex2var_ps(a0_lo, vperm_even, a0_hi)), vscale, vmin), vmin), vone);
                __m512 wb = _mm512_min_ps(_mm512_max_ps(_mm512_fmadd_ps(_mm512_sub_ps(vone, _mm512_permutex2var_ps(a0_lo, vperm_odd,  a0_hi)), vscale, vmin), vmin), vone);
                __m512 wc = _mm512_min_ps(_mm512_max_ps(_mm512_fmadd_ps(_mm512_sub_ps(vone, _mm512_permutex2var_ps(a1_lo, vperm_even, a1_hi)), vscale, vmin), vmin), vone);
                __m512 wd = _mm512_min_ps(_mm512_max_ps(_mm512_fmadd_ps(_mm512_sub_ps(vone, _mm512_permutex2var_ps(a1_lo, vperm_odd,  a1_hi)), vscale, vmin), vmin), vone);

                __m512i pk0 = _mm512_loadu_si512(fr0 + (size_t)ox * 4);
                __m512i pk1 = _mm512_loadu_si512(fr1 + (size_t)ox * 4);

                __m512i mask_byte = _mm512_set1_epi32(0xFF);
                __m512 cy0 = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(pk0, y0_off * 8), mask_byte));
                __m512 cy1 = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(pk0, y1_off * 8), mask_byte));
                __m512 cu0 = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(pk0, u_off * 8),  mask_byte));
                __m512 cv0 = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(pk0, v_off * 8),  mask_byte));

                __m512 cy2 = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(pk1, y0_off * 8), mask_byte));
                __m512 cy3 = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(pk1, y1_off * 8), mask_byte));
                __m512 cu1 = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(pk1, u_off * 8),  mask_byte));
                __m512 cv1 = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(pk1, v_off * 8),  mask_byte));

                __m512 wtop = _mm512_add_ps(wa, wb);
                __m512 wbot = _mm512_add_ps(wc, wd);
                __m512 out_y = _mm512_mul_ps(_mm512_fmadd_ps(cy0, wa, _mm512_fmadd_ps(cy1, wb, _mm512_fmadd_ps(cy2, wc, _mm512_mul_ps(cy3, wd)))), v025);
                __m512 out_u = _mm512_mul_ps(_mm512_fmadd_ps(cu0, wtop, _mm512_mul_ps(cu1, wbot)), v025);
                __m512 out_v = _mm512_mul_ps(_mm512_fmadd_ps(cv0, wtop, _mm512_mul_ps(cv1, wbot)), v025);
                __m512 out_w = _mm512_mul_ps(_mm512_add_ps(wtop, wbot), v025);

                float ya[16]; // replaced scalar scatter with 4x16 SIMD AoS store
                __m512 t0 = _mm512_unpacklo_ps(out_y, out_u);
                __m512 t1 = _mm512_unpackhi_ps(out_y, out_u);
                __m512 t2 = _mm512_unpacklo_ps(out_v, out_w);
                __m512 t3 = _mm512_unpackhi_ps(out_v, out_w);
                __m512 qa = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t0), _mm512_castps_pd(t2)));
                __m512 qb = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t0), _mm512_castps_pd(t2)));
                __m512 qc = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t1), _mm512_castps_pd(t3)));
                __m512 qd = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t1), _mm512_castps_pd(t3)));
                __m512 ab0 = _mm512_shuffle_f32x4(qa, qb, 0x44);
                __m512 ab1 = _mm512_shuffle_f32x4(qa, qb, 0xEE);
                __m512 cd0 = _mm512_shuffle_f32x4(qc, qd, 0x44);
                __m512 cd1 = _mm512_shuffle_f32x4(qc, qd, 0xEE);
                float *dp = dst + (size_t)ox * 4;
                _mm512_storeu_ps(dp +  0, _mm512_shuffle_f32x4(ab0, cd0, 0x88));
                _mm512_storeu_ps(dp + 16, _mm512_shuffle_f32x4(ab0, cd0, 0xDD));
                _mm512_storeu_ps(dp + 32, _mm512_shuffle_f32x4(ab1, cd1, 0x88));
                _mm512_storeu_ps(dp + 48, _mm512_shuffle_f32x4(ab1, cd1, 0xDD));
                (void)ya;
            }
            for (; ox < out_w; ++ox) {
                const int x0 = ox * 2;
                float a00 = ar0[x0], a01 = ar0[x0 + 1], a10 = ar1[x0], a11 = ar1[x0 + 1];
                float wa_s = (1.0f - a00) * (1.0f - 1e-3f) + 1e-3f; wa_s = wa_s < 1e-3f ? 1e-3f : (wa_s > 1.0f ? 1.0f : wa_s);
                float wb_s = (1.0f - a01) * (1.0f - 1e-3f) + 1e-3f; wb_s = wb_s < 1e-3f ? 1e-3f : (wb_s > 1.0f ? 1.0f : wb_s);
                float wc_s = (1.0f - a10) * (1.0f - 1e-3f) + 1e-3f; wc_s = wc_s < 1e-3f ? 1e-3f : (wc_s > 1.0f ? 1.0f : wc_s);
                float wd_s = (1.0f - a11) * (1.0f - 1e-3f) + 1e-3f; wd_s = wd_s < 1e-3f ? 1e-3f : (wd_s > 1.0f ? 1.0f : wd_s);
                const uint8_t *p0 = fr0 + (size_t)ox * 4;
                const uint8_t *p1 = fr1 + (size_t)ox * 4;
                dst[ox * 4 + 0] = 0.25f * ((float)p0[y0_off] * wa_s + (float)p0[y1_off] * wb_s + (float)p1[y0_off] * wc_s + (float)p1[y1_off] * wd_s);
                dst[ox * 4 + 1] = 0.25f * ((float)p0[u_off] * (wa_s + wb_s) + (float)p1[u_off] * (wc_s + wd_s));
                dst[ox * 4 + 2] = 0.25f * ((float)p0[v_off] * (wa_s + wb_s) + (float)p1[v_off] * (wc_s + wd_s));
                dst[ox * 4 + 3] = 0.25f * (wa_s + wb_s + wc_s + wd_s);
            }
        }
        return;
    }

    for (int oy = 0; oy < out_h; ++oy) {
        const int sy0 = oy * 2;
        const int sy1 = (sy0 + 1 < in_h) ? (sy0 + 1) : sy0;
        const uint8_t *fr0 = frame_u8 + (size_t)sy0 * (size_t)in_w * 3;
        const uint8_t *fr1 = frame_u8 + (size_t)sy1 * (size_t)in_w * 3;
        const float *ar0 = alpha_hw + (size_t)sy0 * (size_t)in_w;
        const float *ar1 = alpha_hw + (size_t)sy1 * (size_t)in_w;
        float *dst = out_hw4 + (size_t)oy * (size_t)out_w * 4;

        for (int ox = 0; ox < out_w; ++ox) {
            const int x0 = ox * 2;
            const int x1 = (x0 + 1 < in_w) ? (x0 + 1) : x0;
            float wa_s = (1.0f - ar0[x0]) * (1.0f - 1e-3f) + 1e-3f; wa_s = wa_s < 1e-3f ? 1e-3f : (wa_s > 1.0f ? 1.0f : wa_s);
            float wb_s = (1.0f - ar0[x1]) * (1.0f - 1e-3f) + 1e-3f; wb_s = wb_s < 1e-3f ? 1e-3f : (wb_s > 1.0f ? 1.0f : wb_s);
            float wc_s = (1.0f - ar1[x0]) * (1.0f - 1e-3f) + 1e-3f; wc_s = wc_s < 1e-3f ? 1e-3f : (wc_s > 1.0f ? 1.0f : wc_s);
            float wd_s = (1.0f - ar1[x1]) * (1.0f - 1e-3f) + 1e-3f; wd_s = wd_s < 1e-3f ? 1e-3f : (wd_s > 1.0f ? 1.0f : wd_s);
            const uint8_t *pa = fr0 + (size_t)x0 * 3;
            const uint8_t *pb = fr0 + (size_t)x1 * 3;
            const uint8_t *pc = fr1 + (size_t)x0 * 3;
            const uint8_t *pd = fr1 + (size_t)x1 * 3;
            dst[ox * 4 + 0] = 0.25f * ((float)pa[0] * wa_s + (float)pb[0] * wb_s + (float)pc[0] * wc_s + (float)pd[0] * wd_s);
            dst[ox * 4 + 1] = 0.25f * ((float)pa[1] * wa_s + (float)pb[1] * wb_s + (float)pc[1] * wc_s + (float)pd[1] * wd_s);
            dst[ox * 4 + 2] = 0.25f * ((float)pa[2] * wa_s + (float)pb[2] * wb_s + (float)pc[2] * wc_s + (float)pd[2] * wd_s);
            dst[ox * 4 + 3] = 0.25f * (wa_s + wb_s + wc_s + wd_s);
        }
    }
}

static void _unpremul_upsample_row_x_soa(
    const float *src_hw3,
    float *dst_c0,
    float *dst_c1,
    float *dst_c2,
    const int32_t *x0,
    const int32_t *x1,
    const float *wx,
    int out_w
) {
    for (int ox = 0; ox < out_w; ++ox) {
        const float *p0 = src_hw3 + (size_t)x0[ox] * 3;
        const float *p1 = src_hw3 + (size_t)x1[ox] * 3;
        const float fwx = wx[ox];
        dst_c0[ox] = p0[0] + fwx * (p1[0] - p0[0]);
        dst_c1[ox] = p0[1] + fwx * (p1[1] - p0[1]);
        dst_c2[ox] = p0[2] + fwx * (p1[2] - p0[2]);
    }
}

typedef struct {
    int out_h;
    int out_w;
    int bg_h;
    int bg_w;
    int32_t *ibuf;
    float *fbuf;
    int32_t *bg_i32;
    size_t bg3_cap;
    float *bg3;
} fused_comp_cache_t;

static __thread fused_comp_cache_t g_fused_cache = {0};

/* fmt_code: 0=RGB3/BGR3, 1=YUYV, 2=UYVY */
int unpremul_upsample_bg_composite_u8(
    const uint8_t *fg_u8,
    int fmt_code,
    const float *bg_hw4,
    const float *alpha_hw,
    uint8_t *out_u8,
    int out_h,
    int out_w,
    int bg_h,
    int bg_w,
    float blur_strength
) {
    if (out_h <= 0 || out_w <= 0 || bg_h <= 0 || bg_w <= 0) {
        return -1;
    }
    const size_t bg_n = (size_t)bg_h * (size_t)bg_w;
    if (bg_n * 3 > g_fused_cache.bg3_cap) {
        float *nb = (float *)realloc(g_fused_cache.bg3, bg_n * 3 * sizeof(float));
        if (!nb) return -1;
        g_fused_cache.bg3 = nb;
        g_fused_cache.bg3_cap = bg_n * 3;
    }
    if (
        g_fused_cache.out_h != out_h || g_fused_cache.out_w != out_w ||
        g_fused_cache.bg_h != bg_h || g_fused_cache.bg_w != bg_w
    ) {
        int32_t *ni = (int32_t *)realloc(g_fused_cache.ibuf, ((size_t)out_w * 2 + (size_t)out_h * 2) * sizeof(int32_t));
        float *nf = (float *)realloc(g_fused_cache.fbuf, ((size_t)out_w + (size_t)out_h + (size_t)out_w * 6) * sizeof(float));
        int32_t *nbi = (int32_t *)realloc(g_fused_cache.bg_i32, (size_t)out_w * 3 * sizeof(int32_t));
        if (!ni || !nf || !nbi) {
            if (ni) g_fused_cache.ibuf = ni;
            if (nf) g_fused_cache.fbuf = nf;
            if (nbi) g_fused_cache.bg_i32 = nbi;
            return -1;
        }
        g_fused_cache.ibuf = ni;
        g_fused_cache.fbuf = nf;
        g_fused_cache.bg_i32 = nbi;
        g_fused_cache.out_h = out_h;
        g_fused_cache.out_w = out_w;
        g_fused_cache.bg_h = bg_h;
        g_fused_cache.bg_w = bg_w;
        _build_bilinear_lut_f32(bg_w, out_w, ni, ni + out_w, nf);
        _build_bilinear_lut_f32(bg_h, out_h, ni + out_w * 2, ni + out_w * 2 + out_h, nf + out_w);
    }

    float *bg3 = g_fused_cache.bg3;
    for (size_t i = 0; i < bg_n; ++i) {
        float wv = bg_hw4[i * 4 + 3];
        float inv = 1.0f / (wv > 1e-4f ? wv : 1e-4f);
        bg3[i * 3 + 0] = bg_hw4[i * 4 + 0] * inv;
        bg3[i * 3 + 1] = bg_hw4[i * 4 + 1] * inv;
        bg3[i * 3 + 2] = bg_hw4[i * 4 + 2] * inv;
    }

    const int32_t *x0 = g_fused_cache.ibuf;
    const int32_t *x1 = x0 + out_w;
    const int32_t *y0 = x1 + out_w;
    const int32_t *y1 = y0 + out_h;
    const float *wx = g_fused_cache.fbuf;
    const float *wy = wx + out_w;
    float *slots_base = g_fused_cache.fbuf + out_w + out_h;
    float *s0_c0 = slots_base;
    float *s0_c1 = s0_c0 + out_w;
    float *s0_c2 = s0_c1 + out_w;
    float *s1_c0 = s0_c2 + out_w;
    float *s1_c1 = s1_c0 + out_w;
    float *s1_c2 = s1_c1 + out_w;
    int32_t *bg_c0 = g_fused_cache.bg_i32;
    int32_t *bg_c1 = bg_c0 + out_w;
    int32_t *bg_c2 = bg_c1 + out_w;
    int cached_id[2] = {-1, -1};

    const __m512 vzero = _mm512_setzero_ps();
    const __m512 vone = _mm512_set1_ps(1.0f);
    const __m512 v255 = _mm512_set1_ps(255.0f);
    const __m512 v256 = _mm512_set1_ps(256.0f);
    const __m512i v256i = _mm512_set1_epi32(256);
    const __m512i v128i = _mm512_set1_epi32(128);
    const __m512i vff = _mm512_set1_epi32(0xFF);
    const __m512 va_scale = _mm512_set1_ps(blur_strength);
    const __m512 va_bias = _mm512_set1_ps(1.0f - blur_strength);
    const __m512i vperm_even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i vperm_odd  = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    const size_t bg_row_stride = (size_t)bg_w * 3;
    const int is_yuv = (fmt_code == 1 || fmt_code == 2);
    const int y0_shift = (fmt_code == 2) ? 8 : 0;
    const int u_shift  = (fmt_code == 2) ? 0 : 8;
    const int y1_shift = (fmt_code == 2) ? 24 : 16;
    const int v_shift  = (fmt_code == 2) ? 16 : 24;
    const int y0_off = (fmt_code == 2) ? 1 : 0;
    const int u_off  = (fmt_code == 2) ? 0 : 1;
    const int y1_off = (fmt_code == 2) ? 3 : 2;
    const int v_off  = (fmt_code == 2) ? 2 : 3;

    for (int oy = 0; oy < out_h; ++oy) {
        const int r0 = y0[oy], r1 = y1[oy];
        int p0_idx = -1;
        for (int s = 0; s < 2; ++s) {
            if (cached_id[s] == r0) { p0_idx = s; break; }
        }
        if (p0_idx < 0) {
            p0_idx = (cached_id[0] == r1) ? 1 : 0;
            cached_id[p0_idx] = r0;
            float *dc0 = (p0_idx == 0) ? s0_c0 : s1_c0;
            float *dc1 = (p0_idx == 0) ? s0_c1 : s1_c1;
            float *dc2 = (p0_idx == 0) ? s0_c2 : s1_c2;
            _unpremul_upsample_row_x_soa(bg3 + (size_t)r0 * bg_row_stride, dc0, dc1, dc2, x0, x1, wx, out_w);
        }
        int p1_idx = p0_idx;
        if (r1 != r0 && wy[oy] != 0.0f) {
            p1_idx = -1;
            for (int s = 0; s < 2; ++s) {
                if (cached_id[s] == r1) { p1_idx = s; break; }
            }
            if (p1_idx < 0) {
                p1_idx = (p0_idx == 0) ? 1 : 0;
                cached_id[p1_idx] = r1;
                float *dc0 = (p1_idx == 0) ? s0_c0 : s1_c0;
                float *dc1 = (p1_idx == 0) ? s0_c1 : s1_c1;
                float *dc2 = (p1_idx == 0) ? s0_c2 : s1_c2;
                _unpremul_upsample_row_x_soa(bg3 + (size_t)r1 * bg_row_stride, dc0, dc1, dc2, x0, x1, wx, out_w);
            }
        }

        const float *p0_c0 = (p0_idx == 0) ? s0_c0 : s1_c0;
        const float *p0_c1 = (p0_idx == 0) ? s0_c1 : s1_c1;
        const float *p0_c2 = (p0_idx == 0) ? s0_c2 : s1_c2;
        const float *p1_c0 = (p1_idx == 0) ? s0_c0 : s1_c0;
        const float *p1_c1 = (p1_idx == 0) ? s0_c1 : s1_c1;
        const float *p1_c2 = (p1_idx == 0) ? s0_c2 : s1_c2;

        const __m512 vwy = _mm512_set1_ps(wy[oy]);
        int ox = 0;
        for (; ox + 16 <= out_w; ox += 16) {
            __m512 f0 = _mm512_fmadd_ps(vwy, _mm512_sub_ps(_mm512_loadu_ps(p1_c0 + ox), _mm512_loadu_ps(p0_c0 + ox)), _mm512_loadu_ps(p0_c0 + ox));
            __m512 f1 = _mm512_fmadd_ps(vwy, _mm512_sub_ps(_mm512_loadu_ps(p1_c1 + ox), _mm512_loadu_ps(p0_c1 + ox)), _mm512_loadu_ps(p0_c1 + ox));
            __m512 f2 = _mm512_fmadd_ps(vwy, _mm512_sub_ps(_mm512_loadu_ps(p1_c2 + ox), _mm512_loadu_ps(p0_c2 + ox)), _mm512_loadu_ps(p0_c2 + ox));
            f0 = _mm512_min_ps(_mm512_max_ps(f0, vzero), v255);
            f1 = _mm512_min_ps(_mm512_max_ps(f1, vzero), v255);
            f2 = _mm512_min_ps(_mm512_max_ps(f2, vzero), v255);
            _mm512_storeu_si512(bg_c0 + ox, _mm512_cvtps_epi32(_mm512_roundscale_ps(f0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
            _mm512_storeu_si512(bg_c1 + ox, _mm512_cvtps_epi32(_mm512_roundscale_ps(f1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
            _mm512_storeu_si512(bg_c2 + ox, _mm512_cvtps_epi32(_mm512_roundscale_ps(f2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
        }
        for (; ox < out_w; ++ox) {
            float v0 = p0_c0[ox] + wy[oy] * (p1_c0[ox] - p0_c0[ox]);
            float v1 = p0_c1[ox] + wy[oy] * (p1_c1[ox] - p0_c1[ox]);
            float v2 = p0_c2[ox] + wy[oy] * (p1_c2[ox] - p0_c2[ox]);
            bg_c0[ox] = (int32_t)lrintf(v0 < 0.0f ? 0.0f : (v0 > 255.0f ? 255.0f : v0));
            bg_c1[ox] = (int32_t)lrintf(v1 < 0.0f ? 0.0f : (v1 > 255.0f ? 255.0f : v1));
            bg_c2[ox] = (int32_t)lrintf(v2 < 0.0f ? 0.0f : (v2 > 255.0f ? 255.0f : v2));
        }

        const float *ar = alpha_hw + (size_t)oy * (size_t)out_w;
        if (is_yuv && (out_w & 1) == 0) {
            const int pw = out_w >> 1;
            const uint8_t *fp_row = fg_u8 + (size_t)oy * (size_t)pw * 4;
            uint8_t *op_row = out_u8 + (size_t)oy * (size_t)pw * 4;
            int px_pair = 0;
            for (; px_pair + 16 <= pw; px_pair += 16) {
                const int px = px_pair * 2;
                __m512 va0 = _mm512_loadu_ps(ar + px);
                __m512 va1 = _mm512_loadu_ps(ar + px + 16);
                if (blur_strength < 1.0f) {
                    va0 = _mm512_fmadd_ps(va0, va_scale, va_bias);
                    va1 = _mm512_fmadd_ps(va1, va_scale, va_bias);
                }
                va0 = _mm512_min_ps(_mm512_max_ps(va0, vzero), vone);
                va1 = _mm512_min_ps(_mm512_max_ps(va1, vzero), vone);
                __m512i aq0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(va0, v256), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                __m512i aq1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(va1, v256), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

                __m512i a_even = _mm512_permutex2var_epi32(aq0, vperm_even, aq1);
                __m512i a_odd  = _mm512_permutex2var_epi32(aq0, vperm_odd,  aq1);
                __m512i inv_e  = _mm512_sub_epi32(v256i, a_even);
                __m512i inv_o  = _mm512_sub_epi32(v256i, a_odd);

                __m512i by0_lo = _mm512_loadu_si512(bg_c0 + px), by0_hi = _mm512_loadu_si512(bg_c0 + px + 16);
                __m512i bu0_lo = _mm512_loadu_si512(bg_c1 + px), bu0_hi = _mm512_loadu_si512(bg_c1 + px + 16);
                __m512i bv0_lo = _mm512_loadu_si512(bg_c2 + px), bv0_hi = _mm512_loadu_si512(bg_c2 + px + 16);

                __m512i by_e = _mm512_permutex2var_epi32(by0_lo, vperm_even, by0_hi);
                __m512i by_o = _mm512_permutex2var_epi32(by0_lo, vperm_odd,  by0_hi);
                __m512i bu_e = _mm512_permutex2var_epi32(bu0_lo, vperm_even, bu0_hi);
                __m512i bu_o = _mm512_permutex2var_epi32(bu0_lo, vperm_odd,  bu0_hi);
                __m512i bv_e = _mm512_permutex2var_epi32(bv0_lo, vperm_even, bv0_hi);
                __m512i bv_o = _mm512_permutex2var_epi32(bv0_lo, vperm_odd,  bv0_hi);

                __m512i pk = _mm512_loadu_si512(fp_row + (size_t)px_pair * 4);
                __m512i fy_e = _mm512_and_si512(_mm512_srli_epi32(pk, y0_shift), vff);
                __m512i fy_o = _mm512_and_si512(_mm512_srli_epi32(pk, y1_shift), vff);
                __m512i fu   = _mm512_and_si512(_mm512_srli_epi32(pk, u_shift),  vff);
                __m512i fv   = _mm512_and_si512(_mm512_srli_epi32(pk, v_shift),  vff);

                __m512i oy_e = _mm512_srai_epi32(_mm512_add_epi32(_mm512_add_epi32(_mm512_mullo_epi32(fy_e, a_even), _mm512_mullo_epi32(by_e, inv_e)), v128i), 8);
                __m512i oy_o = _mm512_srai_epi32(_mm512_add_epi32(_mm512_add_epi32(_mm512_mullo_epi32(fy_o, a_odd),  _mm512_mullo_epi32(by_o, inv_o)), v128i), 8);

                __m512i ou_e = _mm512_srai_epi32(_mm512_add_epi32(_mm512_add_epi32(_mm512_mullo_epi32(fu, a_even), _mm512_mullo_epi32(bu_e, inv_e)), v128i), 8);
                __m512i ou_o = _mm512_srai_epi32(_mm512_add_epi32(_mm512_add_epi32(_mm512_mullo_epi32(fu, a_odd),  _mm512_mullo_epi32(bu_o, inv_o)), v128i), 8);
                __m512i ou   = _mm512_srai_epi32(_mm512_add_epi32(ou_e, ou_o), 1);

                __m512i ov_e = _mm512_srai_epi32(_mm512_add_epi32(_mm512_add_epi32(_mm512_mullo_epi32(fv, a_even), _mm512_mullo_epi32(bv_e, inv_e)), v128i), 8);
                __m512i ov_o = _mm512_srai_epi32(_mm512_add_epi32(_mm512_add_epi32(_mm512_mullo_epi32(fv, a_odd),  _mm512_mullo_epi32(bv_o, inv_o)), v128i), 8);
                __m512i ov   = _mm512_srai_epi32(_mm512_add_epi32(ov_e, ov_o), 1);

                __m512i out_pk = _mm512_or_si512(
                    _mm512_or_si512(_mm512_slli_epi32(oy_e, y0_shift), _mm512_slli_epi32(ou, u_shift)),
                    _mm512_or_si512(_mm512_slli_epi32(oy_o, y1_shift), _mm512_slli_epi32(ov, v_shift))
                );
                _mm512_storeu_si512(op_row + (size_t)px_pair * 4, out_pk);
            }
            for (; px_pair < pw; ++px_pair) {
                const int px = px_pair * 2;
                float af0 = ar[px] * blur_strength + (1.0f - blur_strength);
                float af1 = ar[px + 1] * blur_strength + (1.0f - blur_strength);
                af0 = af0 < 0.0f ? 0.0f : (af0 > 1.0f ? 1.0f : af0);
                af1 = af1 < 0.0f ? 0.0f : (af1 > 1.0f ? 1.0f : af1);
                const int32_t a0 = (int32_t)(af0 * 256.0f + 0.5f), inv0 = 256 - a0;
                const int32_t a1 = (int32_t)(af1 * 256.0f + 0.5f), inv1 = 256 - a1;
                const uint8_t *fp = fp_row + (size_t)px_pair * 4;
                uint8_t *op = op_row + (size_t)px_pair * 4;
                op[y0_off] = (uint8_t)((fp[y0_off] * a0 + bg_c0[px] * inv0 + 128) >> 8);
                op[y1_off] = (uint8_t)((fp[y1_off] * a1 + bg_c0[px + 1] * inv1 + 128) >> 8);
                int32_t u0 = (fp[u_off] * a0 + bg_c1[px] * inv0 + 128) >> 8;
                int32_t u1 = (fp[u_off] * a1 + bg_c1[px + 1] * inv1 + 128) >> 8;
                int32_t v0 = (fp[v_off] * a0 + bg_c2[px] * inv0 + 128) >> 8;
                int32_t v1 = (fp[v_off] * a1 + bg_c2[px + 1] * inv1 + 128) >> 8;
                op[u_off] = (uint8_t)((u0 + u1) >> 1);
                op[v_off] = (uint8_t)((v0 + v1) >> 1);
            }
        } else {
            const uint8_t *fp_row = fg_u8 + (size_t)oy * (size_t)out_w * 3;
            uint8_t *op_row = out_u8 + (size_t)oy * (size_t)out_w * 3;
            int px = 0;
            for (; px + 16 <= out_w; px += 16) {
                __m512 va = _mm512_loadu_ps(ar + px);
                if (blur_strength < 1.0f) {
                    va = _mm512_fmadd_ps(va, va_scale, va_bias);
                }
                va = _mm512_min_ps(_mm512_max_ps(va, vzero), vone);
                __m512i aq_vec = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(va, v256), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                int32_t aq[16];
                _mm512_storeu_si512(aq, aq_vec);
                const uint8_t *fp = fp_row + (size_t)px * 3;
                uint8_t *op = op_row + (size_t)px * 3;
                for (int p = 0; p < 16; ++p) {
                    const int32_t a = aq[p], inv = 256 - a;
                    const int o = p * 3;
                    op[o + 0] = (uint8_t)((fp[o + 0] * a + bg_c0[px + p] * inv + 128) >> 8);
                    op[o + 1] = (uint8_t)((fp[o + 1] * a + bg_c1[px + p] * inv + 128) >> 8);
                    op[o + 2] = (uint8_t)((fp[o + 2] * a + bg_c2[px + p] * inv + 128) >> 8);
                }
            }
            for (; px < out_w; ++px) {
                float af = ar[px] * blur_strength + (1.0f - blur_strength);
                af = af < 0.0f ? 0.0f : (af > 1.0f ? 1.0f : af);
                const int32_t a = (int32_t)(af * 256.0f + 0.5f), inv = 256 - a;
                const int o = px * 3;
                op_row[o + 0] = (uint8_t)((fp_row[o + 0] * a + bg_c0[px] * inv + 128) >> 8);
                op_row[o + 1] = (uint8_t)((fp_row[o + 1] * a + bg_c1[px] * inv + 128) >> 8);
                op_row[o + 2] = (uint8_t)((fp_row[o + 2] * a + bg_c2[px] * inv + 128) >> 8);
            }
        }
    }

    return 0;
}
