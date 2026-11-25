#include <immintrin.h>
#include <math.h>
#include <stdint.h>

/*
 * Fast AVX-512 rational tone-curve relighting kernels for Portrait Light:
 *   f(v, g) = (g * v) / (1 + (g - 1) * v), mapping [0, 1] -> [0, 1] smoothly without clipping.
 */

void relight_rgb_u8_f32(
    const uint8_t *src_rgb,
    const float *gain_rgb,
    uint8_t *dst_rgb,
    int64_t num_pixels
) {
    const int64_t n3 = num_pixels * 3;
    const __m512 vinv255 = _mm512_set1_ps(1.0f / 255.0f);
    const __m512 v255 = _mm512_set1_ps(255.0f);
    const __m512 vone = _mm512_set1_ps(1.0f);
    const __m512 vzero = _mm512_setzero_ps();

    int64_t i = 0;
    for (; i + 16 <= n3; i += 16) {
        __m128i u8 = _mm_loadu_si128((const __m128i *)(src_rgb + i));
        __m512i i32 = _mm512_cvtepu8_epi32(u8);
        __m512 v = _mm512_mul_ps(_mm512_cvtepi32_ps(i32), vinv255);
        __m512 g = _mm512_max_ps(_mm512_loadu_ps(gain_rgb + i), vzero);

        __m512 num = _mm512_mul_ps(g, v);
        __m512 den = _mm512_fmadd_ps(_mm512_sub_ps(g, vone), v, vone);
        __m512 out_f = _mm512_mul_ps(_mm512_div_ps(num, den), v255);
        out_f = _mm512_min_ps(_mm512_max_ps(out_f, vzero), v255);

        __m512i vi = _mm512_cvtps_epi32(_mm512_roundscale_ps(out_f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        _mm_storeu_si128((__m128i *)(dst_rgb + i), _mm512_cvtusepi32_epi8(vi));
    }
    for (; i < n3; ++i) {
        float v = (float)src_rgb[i] * (1.0f / 255.0f);
        float g = gain_rgb[i];
        if (g < 0.0f) g = 0.0f;
        float out = (g * v) / (1.0f + (g - 1.0f) * v) * 255.0f;
        if (out < 0.0f) out = 0.0f;
        if (out > 255.0f) out = 255.0f;
        dst_rgb[i] = (uint8_t)lrintf(out);
    }
}

void relight_luma_u8_f32(
    const uint8_t *src_y,
    const float *gain_hw,
    uint8_t *dst_y,
    int64_t num_pixels
) {
    const __m512 vinv255 = _mm512_set1_ps(1.0f / 255.0f);
    const __m512 v255 = _mm512_set1_ps(255.0f);
    const __m512 vone = _mm512_set1_ps(1.0f);
    const __m512 vzero = _mm512_setzero_ps();

    int64_t i = 0;
    for (; i + 16 <= num_pixels; i += 16) {
        __m128i u8 = _mm_loadu_si128((const __m128i *)(src_y + i));
        __m512i i32 = _mm512_cvtepu8_epi32(u8);
        __m512 v = _mm512_mul_ps(_mm512_cvtepi32_ps(i32), vinv255);
        __m512 g = _mm512_max_ps(_mm512_loadu_ps(gain_hw + i), vzero);

        __m512 num = _mm512_mul_ps(g, v);
        __m512 den = _mm512_fmadd_ps(_mm512_sub_ps(g, vone), v, vone);
        __m512 out_f = _mm512_mul_ps(_mm512_div_ps(num, den), v255);
        out_f = _mm512_min_ps(_mm512_max_ps(out_f, vzero), v255);

        __m512i vi = _mm512_cvtps_epi32(_mm512_roundscale_ps(out_f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        _mm_storeu_si128((__m128i *)(dst_y + i), _mm512_cvtusepi32_epi8(vi));
    }
    for (; i < num_pixels; ++i) {
        float v = (float)src_y[i] * (1.0f / 255.0f);
        float g = gain_hw[i];
        if (g < 0.0f) g = 0.0f;
        float out = (g * v) / (1.0f + (g - 1.0f) * v) * 255.0f;
        if (out < 0.0f) out = 0.0f;
        if (out > 255.0f) out = 255.0f;
        dst_y[i] = (uint8_t)lrintf(out);
    }
}
