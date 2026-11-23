#include "qlora.hpp"
#include "qlora_loop.hpp"

#include <algorithm>
#include <cmath>

#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include <immintrin.h>

namespace {

int32_t dot_scalar(const std::uint8_t* a, const std::int8_t* b, int n) {
    std::int32_t acc = 0;
#pragma loop(no_vector)
    for (int i = 0; i < n; ++i) acc += static_cast<std::int32_t>(a[i]) * static_cast<std::int32_t>(b[i]);
    return acc;
}

int32_t dot_avx2(const std::uint8_t* a, const std::int8_t* b, int n) {
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        const __m128i a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + 16));
        const __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
        const __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i + 16));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_cvtepu8_epi16(a0), _mm256_cvtepi8_epi16(b0)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_cvtepu8_epi16(a1), _mm256_cvtepi8_epi16(b1)));
    }
    const __m128i sum4 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    const __m128i sum2 = _mm_hadd_epi32(sum4, sum4);
    const __m128i sum1 = _mm_hadd_epi32(sum2, sum2);
    std::int32_t result = _mm_cvtsi128_si32(sum1);
    for (; i < n; ++i) result += static_cast<std::int32_t>(a[i]) * static_cast<std::int32_t>(b[i]);
    return result;
}

bool vnni_cached() {
    static const int ok = []() {
        int regs[4] = {};
        __cpuidex(regs, 7, 0);
        // EAX=7, ECX=0: ECX bit 11 is AVX512VNNI.
        return (regs[2] & (1 << 11)) != 0 ? 1 : 0;
    }();
    return ok != 0;
}

}  // namespace

void qlora_pack(const float* W, int din, int dout, std::int8_t* q, float* scale, std::int32_t* row_sum) {
    for (int j = 0; j < dout; ++j) {
        float peak = 0.f;
        for (int i = 0; i < din; ++i)
            peak = std::max(peak, std::fabs(W[static_cast<std::size_t>(i) * static_cast<std::size_t>(dout) + j]));
        const float sj = peak > 0.f ? peak / 127.f : 1.f;
        const float inv = 1.f / sj;
        std::int32_t sum = 0;
        std::int8_t* dst = q + static_cast<std::size_t>(j) * static_cast<std::size_t>(din);
        for (int i = 0; i < din; ++i) {
            const int v = qlora_clamp_i8(W[static_cast<std::size_t>(i) * static_cast<std::size_t>(dout) + j] * inv);
            dst[i] = static_cast<std::int8_t>(v);
            sum += v;
        }
        scale[j] = sj;
        row_sum[j] = sum * 128;
    }
}

void qlora_gemm_scalar(const float* X, int rows, int din, const std::int8_t* q, const float* scale,
                       const std::int32_t* row_sum, int dout, float* Y) {
    qlora_gemm_run(dot_scalar, X, rows, din, q, scale, row_sum, dout, Y);
}

void qlora_gemm_avx2(const float* X, int rows, int din, const std::int8_t* q, const float* scale,
                     const std::int32_t* row_sum, int dout, float* Y) {
    qlora_gemm_run(dot_avx2, X, rows, din, q, scale, row_sum, dout, Y);
}

bool qlora_vnni_ok() { return vnni_cached(); }

const char* qlora_kernel() { return qlora_vnni_ok() ? "avx512-vnni" : "avx2"; }

void qlora_gemm(const float* X, int rows, int din, const std::int8_t* q, const float* scale, const std::int32_t* row_sum,
                int dout, float* Y) {
    if (qlora_vnni_ok()) qlora_gemm_vnni(X, rows, din, q, scale, row_sum, dout, Y);
    else qlora_gemm_avx2(X, rows, din, q, scale, row_sum, dout, Y);
}
