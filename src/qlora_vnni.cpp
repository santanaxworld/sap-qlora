#include "qlora.hpp"
#include "qlora_loop.hpp"

#include <immintrin.h>

namespace {

int32_t dot_vnni(const std::uint8_t* a, const std::int8_t* b, int n) {
    __m512i acc = _mm512_setzero_si512();
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        const __m512i au = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i));
        const __m512i bs = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i));
        acc = _mm512_dpbusd_epi32(acc, au, bs);
    }
    alignas(64) std::int32_t tmp[16];
    _mm512_store_si512(tmp, acc);
    std::int32_t result = 0;
    for (int k = 0; k < 16; ++k) result += tmp[k];
    for (; i < n; ++i) result += static_cast<std::int32_t>(a[i]) * static_cast<std::int32_t>(b[i]);
    return result;
}

}  // namespace

void qlora_gemm_vnni(const float* X, int rows, int din, const std::int8_t* q, const float* scale,
                     const std::int32_t* row_sum, int dout, float* Y) {
    qlora_gemm_run(dot_vnni, X, rows, din, q, scale, row_sum, dout, Y);
}
