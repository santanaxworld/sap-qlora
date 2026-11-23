#pragma once

// Shared by the AVX2 and AVX-512 translation units. No SIMD types in here.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

inline int qlora_clamp_i8(float v) {
    int q = static_cast<int>(std::round(v));
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    return q;
}

inline float qlora_quantize_row(const float* x, int n, std::uint8_t* xu) {
    float peak = 0.f;
    for (int i = 0; i < n; ++i) peak = std::max(peak, std::fabs(x[i]));
    const float scale = peak > 0.f ? peak / 127.f : 1.f;
    const float inv = 1.f / scale;
    for (int i = 0; i < n; ++i) xu[i] = static_cast<std::uint8_t>(qlora_clamp_i8(x[i] * inv) + 128);
    return scale;
}

template <typename Dot>
void qlora_gemm_run(Dot&& dot, const float* X, int rows, int din, const std::int8_t* q, const float* scale,
                    const std::int32_t* row_sum, int dout, float* Y) {
    if (rows <= 0 || din <= 0 || dout <= 0) return;
#pragma omp parallel
    {
        std::vector<std::uint8_t> xu(static_cast<std::size_t>(din));
#pragma omp for schedule(static)
        for (int n = 0; n < rows; ++n) {
            const float xs = qlora_quantize_row(X + static_cast<std::size_t>(n) * static_cast<std::size_t>(din), din, xu.data());
            const std::int8_t* w = q;
            float* y = Y + static_cast<std::size_t>(n) * static_cast<std::size_t>(dout);
            for (int j = 0; j < dout; ++j) {
                const std::int32_t acc = dot(xu.data(), w, din) - row_sum[j];
                y[j] = static_cast<float>(acc) * (xs * scale[j]);
                w += din;
            }
        }
    }
}
