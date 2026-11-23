#pragma once

#include <cstdint>

// Frozen W0, packed per output. q is row-major [dout, din]: row j is output j.
// row_sum[j] = 128 * sum_i q[j, i], the uint8 bias of the VNNI dot.
void qlora_pack(const float* W, int din, int dout, std::int8_t* q, float* scale, std::int32_t* row_sum);

// Y = quantize(X) · q, dequantized. X and Y are row-major, X is rows × din.
void qlora_gemm(const float* X, int rows, int din, const std::int8_t* q, const float* scale,
                const std::int32_t* row_sum, int dout, float* Y);

void qlora_gemm_scalar(const float* X, int rows, int din, const std::int8_t* q, const float* scale,
                       const std::int32_t* row_sum, int dout, float* Y);
void qlora_gemm_avx2(const float* X, int rows, int din, const std::int8_t* q, const float* scale,
                     const std::int32_t* row_sum, int dout, float* Y);
void qlora_gemm_vnni(const float* X, int rows, int din, const std::int8_t* q, const float* scale,
                     const std::int32_t* row_sum, int dout, float* Y);

bool qlora_vnni_ok();
const char* qlora_kernel();
