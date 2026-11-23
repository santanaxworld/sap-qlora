#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

// Minimal tensor engine: one contiguous row-major float block.
// The training loop is allowed to call only gemm, transpose, and ReLU.

struct MemStats {
    std::size_t live = 0;
    std::size_t peak = 0;

    void on_alloc(std::size_t bytes);
    void on_free(std::size_t bytes);
};

extern MemStats g_mem;

struct Matrix {
    int rows = 0;
    int cols = 0;
    float* data = nullptr;

    Matrix() = default;
    Matrix(int rows, int cols);
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;
    ~Matrix();

    std::size_t n() const { return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols); }
    std::size_t bytes() const { return capacity * sizeof(float); }

    float& operator()(int i, int j) { return data[static_cast<std::size_t>(i) * cols + j]; }
    float operator()(int i, int j) const { return data[static_cast<std::size_t>(i) * cols + j]; }

    void ensure(int rows, int cols);
    void reshape(int rows, int cols);
    void zero();

    std::size_t capacity = 0;

private:
    void alloc(int rows, int cols);
    void release();
};

// C = alpha * op(A) * op(B) + beta * C
// op(X) is X^T when the matching flag is set. C must already have the result shape.
// A, B, and C must not alias.
void gemm(const Matrix& A, const Matrix& B, Matrix& C,
          bool transA, bool transB, float alpha, float beta);

// dx[n, i] = sum_j dy[n, j] * scale[j] * q[j, i]. q is output-major [dout, din].
// Unpacks a 128-wide input panel and uses the same AVX2 FMA kernel as gemm(..., transB=true).
// The panel is temporary; no FP32 copy of W is kept.
void gemm_dequant_nt(const float* dy, int rows, int dout, const std::int8_t* q, const float* scale, int din, float* dx);

void transpose(const Matrix& A, Matrix& out);

void relu_inplace(Matrix& X);
void relu_into(const Matrix& X, Matrix& Y);

// Y *= relu'(Z), with relu'(z) = 1 when z > 0 and 0 otherwise.
void mul_relu_gate(Matrix& Y, const Matrix& Z);

// Packs relu'(z) = [z > 0] into bits. 32 lanes share one uint32_t; bit 0 is the first lane.
// `words` must hold (n + 31) / 32 entries. Uses AVX2 cmp+movemask when available.
void pack_relu_mask(const float* z, std::uint32_t* words, std::size_t n);

inline bool relu_mask_bit(const std::uint32_t* words, std::size_t index) {
    return ((words[index >> 5] >> (index & 31u)) & 1u) != 0u;
}
