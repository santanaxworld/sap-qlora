#include "matrix.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <string>

#if defined(__AVX2__)
#include <immintrin.h>
#define MINICORE_AVX2 1
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <malloc.h>

MemStats g_mem;

void MemStats::on_alloc(std::size_t bytes) {
    live += bytes;
    if (live > peak) peak = live;
}

void MemStats::on_free(std::size_t bytes) {
    if (bytes > live) throw std::runtime_error("memory tracker underflow");
    live -= bytes;
}

void Matrix::alloc(int r, int c) {
    rows = r;
    cols = c;
    capacity = static_cast<std::size_t>(r) * static_cast<std::size_t>(c);
    if (capacity == 0) {
        data = nullptr;
        return;
    }
    data = static_cast<float*>(_aligned_malloc(capacity * sizeof(float), 32));
    if (!data) throw std::bad_alloc();
    g_mem.on_alloc(capacity * sizeof(float));
}

void Matrix::release() {
    if (data) {
        g_mem.on_free(capacity * sizeof(float));
        _aligned_free(data);
    }
    data = nullptr;
    capacity = 0;
    rows = 0;
    cols = 0;
}

Matrix::Matrix(int r, int c) { alloc(r, c); }

Matrix::Matrix(const Matrix& other) {
    if (other.n() == 0) {
        rows = other.rows;
        cols = other.cols;
        return;
    }
    alloc(other.rows, other.cols);
    std::memcpy(data, other.data, other.n() * sizeof(float));
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other) return *this;
    if (other.n() == 0) {
        release();
        rows = other.rows;
        cols = other.cols;
        return *this;
    }
    ensure(other.rows, other.cols);
    std::memcpy(data, other.data, other.n() * sizeof(float));
    return *this;
}

Matrix::Matrix(Matrix&& other) noexcept
    : rows(other.rows), cols(other.cols), data(other.data), capacity(other.capacity) {
    other.rows = 0;
    other.cols = 0;
    other.data = nullptr;
    other.capacity = 0;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this == &other) return *this;
    release();
    rows = other.rows;
    cols = other.cols;
    data = other.data;
    capacity = other.capacity;
    other.rows = 0;
    other.cols = 0;
    other.data = nullptr;
    other.capacity = 0;
    return *this;
}

Matrix::~Matrix() { release(); }

void Matrix::ensure(int r, int c) {
    if (r < 0 || c < 0) throw std::runtime_error("negative matrix shape");
    const std::size_t need = static_cast<std::size_t>(r) * static_cast<std::size_t>(c);
    if (need <= capacity && (data || need == 0)) {
        rows = r;
        cols = c;
        return;
    }
    release();
    alloc(r, c);
}

void Matrix::reshape(int r, int c) {
    if (r < 0 || c < 0) throw std::runtime_error("negative matrix shape");
    const std::size_t need = static_cast<std::size_t>(r) * static_cast<std::size_t>(c);
    if (need > capacity) throw std::runtime_error("reshape exceeds capacity");
    rows = r;
    cols = c;
}

void Matrix::zero() {
    if (n()) std::memset(data, 0, n() * sizeof(float));
}

namespace {

[[noreturn]] void gemm_error(const std::string& msg) {
    throw std::runtime_error(msg);
}

void apply_beta(Matrix& C, float beta) {
    const std::size_t n = C.n();
    if (n == 0) return;
    if (beta == 0.f) {
        std::memset(C.data, 0, n * sizeof(float));
    } else if (beta != 1.f) {
        for (std::size_t i = 0; i < n; ++i) C.data[i] *= beta;
    }
}

template <bool TransA>
float a_load(const float* A, int lda, int i, int p) {
    if constexpr (TransA) return A[static_cast<std::size_t>(p) * lda + i];
    else return A[static_cast<std::size_t>(i) * lda + p];
}

// C += alpha * op(A) * B, with B row-major and contiguous along N.
// Each output row is owned by one iteration, so OpenMP does not reorder sums.
template <bool TransA>
void gemm_bt(const float* __restrict A, int lda,
             const float* __restrict B, int ldb,
             float* __restrict C, int ldc,
             int M, int N, int K, float alpha) {
    const int tiles = M / 4;
    // Thin products (the 128x10x512 feedback GEMMs) lose to the fork cost.
#if defined(_OPENMP)
    const long long work = static_cast<long long>(M) * N * K;
#pragma omp parallel for schedule(static) if (tiles >= 4 && work >= 1000000LL)
#endif
    for (int t = 0; t < tiles; ++t) {
        const int i0 = t * 4;
        int j = 0;
#if defined(MINICORE_AVX2)
        for (; j + 8 <= N; j += 8) {
            __m256 c0 = _mm256_loadu_ps(C + static_cast<std::size_t>(i0 + 0) * ldc + j);
            __m256 c1 = _mm256_loadu_ps(C + static_cast<std::size_t>(i0 + 1) * ldc + j);
            __m256 c2 = _mm256_loadu_ps(C + static_cast<std::size_t>(i0 + 2) * ldc + j);
            __m256 c3 = _mm256_loadu_ps(C + static_cast<std::size_t>(i0 + 3) * ldc + j);
            for (int p = 0; p < K; ++p) {
                const __m256 b = _mm256_loadu_ps(B + static_cast<std::size_t>(p) * ldb + j);
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(alpha * a_load<TransA>(A, lda, i0 + 0, p)), b, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(alpha * a_load<TransA>(A, lda, i0 + 1, p)), b, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(alpha * a_load<TransA>(A, lda, i0 + 2, p)), b, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(alpha * a_load<TransA>(A, lda, i0 + 3, p)), b, c3);
            }
            _mm256_storeu_ps(C + static_cast<std::size_t>(i0 + 0) * ldc + j, c0);
            _mm256_storeu_ps(C + static_cast<std::size_t>(i0 + 1) * ldc + j, c1);
            _mm256_storeu_ps(C + static_cast<std::size_t>(i0 + 2) * ldc + j, c2);
            _mm256_storeu_ps(C + static_cast<std::size_t>(i0 + 3) * ldc + j, c3);
        }
#endif
        for (; j < N; ++j) {
            for (int r = 0; r < 4; ++r) {
                float s = C[static_cast<std::size_t>(i0 + r) * ldc + j];
                for (int p = 0; p < K; ++p) {
                    s += alpha * a_load<TransA>(A, lda, i0 + r, p) * B[static_cast<std::size_t>(p) * ldb + j];
                }
                C[static_cast<std::size_t>(i0 + r) * ldc + j] = s;
            }
        }
    }

    for (int i = tiles * 4; i < M; ++i) {
        int j = 0;
#if defined(MINICORE_AVX2)
        for (; j + 8 <= N; j += 8) {
            __m256 c = _mm256_loadu_ps(C + static_cast<std::size_t>(i) * ldc + j);
            for (int p = 0; p < K; ++p) {
                const __m256 b = _mm256_loadu_ps(B + static_cast<std::size_t>(p) * ldb + j);
                c = _mm256_fmadd_ps(_mm256_set1_ps(alpha * a_load<TransA>(A, lda, i, p)), b, c);
            }
            _mm256_storeu_ps(C + static_cast<std::size_t>(i) * ldc + j, c);
        }
#endif
        for (; j < N; ++j) {
            float s = C[static_cast<std::size_t>(i) * ldc + j];
            for (int p = 0; p < K; ++p) {
                s += alpha * a_load<TransA>(A, lda, i, p) * B[static_cast<std::size_t>(p) * ldb + j];
            }
            C[static_cast<std::size_t>(i) * ldc + j] = s;
        }
    }
}

// One scratch panel per worker. gemm_bt reads the pointer it is given, so a pack on
// one thread cannot collide with a pack on another.
float* panel_storage(std::size_t nfloat) {
    constexpr int k_slots = 64;
    struct Slot {
        float* buf = nullptr;
        std::size_t cap = 0;
    };
    static Slot slots[k_slots];
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
    if (tid < 0 || tid >= k_slots) tid = 0;
#endif
    Slot& s = slots[tid];
    if (s.buf && nfloat <= s.cap) return s.buf;
#ifdef _OPENMP
#pragma omp critical(minicore_panel)
#endif
    {
        if (!s.buf || nfloat > s.cap) {
            if (s.buf) _aligned_free(s.buf);
            s.buf = static_cast<float*>(_aligned_malloc(nfloat * sizeof(float), 32));
            if (!s.buf) throw std::bad_alloc();
            s.cap = nfloat;
        }
    }
    return s.buf;
}

// C += alpha * A * B^T. B is packed in column panels so the hot kernel stays contiguous.
void gemm_nt(const Matrix& A, const Matrix& B, Matrix& C, float alpha) {
    const int M = A.rows;
    const int K = A.cols;
    const int N = B.rows;
    constexpr int BN = 128;
    for (int j0 = 0; j0 < N; j0 += BN) {
        const int jb = std::min(BN, N - j0);
        float* panel = panel_storage(static_cast<std::size_t>(K) * jb);
        for (int p = 0; p < K; ++p) {
            float* dst = panel + static_cast<std::size_t>(p) * jb;
            for (int j = 0; j < jb; ++j) {
                dst[j] = B.data[static_cast<std::size_t>(j0 + j) * B.cols + p];
            }
        }
        gemm_bt<false>(A.data, A.cols, panel, jb, C.data + j0, C.cols, M, jb, K, alpha);
    }
}

void gemm_tt(const Matrix& A, const Matrix& B, Matrix& C, float alpha) {
    const int M = A.cols;
    const int K = A.rows;
    const int N = B.rows;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (M >= 4)
#endif
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int p = 0; p < K; ++p) {
                const float a = A.data[static_cast<std::size_t>(p) * A.cols + i];
                const float b = B.data[static_cast<std::size_t>(j) * B.cols + p];
                s += static_cast<double>(a) * static_cast<double>(b);
            }
            C.data[static_cast<std::size_t>(i) * C.cols + j] += alpha * static_cast<float>(s);
        }
    }
}

}  // namespace

void gemm(const Matrix& A, const Matrix& B, Matrix& C,
          bool transA, bool transB, float alpha, float beta) {
    const int M = transA ? A.cols : A.rows;
    const int KA = transA ? A.rows : A.cols;
    const int KB = transB ? B.cols : B.rows;
    const int N = transB ? B.rows : B.cols;
    if (KA != KB) {
        std::ostringstream os;
        os << "gemm K mismatch " << KA << " vs " << KB;
        gemm_error(os.str());
    }
    if (C.rows != M || C.cols != N) {
        std::ostringstream os;
        os << "gemm C shape " << C.rows << "x" << C.cols << " expected " << M << "x" << N;
        gemm_error(os.str());
    }
    if (C.data && (C.data == A.data || C.data == B.data)) gemm_error("gemm output aliases an input");

    apply_beta(C, beta);
    if (alpha == 0.f || M == 0 || N == 0 || KA == 0) return;

    if (!transA && !transB) {
        gemm_bt<false>(A.data, A.cols, B.data, B.cols, C.data, C.cols, M, N, KA, alpha);
    } else if (transA && !transB) {
        gemm_bt<true>(A.data, A.cols, B.data, B.cols, C.data, C.cols, M, N, KA, alpha);
    } else if (!transA && transB) {
        gemm_nt(A, B, C, alpha);
    } else {
        gemm_tt(A, B, C, alpha);
    }
}

void gemm_dequant_nt(const float* dy, int rows, int dout, const std::int8_t* q, const float* scale, int din, float* dx) {
    if (rows < 0 || dout < 0 || din < 0) gemm_error("dequant gemm shape");
    if (rows == 0 || din == 0) return;
    std::memset(dx, 0, static_cast<std::size_t>(rows) * static_cast<std::size_t>(din) * sizeof(float));
    if (dout == 0) return;
    constexpr int BN = 128;
    for (int j0 = 0; j0 < din; j0 += BN) {
        const int jb = std::min(BN, din - j0);
        float* panel = panel_storage(static_cast<std::size_t>(dout) * static_cast<std::size_t>(jb));
        for (int p = 0; p < dout; ++p) {
            const float s = scale[p];
            const std::int8_t* src = q + static_cast<std::size_t>(p) * static_cast<std::size_t>(din) + static_cast<std::size_t>(j0);
            float* dst = panel + static_cast<std::size_t>(p) * static_cast<std::size_t>(jb);
            int j = 0;
#if defined(MINICORE_AVX2)
            const __m256 sv = _mm256_set1_ps(s);
            for (; j + 16 <= jb; j += 16) {
                const __m128i b0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + j));
                const __m128i b1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + j + 8));
                _mm256_storeu_ps(dst + j, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)), sv));
                _mm256_storeu_ps(dst + j + 8, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1)), sv));
            }
#endif
            for (; j < jb; ++j) dst[j] = static_cast<float>(src[j]) * s;
        }
        gemm_bt<false>(dy, dout, panel, jb, dx + j0, din, rows, jb, dout, 1.f);
    }
}

void transpose(const Matrix& A, Matrix& out) {
    out.ensure(A.cols, A.rows);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            out(j, i) = A(i, j);
        }
    }
}

void relu_inplace(Matrix& X) {
    const std::size_t n = X.n();
    for (std::size_t i = 0; i < n; ++i) {
        if (X.data[i] < 0.f) X.data[i] = 0.f;
    }
}

void relu_into(const Matrix& X, Matrix& Y) {
    if (Y.rows != X.rows || Y.cols != X.cols) throw std::runtime_error("relu shape mismatch");
    const std::size_t n = X.n();
    for (std::size_t i = 0; i < n; ++i) Y.data[i] = X.data[i] > 0.f ? X.data[i] : 0.f;
}

void mul_relu_gate(Matrix& Y, const Matrix& Z) {
    if (Y.rows != Z.rows || Y.cols != Z.cols) throw std::runtime_error("relu gate shape mismatch");
    const std::size_t n = Y.n();
    for (std::size_t i = 0; i < n; ++i) {
        if (!(Z.data[i] > 0.f)) Y.data[i] = 0.f;
    }
}

void pack_relu_mask(const float* z, std::uint32_t* words, std::size_t n) {
    std::size_t i = 0;
#if defined(MINICORE_AVX2)
    const __m256 zero = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        std::uint32_t mask = 0;
        for (int lane = 0; lane < 4; ++lane) {
            const __m256 v = _mm256_loadu_ps(z + i + static_cast<std::size_t>(lane) * 8u);
            const __m256 pos = _mm256_cmp_ps(v, zero, _CMP_GT_OQ);
            mask |= static_cast<std::uint32_t>(_mm256_movemask_ps(pos)) << (lane * 8);
        }
        words[i >> 5] = mask;
    }
#else
    (void)z;
    (void)words;
#endif
    for (; i + 32 <= n; i += 32) {
        std::uint32_t mask = 0;
        for (int t = 0; t < 32; ++t) {
            if (z[i + static_cast<std::size_t>(t)] > 0.f) mask |= 1u << t;
        }
        words[i >> 5] = mask;
    }
    if (i >= n) return;
    std::uint32_t mask = 0;
    for (std::size_t t = 0; t < n - i; ++t) {
        if (z[i + t] > 0.f) mask |= 1u << static_cast<unsigned>(t);
    }
    words[i >> 5] = mask;
}
