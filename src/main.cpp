// Synthetic training comparison.
//
// Target: y = sin(a·x) + cos(b·x), drawn in-process. No dataset is loaded.
//
// A keeps every hidden activation and applies the chain rule.
// B is feedback alignment with no replay and no full delta tape:
//   forward stays in float; the tape stores a 1-bit ReLU gate and an INT8
//   copy of each layer input. Each layer writes P_l e into one 512x128
//   scratch, applies its mask, and updates W. P is realigned every K steps.
//
// Both nets start from the same He/Xavier weights and see the same minibatches.

#include "matrix.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <malloc.h>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>

#include "qlora.hpp"

#pragma comment(lib, "psapi.lib")

namespace {

void expect(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

const char* kernel_name() {
#if defined(__AVX2__)
    return "AVX2+FMA";
#else
    return "scalar";
#endif
}

int thread_count() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

float mean_square(const Matrix& P, const Matrix& Y) {
    expect(P.rows == Y.rows && P.cols == Y.cols, "mean_square shape");
    const std::size_t n = P.n();
    expect(n > 0, "mean_square empty");
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(P.data[i]) - static_cast<double>(Y.data[i]);
        s += d * d;
    }
    return static_cast<float>(s / static_cast<double>(n));
}

double sum_square(const Matrix& P, const Matrix& Y) {
    expect(P.rows == Y.rows && P.cols == Y.cols, "sum_square shape");
    double s = 0.0;
    for (std::size_t i = 0; i < P.n(); ++i) {
        const double d = static_cast<double>(P.data[i]) - static_cast<double>(Y.data[i]);
        s += d * d;
    }
    return s;
}

// d(mean square)/dP = 2 (P - Y) / n
void output_grad(const Matrix& P, const Matrix& Y, Matrix& G) {
    expect(P.rows == Y.rows && P.cols == Y.cols, "output_grad shape");
    G.ensure(P.rows, P.cols);
    const float scale = 2.f / static_cast<float>(P.n());
    for (std::size_t i = 0; i < P.n(); ++i) {
        G.data[i] = (P.data[i] - Y.data[i]) * scale;
    }
}

// Mean cross-entropy. When G is set, it receives d(mean CE)/dZ = (p - Y) / batch.
float softmax_ce(const Matrix& Z, const Matrix& Y, Matrix* G) {
    expect(Z.rows == Y.rows && Z.cols == Y.cols, "softmax shape");
    const int B = Z.rows;
    const int K = Z.cols;
    expect(B > 0 && K > 0, "softmax empty");
    if (G) G->ensure(B, K);
    std::vector<float> e(static_cast<std::size_t>(K));
    double loss = 0.0;
    const float inv_b = 1.f / static_cast<float>(B);
    for (int b = 0; b < B; ++b) {
        const float* z = Z.data + static_cast<std::size_t>(b) * static_cast<std::size_t>(K);
        const float* y = Y.data + static_cast<std::size_t>(b) * static_cast<std::size_t>(K);
        float m = z[0];
        for (int k = 1; k < K; ++k) m = std::max(m, z[k]);
        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            const float ek = std::exp(z[k] - m);
            e[static_cast<std::size_t>(k)] = ek;
            sum += ek;
        }
        const float inv = static_cast<float>(1.0 / sum);
        for (int k = 0; k < K; ++k) {
            const float p = e[static_cast<std::size_t>(k)] * inv;
            if (G) (*G)(b, k) = (p - y[k]) * inv_b;
            if (y[k] > 0.f) loss -= static_cast<double>(y[k]) * std::log(static_cast<double>(std::max(p, 1e-12f)));
        }
    }
    return static_cast<float>(loss / static_cast<double>(B));
}

void fill_normal(Matrix& M, std::mt19937& rng, float stddev) {
    std::normal_distribution<float> dist(0.f, stddev);
    for (std::size_t i = 0; i < M.n(); ++i) M.data[i] = dist(rng);
}

void fill_uniform(Matrix& M, std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (std::size_t i = 0; i < M.n(); ++i) M.data[i] = dist(rng);
}

// Modified Gram-Schmidt. Columns of M become orthonormal (M.rows >= M.cols).
void orthonormal_columns(Matrix& M) {
    expect(M.rows >= M.cols && M.cols > 0, "orthonormal shape");
    const int R = M.rows;
    const int C = M.cols;
    for (int j = 0; j < C; ++j) {
        for (int pass = 0; pass < 2; ++pass) {
            for (int i = 0; i < j; ++i) {
                double dot = 0.0;
                for (int r = 0; r < R; ++r) {
                    dot += static_cast<double>(M(r, i)) * static_cast<double>(M(r, j));
                }
                for (int r = 0; r < R; ++r) {
                    M(r, j) = static_cast<float>(static_cast<double>(M(r, j)) - dot * M(r, i));
                }
            }
        }
        double nrm = 0.0;
        for (int r = 0; r < R; ++r) {
            nrm += static_cast<double>(M(r, j)) * static_cast<double>(M(r, j));
        }
        nrm = std::sqrt(nrm);
        expect(nrm > 1e-8, "orthonormal column vanished");
        const float inv = static_cast<float>(1.0 / nrm);
        for (int r = 0; r < R; ++r) M(r, j) *= inv;
    }
}

// Off-diagonal column cosine. Scaled orthogonal columns stay near 0 here;
// the diagonal is their energy and is intentionally not 1.
float column_ortho_error(const Matrix& M) {
    float worst = 0.f;
    for (int i = 0; i < M.cols; ++i) {
        for (int j = i + 1; j < M.cols; ++j) {
            double dot = 0.0;
            double ni = 0.0;
            double nj = 0.0;
            for (int r = 0; r < M.rows; ++r) {
                const double a = M(r, i);
                const double b = M(r, j);
                dot += a * b;
                ni += a * a;
                nj += b * b;
            }
            if (ni < 1e-18 || nj < 1e-18) continue;
            worst = std::max(worst, static_cast<float>(std::fabs(dot) / std::sqrt(ni * nj)));
        }
    }
    return worst;
}

float gram_error(const Matrix& M) {
    float worst = 0.f;
    for (int i = 0; i < M.cols; ++i) {
        for (int j = 0; j < M.cols; ++j) {
            double s = 0.0;
            for (int r = 0; r < M.rows; ++r) {
                s += static_cast<double>(M(r, i)) * static_cast<double>(M(r, j));
            }
            const double target = (i == j) ? 1.0 : 0.0;
            worst = std::max(worst, static_cast<float>(std::fabs(s - target)));
        }
    }
    return worst;
}

float max_abs_diff(const Matrix& A, const Matrix& B) {
    expect(A.rows == B.rows && A.cols == B.cols, "diff shape");
    float m = 0.f;
    for (std::size_t i = 0; i < A.n(); ++i) {
        m = std::max(m, std::fabs(A.data[i] - B.data[i]));
    }
    return m;
}

struct Net {
    std::vector<int> dims;
    std::vector<Matrix> W;
    bool residual = false;
};

int n_layers(const Net& net) { return static_cast<int>(net.W.size()); }

// Square hidden layer: h <- gamma_res * (h + alpha * ReLU(h W)).
// alpha = 1/sqrt(10). gamma_res cancels the positive ReLU cross term.
constexpr float alpha = 0.316227766f;
constexpr float gamma_res = 0.876f;

bool skip_layer(const Net& net, int layer) {
    return net.residual && layer + 1 < n_layers(net) && net.dims[static_cast<std::size_t>(layer)] ==
                                                            net.dims[static_cast<std::size_t>(layer + 1)];
}

void axpy(float a, const Matrix& X, Matrix& Y);
void scale_inplace(Matrix& Y, float a);
int capped_update(Matrix& W, const Matrix& Xh, const Matrix& Gmat, float eta, bool clip, float clip_gamma);
float commit_grads(std::vector<Matrix>& W, const std::vector<Matrix>& dW, float lr, bool clip, float max_norm,
                   int& hits, double& scale_sum, int& scale_n, float& last_scale);

void apply_bit_mask(Matrix& G, const std::uint32_t* bits) {
    for (std::size_t i = 0; i < G.n(); ++i) {
        if (!relu_mask_bit(bits, i)) G.data[i] = 0.f;
    }
}

std::size_t bytes_of(const std::vector<Matrix>& ms) {
    std::size_t s = 0;
    for (const Matrix& m : ms) s += m.bytes();
    return s;
}

std::vector<int> make_dims(int in_dim, int hidden, int out_dim, int squares) {
    expect(in_dim > 0 && hidden > 0 && out_dim > 0 && squares >= 0, "bad architecture");
    expect(hidden >= out_dim, "hidden width must be >= output width for orthonormal feedback");
    std::vector<int> dims;
    dims.push_back(in_dim);
    for (int i = 0; i < squares + 1; ++i) dims.push_back(hidden);
    dims.push_back(out_dim);
    return dims;
}

Net init_net(const std::vector<int>& dims, std::uint32_t seed) {
    expect(dims.size() >= 2, "net needs input and output");
    Net net;
    net.dims = dims;
    const int L = static_cast<int>(dims.size()) - 1;
    net.W.resize(L);
    std::mt19937 rng(seed);
    for (int l = 0; l < L; ++l) {
        const int fan_in = dims[l];
        const int fan_out = dims[l + 1];
        const float stddev = (l == L - 1) ? std::sqrt(1.f / static_cast<float>(fan_in))
                                           : std::sqrt(2.f / static_cast<float>(fan_in));
        net.W[l].ensure(fan_in, fan_out);
        fill_normal(net.W[l], rng, stddev);
    }
    return net;
}

std::string arch_string(const Net& net) {
    std::ostringstream os;
    for (int i = 0; i < static_cast<int>(net.dims.size()); ++i) {
        if (i) os << " -> ";
        os << net.dims[i];
    }
    return os.str();
}

int count_square(const Net& net, int hidden) {
    int n = 0;
    for (int l = 0; l < n_layers(net); ++l) {
        if (net.dims[l] == hidden && net.dims[l + 1] == hidden) ++n;
    }
    return n;
}

// Single forward. Hidden activations are overwritten in `a` and `b`.
double forward_sumsq(const Net& net, const Matrix& X, const Matrix& Y, Matrix& a, Matrix& b) {
    const int batch = X.rows;
    const int L = n_layers(net);
    expect(L >= 1, "empty net");
    expect(X.cols == net.dims[0] && Y.rows == batch && Y.cols == net.dims.back(), "forward shape");

    if (L == 1) {
        a.ensure(batch, net.dims[1]);
        gemm(X, net.W[0], a, false, false, 1.f, 0.f);
        return sum_square(a, Y);
    }

    const Matrix* h = &X;
    Matrix* cur = &a;
    Matrix* nxt = &b;
    for (int l = 0; l < L - 1; ++l) {
        cur->ensure(batch, net.dims[l + 1]);
        gemm(*h, net.W[l], *cur, false, false, 1.f, 0.f);
        if (skip_layer(net, l)) {
            relu_inplace(*cur);
            scale_inplace(*cur, alpha);
            axpy(1.f, *h, *cur);
            scale_inplace(*cur, gamma_res);
        } else {
            relu_inplace(*cur);
        }
        h = cur;
        std::swap(cur, nxt);
    }
    cur->ensure(batch, net.dims[L]);
    gemm(*h, net.W[L - 1], *cur, false, false, 1.f, 0.f);
    return sum_square(*cur, Y);
}

void forward_logits(const Net& net, const Matrix& X, Matrix& a, Matrix& b, Matrix& logits) {
    const int batch = X.rows;
    const int L = n_layers(net);
    expect(L >= 1 && X.cols == net.dims[0], "logit forward");
    if (L == 1) {
        logits.ensure(batch, net.dims[1]);
        gemm(X, net.W[0], logits, false, false, 1.f, 0.f);
        return;
    }
    const Matrix* h = &X;
    Matrix* cur = &a;
    Matrix* nxt = &b;
    for (int l = 0; l < L - 1; ++l) {
        cur->ensure(batch, net.dims[static_cast<std::size_t>(l + 1)]);
        gemm(*h, net.W[static_cast<std::size_t>(l)], *cur, false, false, 1.f, 0.f);
        if (skip_layer(net, l)) {
            relu_inplace(*cur);
            scale_inplace(*cur, alpha);
            axpy(1.f, *h, *cur);
            scale_inplace(*cur, gamma_res);
        } else {
            relu_inplace(*cur);
        }
        h = cur;
        std::swap(cur, nxt);
    }
    logits.ensure(batch, net.dims[static_cast<std::size_t>(L)]);
    gemm(*h, net.W[static_cast<std::size_t>(L - 1)], logits, false, false, 1.f, 0.f);
}

float eval_mse(const Net& net, const Matrix& X, const Matrix& Y, int chunk, Matrix& a, Matrix& b,
               Matrix& xs, Matrix& ys) {
    expect(chunk > 0, "chunk");
    double acc = 0.0;
    int rows = 0;
    for (int i = 0; i < X.rows; i += chunk) {
        const int bsz = std::min(chunk, X.rows - i);
        xs.ensure(bsz, X.cols);
        ys.ensure(bsz, Y.cols);
        for (int r = 0; r < bsz; ++r) {
            std::memcpy(xs.data + static_cast<std::size_t>(r) * xs.cols,
                        X.data + static_cast<std::size_t>(i + r) * X.cols,
                        static_cast<std::size_t>(X.cols) * sizeof(float));
            std::memcpy(ys.data + static_cast<std::size_t>(r) * ys.cols,
                        Y.data + static_cast<std::size_t>(i + r) * Y.cols,
                        static_cast<std::size_t>(Y.cols) * sizeof(float));
        }
        acc += forward_sumsq(net, xs, ys, a, b);
        rows += bsz;
    }
    return static_cast<float>(acc / (static_cast<double>(rows) * static_cast<double>(Y.cols)));
}

struct Synth {
    Matrix fa;
    Matrix fb;

    void init(int in_dim, int out_dim, std::uint32_t seed) {
        fa.ensure(out_dim, in_dim);
        fb.ensure(out_dim, in_dim);
        std::mt19937 rng(seed);
        fill_uniform(fa, rng, -1.f, 1.f);
        fill_uniform(fb, rng, -1.f, 1.f);
    }

    void sample(Matrix& X, Matrix& Y, std::mt19937& rng) const {
        expect(X.cols == fa.cols && Y.cols == fa.rows && X.rows == Y.rows, "sample shape");
        std::normal_distribution<float> normal(0.f, 1.f);
        for (std::size_t i = 0; i < X.n(); ++i) X.data[i] = normal(rng);
        for (int b = 0; b < X.rows; ++b) {
            for (int k = 0; k < Y.cols; ++k) {
                double da = 0.0;
                double db = 0.0;
                for (int d = 0; d < X.cols; ++d) {
                    const double x = X(b, d);
                    da += x * fa(k, d);
                    db += x * fb(k, d);
                }
                Y(b, k) = std::sin(static_cast<float>(da)) + std::cos(static_cast<float>(db));
            }
        }
    }
};

float frobenius(const Matrix& M);

struct TrainerA {
    Net net;
    std::vector<Matrix> H;
    Matrix G;
    Matrix Gprev;
    Matrix Dz;
    std::vector<std::uint32_t> mask_words;
    std::vector<std::size_t> mask_off;
    bool clip = false;
    float clip_gamma = 0.05f;
    float max_norm = 100.f;
    int clip_hits = 0;
    float last_scale = 1.f;
    double scale_sum = 0.0;
    int scale_n = 0;
    std::vector<Matrix> pending;
    bool classify = false;
    // Filled with dL/dx after a residual backward. Off for every existing test.
    bool keep_input_grad = false;
    Matrix Gin;
    // Diagnostic only. Raw ||X^T g|| before the clip, plus ||dX|| for the embedding.
    bool capture_grad = false;
    std::vector<float> grad_norm;
    float embed_gnorm = 0.f;
    Matrix grad_scratch;

    void note_layer_grad(int layer, const Matrix& Xh, const Matrix& Gmat) {
        if (!capture_grad) return;
        const int L = n_layers(net);
        if (static_cast<int>(grad_norm.size()) != L) grad_norm.assign(static_cast<std::size_t>(L), 0.f);
        const Matrix& W = net.W[static_cast<std::size_t>(layer)];
        grad_scratch.ensure(W.rows, W.cols);
        gemm(Xh, Gmat, grad_scratch, true, false, 1.f, 0.f);
        grad_norm[static_cast<std::size_t>(layer)] = frobenius(grad_scratch);
    }

    void prepare(int batch) {
        const int L = n_layers(net);
        H.resize(static_cast<std::size_t>(L + 1));
        int maxd = 0;
        for (int d : net.dims) maxd = std::max(maxd, d);
        for (int l = 0; l <= L; ++l) H[static_cast<std::size_t>(l)].ensure(batch, net.dims[l]);
        G.ensure(batch, maxd);
        Gprev.ensure(batch, maxd);
        mask_words.clear();
        mask_off.clear();
        if (!net.residual) return;
        Dz.ensure(batch, maxd);
        mask_off.assign(static_cast<std::size_t>(std::max(L, 1)), 0);
        std::size_t words = 0;
        for (int l = 0; l < L - 1; ++l) {
            mask_off[static_cast<std::size_t>(l)] = words;
            const std::size_t nelt = static_cast<std::size_t>(batch) *
                                     static_cast<std::size_t>(net.dims[static_cast<std::size_t>(l + 1)]);
            words += (nelt + 31u) / 32u;
        }
        mask_words.assign(words, 0u);
    }

    std::size_t bytes() const {
        return bytes_of(net.W) + bytes_of(H) + G.bytes() + Gprev.bytes() + Dz.bytes() +
               mask_words.size() * sizeof(std::uint32_t);
    }

    std::uint32_t* mask_at(int layer) { return mask_words.data() + mask_off[static_cast<std::size_t>(layer)]; }

    void forward(const Matrix& X) {
        const int L = n_layers(net);
        const int batch = X.rows;
        expect(H[0].rows == batch && H[0].cols == X.cols, "A batch buffer");
        std::memcpy(H[0].data, X.data, X.n() * sizeof(float));
        for (int l = 0; l < L; ++l) {
            gemm(H[static_cast<std::size_t>(l)], net.W[static_cast<std::size_t>(l)],
                 H[static_cast<std::size_t>(l + 1)], false, false, 1.f, 0.f);
            if (l + 1 < L) {
                if (skip_layer(net, l)) {
                    pack_relu_mask(H[static_cast<std::size_t>(l + 1)].data, mask_at(l),
                                   H[static_cast<std::size_t>(l + 1)].n());
                    relu_inplace(H[static_cast<std::size_t>(l + 1)]);
                    scale_inplace(H[static_cast<std::size_t>(l + 1)], alpha);
                    axpy(1.f, H[static_cast<std::size_t>(l)], H[static_cast<std::size_t>(l + 1)]);
                    scale_inplace(H[static_cast<std::size_t>(l + 1)], gamma_res);
                } else {
                    relu_inplace(H[static_cast<std::size_t>(l + 1)]);
                }
            }
        }
    }

    // When store_grad is set, raw dL/dW is written and weights stay put.
    void backward(const Matrix& Y, float lr, bool apply, std::vector<Matrix>* store_grad) {
        const int L = n_layers(net);
        const int batch = Y.rows;
        if (classify) softmax_ce(H[static_cast<std::size_t>(L)], Y, &G);
        else output_grad(H[static_cast<std::size_t>(L)], Y, G);
        std::vector<Matrix>* sink = store_grad;
        if (apply) {
            if (static_cast<int>(pending.size()) != L) pending.assign(static_cast<std::size_t>(L), Matrix{});
            sink = &pending;
        }
        if (net.residual) {
            for (int l = L - 1; l >= 0; --l) {
                const bool hidden = l < L - 1;
                const bool skip = skip_layer(net, l);
                const Matrix* grad = &G;
                if (hidden) {
                    Dz.ensure(batch, net.dims[static_cast<std::size_t>(l + 1)]);
                    std::memcpy(Dz.data, G.data, G.n() * sizeof(float));
                    if (skip) {
                        apply_bit_mask(Dz, mask_at(l));
                        scale_inplace(Dz, gamma_res * alpha);
                    } else {
                        mul_relu_gate(Dz, H[static_cast<std::size_t>(l + 1)]);
                    }
                    grad = &Dz;
                }
                Matrix& W = net.W[static_cast<std::size_t>(l)];
                if (sink) {
                    Matrix& dW = (*sink)[static_cast<std::size_t>(l)];
                    dW.ensure(W.rows, W.cols);
                    gemm(H[static_cast<std::size_t>(l)], *grad, dW, true, false, 1.f, 0.f);
                }
                note_layer_grad(l, H[static_cast<std::size_t>(l)], *grad);
                if (keep_input_grad && l == 0) {
                    Gin.ensure(batch, net.dims[0]);
                    gemm(*grad, W, Gin, false, true, 1.f, 0.f);
                    if (skip) axpy(gamma_res, G, Gin);
                    if (capture_grad) embed_gnorm = frobenius(Gin);
                }
                if (l > 0) {
                    Gprev.ensure(batch, net.dims[static_cast<std::size_t>(l)]);
                    gemm(*grad, W, Gprev, false, true, 1.f, 0.f);
                    if (skip) axpy(gamma_res, G, Gprev);
                    std::swap(G, Gprev);
                }
            }
            if (apply) {
                commit_grads(net.W, pending, lr, clip, max_norm, clip_hits, scale_sum, scale_n, last_scale);
            }
            return;
        }
        for (int l = L - 1; l >= 0; --l) {
            if (l > 0) {
                Gprev.ensure(batch, net.dims[l]);
                gemm(G, net.W[static_cast<std::size_t>(l)], Gprev, false, true, 1.f, 0.f);
                mul_relu_gate(Gprev, H[static_cast<std::size_t>(l)]);
            }
            Matrix& W = net.W[static_cast<std::size_t>(l)];
            if (sink) {
                Matrix& dW = (*sink)[static_cast<std::size_t>(l)];
                dW.ensure(W.rows, W.cols);
                gemm(H[static_cast<std::size_t>(l)], G, dW, true, false, 1.f, 0.f);
            }
            note_layer_grad(l, H[static_cast<std::size_t>(l)], G);
            if (l > 0) std::swap(G, Gprev);
        }
        if (apply) commit_grads(net.W, pending, lr, clip, max_norm, clip_hits, scale_sum, scale_n, last_scale);
    }

    float step(const Matrix& X, const Matrix& Y, float lr) {
        forward(X);
        const float loss = classify ? softmax_ce(H.back(), Y, nullptr) : mean_square(H.back(), Y);
        backward(Y, lr, true, nullptr);
        return loss;
    }
};

struct Blob {
    void* data = nullptr;
    std::size_t bytes = 0;

    Blob() = default;
    Blob(const Blob&) = delete;
    Blob& operator=(const Blob&) = delete;
    Blob(Blob&& other) noexcept : data(other.data), bytes(other.bytes) {
        other.data = nullptr;
        other.bytes = 0;
    }
    Blob& operator=(Blob&& other) noexcept {
        if (this != &other) {
            release();
            data = other.data;
            bytes = other.bytes;
            other.data = nullptr;
            other.bytes = 0;
        }
        return *this;
    }
    ~Blob() { release(); }

    void ensure(std::size_t n) {
        if (n <= bytes && (data || n == 0)) return;
        release();
        if (n == 0) return;
        data = _aligned_malloc(n, 32);
        if (!data) throw std::bad_alloc();
        bytes = n;
        g_mem.on_alloc(n);
    }

    void release() {
        if (!data) return;
        g_mem.on_free(bytes);
        _aligned_free(data);
        data = nullptr;
        bytes = 0;
    }
};

void quantize_int8(const float* x, std::int8_t* q, std::size_t n, float& scale) {
    float peak = 0.f;
    for (std::size_t i = 0; i < n; ++i) peak = std::max(peak, std::fabs(x[i]));
    scale = peak > 1e-12f ? peak / 127.f : 1.f;
    const float inv = 1.f / scale;
    for (std::size_t i = 0; i < n; ++i) {
        int v = static_cast<int>(std::round(x[i] * inv));
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        q[i] = static_cast<std::int8_t>(v);
    }
}

// Peak is already known. Only the convert pass: OpenMP over 8-wide AVX2 chunks.
void quantize_int8_peak(const float* x, std::int8_t* q, std::size_t n, float peak, float& scale) {
    scale = peak > 1e-12f ? peak / 127.f : 1.f;
    const float inv = 1.f / scale;
    const int chunks = static_cast<int>(n >> 3);
#if defined(__AVX2__)
    const __m256 vinv = _mm256_set1_ps(inv);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 signbit = _mm256_set1_ps(-0.f);
    const __m256i lo = _mm256_set1_epi32(-127);
    const __m256i hi = _mm256_set1_epi32(127);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int c = 0; c < chunks; ++c) {
        const __m256 v = _mm256_mul_ps(_mm256_loadu_ps(x + static_cast<std::size_t>(c) * 8u), vinv);
        const __m256 sgn = _mm256_and_ps(v, signbit);
        const __m256 biased = _mm256_add_ps(v, _mm256_or_ps(sgn, half));
        __m256i i = _mm256_cvttps_epi32(biased);
        i = _mm256_max_epi32(i, lo);
        i = _mm256_min_epi32(i, hi);
        const __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(i), _mm256_extracti128_si256(i, 1));
        const __m128i p8 = _mm_packs_epi16(p16, p16);
        alignas(16) std::int8_t tmp[16];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), p8);
        std::memcpy(q + static_cast<std::size_t>(c) * 8u, tmp, 8);
    }
#else
    for (int c = 0; c < chunks; ++c) {
        for (int k = 0; k < 8; ++k) {
            const std::size_t i = static_cast<std::size_t>(c) * 8u + static_cast<std::size_t>(k);
            int v = static_cast<int>(std::round(x[i] * inv));
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            q[i] = static_cast<std::int8_t>(v);
        }
    }
#endif
    for (std::size_t i = static_cast<std::size_t>(chunks) * 8u; i < n; ++i) {
        int v = static_cast<int>(std::round(x[i] * inv));
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        q[i] = static_cast<std::int8_t>(v);
    }
}

void dequantize_int8(const std::int8_t* q, std::size_t n, float scale, float* x) {
    for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<float>(q[i]) * scale;
}

// crest = peak/rms. active_zero = fraction of strictly positive entries that the
// per-tensor INT8 scale (peak/127, round-to-nearest) collapses to 0.
void tape_stats(const float* x, std::size_t n, float& crest, float& active_zero) {
    float peak = 0.f;
    double ss = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        peak = std::max(peak, std::fabs(x[i]));
        ss += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }
    const double rms = n > 0 ? std::sqrt(ss / static_cast<double>(n)) : 0.0;
    crest = rms > 0.0 ? static_cast<float>(static_cast<double>(peak) / rms) : 0.f;
    const float scale = peak > 1e-12f ? peak / 127.f : 1.f;
    const float inv = 1.f / scale;
    std::size_t pos = 0;
    std::size_t killed = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!(x[i] > 0.f)) continue;
        ++pos;
        int v = static_cast<int>(std::round(x[i] * inv));
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        if (v == 0) ++killed;
    }
    active_zero = pos > 0 ? static_cast<float>(killed) / static_cast<float>(pos) : 0.f;
}

enum class Align { Off, Covariance, Jacobian };

float frobenius(const Matrix& M) {
    const std::size_t n = M.n();
    const float* p = M.data;
    double s = 0.0;
    std::size_t i = 0;
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(p + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    for (float v : tmp) s += static_cast<double>(v);
#endif
    for (; i < n; ++i) {
        const double v = p[i];
        s += v * v;
    }
    return static_cast<float>(std::sqrt(s));
}

void axpy(float a, const Matrix& X, Matrix& Y) {
    const std::size_t n = Y.n();
    float* y = Y.data;
    const float* x = X.data;
    std::size_t i = 0;
#if defined(__AVX2__)
    const __m256 va = _mm256_set1_ps(a);
    for (; i + 8 <= n; i += 8) {
        const __m256 s = _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i));
        _mm256_storeu_ps(y + i, s);
    }
#endif
    for (; i < n; ++i) y[i] += a * x[i];
}

void scale_inplace(Matrix& Y, float a) {
    const std::size_t n = Y.n();
    float* y = Y.data;
    std::size_t i = 0;
#if defined(__AVX2__)
    const __m256 va = _mm256_set1_ps(a);
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_mul_ps(va, _mm256_loadu_ps(y + i)));
    }
#endif
    for (; i < n; ++i) y[i] *= a;
}

int capped_update(Matrix& W, const Matrix& Xh, const Matrix& Gmat, float eta, bool clip, float clip_gamma) {
    if (clip) {
        const float nW = frobenius(W);
        const float cap = clip_gamma * nW;
        const float bound = eta * frobenius(Xh) * frobenius(Gmat);
        if (nW > 0.f && bound > cap) {
            Matrix raw(W.rows, W.cols);
            gemm(Xh, Gmat, raw, true, false, 1.f, 0.f);
            const float nG = frobenius(raw);
            const float nStep = eta * nG;
            float scale = -eta;
            int hit = 0;
            if (nG > 0.f && nStep > cap) {
                scale = -cap / nG;
                hit = 1;
            }
            axpy(scale, raw, W);
            return hit;
        }
    }
    gemm(Xh, Gmat, W, true, false, -eta, 1.f);
    return 0;
}

float commit_grads(std::vector<Matrix>& W, const std::vector<Matrix>& dW, float lr, bool clip, float max_norm,
                   int& hits, double& scale_sum, int& scale_n, float& last_scale) {
    double ss = 0.0;
    const int L = static_cast<int>(std::min(W.size(), dW.size()));
    for (int l = 0; l < L; ++l) {
        if (dW[static_cast<std::size_t>(l)].n() == 0) continue;
        const double f = frobenius(dW[static_cast<std::size_t>(l)]);
        ss += f * f;
    }
    const float gnorm = static_cast<float>(std::sqrt(ss));
    float scale = 1.f;
    if (clip && max_norm > 0.f && gnorm > max_norm && std::isfinite(gnorm)) {
        scale = max_norm / gnorm;
        ++hits;
    }
    last_scale = scale;
    scale_sum += scale;
    ++scale_n;
    for (int l = 0; l < L; ++l) {
        const Matrix& G = dW[static_cast<std::size_t>(l)];
        if (G.n() == 0 || G.rows != W[static_cast<std::size_t>(l)].rows || G.cols != W[static_cast<std::size_t>(l)].cols) {
            continue;
        }
        axpy(-lr * scale, G, W[static_cast<std::size_t>(l)]);
    }
    return scale;
}

float cosine_matrix(const Matrix& A, const Matrix& B) {
    expect(A.rows == B.rows && A.cols == B.cols, "cosine shape");
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (std::size_t i = 0; i < A.n(); ++i) {
        const double a = A.data[i];
        const double b = B.data[i];
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    if (na < 1e-18 || nb < 1e-18) return 0.f;
    return static_cast<float>(dot / std::sqrt(na * nb));
}

void column_energy(const Matrix& M, std::vector<double>& s, double& ss) {
    s.assign(static_cast<std::size_t>(M.cols), 0.0);
    for (int j = 0; j < M.cols; ++j) {
        double acc = 0.0;
        for (int r = 0; r < M.rows; ++r) {
            const double v = M(r, j);
            acc += v * v;
        }
        s[static_cast<std::size_t>(j)] = std::sqrt(acc);
    }
    ss = 0.0;
    for (double v : s) ss += v * v;
}

// A collapsed column stays 0 so one dead direction does not abort the sweep.
void orthonormal_columns_soft(Matrix& M) {
    expect(M.rows >= M.cols && M.cols > 0, "orthonormal shape");
    const int R = M.rows;
    const int C = M.cols;
    for (int j = 0; j < C; ++j) {
        for (int pass = 0; pass < 2; ++pass) {
            for (int i = 0; i < j; ++i) {
                double dot = 0.0;
                for (int r = 0; r < R; ++r) {
                    dot += static_cast<double>(M(r, i)) * static_cast<double>(M(r, j));
                }
                for (int r = 0; r < R; ++r) {
                    M(r, j) = static_cast<float>(static_cast<double>(M(r, j)) - dot * M(r, i));
                }
            }
        }
        double nrm = 0.0;
        for (int r = 0; r < R; ++r) {
            const double v = M(r, j);
            nrm += v * v;
        }
        nrm = std::sqrt(nrm);
        if (!(nrm > 1e-8) || !std::isfinite(nrm)) {
            for (int r = 0; r < R; ++r) M(r, j) = 0.f;
            continue;
        }
        const float inv = static_cast<float>(1.0 / nrm);
        for (int r = 0; r < R; ++r) M(r, j) *= inv;
    }
}

// Same MGS, but a column left with norm under 1e-8 is replaced by a random unit vector
// and projected against the columns already accepted. One dead byte-subspace cannot zero R.
void orthonormal_columns_reseed(Matrix& M, std::mt19937& rng) {
    expect(M.rows >= M.cols && M.cols > 0, "orthonormal shape");
    std::normal_distribution<float> normal(0.f, 1.f);
    const int rows = M.rows;
    const int cols = M.cols;
    auto project = [&](int j) {
        for (int pass = 0; pass < 2; ++pass) {
            for (int i = 0; i < j; ++i) {
                double dot = 0.0;
                for (int r = 0; r < rows; ++r) {
                    dot += static_cast<double>(M(r, i)) * static_cast<double>(M(r, j));
                }
                for (int r = 0; r < rows; ++r) {
                    M(r, j) = static_cast<float>(static_cast<double>(M(r, j)) - dot * M(r, i));
                }
            }
        }
    };
    auto column_norm = [&](int j) {
        double nrm = 0.0;
        for (int r = 0; r < rows; ++r) {
            const double v = M(r, j);
            nrm += v * v;
        }
        return std::sqrt(nrm);
    };
    for (int j = 0; j < cols; ++j) {
        project(j);
        double nrm = column_norm(j);
        if (!(nrm > 1e-8) || !std::isfinite(nrm)) {
            for (int attempt = 0; attempt < 4 && (!(nrm > 1e-8) || !std::isfinite(nrm)); ++attempt) {
                for (int r = 0; r < rows; ++r) M(r, j) = normal(rng);
                project(j);
                nrm = column_norm(j);
            }
        }
        expect(nrm > 1e-8 && std::isfinite(nrm), "reseed column vanished");
        const float inv = static_cast<float>(1.0 / nrm);
        for (int r = 0; r < rows; ++r) M(r, j) *= inv;
    }
}

// P = Q * diag(s / ||s||) * sqrt(cols). Orthogonal, but weak columns stay weak.
void apply_column_scales(Matrix& Q, const std::vector<double>& s, double ss) {
    const int C = Q.cols;
    const float gain = std::sqrt(static_cast<float>(C));
    const double inv = ss > 1e-18 && std::isfinite(ss) ? 1.0 / std::sqrt(ss) : 0.0;
    for (int j = 0; j < C; ++j) {
        const float scale = inv == 0.0 ? gain / std::sqrt(static_cast<float>(C))
                                       : static_cast<float>(s[static_cast<std::size_t>(j)] * inv) * gain;
        if (!std::isfinite(scale)) continue;
        for (int r = 0; r < Q.rows; ++r) Q(r, j) *= scale;
    }
}

void spectral_assign(Matrix& P, const Matrix& target) {
    expect(target.rows == P.rows && target.cols == P.cols, "spectral shape");
    std::vector<double> s;
    double ss = 0.0;
    column_energy(target, s, ss);
    P = target;
    orthonormal_columns_soft(P);
    apply_column_scales(P, s, ss);
}

// Directions mix after unit-normalizing. Scales come from the live target columns.
void ema_spectral(Matrix& P, const Matrix& target, float beta) {
    expect(P.rows == target.rows && P.cols == target.cols, "ema shape");
    Matrix kept = P;
    std::vector<double> s;
    double ss = 0.0;
    column_energy(target, s, ss);
    if (!std::isfinite(ss)) return;
    Matrix blend(P.rows, P.cols);
    const float keep = 1.f - beta;
    for (int j = 0; j < P.cols; ++j) {
        double np = 0.0;
        for (int r = 0; r < P.rows; ++r) {
            const double v = P(r, j);
            np += v * v;
        }
        np = std::sqrt(np);
        const double nt = s[static_cast<std::size_t>(j)];
        for (int r = 0; r < P.rows; ++r) {
            const float tp = nt > 1e-12 ? static_cast<float>(target(r, j) / nt) : 0.f;
            const float pp = np > 1e-8 ? static_cast<float>(P(r, j) / np) : tp;
            blend(r, j) = keep * pp + beta * tp;
        }
    }
    orthonormal_columns_soft(blend);
    apply_column_scales(blend, s, ss);
    for (std::size_t i = 0; i < blend.n(); ++i) {
        if (!std::isfinite(blend.data[i])) {
            P = std::move(kept);
            return;
        }
    }
    P = std::move(blend);
}

struct TrainerB {
    Net net;
    std::vector<Matrix> P;
    Blob xq;
    Blob mask;
    std::vector<std::size_t> xq_off;
    std::vector<std::size_t> mask_off;
    std::vector<float> scale;
    std::vector<std::vector<float>> rate_sum;
    int rate_fill = 0;
    std::vector<Matrix> U;
    std::vector<Matrix> aux_err;
    Matrix buf0, buf1, delta, dW, err, pred;
    int batch_ = 0;
    bool clip = false;
    float clip_gamma = 0.05f;
    float max_norm = 100.f;
    int clip_hits = 0;
    float last_scale = 1.f;
    double scale_sum = 0.0;
    int scale_n = 0;
    std::vector<Matrix> pending;
    bool classify = false;
    // Layer-0 residual input gradient: gamma * (e P^T) + delta @ W^T. Off unless a caller asks.
    bool keep_input_grad = false;
    Matrix embed_g;
    Align align = Align::Off;
    int align_k = 10;
    float align_beta = 0.05f;
    float align_beta_fast = 0.25f;
    int align_fast_until = 50;
    int ticks = 0;
    // 0: every sweep orthogonalizes. Positive: full MGS only when ticks is a multiple.
    int mgs_k = 0;
    bool mgs_now = true;
    std::vector<std::vector<float>> col_norm;
    double sweep_us = 0.0;
    double mgs_us = 0.0;
    double scale_us = 0.0;
    int sweep_calls = 0;
    int mgs_calls = 0;
    int scale_calls = 0;
    double r_us = 0.0;
    int r_calls = 0;
    // Two anchors. 0.10 is the dose that kept lr=0.01 finite; 0.70 replaced SAP.
    float lambda_peak = 0.10f;
    // 0 keeps P at the full output width. A positive rank compresses e through R (out x rank).
    int rank = 0;
    Matrix R;
    Matrix etilde;
    Matrix Yacc;
    std::mt19937 rank_rng;
    // Diagnostic. float_strip keeps the pre-quant activation and uses it in dW = X^T delta.
    bool float_strip = false;
    bool capture_stats = false;
    bool capture_grad = false;
    bool report_embed = false;
    std::vector<Matrix> xfloat;
    std::vector<float> crest;
    std::vector<float> active_zero;
    std::vector<float> grad_norm;
    float embed_gnorm = 0.f;

    int n_hidden() const { return std::max(n_layers(net) - 1, 0); }

    int feedback_cols() const { return rank > 0 ? rank : net.dims.back(); }

    // 1-based layers 2 and 4 only. Every other hidden layer stays pure SAP.
    float lambda_at(int layer0) const {
        if (lambda_peak <= 0.f) return 0.f;
        const int k = layer0 + 1;
        if (k == 2 || k == 4) return lambda_peak;
        return 0.f;
    }

    void allocate_feedback() {
        const int L = n_layers(net);
        const int cols = feedback_cols();
        if (rank > 0) expect(rank <= net.dims.back(), "rank exceeds the output width");
        P.assign(static_cast<std::size_t>(std::max(L - 1, 0)), Matrix{});
        for (int l = 0; l < L - 1; ++l) {
            const int fan_out = net.dims[static_cast<std::size_t>(l + 1)];
            expect(fan_out >= cols, "feedback matrix needs fan_out >= feedback width");
            P[static_cast<std::size_t>(l)].ensure(fan_out, cols);
        }
    }

    void refresh_R() {
        expect(rank > 0 && R.cols == rank && R.rows == net.dims.back(), "R shape");
        fill_normal(R, rank_rng, 1.f);
        orthonormal_columns(R);
    }

    // Add this step's e^T (e R) into Yacc. etilde already holds e R.
    void accumulate_error_R() {
        expect(rank > 0 && etilde.rows == err.rows && etilde.cols == rank, "etilde missing");
        if (Yacc.rows != err.cols || Yacc.cols != rank) {
            Yacc.ensure(err.cols, rank);
            Yacc.zero();
        }
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();
        gemm(err, etilde, Yacc, true, false, 1.f, 1.f);
        r_us += std::chrono::duration<double, std::micro>(clock::now() - t0).count();
    }

    // R = MGS of the outer products gathered since the previous refresh, then the sum is cleared.
    void spectral_R() {
        expect(rank > 0 && R.cols == rank && Yacc.rows == R.rows && Yacc.cols == R.cols, "Yacc shape");
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();
        R = Yacc;
        orthonormal_columns_reseed(R, rank_rng);
        Yacc.zero();
        r_us += std::chrono::duration<double, std::micro>(clock::now() - t0).count();
        ++r_calls;
    }

    void init_feedback(std::uint32_t seed) {
        allocate_feedback();
        rank_rng.seed(seed + 17u);
        if (rank > 0) {
            R.ensure(net.dims.back(), rank);
            refresh_R();
        }
        if (align == Align::Jacobian) {
            init_aux(seed + 123u);
            sweep_jacobian(1.f, false);
            return;
        }
        std::mt19937 rng(seed);
        for (Matrix& block : P) {
            fill_normal(block, rng, 1.f);
            orthonormal_columns(block);
        }
    }

    float beta_now() const {
        return ticks <= align_fast_until ? align_beta_fast : align_beta;
    }

    void init_aux(std::uint32_t seed) {
        const int n = n_hidden();
        const int out = net.dims.back();
        U.assign(static_cast<std::size_t>(n), Matrix{});
        std::mt19937 rng(seed);
        for (int l = 0; l < n; ++l) {
            if (lambda_at(l) <= 0.f) continue;
            const int fan = net.dims[static_cast<std::size_t>(l + 1)];
            U[static_cast<std::size_t>(l)].ensure(fan, out);
            fill_normal(U[static_cast<std::size_t>(l)], rng, std::sqrt(1.f / static_cast<float>(fan)));
        }
    }

    void prepare(int batch) {
        const int L = n_layers(net);
        batch_ = batch;
        int maxd = 0;
        int max_h = 0;
        for (int d : net.dims) maxd = std::max(maxd, d);
        for (int l = 0; l < L - 1; ++l) max_h = std::max(max_h, net.dims[static_cast<std::size_t>(l + 1)]);
        buf0.ensure(batch, maxd);
        buf1.ensure(batch, maxd);
        if (max_h > 0) delta.ensure(batch, max_h);
        err.ensure(batch, net.dims.back());
        pred.ensure(batch, net.dims.back());
        aux_err.assign(U.size(), Matrix{});
        for (int l = 0; l < static_cast<int>(U.size()); ++l) {
            if (U[static_cast<std::size_t>(l)].n() == 0) continue;
            aux_err[static_cast<std::size_t>(l)].ensure(batch, net.dims.back());
        }

        xq_off.assign(static_cast<std::size_t>(L + 1), 0);
        std::size_t xq_n = 0;
        for (int l = 0; l < L; ++l) {
            xq_off[static_cast<std::size_t>(l)] = xq_n;
            xq_n += static_cast<std::size_t>(batch) * static_cast<std::size_t>(net.dims[static_cast<std::size_t>(l)]);
        }
        xq_off[static_cast<std::size_t>(L)] = xq_n;
        xq.ensure(xq_n);

        mask_off.assign(static_cast<std::size_t>(std::max(L, 1)), 0);
        std::size_t words = 0;
        for (int l = 0; l < L - 1; ++l) {
            mask_off[static_cast<std::size_t>(l)] = words;
            const std::size_t nelt = static_cast<std::size_t>(batch) *
                                     static_cast<std::size_t>(net.dims[static_cast<std::size_t>(l + 1)]);
            words += (nelt + 31u) / 32u;
        }
        if (L >= 1) mask_off[static_cast<std::size_t>(L - 1)] = words;
        mask.ensure(words * sizeof(std::uint32_t));
        scale.assign(static_cast<std::size_t>(L), 1.f);
        xfloat.clear();
        if (float_strip) {
            xfloat.assign(static_cast<std::size_t>(L), Matrix{});
            for (int l = 0; l < L; ++l) {
                xfloat[static_cast<std::size_t>(l)].ensure(batch, net.dims[static_cast<std::size_t>(l)]);
            }
        }
    }

    std::size_t bytes() const {
        return bytes_of(net.W) + bytes_of(P) + bytes_of(U) + bytes_of(aux_err) + xq.bytes + mask.bytes +
               delta.bytes() + dW.bytes() + buf0.bytes() + buf1.bytes() + err.bytes() + pred.bytes() + R.bytes() +
               etilde.bytes() + Yacc.bytes() + bytes_of(xfloat);
    }

    std::int8_t* xq_at(int layer) {
        return static_cast<std::int8_t*>(xq.data) + xq_off[static_cast<std::size_t>(layer)];
    }
    std::uint32_t* mask_at(int layer) {
        return static_cast<std::uint32_t*>(mask.data) + mask_off[static_cast<std::size_t>(layer)];
    }

    void dequant_input(int layer, Matrix& dst) {
        const int fan_in = net.dims[static_cast<std::size_t>(layer)];
        const std::size_t n = static_cast<std::size_t>(batch_) * static_cast<std::size_t>(fan_in);
        dst.ensure(batch_, fan_in);
        if (float_strip) {
            expect(static_cast<int>(xfloat.size()) > layer && xfloat[static_cast<std::size_t>(layer)].n() == n,
                   "float strip missing");
            std::memcpy(dst.data, xfloat[static_cast<std::size_t>(layer)].data, n * sizeof(float));
            return;
        }
        dequantize_int8(xq_at(layer), n, scale[static_cast<std::size_t>(layer)], dst.data);
    }

    void remember_input(int layer, const Matrix& h) {
        if (float_strip) {
            Matrix& slot = xfloat[static_cast<std::size_t>(layer)];
            expect(slot.rows == h.rows && slot.cols == h.cols, "float strip shape");
            std::memcpy(slot.data, h.data, h.n() * sizeof(float));
        }
        if (!capture_stats) return;
        const int L = n_layers(net);
        if (static_cast<int>(crest.size()) != L) {
            crest.assign(static_cast<std::size_t>(L), 0.f);
            active_zero.assign(static_cast<std::size_t>(L), 0.f);
        }
        tape_stats(h.data, h.n(), crest[static_cast<std::size_t>(layer)], active_zero[static_cast<std::size_t>(layer)]);
    }

    void note_layer_grad(int layer, const Matrix& Xh, const Matrix& Gmat) {
        if (!capture_grad) return;
        const int L = n_layers(net);
        if (static_cast<int>(grad_norm.size()) != L) grad_norm.assign(static_cast<std::size_t>(L), 0.f);
        const Matrix& W = net.W[static_cast<std::size_t>(layer)];
        dW.ensure(W.rows, W.cols);
        gemm(Xh, Gmat, dW, true, false, 1.f, 0.f);
        grad_norm[static_cast<std::size_t>(layer)] = frobenius(dW);
    }

    void apply_mask(Matrix& G, const std::uint32_t* bits) const {
        for (std::size_t i = 0; i < G.n(); ++i) {
            if (!relu_mask_bit(bits, i)) G.data[i] = 0.f;
        }
    }

    void stash_grad(int layer, const Matrix& Xh, const Matrix& G) {
        const int L = n_layers(net);
        if (static_cast<int>(pending.size()) != L) pending.assign(static_cast<std::size_t>(L), Matrix{});
        const Matrix& W = net.W[static_cast<std::size_t>(layer)];
        Matrix& slot = pending[static_cast<std::size_t>(layer)];
        slot.ensure(W.rows, W.cols);
        gemm(Xh, G, slot, true, false, 1.f, 0.f);
    }

    void observe_masks() {
        if (rate_sum.size() != P.size()) rate_sum.assign(P.size(), {});
        for (int l = 0; l < static_cast<int>(P.size()); ++l) {
            const int fan = P[static_cast<std::size_t>(l)].rows;
            auto& acc = rate_sum[static_cast<std::size_t>(l)];
            if (static_cast<int>(acc.size()) != fan) acc.assign(static_cast<std::size_t>(fan), 0.f);
            const std::uint32_t* bits = mask_at(l);
            for (int j = 0; j < fan; ++j) {
                int on = 0;
                for (int b = 0; b < batch_; ++b) {
                    const std::size_t idx = static_cast<std::size_t>(b) * static_cast<std::size_t>(fan) + j;
                    if (relu_mask_bit(bits, idx)) ++on;
                }
                acc[static_cast<std::size_t>(j)] += static_cast<float>(on) / static_cast<float>(batch_);
            }
        }
        ++rate_fill;
    }

    const Matrix& feedback_error() {
        if (rank <= 0) return err;
        etilde.ensure(batch_, rank);
        gemm(err, R, etilde, false, false, 1.f, 0.f);
        return etilde;
    }

    void align_covariance() {
        const float beta = beta_now();
        const Matrix& fb = feedback_error();
        for (int l = 0; l < static_cast<int>(P.size()); ++l) {
            dequant_input(l + 1, buf0);
            gemm(buf0, fb, P[static_cast<std::size_t>(l)], true, false, beta, 1.f - beta);
            orthonormal_columns(P[static_cast<std::size_t>(l)]);
        }
    }

    void remember_column_norms(int layer, const Matrix& M) {
        if (static_cast<int>(col_norm.size()) != static_cast<int>(P.size())) col_norm.assign(P.size(), {});
        auto& dst = col_norm[static_cast<std::size_t>(layer)];
        dst.assign(static_cast<std::size_t>(M.cols), 0.f);
        for (int j = 0; j < M.cols; ++j) {
            double acc = 0.0;
            for (int r = 0; r < M.rows; ++r) {
                const double v = M(r, j);
                acc += v * v;
            }
            dst[static_cast<std::size_t>(j)] = static_cast<float>(std::sqrt(acc));
        }
    }

    // Fresh directions, column L2 snapped to the last MGS energy. No inner products.
    void scale_columns(Matrix& block, const Matrix& target, int layer) {
        expect(target.rows == block.rows && target.cols == block.cols, "scale shape");
        block = target;
        const bool saved = static_cast<int>(col_norm.size()) == static_cast<int>(P.size()) &&
                           layer >= 0 && layer < static_cast<int>(col_norm.size()) &&
                           static_cast<int>(col_norm[static_cast<std::size_t>(layer)].size()) == block.cols;
        if (!saved) {
            std::vector<double> s;
            double ss = 0.0;
            column_energy(block, s, ss);
            const float gain = std::sqrt(static_cast<float>(block.cols));
            const float mul = ss > 1e-18 && std::isfinite(ss) ? static_cast<float>(gain / std::sqrt(ss)) : 1.f;
            if (std::isfinite(mul)) scale_inplace(block, mul);
            return;
        }
        const auto& want = col_norm[static_cast<std::size_t>(layer)];
        for (int j = 0; j < block.cols; ++j) {
            double acc = 0.0;
            for (int r = 0; r < block.rows; ++r) {
                const double v = block(r, j);
                acc += v * v;
            }
            const double nrm = std::sqrt(acc);
            const float target_norm = want[static_cast<std::size_t>(j)];
            if (!(target_norm > 0.f) || !std::isfinite(target_norm)) {
                for (int r = 0; r < block.rows; ++r) block(r, j) = 0.f;
                continue;
            }
            if (!(nrm > 1e-8) || !std::isfinite(nrm)) continue;
            const float mul = target_norm / static_cast<float>(nrm);
            if (!std::isfinite(mul)) continue;
            for (int r = 0; r < block.rows; ++r) block(r, j) *= mul;
        }
    }

    void place_feedback(Matrix& block, const Matrix& target, float beta, int layer) {
        using clock = std::chrono::steady_clock;
        if (beta >= 0.999f && !mgs_now) {
            const auto t0 = clock::now();
            scale_columns(block, target, layer);
            scale_us += std::chrono::duration<double, std::micro>(clock::now() - t0).count();
            ++scale_calls;
            return;
        }
        if (beta >= 0.999f) {
            const auto t0 = clock::now();
            spectral_assign(block, target);
            remember_column_norms(layer, block);
            mgs_us += std::chrono::duration<double, std::micro>(clock::now() - t0).count();
            ++mgs_calls;
            return;
        }
        ema_spectral(block, target, beta);
    }

    // Raw Jacobian carried backward, then MGS with the target's column scales restored.
    // use_mask == false is the t = 0 seed.
    void sweep_jacobian(float beta, bool use_mask) {
        const int L = n_layers(net);
        const int last = L - 2;
        if (last < 0) return;
        if (use_mask && rate_fill <= 0) return;
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();
        const float inv = use_mask ? 1.f / static_cast<float>(rate_fill) : 1.f;
        Matrix carried;
        if (rank > 0) {
            const Matrix& Wout = net.W[static_cast<std::size_t>(L - 1)];
            carried.ensure(Wout.rows, rank);
            gemm(Wout, R, carried, false, false, 1.f, 0.f);
        } else {
            carried = net.W[static_cast<std::size_t>(L - 1)];
        }
        place_feedback(P[static_cast<std::size_t>(last)], carried, beta, last);
        for (int l = last - 1; l >= 0; --l) {
            Matrix scaled = carried;
            if (use_mask) {
                const auto& acc = rate_sum[static_cast<std::size_t>(l + 1)];
                for (int r = 0; r < scaled.rows; ++r) {
                    const float m = acc[static_cast<std::size_t>(r)] * inv;
                    for (int c = 0; c < scaled.cols; ++c) scaled(r, c) *= m;
                }
            }
            Matrix next(net.W[static_cast<std::size_t>(l + 1)].rows, carried.cols);
            const bool skip = skip_layer(net, l + 1);
            gemm(net.W[static_cast<std::size_t>(l + 1)], scaled, next, false, false, skip ? alpha : 1.f, 0.f);
            if (skip) {
                axpy(1.f, carried, next);
                scale_inplace(next, gamma_res);
            }
            bool finite = true;
            for (std::size_t i = 0; i < next.n(); ++i) {
                if (!std::isfinite(next.data[i])) {
                    finite = false;
                    break;
                }
            }
            if (!finite) break;
            carried = std::move(next);
            place_feedback(P[static_cast<std::size_t>(l)], carried, beta, l);
        }
        if (use_mask) {
            for (auto& acc : rate_sum) std::fill(acc.begin(), acc.end(), 0.f);
            rate_fill = 0;
        }
        sweep_us += std::chrono::duration<double, std::micro>(clock::now() - t0).count();
        ++sweep_calls;
    }

    void align_jacobian() {
        mgs_now = mgs_k == 0 || (mgs_k > 0 && ticks % mgs_k == 0);
        sweep_jacobian(beta_now(), true);
    }

    void report_cosine(const Matrix& X, const Matrix& Y, int step_id, const char* tag = nullptr) {
        const int L = n_layers(net);
        const int batch = X.rows;
        std::vector<Matrix> H(static_cast<std::size_t>(L + 1));
        H[0] = X;
        std::vector<std::uint32_t> bits;
        std::vector<std::size_t> bit_off(static_cast<std::size_t>(std::max(L, 1)), 0);
        if (net.residual) {
            std::size_t words = 0;
            for (int l = 0; l < L - 1; ++l) {
                bit_off[static_cast<std::size_t>(l)] = words;
                const std::size_t nelt = static_cast<std::size_t>(batch) *
                                         static_cast<std::size_t>(net.dims[static_cast<std::size_t>(l + 1)]);
                words += (nelt + 31u) / 32u;
            }
            bits.assign(words, 0u);
        }
        for (int l = 0; l < L; ++l) {
            H[static_cast<std::size_t>(l + 1)].ensure(batch, net.dims[static_cast<std::size_t>(l + 1)]);
            gemm(H[static_cast<std::size_t>(l)], net.W[static_cast<std::size_t>(l)],
                 H[static_cast<std::size_t>(l + 1)], false, false, 1.f, 0.f);
            if (l + 1 < L) {
                if (skip_layer(net, l)) {
                    pack_relu_mask(H[static_cast<std::size_t>(l + 1)].data,
                                   bits.data() + bit_off[static_cast<std::size_t>(l)],
                                   H[static_cast<std::size_t>(l + 1)].n());
                    relu_inplace(H[static_cast<std::size_t>(l + 1)]);
                    scale_inplace(H[static_cast<std::size_t>(l + 1)], alpha);
                    axpy(1.f, H[static_cast<std::size_t>(l)], H[static_cast<std::size_t>(l + 1)]);
                    scale_inplace(H[static_cast<std::size_t>(l + 1)], gamma_res);
                } else {
                    relu_inplace(H[static_cast<std::size_t>(l + 1)]);
                }
            }
        }
        Matrix E;
        Matrix G;
        Matrix Gprev;
        Matrix Sap;
        if (classify) softmax_ce(H[static_cast<std::size_t>(L)], Y, &E);
        else output_grad(H[static_cast<std::size_t>(L)], Y, E);
        Matrix Et;
        const Matrix* fb = &E;
        if (rank > 0) {
            Et.ensure(batch, rank);
            gemm(E, R, Et, false, false, 1.f, 0.f);
            fb = &Et;
        }
        G = E;
        std::vector<float> cosv(static_cast<std::size_t>(std::max(L - 1, 0)), 0.f);
        Matrix raw;
        Matrix sap_raw;
        float embed_cos = 0.f;
        const float ninv = 2.f / static_cast<float>(batch * Y.cols);
        Matrix Dz;
        for (int l = L - 1; l >= 0; --l) {
            const bool hidden = l < L - 1;
            const bool skip = skip_layer(net, l);
            if (hidden) {
                const float lam = l < static_cast<int>(U.size()) && U[static_cast<std::size_t>(l)].n() > 0
                                      ? lambda_at(l)
                                      : 0.f;
                Sap.ensure(batch, net.dims[static_cast<std::size_t>(l + 1)]);
                if (lam > 0.f) {
                    raw.ensure(batch, Y.cols);
                    gemm(H[static_cast<std::size_t>(l + 1)], U[static_cast<std::size_t>(l)], raw, false, false, 1.f,
                         0.f);
                    for (std::size_t i = 0; i < raw.n(); ++i) raw.data[i] -= Y.data[i];
                    gemm(*fb, P[static_cast<std::size_t>(l)], Sap, false, true, 1.f - lam, 0.f);
                    gemm(raw, U[static_cast<std::size_t>(l)], Sap, false, true, lam * ninv, 1.f);
                } else {
                    gemm(*fb, P[static_cast<std::size_t>(l)], Sap, false, true, 1.f, 0.f);
                }
                if (report_embed && l == 0) sap_raw = Sap;
                if (skip) {
                    Dz.ensure(G.rows, G.cols);
                    std::memcpy(Dz.data, G.data, G.n() * sizeof(float));
                    apply_bit_mask(Dz, bits.data() + bit_off[static_cast<std::size_t>(l)]);
                    apply_bit_mask(Sap, bits.data() + bit_off[static_cast<std::size_t>(l)]);
                    scale_inplace(Dz, gamma_res * alpha);
                    scale_inplace(Sap, gamma_res * alpha);
                    cosv[static_cast<std::size_t>(l)] = cosine_matrix(Dz, Sap);
                } else if (net.residual) {
                    mul_relu_gate(G, H[static_cast<std::size_t>(l + 1)]);
                    mul_relu_gate(Sap, H[static_cast<std::size_t>(l + 1)]);
                    cosv[static_cast<std::size_t>(l)] = cosine_matrix(G, Sap);
                } else {
                    mul_relu_gate(Sap, H[static_cast<std::size_t>(l + 1)]);
                    cosv[static_cast<std::size_t>(l)] = cosine_matrix(G, Sap);
                }
                if (report_embed && l == 0) {
                    Matrix gin_bp(batch, net.dims[0]);
                    Matrix gin_sap(batch, net.dims[0]);
                    const Matrix& g_bp = skip ? Dz : G;
                    gemm(g_bp, net.W[0], gin_bp, false, true, 1.f, 0.f);
                    gemm(Sap, net.W[0], gin_sap, false, true, 1.f, 0.f);
                    if (skip) {
                        axpy(gamma_res, G, gin_bp);
                        axpy(gamma_res, sap_raw, gin_sap);
                    }
                    embed_cos = cosine_matrix(gin_bp, gin_sap);
                }
            }
            if (l > 0) {
                Gprev.ensure(batch, net.dims[static_cast<std::size_t>(l)]);
                if (skip) {
                    gemm(Dz, net.W[static_cast<std::size_t>(l)], Gprev, false, true, 1.f, 0.f);
                    axpy(gamma_res, G, Gprev);
                } else {
                    gemm(G, net.W[static_cast<std::size_t>(l)], Gprev, false, true, 1.f, 0.f);
                    if (!net.residual) mul_relu_gate(Gprev, H[static_cast<std::size_t>(l)]);
                }
                std::swap(G, Gprev);
            }
        }
        std::cout << "cos";
        if (tag) std::cout << " " << tag;
        std::cout << " " << step_id;
        for (float c : cosv) {
            char buf[16];
            std::snprintf(buf, sizeof(buf), " %6.3f", c);
            std::cout << buf;
        }
        if (report_embed) {
            char buf[24];
            std::snprintf(buf, sizeof(buf), " E %6.3f", embed_cos);
            std::cout << buf;
        }
        std::cout << "\n" << std::flush;
    }

    float step(const Matrix& X, const Matrix& Y, float lr) {
        const int L = n_layers(net);
        const int batch = X.rows;
        const int out = net.dims.back();
        expect(batch == batch_ && Y.rows == batch && Y.cols == out, "B batch shape");

        const Matrix* h = &X;
        Matrix* dest = &buf0;
        Matrix* spare = &buf1;
        for (int l = 0; l < L - 1; ++l) {
            const std::size_t n_in = h->n();
            expect(n_in == xq_off[static_cast<std::size_t>(l + 1)] - xq_off[static_cast<std::size_t>(l)],
                   "input tape size");
            remember_input(l, *h);
            quantize_int8(h->data, xq_at(l), n_in, scale[static_cast<std::size_t>(l)]);
            const int fan_out = net.dims[static_cast<std::size_t>(l + 1)];
            dest->ensure(batch, fan_out);
            gemm(*h, net.W[static_cast<std::size_t>(l)], *dest, false, false, 1.f, 0.f);
            pack_relu_mask(dest->data, mask_at(l), dest->n());
            relu_inplace(*dest);
            if (skip_layer(net, l)) {
                scale_inplace(*dest, alpha);
                axpy(1.f, *h, *dest);
                scale_inplace(*dest, gamma_res);
            }
            if (l < static_cast<int>(U.size()) && U[static_cast<std::size_t>(l)].n() > 0) {
                Matrix& el = aux_err[static_cast<std::size_t>(l)];
                el.ensure(batch, out);
                gemm(*dest, U[static_cast<std::size_t>(l)], el, false, false, 1.f, 0.f);
                for (std::size_t i = 0; i < el.n(); ++i) el.data[i] -= Y.data[i];
            }
            h = dest;
            std::swap(dest, spare);
        }
        expect(h->n() == xq_off[static_cast<std::size_t>(L)] - xq_off[static_cast<std::size_t>(L - 1)],
               "last input tape size");
        remember_input(L - 1, *h);
        quantize_int8(h->data, xq_at(L - 1), h->n(), scale[static_cast<std::size_t>(L - 1)]);
        pred.ensure(batch, out);
        gemm(*h, net.W[static_cast<std::size_t>(L - 1)], pred, false, false, 1.f, 0.f);
        float loss = 0.f;
        if (classify) loss = softmax_ce(pred, Y, &err);
        else {
            loss = mean_square(pred, Y);
            output_grad(pred, Y, err);
        }

        const float ninv = 2.f / static_cast<float>(batch * out);
        const Matrix* fb = &err;
        if (rank > 0) {
            etilde.ensure(batch, rank);
            gemm(err, R, etilde, false, false, 1.f, 0.f);
            fb = &etilde;
        }
        for (int l = 0; l < L - 1; ++l) {
            const int fan_out = net.dims[static_cast<std::size_t>(l + 1)];
            const float lam = l < static_cast<int>(U.size()) && U[static_cast<std::size_t>(l)].n() > 0 ? lambda_at(l)
                                                                                                        : 0.f;
            delta.reshape(batch, fan_out);
            if (lam > 0.f) {
                gemm(*fb, P[static_cast<std::size_t>(l)], delta, false, true, 1.f - lam, 0.f);
                gemm(aux_err[static_cast<std::size_t>(l)], U[static_cast<std::size_t>(l)], delta, false, true,
                     lam * ninv, 1.f);
            } else {
                gemm(*fb, P[static_cast<std::size_t>(l)], delta, false, true, 1.f, 0.f);
            }
            if (keep_input_grad && l == 0) {
                embed_g.ensure(batch, fan_out);
                std::memcpy(embed_g.data, delta.data, delta.n() * sizeof(float));
            }
            const bool skip = skip_layer(net, l);
            if (skip) scale_inplace(delta, gamma_res * alpha);
            apply_mask(delta, mask_at(l));
            if (keep_input_grad && l == 0) {
                const int fan_in = net.dims[0];
                buf1.reshape(batch, fan_in);
                gemm(delta, net.W[0], buf1, false, true, 1.f, 0.f);
                if (skip) axpy(gamma_res, embed_g, buf1);
                embed_g.ensure(batch, fan_in);
                std::memcpy(embed_g.data, buf1.data, embed_g.n() * sizeof(float));
                if (capture_grad) embed_gnorm = frobenius(embed_g);
            }
            dequant_input(l, buf0);
            note_layer_grad(l, buf0, delta);
            stash_grad(l, buf0, delta);
            if (lam > 0.f) {
                dequant_input(l + 1, buf1);
                gemm(buf1, aux_err[static_cast<std::size_t>(l)], U[static_cast<std::size_t>(l)], true, false,
                     -lr / static_cast<float>(batch), 1.f);
            }
        }
        dequant_input(L - 1, buf0);
        note_layer_grad(L - 1, buf0, err);
        stash_grad(L - 1, buf0, err);
        commit_grads(net.W, pending, lr, clip, max_norm, clip_hits, scale_sum, scale_n, last_scale);

        if (align == Align::Jacobian) observe_masks();
        ++ticks;
        if (rank > 0 && align != Align::Off) accumulate_error_R();
        if (align != Align::Off && align_k > 0 && ticks % align_k == 0) {
            if (rank > 0) spectral_R();
            if (align == Align::Covariance) align_covariance();
            else if (rank > 0) {
                mgs_now = mgs_k == 0 || (mgs_k > 0 && ticks % mgs_k == 0);
                sweep_jacobian(1.f, true);
            } else align_jacobian();
        }
        return loss;
    }
};

struct Config {
    int steps = 300;
    int batch = 128;
    int hidden = 512;
    int squares = 10;
    int in_dim = 10;
    int out_dim = 10;
    // 0.01 keeps lowering A's loss but the orthonormal feedback update diverges after ~100 steps.
    float lr = 0.002f;
    int every = 100;
    std::uint32_t seed = 1;
    int probe = 256;
    int align = 2;
    bool clip = true;
    float clip_gamma = 0.05f;
    int cos_every = 50;
    float lambda_peak = 0.10f;
    bool residual = false;
    double seconds = 0.0;
    std::string data;
    std::string lm;
    std::string shake;
    int epochs = 3;
    int align_k = 0;
    int k2 = 0;
    // Negative leaves the trainer's 0.25-then-0.05 schedule alone.
    float beta = -1.f;
    int mgs_k = 0;
    float max_norm = 100.f;
    int rank = 0;
    bool float_strip = false;
    bool diag = false;
    bool trio = false;
    bool adam = false;
    bool phase2 = false;
    bool scale = false;
    bool lora = false;
    bool qlora = false;
    int qsuite = 0;
    float wd = 0.f;
};

double mb(std::size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

void naive_reference(const Matrix& A, bool transA, const Matrix& B, bool transB, Matrix& C,
                     float alpha, float beta) {
    const int M = transA ? A.cols : A.rows;
    const int K = transA ? A.rows : A.cols;
    const int N = transB ? B.rows : B.cols;
    expect(C.rows == M && C.cols == N, "naive shape");
    auto a_at = [&](int i, int p) {
        return transA ? A(p, i) : A(i, p);
    };
    auto b_at = [&](int p, int j) {
        return transB ? B(j, p) : B(p, j);
    };
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int p = 0; p < K; ++p) s += static_cast<double>(a_at(i, p)) * b_at(p, j);
            const double old = C(i, j);
            C(i, j) = static_cast<float>(static_cast<double>(beta) * old + static_cast<double>(alpha) * s);
        }
    }
}

void check_gemm_case(std::mt19937& rng, int m, int k, int n, bool transA, bool transB, float alpha,
                     float beta) {
    Matrix A(transA ? k : m, transA ? m : k);
    Matrix B(transB ? n : k, transB ? k : n);
    Matrix C(m, n);
    Matrix Ref(m, n);
    fill_uniform(A, rng, -1.f, 1.f);
    fill_uniform(B, rng, -1.f, 1.f);
    fill_uniform(C, rng, -1.f, 1.f);
    Ref = C;
    gemm(A, B, C, transA, transB, alpha, beta);
    naive_reference(A, transA, B, transB, Ref, alpha, beta);
    const float err = max_abs_diff(C, Ref);
    const float limit = 2e-3f + 1e-5f * static_cast<float>(k);
    if (err > limit) {
        std::ostringstream os;
        os << "gemm mismatch m=" << m << " k=" << k << " n=" << n << " transA=" << transA
           << " transB=" << transB << " err=" << err;
        throw std::runtime_error(os.str());
    }
}

void naive_dfa(const std::vector<Matrix>& W0, const std::vector<Matrix>& P, const Matrix& X,
               const Matrix& Y, float lr, std::vector<Matrix>& Wnew) {
    const int L = static_cast<int>(W0.size());
    const int B = X.rows;
    std::vector<Matrix> H(static_cast<std::size_t>(L + 1));
    std::vector<Matrix> Z(static_cast<std::size_t>(L));
    H[0] = X;
    for (int l = 0; l < L; ++l) {
        const int fo = W0[static_cast<std::size_t>(l)].cols;
        const int fi = W0[static_cast<std::size_t>(l)].rows;
        Z[static_cast<std::size_t>(l)].ensure(B, fo);
        for (int b = 0; b < B; ++b) {
            for (int j = 0; j < fo; ++j) {
                double s = 0.0;
                for (int i = 0; i < fi; ++i) {
                    s += static_cast<double>(H[static_cast<std::size_t>(l)](b, i)) *
                         W0[static_cast<std::size_t>(l)](i, j);
                }
                Z[static_cast<std::size_t>(l)](b, j) = static_cast<float>(s);
            }
        }
        H[static_cast<std::size_t>(l + 1)].ensure(B, fo);
        if (l + 1 < L) {
            for (std::size_t i = 0; i < Z[static_cast<std::size_t>(l)].n(); ++i) {
                const float z = Z[static_cast<std::size_t>(l)].data[i];
                H[static_cast<std::size_t>(l + 1)].data[i] = z > 0.f ? z : 0.f;
            }
        } else {
            H[static_cast<std::size_t>(l + 1)] = Z[static_cast<std::size_t>(l)];
        }
    }

    Matrix E;
    output_grad(H[static_cast<std::size_t>(L)], Y, E);
    Wnew.resize(static_cast<std::size_t>(L));
    for (int l = 0; l < L; ++l) {
        const int fi = W0[static_cast<std::size_t>(l)].rows;
        const int fo = W0[static_cast<std::size_t>(l)].cols;
        Matrix delta(B, fo);
        if (l == L - 1) {
            delta = E;
        } else {
            for (int b = 0; b < B; ++b) {
                for (int j = 0; j < fo; ++j) {
                    double s = 0.0;
                    for (int k = 0; k < E.cols; ++k) {
                        s += static_cast<double>(E(b, k)) * P[static_cast<std::size_t>(l)](j, k);
                    }
                    const float gate = Z[static_cast<std::size_t>(l)](b, j) > 0.f ? 1.f : 0.f;
                    delta(b, j) = static_cast<float>(s) * gate;
                }
            }
        }
        Wnew[static_cast<std::size_t>(l)] = W0[static_cast<std::size_t>(l)];
        const std::size_t n_in = static_cast<std::size_t>(B) * static_cast<std::size_t>(fi);
        std::vector<std::int8_t> q(n_in);
        std::vector<float> xhat(n_in);
        float qscale = 1.f;
        quantize_int8(H[static_cast<std::size_t>(l)].data, q.data(), n_in, qscale);
        dequantize_int8(q.data(), n_in, qscale, xhat.data());
        for (int i = 0; i < fi; ++i) {
            for (int j = 0; j < fo; ++j) {
                double s = 0.0;
                for (int b = 0; b < B; ++b) {
                    s += static_cast<double>(xhat[static_cast<std::size_t>(b) * static_cast<std::size_t>(fi) + i]) *
                         delta(b, j);
                }
                Wnew[static_cast<std::size_t>(l)](i, j) -= lr * static_cast<float>(s);
            }
        }
    }
}

void shake_self_test();

void self_test() {
    std::cout << "self-test  kernel=" << kernel_name() << "  threads=" << thread_count() << "\n";
    std::mt19937 rng(123);
    const std::size_t live0 = g_mem.live;

    check_gemm_case(rng, 1, 1, 1, false, false, 1.f, 0.f);
    check_gemm_case(rng, 7, 5, 3, false, false, 1.f, 0.f);
    check_gemm_case(rng, 8, 16, 12, false, false, -0.5f, 0.25f);
    check_gemm_case(rng, 64, 32, 48, false, false, 1.f, 0.f);
    check_gemm_case(rng, 13, 17, 9, true, false, 1.f, 0.f);
    check_gemm_case(rng, 15, 11, 14, false, true, 0.5f, 1.f);
    check_gemm_case(rng, 6, 8, 5, true, true, 1.f, 0.f);
    check_gemm_case(rng, 128, 64, 96, true, false, 1.f, 0.f);
    check_gemm_case(rng, 96, 40, 80, false, true, -1.f, 0.f);
    std::cout << "[ok] gemm NN/TN/NT/TT\n";

    {
        Matrix A(5, 7);
        fill_uniform(A, rng, -1.f, 1.f);
        Matrix At, Att;
        transpose(A, At);
        transpose(At, Att);
        expect(max_abs_diff(A, Att) == 0.f, "transpose roundtrip");
        Matrix P(32, 4);
        fill_normal(P, rng, 1.f);
        orthonormal_columns(P);
        expect(gram_error(P) < 1e-4f, "orthonormal columns");
    }
    std::cout << "[ok] transpose + orthonormal columns\n";

    {
        std::vector<float> z(64);
        for (std::size_t i = 0; i < z.size(); ++i) {
            const int k = static_cast<int>(i) % 7;
            z[i] = (k == 0) ? 0.f : (k == 1) ? -0.f : (k < 4) ? -0.3f * k : 0.15f * k;
        }
        std::vector<std::uint32_t> bits(2);
        pack_relu_mask(z.data(), bits.data(), z.size());
        for (std::size_t i = 0; i < z.size(); ++i) {
            expect(relu_mask_bit(bits.data(), i) == (z[i] > 0.f), "relu mask bit");
        }
        std::vector<float> tail(20, -1.f);
        tail[0] = 0.f;
        tail[3] = 0.5f;
        tail[19] = 1.f;
        std::vector<std::uint32_t> tail_bits(1);
        pack_relu_mask(tail.data(), tail_bits.data(), tail.size());
        for (std::size_t i = 0; i < tail.size(); ++i) {
            expect(relu_mask_bit(tail_bits.data(), i) == (tail[i] > 0.f), "relu mask tail");
        }

        std::vector<float> x = {0.f, 1.f, -1.f, 0.5f, 3.3f, -2.4f, 127.f};
        std::vector<std::int8_t> q(x.size());
        std::vector<float> y(x.size());
        float qscale = 0.f;
        quantize_int8(x.data(), q.data(), x.size(), qscale);
        dequantize_int8(q.data(), q.size(), qscale, y.data());
        for (std::size_t i = 0; i < x.size(); ++i) {
            expect(std::fabs(x[i] - y[i]) <= qscale * 0.5f + 1e-5f, "int8 quant error");
        }

        Matrix E(6, 3);
        Matrix P0(5, 3);
        Matrix P1(4, 3);
        fill_uniform(E, rng, -1.f, 1.f);
        fill_uniform(P0, rng, -1.f, 1.f);
        fill_uniform(P1, rng, -1.f, 1.f);
        Matrix stacked(9, 3);
        for (int r = 0; r < 5; ++r)
            std::memcpy(stacked.data + static_cast<std::size_t>(r) * 3, P0.data + static_cast<std::size_t>(r) * 3,
                        3 * sizeof(float));
        for (int r = 0; r < 4; ++r)
            std::memcpy(stacked.data + static_cast<std::size_t>(5 + r) * 3, P1.data + static_cast<std::size_t>(r) * 3,
                        3 * sizeof(float));
        Matrix Dall(6, 9);
        Matrix D0(6, 5);
        Matrix D1(6, 4);
        gemm(E, stacked, Dall, false, true, 1.f, 0.f);
        gemm(E, P0, D0, false, true, 1.f, 0.f);
        gemm(E, P1, D1, false, true, 1.f, 0.f);
        float split = 0.f;
        for (int b = 0; b < 6; ++b) {
            for (int j = 0; j < 5; ++j) split = std::max(split, std::fabs(Dall(b, j) - D0(b, j)));
            for (int j = 0; j < 4; ++j) split = std::max(split, std::fabs(Dall(b, 5 + j) - D1(b, j)));
        }
        expect(split < 1e-5f, "stacked feedback gemm mismatch " + std::to_string(split));
    }
    std::cout << "[ok] 1-bit relu mask, int8 roundtrip, stacked projection\n";

    {
        float spiked[4] = {100.f, 0.2f, 0.2f, -1.f};
        float crest = 0.f;
        float active_zero = 0.f;
        tape_stats(spiked, 4, crest, active_zero);
        expect(crest > 1.5f && crest < 2.5f, "tape crest");
        expect(std::fabs(active_zero - (2.f / 3.f)) < 1e-6f, "tape active zero");
        std::cout << "[ok] tape crest and active-zero\n";
    }

    {
        // Linear map: finite-difference gradient must match backprop.
        TrainerA lin;
        lin.net = init_net({4, 3}, 11);
        constexpr int batch = 5;
        Matrix X(batch, 4);
        Matrix Y(batch, 3);
        fill_uniform(X, rng, -1.f, 1.f);
        fill_uniform(Y, rng, -1.f, 1.f);
        lin.prepare(batch);
        lin.forward(X);
        std::vector<Matrix> dW(1);
        lin.backward(Y, 0.f, false, &dW);
        const float eps = 1e-3f;
        float worst = 0.f;
        for (std::size_t i = 0; i < lin.net.W[0].n(); ++i) {
            const float orig = lin.net.W[0].data[i];
            lin.net.W[0].data[i] = orig + eps;
            lin.forward(X);
            const float lp = mean_square(lin.H.back(), Y);
            lin.net.W[0].data[i] = orig - eps;
            lin.forward(X);
            const float lm = mean_square(lin.H.back(), Y);
            lin.net.W[0].data[i] = orig;
            const float num = (lp - lm) / (2.f * eps);
            worst = std::max(worst, std::fabs(num - dW[0].data[i]));
        }
        expect(worst < 2e-3f, "linear finite-difference mismatch " + std::to_string(worst));
    }
    std::cout << "[ok] backprop finite difference (linear)\n";

    {
        // Two residual blocks plus a linear head. Pre-activations stay away from zero.
        TrainerA res;
        res.net = init_net({2, 2, 2, 1}, 5);
        res.net.residual = true;
        res.net.W[0].data[0] = 0.4f;
        res.net.W[0].data[1] = 0.3f;
        res.net.W[0].data[2] = 0.2f;
        res.net.W[0].data[3] = 0.5f;
        res.net.W[1].data[0] = 0.3f;
        res.net.W[1].data[1] = -0.2f;
        res.net.W[1].data[2] = 0.4f;
        res.net.W[1].data[3] = 0.25f;
        res.net.W[2].data[0] = 0.3f;
        res.net.W[2].data[1] = 0.4f;
        constexpr int batch = 1;
        Matrix X(batch, 2);
        Matrix Y(batch, 1);
        X.data[0] = 1.f;
        X.data[1] = 1.f;
        Y.data[0] = 0.f;
        res.prepare(batch);
        res.forward(X);
        std::vector<Matrix> dW(3);
        res.backward(Y, 0.f, false, &dW);
        const float eps = 1e-3f;
        float worst = 0.f;
        for (int layer = 0; layer < 3; ++layer) {
            for (std::size_t i = 0; i < res.net.W[static_cast<std::size_t>(layer)].n(); ++i) {
                float& w = res.net.W[static_cast<std::size_t>(layer)].data[i];
                const float orig = w;
                w = orig + eps;
                res.forward(X);
                const float lp = mean_square(res.H.back(), Y);
                w = orig - eps;
                res.forward(X);
                const float lm = mean_square(res.H.back(), Y);
                w = orig;
                const float num = (lp - lm) / (2.f * eps);
                worst = std::max(worst, std::fabs(num - dW[static_cast<std::size_t>(layer)].data[i]));
            }
        }
        expect(worst < 2e-3f, "residual finite-difference mismatch " + std::to_string(worst));
    }
    std::cout << "[ok] backprop finite difference (residual)\n";

    {
        Matrix Z(1, 2);
        Z(0, 0) = 0.f;
        Z(0, 1) = 0.f;
        Matrix Y(1, 2);
        Y(0, 0) = 1.f;
        Y(0, 1) = 0.f;
        Matrix G;
        const float ce = softmax_ce(Z, Y, &G);
        expect(std::fabs(ce - std::log(2.f)) < 1e-5f, "softmax loss");
        expect(std::fabs(G(0, 0) + 0.5f) < 1e-5f && std::fabs(G(0, 1) - 0.5f) < 1e-5f, "softmax grad");

        TrainerA cls;
        cls.classify = true;
        cls.net = init_net({3, 2}, 9);
        constexpr int batch = 4;
        cls.prepare(batch);
        Matrix X(batch, 3);
        Matrix Yb(batch, 2);
        fill_uniform(X, rng, -1.f, 1.f);
        for (int b = 0; b < batch; ++b) {
            Yb(b, 0) = (b % 2 == 0) ? 1.f : 0.f;
            Yb(b, 1) = 1.f - Yb(b, 0);
        }
        cls.forward(X);
        std::vector<Matrix> dW(1);
        cls.backward(Yb, 0.f, false, &dW);
        const float eps = 1e-3f;
        float worst = 0.f;
        for (std::size_t i = 0; i < cls.net.W[0].n(); ++i) {
            float& w = cls.net.W[0].data[i];
            const float orig = w;
            w = orig + eps;
            cls.forward(X);
            const float lp = softmax_ce(cls.H.back(), Yb, nullptr);
            w = orig - eps;
            cls.forward(X);
            const float lm = softmax_ce(cls.H.back(), Yb, nullptr);
            w = orig;
            worst = std::max(worst, std::fabs((lp - lm) / (2.f * eps) - dW[0].data[i]));
        }
        expect(worst < 2e-3f, "cross-entropy finite-difference " + std::to_string(worst));
    }
    std::cout << "[ok] softmax cross-entropy gradient\n";

    {
        // Two hidden widths, one square. Naive DFA is the definition of B.
        // A's last layer is ordinary backprop, so it must match B's last layer.
        const std::vector<int> dims = make_dims(3, 5, 2, 1);
        const float lr = 0.07f;
        constexpr int batch = 4;
        Matrix X(batch, 3);
        Matrix Y(batch, 2);
        fill_uniform(X, rng, -1.5f, 1.5f);
        Synth synth;
        synth.init(3, 2, 99);
        std::mt19937 data_rng(5);
        synth.sample(X, Y, data_rng);

        TrainerA A;
        TrainerB B;
        A.net = init_net(dims, 21);
        B.net = A.net;
        B.init_feedback(77);
        A.prepare(batch);
        B.prepare(batch);
        std::vector<Matrix> W0 = A.net.W;
        std::vector<Matrix> expected;
        naive_dfa(W0, B.P, X, Y, lr, expected);
        A.step(X, Y, lr);
        B.step(X, Y, lr);
        float worst = 0.f;
        for (int l = 0; l < n_layers(B.net); ++l) {
            worst = std::max(worst, max_abs_diff(B.net.W[static_cast<std::size_t>(l)],
                                                 expected[static_cast<std::size_t>(l)]));
        }
        expect(worst < 2e-3f, "quantized DFA mismatch " + std::to_string(worst));
        const float hidden = max_abs_diff(A.net.W[0], B.net.W[0]);
        expect(hidden > 1e-5f, "A and B hidden updates unexpectedly identical");
    }
    std::cout << "[ok] int8 DFA matches the direct formula\n";

    {
        Matrix src(6, 3);
        src.zero();
        src(0, 0) = 3.f;
        src(1, 1) = 1.f;
        src(0, 2) = 0.5f;
        src(1, 2) = 0.5f;
        src(2, 2) = 2.f;
        std::vector<double> s;
        double ss = 0.0;
        column_energy(src, s, ss);
        Matrix Sp(6, 3);
        spectral_assign(Sp, src);
        expect(column_ortho_error(Sp) < 1e-4f, "spectral columns not orthogonal");
        const float gain = std::sqrt(3.f);
        const double inv = 1.0 / std::sqrt(ss);
        for (int j = 0; j < 3; ++j) {
            double n = 0.0;
            for (int r = 0; r < Sp.rows; ++r) n += static_cast<double>(Sp(r, j)) * Sp(r, j);
            const double want = s[static_cast<std::size_t>(j)] * inv * gain;
            expect(want > 1e-8 && std::fabs(std::sqrt(n) - want) / want < 1e-4, "spectral column scale");
        }
    }
    std::cout << "[ok] spectral scales keep orthogonal columns\n";

    {
        auto check_global = [](const std::vector<Matrix>& before, const std::vector<Matrix>& full,
                               const std::vector<Matrix>& clipped, float max_norm, int hits) {
            double nrss = 0.0;
            double ncss = 0.0;
            for (int l = 0; l < static_cast<int>(before.size()); ++l) {
                const Matrix& oldW = before[static_cast<std::size_t>(l)];
                const Matrix& Wr = full[static_cast<std::size_t>(l)];
                const Matrix& Wc = clipped[static_cast<std::size_t>(l)];
                double nr = 0.0;
                double nc = 0.0;
                double dot = 0.0;
                for (std::size_t i = 0; i < oldW.n(); ++i) {
                    const double dr = static_cast<double>(Wr.data[i]) - oldW.data[i];
                    const double dc = static_cast<double>(Wc.data[i]) - oldW.data[i];
                    nr += dr * dr;
                    nc += dc * dc;
                    dot += dr * dc;
                }
                nr = std::sqrt(nr);
                nc = std::sqrt(nc);
                nrss += nr * nr;
                ncss += nc * nc;
                expect(nc > 0.0 && nr > 0.0 && dot / (nr * nc) > 0.99, "global clip changed a layer direction");
            }
            const double nraw = std::sqrt(nrss);
            const double ncut = std::sqrt(ncss);
            expect(nraw > max_norm * 2.0, "global clip test did not exceed max-norm");
            expect(std::fabs(ncut - max_norm) / max_norm < 1e-3, "global clip missed max-norm");
            expect(hits == 1, "global clip never fired");
        };
        Matrix Xt(4, 3);
        Matrix Yt(4, 2);
        fill_uniform(Xt, rng, -1.f, 1.f);
        fill_uniform(Yt, rng, -1.f, 1.f);
        {
            TrainerA raw;
            TrainerA cut;
            raw.net = init_net(make_dims(3, 5, 2, 0), 3);
            cut.net = raw.net;
            raw.prepare(4);
            cut.prepare(4);
            cut.clip = true;
            cut.max_norm = 1e-3f;
            const std::vector<Matrix> before = raw.net.W;
            raw.step(Xt, Yt, 1.f);
            cut.step(Xt, Yt, 1.f);
            check_global(before, raw.net.W, cut.net.W, cut.max_norm, cut.clip_hits);
        }
        {
            TrainerB raw;
            TrainerB cut;
            raw.net = init_net(make_dims(3, 5, 2, 0), 3);
            cut.net = raw.net;
            raw.init_feedback(1);
            cut.P = raw.P;
            raw.prepare(4);
            cut.prepare(4);
            cut.clip = true;
            cut.max_norm = 1e-3f;
            const std::vector<Matrix> before = raw.net.W;
            raw.step(Xt, Yt, 1.f);
            cut.step(Xt, Yt, 1.f);
            check_global(before, raw.net.W, cut.net.W, cut.max_norm, cut.clip_hits);
        }
    }
    std::cout << "[ok] global grad clip scales every layer by one factor\n";

    {
        // Shallow net must reduce probe error. This locks the training loop, not the deep run.
        Synth synth;
        synth.init(2, 2, 3);
        std::mt19937 data_rng(4);
        TrainerA A;
        A.net = init_net(make_dims(2, 32, 2, 0), 8);
        A.prepare(64);
        Matrix probeX(256, 2);
        Matrix probeY(256, 2);
        synth.sample(probeX, probeY, data_rng);
        Matrix a, b, xs, ys;
        const float before = eval_mse(A.net, probeX, probeY, 64, a, b, xs, ys);
        Matrix X(64, 2);
        Matrix Y(64, 2);
        for (int step = 0; step < 200; ++step) {
            synth.sample(X, Y, data_rng);
            A.step(X, Y, 0.2f);
        }
        const float after = eval_mse(A.net, probeX, probeY, 64, a, b, xs, ys);
        std::cout << "[info] shallow backprop probe " << before << " -> " << after << "\n";
        expect(std::isfinite(after) && after < before * 0.8f, "shallow backprop did not learn");

        TrainerB Bnet;
        Bnet.net = init_net(make_dims(2, 32, 2, 0), 8);
        Bnet.init_feedback(19);
        Bnet.prepare(64);
        data_rng.seed(4);
        synth.sample(probeX, probeY, data_rng);
        const float before_b = eval_mse(Bnet.net, probeX, probeY, 64, a, b, xs, ys);
        for (int step = 0; step < 200; ++step) {
            synth.sample(X, Y, data_rng);
            Bnet.step(X, Y, 0.05f);
        }
        const float after_b = eval_mse(Bnet.net, probeX, probeY, 64, a, b, xs, ys);
        std::cout << "[info] shallow DFA probe " << before_b << " -> " << after_b << "\n";
        expect(std::isfinite(after_b) && after_b < before_b * 0.85f, "shallow DFA did not learn");
    }
    std::cout << "[ok] shallow backprop and DFA reduce probe loss\n";

    shake_self_test();
    expect(g_mem.live == live0, "self-test leaked allocations");
    std::cout << "[ok] no allocation leak\n";
}

void print_usage() {
    std::cout << "minicore [--steps N] [--batch N] [--hidden N] [--squares N]\n"
              << "         [--lr F] [--every N] [--seed N] [--in N] [--out N] [--probe N]\n"
              << "         [--align cov|jac|off] [--no-clip] [--clip-gamma F] [--max-norm F] [--cos-every N] [--lambda F]\n"
              << "         [--residual] [--seconds F] [--data DIR] [--epochs N] [--lm FILE] [--align-k N] [--beta F] [--mgs-k N] [--rank N]\n"
              << "         [--float-strip] [--diag] [--shakespeare FILE] [--trio] [--adam] [--phase2] [--scale] [--lora] [--qlora] [--qsuite 1|2|3|4|5|6|7] [--wd F]\n"
              << "defaults: steps=300 batch=128 hidden=512 squares=10 lr=0.002 every=100\n"
              << "          seed=1 in=10 out=10 probe=256 align=jac clip=on gamma=0.05 cos-every=50 lambda=0.10\n";
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    auto need = [&](int& i, const char* flag) {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            print_usage();
            std::exit(0);
        } else if (a == "--steps") cfg.steps = std::stoi(need(i, "--steps"));
        else if (a == "--batch") cfg.batch = std::stoi(need(i, "--batch"));
        else if (a == "--hidden") cfg.hidden = std::stoi(need(i, "--hidden"));
        else if (a == "--squares") cfg.squares = std::stoi(need(i, "--squares"));
        else if (a == "--lr") cfg.lr = std::stof(need(i, "--lr"));
        else if (a == "--every") cfg.every = std::stoi(need(i, "--every"));
        else if (a == "--seed") cfg.seed = static_cast<std::uint32_t>(std::stoul(need(i, "--seed")));
        else if (a == "--in") cfg.in_dim = std::stoi(need(i, "--in"));
        else if (a == "--out") cfg.out_dim = std::stoi(need(i, "--out"));
        else if (a == "--probe") cfg.probe = std::stoi(need(i, "--probe"));
        else if (a == "--cos-every") cfg.cos_every = std::stoi(need(i, "--cos-every"));
        else if (a == "--align") {
            const std::string v = need(i, "--align");
            if (v == "cov") cfg.align = 1;
            else if (v == "jac") cfg.align = 2;
            else if (v == "off") cfg.align = 0;
            else throw std::runtime_error("align must be cov, jac, or off");
        } else if (a == "--clip-gamma") cfg.clip_gamma = std::stof(need(i, "--clip-gamma"));
        else if (a == "--no-clip" || a == "--no-trust") cfg.clip = false;
        else if (a == "--lambda") cfg.lambda_peak = std::stof(need(i, "--lambda"));
        else if (a == "--residual") cfg.residual = true;
        else if (a == "--seconds") cfg.seconds = std::stod(need(i, "--seconds"));
        else if (a == "--data") cfg.data = need(i, "--data");
        else if (a == "--lm") cfg.lm = need(i, "--lm");
        else if (a == "--shakespeare") cfg.shake = need(i, "--shakespeare");
        else if (a == "--trio") cfg.trio = true;
        else if (a == "--adam") cfg.adam = true;
        else if (a == "--phase2") cfg.phase2 = true;
        else if (a == "--scale") cfg.scale = true;
        else if (a == "--lora") cfg.lora = true;
        else if (a == "--qlora") cfg.qlora = true;
        else if (a == "--qsuite") cfg.qsuite = std::stoi(need(i, "--qsuite"));
        else if (a == "--wd") cfg.wd = std::stof(need(i, "--wd"));
        else if (a == "--align-k") cfg.align_k = std::stoi(need(i, "--align-k"));
        else if (a == "--k2") cfg.k2 = std::stoi(need(i, "--k2"));
        else if (a == "--beta") cfg.beta = std::stof(need(i, "--beta"));
        else if (a == "--mgs-k") cfg.mgs_k = std::stoi(need(i, "--mgs-k"));
        else if (a == "--max-norm") cfg.max_norm = std::stof(need(i, "--max-norm"));
        else if (a == "--rank") cfg.rank = std::stoi(need(i, "--rank"));
        else if (a == "--float-strip") cfg.float_strip = true;
        else if (a == "--diag") cfg.diag = true;
        else if (a == "--epochs") cfg.epochs = std::stoi(need(i, "--epochs"));
        else throw std::runtime_error("unknown argument " + a);
    }
    expect(cfg.steps >= 0 && cfg.batch > 0 && cfg.every > 0 && cfg.probe > 0 && cfg.seconds >= 0.0 &&
               cfg.epochs > 0 && cfg.rank >= 0 && cfg.beta <= 1.f && cfg.max_norm > 0.f,
           "bad numeric argument");
    return cfg;
}

void report_line(int step, float loss_a, double us_a, double mem_a, float loss_b, double us_b,
                 double mem_b) {
    char line[320];
    std::snprintf(line, sizeof(line),
                  "step %4d | A loss %8.5f  time %10.1f us  mem %7.3f MB | "
                  "B loss %8.5f  time %10.1f us  mem %7.3f MB | "
                  "B/A loss %5.3f  time %5.3f  mem %5.3f\n",
                  step, loss_a, us_a, mem_a, loss_b, us_b, mem_b,
                  loss_a == 0.f ? 0.f : loss_b / loss_a, us_a == 0.0 ? 0.0 : us_b / us_a,
                  mem_a == 0.0 ? 0.0 : mem_b / mem_a);
    std::cout << line << std::flush;
}

void run_experiment(const Config& cfg) {
    const std::vector<int> dims = make_dims(cfg.in_dim, cfg.hidden, cfg.out_dim, cfg.squares);
    TrainerA A;
    TrainerB B;
    A.net = init_net(dims, cfg.seed);
    A.net.residual = cfg.residual;
    B.net = A.net;
    A.clip = cfg.clip;
    A.clip_gamma = cfg.clip_gamma;
    B.clip = cfg.clip;
    B.clip_gamma = cfg.clip_gamma;
    B.align = cfg.align == 2 ? Align::Jacobian : cfg.align == 1 ? Align::Covariance : Align::Off;
    B.lambda_peak = cfg.lambda_peak;
    if (cfg.align_k > 0) B.align_k = cfg.align_k;
    if (cfg.beta > 0.f) {
        B.align_beta = cfg.beta;
        B.align_beta_fast = cfg.beta;
    }
    if (cfg.mgs_k != 0) {
        B.mgs_k = cfg.mgs_k;
        B.mgs_now = cfg.mgs_k > 0;
    }
    A.max_norm = cfg.max_norm;
    B.max_norm = cfg.max_norm;
    B.rank = cfg.rank;
    B.init_feedback(cfg.seed + 10007u);
    A.prepare(cfg.batch);
    B.prepare(cfg.batch);

    for (int l = 0; l < n_layers(B.net) - 1; ++l) {
        const float nP = frobenius(B.P[static_cast<std::size_t>(l)]);
        expect(std::isfinite(nP) && nP > 1e-4f, "feedback norm collapsed");
        if (B.align != Align::Jacobian) {
            expect(column_ortho_error(B.P[static_cast<std::size_t>(l)]) < 1e-3f, "feedback columns not orthogonal");
        }
    }

    Synth synth;
    synth.init(cfg.in_dim, cfg.out_dim, cfg.seed + 17u);
    std::mt19937 data_rng(cfg.seed + 99u);
    std::mt19937 probe_rng(cfg.seed + 1234u);

    Matrix X(cfg.batch, cfg.in_dim);
    Matrix Y(cfg.batch, cfg.out_dim);
    Matrix probeX(cfg.probe, cfg.in_dim);
    Matrix probeY(cfg.probe, cfg.out_dim);
    synth.sample(probeX, probeY, probe_rng);

    Matrix ev_a, ev_b, ev_x, ev_y;
    const float loss0_a = eval_mse(A.net, probeX, probeY, cfg.batch, ev_a, ev_b, ev_x, ev_y);
    const float loss0_b = eval_mse(B.net, probeX, probeY, cfg.batch, ev_a, ev_b, ev_x, ev_y);
    expect(std::fabs(loss0_a - loss0_b) < 1e-5f, "A and B diverged before training");

    const std::size_t a_w = bytes_of(A.net.W);
    const std::size_t a_h = bytes_of(A.H);
    const std::size_t a_g = A.G.bytes() + A.Gprev.bytes();
    const std::size_t b_w = bytes_of(B.net.W);
    const std::size_t b_p = bytes_of(B.P);
    const std::size_t b_u = bytes_of(B.U) + bytes_of(B.aux_err);
    const std::size_t b_x = B.xq.bytes;
    const std::size_t b_m = B.mask.bytes;
    const std::size_t b_d = B.delta.bytes();
    const std::size_t b_dw = B.dW.bytes();
    const std::size_t b_s = B.buf0.bytes() + B.buf1.bytes() + B.err.bytes() + B.pred.bytes();
    const int L = n_layers(A.net);
    const double gemm_a = static_cast<double>(3 * L - 1);
    int n_aux = 0;
    for (const Matrix& u : B.U)
        if (u.n() > 0) ++n_aux;
    const double gemm_b = static_cast<double>(3 * L - 1 + 3 * n_aux);

    std::cout << "\nminicore\n";
    std::cout << "target     y = sin(a.x) + cos(b.x)   x ~ N(0,I)  in=" << cfg.in_dim
              << " out=" << cfg.out_dim << "\n";
    std::cout << "kernel  " << kernel_name() << "   omp threads=" << thread_count() << "\n";
    std::cout << "arch    " << arch_string(A.net) << "\n";
    std::cout << "layer    " << L << " weight matrices, " << count_square(A.net, cfg.hidden)
              << " of them " << cfg.hidden << "x" << cfg.hidden << "\n";
    const char* align_name = B.align == Align::Jacobian ? "jacobian" : B.align == Align::Covariance ? "covariance" : "fixed";
    std::cout << "A         backprop: float activation strip, chain rule"
              << (cfg.residual ? ", residual h <- 0.876 (h + 0.316 ReLU(h W))" : "")
              << ", clip=" << (A.clip ? "on" : "off") << " gamma=" << A.clip_gamma << "\n";
    std::cout << "B         SAP+U: MGS, auxiliary heads on layers 2 and 4, lambda " << B.lambda_peak
              << ", beta "
              << B.align_beta_fast << " (first " << B.align_fast_until << " steps) then " << B.align_beta
              << ", P align=" << align_name << " every " << B.align_k << " steps, clip="
              << (B.clip ? "on" : "off") << " gamma=" << B.clip_gamma << "\n";
    std::cout << "measure     loss = fixed probe MSE (" << cfg.probe
              << " samples)   time = training step   mem = this arm's own buffers\n";
    std::cout << "          cos = hybrid delta against delta_BP, layers 1..N, not included in the time\n";
    std::cout << "A memory  " << mb(A.bytes()) << " MB  (weight " << mb(a_w) << " + activation " << mb(a_h)
              << " + gradient buffer " << mb(a_g);
    if (cfg.residual) {
        std::cout << " + residual " << mb(A.Dz.bytes() + A.mask_words.size() * sizeof(std::uint32_t));
    }
    std::cout << ")\n";
    std::cout << "B memory  " << mb(B.bytes()) << " MB  (weight " << mb(b_w) << " + P " << mb(b_p)
              << " + U " << mb(b_u) << " + int8 xi " << mb(b_x)
              << " + bit mask " << (static_cast<double>(b_m) / 1024.0)
              << " KB + delta " << mb(b_d) << " + dW " << mb(b_dw) << " + scratch " << mb(b_s) << ")\n";
    std::cout << "gemm/step A ~ " << gemm_a << "   B ~ " << gemm_b
              << "   (B: " << L << " fwd + " << (L - 1) << " SAP + " << n_aux
              << " local read + " << n_aux << " U mix + " << n_aux
              << " U update + " << L << " W update)\n";
    std::cout << "lr=" << cfg.lr << "  batch=" << cfg.batch;
    if (cfg.seconds > 0.0) std::cout << "  wall=" << cfg.seconds << " s";
    else std::cout << "  steps=" << cfg.steps;
    std::cout << "\n\n";

    report_line(0, loss0_a, 0.0, mb(A.bytes()), loss0_b, 0.0, mb(B.bytes()));
    if (cfg.cos_every > 0 && L > 1) {
        std::cout << "cos columns: layer 1.." << (L - 1) << " (1 = near the input, " << (L - 1)
                  << " = near the output)\n";
        B.report_cosine(probeX, probeY, 0);
    }
    if (cfg.steps == 0 && cfg.seconds <= 0.0) return;

    double acc_us_a = 0.0;
    double acc_us_b = 0.0;
    int win_a = 0;
    int win_b = 0;
    bool a_alive = true;
    bool b_alive = true;
    using clock = std::chrono::steady_clock;
    const bool by_time = cfg.seconds > 0.0;
    const double budget_us = cfg.seconds * 1.0e6;
    double used_a = 0.0;
    double used_b = 0.0;
    int steps_a = 0;
    int steps_b = 0;

    for (int step = 1;; ++step) {
        if (by_time) {
            if ((!a_alive || used_a >= budget_us) && (!b_alive || used_b >= budget_us)) break;
        } else if (step > cfg.steps) {
            break;
        }
        synth.sample(X, Y, data_rng);

        const bool run_a = a_alive && (!by_time || used_a < budget_us);
        const bool run_b = b_alive && (!by_time || used_b < budget_us);
        if (run_a) {
            const auto ta0 = clock::now();
            A.step(X, Y, cfg.lr);
            const auto ta1 = clock::now();
            const double dt = std::chrono::duration<double, std::micro>(ta1 - ta0).count();
            acc_us_a += dt;
            used_a += dt;
            ++steps_a;
        }
        if (run_b) {
            const auto tb0 = clock::now();
            B.step(X, Y, cfg.lr);
            const auto tb1 = clock::now();
            const double dt = std::chrono::duration<double, std::micro>(tb1 - tb0).count();
            acc_us_b += dt;
            used_b += dt;
            ++steps_b;
        }
        if (run_a) ++win_a;
        if (run_b) ++win_b;
        if (run_b && cfg.cos_every > 0 && steps_b % cfg.cos_every == 0) B.report_cosine(X, Y, steps_b);

        if (step % cfg.every == 0) {
            const float la = a_alive ? eval_mse(A.net, probeX, probeY, cfg.batch, ev_a, ev_b, ev_x, ev_y)
                                     : std::numeric_limits<float>::quiet_NaN();
            const float lb = b_alive ? eval_mse(B.net, probeX, probeY, cfg.batch, ev_a, ev_b, ev_x, ev_y)
                                     : std::numeric_limits<float>::quiet_NaN();
            if (b_alive && !std::isfinite(lb)) {
                std::cout << "B non-finite probe " << lb << " at step " << step << "\n";
                b_alive = false;
                if (!by_time) break;
            }
            if (a_alive && !std::isfinite(la)) {
                std::cout << "A non-finite probe " << la << " at step " << step << ", B continues\n";
                a_alive = false;
            }
            report_line(step, la, win_a > 0 ? acc_us_a / win_a : 0.0, mb(A.bytes()), lb,
                        win_b > 0 ? acc_us_b / win_b : 0.0, mb(B.bytes()));
            acc_us_a = 0.0;
            acc_us_b = 0.0;
            win_a = 0;
            win_b = 0;
        }
    }

    if (by_time) {
        const float la = a_alive ? eval_mse(A.net, probeX, probeY, cfg.batch, ev_a, ev_b, ev_x, ev_y)
                                 : std::numeric_limits<float>::quiet_NaN();
        const float lb = b_alive ? eval_mse(B.net, probeX, probeY, cfg.batch, ev_a, ev_b, ev_x, ev_y)
                                 : std::numeric_limits<float>::quiet_NaN();
        char line[240];
        std::snprintf(line, sizeof(line),
                      "wall A step %d time %.3f s loss %.5f | B step %d time %.3f s loss %.5f\n"
                      "wall note: time is the sum of step times, excluding data generation and cosine\n",
                      steps_a, used_a / 1.0e6, la, steps_b, used_b / 1.0e6, lb);
        std::cout << line << std::flush;
    }

    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        std::cout << "process peak working set " << mb(pmc.PeakWorkingSetSize)
                  << " MB  (A+B+probe together, OS measurement)\n";
    }
    std::cout << "allocator live " << mb(g_mem.live) << " MB   allocator peak " << mb(g_mem.peak) << " MB\n";
    std::cout << "clip fires A " << A.clip_hits << "  B " << B.clip_hits << "\n";
}

struct IdxSet {
    int n = 0;
    int dim = 0;
    int classes = 10;
    std::vector<float> x;
    std::vector<std::uint8_t> y;
};

std::uint32_t read_u32be(std::ifstream& in) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    expect(static_cast<bool>(in), "idx header truncated");
    return (static_cast<std::uint32_t>(b[0]) << 24) | (static_cast<std::uint32_t>(b[1]) << 16) |
           (static_cast<std::uint32_t>(b[2]) << 8) | b[3];
}

IdxSet load_idx(const std::string& dir, const char* images, const char* labels) {
    std::ifstream img(dir + "\\" + images, std::ios::binary);
    std::ifstream lab(dir + "\\" + labels, std::ios::binary);
    expect(static_cast<bool>(img) && static_cast<bool>(lab), "idx open failed in " + dir);
    const std::uint32_t magic_i = read_u32be(img);
    const std::uint32_t n_i = read_u32be(img);
    const std::uint32_t rows = read_u32be(img);
    const std::uint32_t cols = read_u32be(img);
    const std::uint32_t magic_l = read_u32be(lab);
    const std::uint32_t n_l = read_u32be(lab);
    expect(magic_i == 2051u && magic_l == 2049u, "idx magic");
    expect(n_i == n_l && n_i > 0 && rows * cols == 784u, "idx shape");
    IdxSet d;
    d.n = static_cast<int>(n_i);
    d.dim = 784;
    d.x.resize(static_cast<std::size_t>(d.n) * 784u);
    std::vector<unsigned char> raw(d.x.size());
    img.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    expect(static_cast<bool>(img), "idx image body");
    for (std::size_t i = 0; i < raw.size(); ++i) d.x[i] = static_cast<float>(raw[i]) / 255.f;
    d.y.resize(static_cast<std::size_t>(d.n));
    lab.read(reinterpret_cast<char*>(d.y.data()), static_cast<std::streamsize>(d.n));
    expect(static_cast<bool>(lab), "idx label body");
    return d;
}

void class_batch(const IdxSet& data, const std::vector<int>& order, int begin, Matrix& X, Matrix& Y) {
    const int batch = X.rows;
    expect(Y.rows == batch && X.cols == data.dim && Y.cols == data.classes, "class batch");
    std::memset(Y.data, 0, Y.n() * sizeof(float));
    for (int r = 0; r < batch; ++r) {
        const int i = order[static_cast<std::size_t>(begin + r)];
        std::memcpy(X.data + static_cast<std::size_t>(r) * static_cast<std::size_t>(data.dim),
                    data.x.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(data.dim),
                    static_cast<std::size_t>(data.dim) * sizeof(float));
        Y(r, data.y[static_cast<std::size_t>(i)]) = 1.f;
    }
}

void eval_classify(const Net& net, const IdxSet& set, int chunk, Matrix& a, Matrix& b, Matrix& xs, Matrix& ys,
                   Matrix& logits, float& loss, float& acc) {
    double loss_sum = 0.0;
    int correct = 0;
    for (int i = 0; i < set.n; i += chunk) {
        const int bsz = std::min(chunk, set.n - i);
        xs.ensure(bsz, set.dim);
        ys.ensure(bsz, set.classes);
        std::memset(ys.data, 0, ys.n() * sizeof(float));
        for (int r = 0; r < bsz; ++r) {
            std::memcpy(xs.data + static_cast<std::size_t>(r) * static_cast<std::size_t>(set.dim),
                        set.x.data() + static_cast<std::size_t>(i + r) * static_cast<std::size_t>(set.dim),
                        static_cast<std::size_t>(set.dim) * sizeof(float));
            ys(r, set.y[static_cast<std::size_t>(i + r)]) = 1.f;
        }
        forward_logits(net, xs, a, b, logits);
        loss_sum += static_cast<double>(softmax_ce(logits, ys, nullptr)) * bsz;
        for (int r = 0; r < bsz; ++r) {
            int arg = 0;
            float best = logits(r, 0);
            for (int k = 1; k < set.classes; ++k) {
                if (logits(r, k) > best) {
                    best = logits(r, k);
                    arg = k;
                }
            }
            if (arg == static_cast<int>(set.y[static_cast<std::size_t>(i + r)])) ++correct;
        }
    }
    loss = static_cast<float>(loss_sum / set.n);
    acc = 100.f * static_cast<float>(correct) / static_cast<float>(set.n);
}

void run_classify(const Config& cfg) {
    const IdxSet train = load_idx(cfg.data, "train-images-idx3-ubyte", "train-labels-idx1-ubyte");
    const IdxSet test = load_idx(cfg.data, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
    const int steps_per_epoch = train.n / cfg.batch;
    expect(steps_per_epoch > 0, "batch larger than the training set");

    const std::vector<int> dims = make_dims(train.dim, cfg.hidden, train.classes, cfg.squares);
    TrainerA A;
    TrainerB B;
    A.net = init_net(dims, cfg.seed);
    A.net.residual = true;
    B.net = A.net;
    A.classify = true;
    B.classify = true;
    A.clip = cfg.clip;
    A.clip_gamma = cfg.clip_gamma;
    B.clip = cfg.clip;
    B.clip_gamma = cfg.clip_gamma;
    B.align = cfg.align == 2 ? Align::Jacobian : cfg.align == 1 ? Align::Covariance : Align::Off;
    B.lambda_peak = cfg.lambda_peak;
    if (cfg.align_k > 0) B.align_k = cfg.align_k;
    if (cfg.beta > 0.f) {
        B.align_beta = cfg.beta;
        B.align_beta_fast = cfg.beta;
    }
    if (cfg.mgs_k != 0) {
        B.mgs_k = cfg.mgs_k;
        B.mgs_now = cfg.mgs_k > 0;
    }
    A.max_norm = cfg.max_norm;
    B.max_norm = cfg.max_norm;
    B.rank = cfg.rank;
    B.init_feedback(cfg.seed + 10007u);
    A.prepare(cfg.batch);
    B.prepare(cfg.batch);

    std::cout << "\nidx       " << cfg.data << "  train " << train.n << "  test " << test.n
              << "  input " << train.dim << "  classes " << train.classes << "\n";
    std::cout << "arch    " << arch_string(A.net) << "  residual gamma_res=0.876 alpha=0.316\n";
    std::cout << "loss     mean cross-entropy   e = (p - y) / batch\n";
    std::cout << "A memory  " << mb(A.bytes()) << " MB   B memory " << mb(B.bytes())
              << " MB   veri " << mb((train.x.size() + test.x.size()) * sizeof(float)) << " MB\n";
    std::cout << "lambda    " << B.lambda_peak << "  lr=" << cfg.lr << "  batch=" << cfg.batch
              << "  clip gamma=" << cfg.clip_gamma << "\n";

    Matrix X(cfg.batch, train.dim);
    Matrix Y(cfg.batch, train.classes);
    Matrix ev_a, ev_b, ev_x, ev_y, logits;
    std::mt19937 rng(cfg.seed + 99u);
    std::vector<int> order(static_cast<std::size_t>(train.n));
    std::iota(order.begin(), order.end(), 0);
    using clock = std::chrono::steady_clock;

    auto score = [&](const Net& net, float& loss, float& acc) {
        eval_classify(net, test, cfg.batch, ev_a, ev_b, ev_x, ev_y, logits, loss, acc);
    };

    if (cfg.seconds > 0.0) {
        const double budget_us = cfg.seconds * 1.0e6;
        double used_a = 0.0;
        double used_b = 0.0;
        int steps_a = 0;
        int steps_b = 0;
        int cursor = train.n;
        bool a_alive = true;
        bool b_alive = true;
        while ((a_alive && used_a < budget_us) || (b_alive && used_b < budget_us)) {
            if (cursor + cfg.batch > static_cast<int>(order.size())) {
                std::shuffle(order.begin(), order.end(), rng);
                cursor = 0;
            }
            class_batch(train, order, cursor, X, Y);
            cursor += cfg.batch;
            if (a_alive && used_a < budget_us) {
                const auto t0 = clock::now();
                const float la = A.step(X, Y, cfg.lr);
                used_a += std::chrono::duration<double, std::micro>(clock::now() - t0).count();
                ++steps_a;
                if (!std::isfinite(la)) a_alive = false;
            }
            if (b_alive && used_b < budget_us) {
                const auto t0 = clock::now();
                const float lb = B.step(X, Y, cfg.lr);
                used_b += std::chrono::duration<double, std::micro>(clock::now() - t0).count();
                ++steps_b;
                if (!std::isfinite(lb)) b_alive = false;
            }
        }
        float la = 0.f, aa = 0.f, lb = 0.f, ab = 0.f;
        if (a_alive) score(A.net, la, aa);
        if (b_alive) score(B.net, lb, ab);
        char line[320];
        std::snprintf(line, sizeof(line),
                      "wall %.1f s | A step %d time %.3f s test %.2f%% loss %.4f | "
                      "B step %d time %.3f s test %.2f%% loss %.4f\n",
                      cfg.seconds, steps_a, used_a / 1.0e6, aa, la, steps_b, used_b / 1.0e6, ab, lb);
        std::cout << line << "clip A " << A.clip_hits << "  B " << B.clip_hits << "\n";
        return;
    }

    double used_a = 0.0;
    double used_b = 0.0;
    bool a_alive = true;
    bool b_alive = true;
    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        double sum_a = 0.0;
        double sum_b = 0.0;
        int n_a = 0;
        int n_b = 0;
        for (int s = 0; s < steps_per_epoch; ++s) {
            class_batch(train, order, s * cfg.batch, X, Y);
            if (a_alive) {
                const auto t0 = clock::now();
                const float la = A.step(X, Y, cfg.lr);
                used_a += std::chrono::duration<double, std::micro>(clock::now() - t0).count();
                if (!std::isfinite(la)) a_alive = false;
                else {
                    sum_a += la;
                    ++n_a;
                }
            }
            if (b_alive) {
                const auto t0 = clock::now();
                const float lb = B.step(X, Y, cfg.lr);
                used_b += std::chrono::duration<double, std::micro>(clock::now() - t0).count();
                if (!std::isfinite(lb)) b_alive = false;
                else {
                    sum_b += lb;
                    ++n_b;
                }
            }
        }
        float tea = 0.f, aa = 0.f, teb = 0.f, ab = 0.f;
        if (a_alive) score(A.net, tea, aa);
        if (b_alive) score(B.net, teb, ab);
        char line[360];
        std::snprintf(line, sizeof(line),
                      "epoch %d | A train %.4f testloss %.4f acc %6.2f%% time %6.2f s | "
                      "B train %.4f testloss %.4f acc %6.2f%% time %6.2f s\n",
                      epoch, n_a ? sum_a / n_a : std::numeric_limits<float>::quiet_NaN(), tea, aa, used_a / 1.0e6,
                      n_b ? sum_b / n_b : std::numeric_limits<float>::quiet_NaN(), teb, ab, used_b / 1.0e6);
        std::cout << line << std::flush;
    }
    std::cout << "clip A " << A.clip_hits << "  B " << B.clip_hits << "\n";
    std::cout << "memory A " << mb(A.bytes()) << " MB  B " << mb(B.bytes()) << " MB\n";
}

constexpr int k_lm_ctx = 32;
constexpr int k_lm_dim = 16;
constexpr int k_lm_vocab = 256;
constexpr float k_ln2 = 0.69314718056f;

std::vector<std::uint8_t> load_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    expect(static_cast<bool>(in), "lm file open failed: " + path);
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    expect(n >= 100000000, "enwik8 must be 100000000 bytes");
    in.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> text(100000000u);
    in.read(reinterpret_cast<char*>(text.data()), static_cast<std::streamsize>(text.size()));
    expect(static_cast<bool>(in), "enwik8 read failed");
    return text;
}

void embed_gather(const Matrix& E, const std::uint8_t* tok, int batch, Matrix& X) {
    X.ensure(batch, k_lm_ctx * k_lm_dim);
    for (int b = 0; b < batch; ++b) {
        float* row = X.data + static_cast<std::size_t>(b) * static_cast<std::size_t>(X.cols);
        for (int c = 0; c < k_lm_ctx; ++c) {
            const int id = tok[static_cast<std::size_t>(b) * k_lm_ctx + c];
            std::memcpy(row + static_cast<std::size_t>(c) * k_lm_dim,
                        E.data + static_cast<std::size_t>(id) * k_lm_dim, k_lm_dim * sizeof(float));
        }
    }
}

void embed_update(Matrix& E, const Matrix& Gx, const std::uint8_t* tok, int batch, float lr) {
    expect(Gx.rows == batch && Gx.cols == k_lm_ctx * k_lm_dim, "embed grad shape");
    for (int b = 0; b < batch; ++b) {
        const float* g = Gx.data + static_cast<std::size_t>(b) * static_cast<std::size_t>(Gx.cols);
        for (int c = 0; c < k_lm_ctx; ++c) {
            const int id = tok[static_cast<std::size_t>(b) * k_lm_ctx + c];
            float* row = E.data + static_cast<std::size_t>(id) * k_lm_dim;
            const float* gc = g + static_cast<std::size_t>(c) * k_lm_dim;
            for (int k = 0; k < k_lm_dim; ++k) row[k] -= lr * gc[k];
        }
    }
}

void fill_lm_batch(const std::vector<std::uint8_t>& text, const std::vector<int>& pos, std::uint8_t* tok,
                   Matrix& Y) {
    const int batch = static_cast<int>(pos.size());
    Y.ensure(batch, k_lm_vocab);
    std::memset(Y.data, 0, Y.n() * sizeof(float));
    for (int b = 0; b < batch; ++b) {
        const int t = pos[static_cast<std::size_t>(b)];
        for (int c = 0; c < k_lm_ctx; ++c) {
            tok[static_cast<std::size_t>(b) * k_lm_ctx + c] = text[static_cast<std::size_t>(t - k_lm_ctx + c)];
        }
        Y(b, text[static_cast<std::size_t>(t)]) = 1.f;
    }
}

float lm_bpc(const Net& net, const Matrix& E, const std::vector<std::uint8_t>& text, int begin, int end,
             int batch, int probe, Matrix& a, Matrix& b, Matrix& X, Matrix& Y, Matrix& logits) {
    const int span = end - begin;
    const int npos = std::min(probe, span);
    const int stride = std::max(1, span / npos);
    double ce_sum = 0.0;
    int seen = 0;
    std::vector<std::uint8_t> tok(static_cast<std::size_t>(batch) * k_lm_ctx);
    std::vector<int> pos;
    pos.reserve(static_cast<std::size_t>(batch));
    for (int i = 0; i < npos;) {
        pos.clear();
        const int room = std::min(batch, npos - i);
        for (int r = 0; r < room; ++r) {
            const int t = begin + i * stride;
            ++i;
            if (t >= end) {
                i = npos;
                break;
            }
            pos.push_back(t);
        }
        if (pos.empty()) break;
        fill_lm_batch(text, pos, tok.data(), Y);
        embed_gather(E, tok.data(), static_cast<int>(pos.size()), X);
        forward_logits(net, X, a, b, logits);
        ce_sum += static_cast<double>(softmax_ce(logits, Y, nullptr)) * static_cast<double>(pos.size());
        seen += static_cast<int>(pos.size());
    }
    expect(seen > 0, "empty val probe");
    return static_cast<float>((ce_sum / seen) / k_ln2);
}

std::string show_byte(unsigned char c) {
    if (c == '\n') return "\\n";
    if (c == '\t') return "\\t";
    if (c >= 32 && c < 127) return std::string(1, static_cast<char>(c));
    char buf[8];
    std::snprintf(buf, sizeof(buf), "\\x%02x", c);
    return buf;
}

std::string lm_generate(const Net& net, const Matrix& E, const std::string& prompt, int nchar, bool greedy,
                        std::uint32_t seed) {
    expect(static_cast<int>(prompt.size()) <= k_lm_ctx, "prompt longer than the window");
    std::vector<std::uint8_t> ctx(static_cast<std::size_t>(k_lm_ctx), static_cast<std::uint8_t>(' '));
    std::memcpy(ctx.data() + (k_lm_ctx - static_cast<int>(prompt.size())), prompt.data(), prompt.size());
    Matrix X, a, b, logits;
    std::mt19937 rng(seed);
    std::string out;
    out.reserve(static_cast<std::size_t>(nchar) * 2);
    for (int i = 0; i < nchar; ++i) {
        embed_gather(E, ctx.data(), 1, X);
        forward_logits(net, X, a, b, logits);
        int pick = 0;
        if (greedy) {
            float best = logits(0, 0);
            for (int k = 1; k < k_lm_vocab; ++k) {
                if (logits(0, k) > best) {
                    best = logits(0, k);
                    pick = k;
                }
            }
        } else {
            float m = logits(0, 0);
            for (int k = 1; k < k_lm_vocab; ++k) m = std::max(m, logits(0, k));
            double sum = 0.0;
            double acc[256];
            for (int k = 0; k < k_lm_vocab; ++k) {
                acc[k] = std::exp(static_cast<double>(logits(0, k) - m));
                sum += acc[k];
            }
            std::uniform_real_distribution<double> uni(0.0, sum);
            double draw = uni(rng);
            pick = k_lm_vocab - 1;
            for (int k = 0; k < k_lm_vocab; ++k) {
                draw -= acc[k];
                if (draw <= 0.0) {
                    pick = k;
                    break;
                }
            }
        }
        out += show_byte(static_cast<unsigned char>(pick));
        std::memmove(ctx.data(), ctx.data() + 1, static_cast<std::size_t>(k_lm_ctx - 1));
        ctx.back() = static_cast<std::uint8_t>(pick);
    }
    return out;
}

float mean_abs_col_cosine(const Matrix& M, int pairs, std::uint32_t seed) {
    if (M.cols < 2 || M.rows <= 0 || pairs <= 0) return 0.f;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, M.cols - 1);
    double sum = 0.0;
    for (int n = 0; n < pairs; ++n) {
        const int i = pick(rng);
        int j = pick(rng);
        if (j == i) j = (j + 1) % M.cols;
        double dot = 0.0;
        double ni = 0.0;
        double nj = 0.0;
        for (int r = 0; r < M.rows; ++r) {
            const double a = M(r, i);
            const double b = M(r, j);
            dot += a * b;
            ni += a * a;
            nj += b * b;
        }
        const double den = std::sqrt(ni * nj);
        if (den > 1e-18) sum += std::fabs(dot / den);
    }
    return static_cast<float>(sum / static_cast<double>(pairs));
}

void run_lm_diag(const Config& cfg, const std::vector<std::uint8_t>& text, TrainerA& A, TrainerB& Bint, Matrix& Ea,
                 Matrix& Eb) {
    constexpr int train_end = 90000000;
    constexpr int val_begin = 90000000;
    constexpr int val_end = 95000000;
    constexpr int val_probe = 131072;

    TrainerB Bf;
    Bf.net = A.net;
    Bf.classify = true;
    Bf.keep_input_grad = true;
    Bf.clip = Bint.clip;
    Bf.clip_gamma = Bint.clip_gamma;
    Bf.align = Bint.align;
    Bf.lambda_peak = Bint.lambda_peak;
    Bf.align_k = Bint.align_k;
    Bf.rank = Bint.rank;
    Bf.float_strip = true;
    Bf.report_embed = true;
    Bint.report_embed = true;
    Bf.init_feedback(cfg.seed + 10007u);
    Bf.prepare(cfg.batch);
    Matrix Ebf = Ea;

    auto stat_step = [](int s) { return s == 1 || s == 100 || s == 300; };
    auto cos_step = [](int s) { return s == 0 || s == 50 || s == 100 || s == 200 || s == 300; };
    auto val_step = [](int s) { return s == 0 || s == 100 || s == 200 || s == 300; };
    auto ratio = [](float b, float a) {
        if (!(a > 0.f) || !std::isfinite(a) || !std::isfinite(b)) return std::numeric_limits<float>::quiet_NaN();
        return b / a;
    };
    auto print_nums = [](const char* kind, const char* tag, int step, const std::vector<float>& v, const char* fmt) {
        std::cout << kind << " " << tag << " " << step;
        for (float x : v) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), fmt, static_cast<double>(x));
            std::cout << buf;
        }
        std::cout << "\n" << std::flush;
    };

    std::cout << "\ndiag      A + B-int8 + B-float   same minibatch   float-strip dW = X^T delta\n";
    std::cout << "cos       on the same net, SAP delta versus backprop; columns run from the residual layer near the input to the output, then E\n";
    std::cout << "tape      crest = peak/rms    azero = share of positive elements rounded to INT8 0\n";
    std::cout << "gnorm     ||dW|| before clipping, B/A. Columns: E, then weight layers 0..last\n";
    std::cout << "float memory " << mb(Bf.bytes() + Ebf.bytes()) << " MB\n";

    Matrix X, Y, ev_a, ev_b, ev_x, ev_y, logits;
    std::vector<std::uint8_t> tok(static_cast<std::size_t>(cfg.batch) * k_lm_ctx);
    std::vector<int> pos(static_cast<std::size_t>(cfg.batch));
    std::mt19937 rng(cfg.seed + 99u);
    std::uniform_int_distribution<int> pick(k_lm_ctx, train_end - 1);

    auto draw = [&]() {
        for (int i = 0; i < cfg.batch; ++i) pos[static_cast<std::size_t>(i)] = pick(rng);
        fill_lm_batch(text, pos, tok.data(), Y);
    };
    auto score = [&](const Net& net, const Matrix& E) {
        return lm_bpc(net, E, text, val_begin, val_end, cfg.batch, val_probe, ev_a, ev_b, ev_x, ev_y, logits);
    };
    auto emit_cos = [&](int step) {
        embed_gather(Eb, tok.data(), cfg.batch, X);
        Bint.report_cosine(X, Y, step, "int8");
        embed_gather(Ebf, tok.data(), cfg.batch, X);
        Bf.report_cosine(X, Y, step, "float");
    };
    auto emit_val = [&](int step, float train_a, float train_i, float train_f) {
        const float ba = score(A.net, Ea);
        const float bi = score(Bint.net, Eb);
        const float bf = score(Bf.net, Ebf);
        char line[240];
        std::snprintf(line, sizeof(line),
                      "val %4d | A train %6.3f val %6.3f | int8 train %6.3f val %6.3f | float train %6.3f val %6.3f\n",
                      step, train_a, ba, train_i, bi, train_f, bf);
        std::cout << line << std::flush;
    };
    auto emit_tape = [&](const char* tag, int step, const TrainerB& B) {
        print_nums("crest", tag, step, B.crest, " %7.2f");
        print_nums("azero", tag, step, B.active_zero, " %6.3f");
    };
    auto emit_gnorm = [&](const char* tag, int step, const TrainerB& B) {
        std::vector<float> r;
        r.reserve(B.grad_norm.size() + 1);
        r.push_back(ratio(B.embed_gnorm, A.embed_gnorm));
        const int n = static_cast<int>(std::min(B.grad_norm.size(), A.grad_norm.size()));
        for (int i = 0; i < n; ++i) r.push_back(ratio(B.grad_norm[static_cast<std::size_t>(i)], A.grad_norm[static_cast<std::size_t>(i)]));
        print_nums("gnorm", tag, step, r, " %6.3f");
    };

    draw();
    emit_cos(0);
    emit_val(0, 0.f, 0.f, 0.f);

    double sum_a = 0.0;
    double sum_i = 0.0;
    double sum_f = 0.0;
    int seen = 0;
    for (int step = 1; step <= cfg.steps; ++step) {
        draw();
        const bool capture = stat_step(step);
        A.capture_grad = capture;
        Bint.capture_grad = capture;
        Bint.capture_stats = capture;
        Bf.capture_grad = capture;
        Bf.capture_stats = capture;

        embed_gather(Ea, tok.data(), cfg.batch, X);
        const float la = A.step(X, Y, cfg.lr);
        embed_update(Ea, A.Gin, tok.data(), cfg.batch, cfg.lr);
        embed_gather(Eb, tok.data(), cfg.batch, X);
        const float li = Bint.step(X, Y, cfg.lr);
        embed_update(Eb, Bint.embed_g, tok.data(), cfg.batch, cfg.lr);
        embed_gather(Ebf, tok.data(), cfg.batch, X);
        const float lf = Bf.step(X, Y, cfg.lr);
        embed_update(Ebf, Bf.embed_g, tok.data(), cfg.batch, cfg.lr);

        A.capture_grad = false;
        Bint.capture_grad = false;
        Bint.capture_stats = false;
        Bf.capture_grad = false;
        Bf.capture_stats = false;

        if (std::isfinite(la)) sum_a += la;
        if (std::isfinite(li)) sum_i += li;
        if (std::isfinite(lf)) sum_f += lf;
        ++seen;

        if (capture) {
            emit_tape("int8", step, Bint);
            emit_tape("float", step, Bf);
            emit_gnorm("int8", step, Bint);
            emit_gnorm("float", step, Bf);
        }
        if (cos_step(step)) emit_cos(step);
        if (val_step(step)) {
            const float inv = seen > 0 ? static_cast<float>(1.0 / static_cast<double>(seen)) : 0.f;
            emit_val(step, static_cast<float>(sum_a * inv / k_ln2), static_cast<float>(sum_i * inv / k_ln2),
                     static_cast<float>(sum_f * inv / k_ln2));
            sum_a = 0.0;
            sum_i = 0.0;
            sum_f = 0.0;
            seen = 0;
        }
        if (!std::isfinite(la) || !std::isfinite(li) || !std::isfinite(lf)) {
            std::cout << "nonfinite step " << step << " A " << la << " int8 " << li << " float " << lf << "\n";
        }
    }
    std::cout << "clip A " << A.clip_hits << "  int8 " << Bint.clip_hits << "  float " << Bf.clip_hits << "\n";
}

void run_lm(const Config& cfg) {
    const std::vector<std::uint8_t> text = load_bytes(cfg.lm);
    constexpr int train_end = 90000000;
    constexpr int val_begin = 90000000;
    constexpr int val_end = 95000000;
    constexpr int val_probe = 131072;

    const std::vector<int> dims = make_dims(k_lm_ctx * k_lm_dim, cfg.hidden, k_lm_vocab, 9);
    TrainerA A;
    TrainerB B;
    A.net = init_net(dims, cfg.seed);
    A.net.residual = true;
    B.net = A.net;
    A.classify = true;
    B.classify = true;
    A.keep_input_grad = true;
    B.keep_input_grad = true;
    A.clip = cfg.clip;
    A.clip_gamma = cfg.clip_gamma;
    B.clip = cfg.clip;
    B.clip_gamma = cfg.clip_gamma;
    B.align = cfg.align == 2 ? Align::Jacobian : cfg.align == 1 ? Align::Covariance : Align::Off;
    B.lambda_peak = cfg.lambda_peak;
    B.align_k = cfg.align_k > 0 ? cfg.align_k : 50;
    if (cfg.beta > 0.f) {
        B.align_beta = cfg.beta;
        B.align_beta_fast = cfg.beta;
    }
    if (cfg.mgs_k != 0) {
        B.mgs_k = cfg.mgs_k;
        B.mgs_now = cfg.mgs_k > 0;
    }
    A.max_norm = cfg.max_norm;
    B.max_norm = cfg.max_norm;
    B.rank = cfg.rank;
    B.float_strip = cfg.diag ? false : cfg.float_strip;
    B.init_feedback(cfg.seed + 10007u);
    if (B.rank > 0) {
        expect(B.P[0].cols == B.rank, "low-rank P width");
        expect(std::fabs(frobenius(B.R) - std::sqrt(static_cast<float>(B.rank))) < 1e-3f, "R not orthonormal");
    }
    A.prepare(cfg.batch);
    B.prepare(cfg.batch);
    B.report_embed = true;

    Matrix Ea(k_lm_vocab, k_lm_dim);
    Matrix Eb;
    {
        std::mt19937 rng(cfg.seed + 17u);
        fill_normal(Ea, rng, 0.02f);
    }
    Eb = Ea;

    auto mem_a = [&]() { return A.bytes() + Ea.bytes() + A.Gin.bytes(); };
    auto mem_b = [&]() { return B.bytes() + Eb.bytes() + B.embed_g.bytes(); };

    std::cout << "\nlm        " << cfg.lm << "  bytes " << text.size() << "  train " << train_end
              << "  val 5000000  test 5000000\n";
    std::cout << "arch    embed 256x16  window 32 -> " << arch_string(A.net)
              << "  residual gamma_res=0.876 alpha=0.316\n";
    std::cout << "loss     mean cross-entropy / ln(2) = bpc   e = (p - y) / batch\n";
    std::cout << "sap       lambda " << B.lambda_peak << "  K " << B.align_k << "  beta " << B.align_beta_fast
              << "/" << B.align_beta << "  mgs "
              << (B.mgs_k < 0 ? std::string("none") : std::to_string(B.mgs_k)) << "  rank "
              << (B.rank > 0 ? B.rank : k_lm_vocab) << "  lr " << cfg.lr << "  batch " << cfg.batch
              << "  max-norm " << cfg.max_norm << "\n";
    std::cout << "A memory  " << mb(mem_a()) << " MB   B memory " << mb(mem_b()) << " MB   text "
              << mb(text.size()) << " MB\n";
    std::cout << "val       " << val_probe << " positions, step " << (5000000 / val_probe) << "\n";

    Matrix X, Y, ev_a, ev_b, ev_x, ev_y, logits;
    std::vector<std::uint8_t> tok(static_cast<std::size_t>(cfg.batch) * k_lm_ctx);
    std::vector<int> pos(static_cast<std::size_t>(cfg.batch));
    std::mt19937 rng(cfg.seed + 99u);
    std::uniform_int_distribution<int> pick(k_lm_ctx, train_end - 1);
    using clock = std::chrono::steady_clock;

    auto score = [&](const Net& net, const Matrix& E) {
        return lm_bpc(net, E, text, val_begin, val_end, cfg.batch, val_probe, ev_a, ev_b, ev_x, ev_y, logits);
    };

    if (cfg.diag) {
        run_lm_diag(cfg, text, A, B, Ea, Eb);
        return;
    }
    B.sweep_us = B.mgs_us = B.scale_us = B.r_us = 0.0;
    B.sweep_calls = B.mgs_calls = B.scale_calls = B.r_calls = 0;

    if (cfg.seconds > 0.0) {
        const double budget_us = cfg.seconds * 1.0e6;
        double used_a = 0.0;
        double used_b = 0.0;
        int steps_a = 0;
        int steps_b = 0;
        bool a_alive = true;
        bool b_alive = true;
        while ((a_alive && used_a < budget_us) || (b_alive && used_b < budget_us)) {
            const bool run_a = a_alive && used_a < budget_us;
            const bool run_b = b_alive && used_b < budget_us;
            if (run_a || run_b) {
                for (int i = 0; i < cfg.batch; ++i) pos[static_cast<std::size_t>(i)] = pick(rng);
                fill_lm_batch(text, pos, tok.data(), Y);
            }
            if (run_a) {
                const auto t0 = clock::now();
                embed_gather(Ea, tok.data(), cfg.batch, X);
                const float la = A.step(X, Y, cfg.lr);
                embed_update(Ea, A.Gin, tok.data(), cfg.batch, cfg.lr * A.last_scale);
                used_a += std::chrono::duration<double, std::micro>(clock::now() - t0).count();
                ++steps_a;
                if (!std::isfinite(la)) a_alive = false;
            }
            if (run_b) {
                const auto t0 = clock::now();
                embed_gather(Eb, tok.data(), cfg.batch, X);
                const float lb = B.step(X, Y, cfg.lr);
                embed_update(Eb, B.embed_g, tok.data(), cfg.batch, cfg.lr * B.last_scale);
                used_b += std::chrono::duration<double, std::micro>(clock::now() - t0).count();
                ++steps_b;
                if (!std::isfinite(lb)) b_alive = false;
            }
        }
        const float ba = a_alive ? score(A.net, Ea) : std::numeric_limits<float>::quiet_NaN();
        const float bb = b_alive ? score(B.net, Eb) : std::numeric_limits<float>::quiet_NaN();
        char line[320];
        std::snprintf(line, sizeof(line),
                      "wall %.1f s | A step %d time %.3f s val bpc %.3f | B step %d time %.3f s val bpc %.3f\n",
                      cfg.seconds, steps_a, used_a / 1.0e6, ba, steps_b, used_b / 1.0e6, bb);
        std::cout << line << "clip A " << A.clip_hits << " scale "
                  << (A.scale_n ? A.scale_sum / A.scale_n : 1.0) << "  B " << B.clip_hits << " scale "
                  << (B.scale_n ? B.scale_sum / B.scale_n : 1.0) << "\n";
        std::cout << "memory A " << mb(mem_a()) << " MB  B " << mb(mem_b()) << " MB\n";
        return;
    }

    double used_a = 0.0;
    double used_b = 0.0;
    double sum_a = 0.0;
    double sum_b = 0.0;
    int n_a = 0;
    int n_b = 0;
    int steps_a = 0;
    int steps_b = 0;
    bool a_alive = true;
    bool b_alive = true;
    for (int step = 1; step <= cfg.steps; ++step) {
        float la = 0.f;
        float lb = 0.f;
        for (int i = 0; i < cfg.batch; ++i) pos[static_cast<std::size_t>(i)] = pick(rng);
        fill_lm_batch(text, pos, tok.data(), Y);
        if (a_alive) {
            const auto t0 = clock::now();
            embed_gather(Ea, tok.data(), cfg.batch, X);
            la = A.step(X, Y, cfg.lr);
            embed_update(Ea, A.Gin, tok.data(), cfg.batch, cfg.lr * A.last_scale);
            used_a += std::chrono::duration<double, std::micro>(clock::now() - t0).count();
            ++steps_a;
            if (!std::isfinite(la)) a_alive = false;
            else {
                sum_a += la;
                ++n_a;
            }
        }
        if (b_alive) {
            const auto t0 = clock::now();
            embed_gather(Eb, tok.data(), cfg.batch, X);
            lb = B.step(X, Y, cfg.lr);
            embed_update(Eb, B.embed_g, tok.data(), cfg.batch, cfg.lr * B.last_scale);
            used_b += std::chrono::duration<double, std::micro>(clock::now() - t0).count();
            ++steps_b;
            if (!std::isfinite(lb)) b_alive = false;
            else {
                sum_b += lb;
                ++n_b;
            }
        }
        const bool sense = step == 300 || step == 1000 || step == 3000;
        if (sense && b_alive && !B.P.empty()) {
            embed_gather(Eb, tok.data(), cfg.batch, X);
            B.report_cosine(X, Y, step, "B");
            char crow[80];
            std::snprintf(crow, sizeof(crow), "colcos P1 %4d %.3f\n", step, mean_abs_col_cosine(B.P[0], 100, 1u));
            std::cout << crow << std::flush;
        }
        const bool report = step == 200 || step == 300 || step == 1000 || step == 3000 || step == cfg.steps;
        if (!report) continue;
        const float ba = a_alive ? score(A.net, Ea) : std::numeric_limits<float>::quiet_NaN();
        const float bb = b_alive ? score(B.net, Eb) : std::numeric_limits<float>::quiet_NaN();
        const float tra = n_a ? static_cast<float>((sum_a / n_a) / k_ln2) : std::numeric_limits<float>::quiet_NaN();
        const float trb = n_b ? static_cast<float>((sum_b / n_b) / k_ln2) : std::numeric_limits<float>::quiet_NaN();
        char line[480];
        std::snprintf(line, sizeof(line),
                      "step %4d | A train %6.3f val %6.3f bpc  time %6.2f ms mem %6.3f MB scl %.3f | "
                      "B train %6.3f val %6.3f bpc  time %6.2f ms mem %6.3f MB scl %.3f\n",
                      step, tra, ba, steps_a ? (used_a / steps_a) / 1000.0 : 0.0, mb(mem_a()),
                      A.scale_n ? static_cast<float>(A.scale_sum / A.scale_n) : 1.f, trb, bb,
                      steps_b ? (used_b / steps_b) / 1000.0 : 0.0, mb(mem_b()),
                      B.scale_n ? static_cast<float>(B.scale_sum / B.scale_n) : 1.f);
        std::cout << line << std::flush;
        sum_a = 0.0;
        sum_b = 0.0;
        n_a = 0;
        n_b = 0;
    }
    std::cout << "clip A " << A.clip_hits << " scale " << (A.scale_n ? A.scale_sum / A.scale_n : 1.0) << "  B "
              << B.clip_hits << " scale " << (B.scale_n ? B.scale_sum / B.scale_n : 1.0) << "\n";
    if (B.sweep_calls > 0 && steps_b > 0) {
        const double inv = 1.0 / static_cast<double>(steps_b) / 1000.0;
        const double transport = B.sweep_us - B.mgs_us - B.scale_us;
        char line[400];
        std::snprintf(line, sizeof(line),
                      "sweep %d  mgs-layers %d  scale-layers %d | ms/step transport %.2f  mgs %.2f  scale %.2f  sweep %.2f | R %d  R-mgs %.3f\n",
                      B.sweep_calls, B.mgs_calls, B.scale_calls, transport * inv, B.mgs_us * inv, B.scale_us * inv,
                      B.sweep_us * inv, B.r_calls, B.r_us * inv);
        std::cout << line;
    }
    const std::string prompt = "The history of ";
    if (a_alive) {
        std::cout << "A greedy: " << lm_generate(A.net, Ea, prompt, 200, true, cfg.seed) << "\n";
        std::cout << "A sample: " << lm_generate(A.net, Ea, prompt, 200, false, cfg.seed + 3u) << "\n";
    }
    if (b_alive) {
        std::cout << "B greedy: " << lm_generate(B.net, Eb, prompt, 200, true, cfg.seed) << "\n";
        std::cout << "B sample: " << lm_generate(B.net, Eb, prompt, 200, false, cfg.seed + 3u) << "\n";
    }
}

#include "shake.inc"

}  // namespace

int main(int argc, char** argv) {
    SetConsoleOutputCP(CP_UTF8);
    try {
        const Config cfg = parse_args(argc, argv);
        self_test();
        if (!cfg.shake.empty()) run_shake(cfg);
        else if (!cfg.lm.empty()) run_lm(cfg);
        else if (!cfg.data.empty()) run_classify(cfg);
        else if (cfg.steps > 0 || cfg.seconds > 0.0) run_experiment(cfg);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "hata: " << ex.what() << "\n";
        return 1;
    }
}
