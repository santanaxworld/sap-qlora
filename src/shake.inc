// Causal mixer for TinyShakespeare. The enwik8 trainer is untouched.
// Time mixing is a fixed multi-scale exponential decay, not a learned matrix.
// Channel weights: backprop on A, MGS-free Jacobian SAP on B.

constexpr int k_shake_T = 128;
constexpr int k_shake_D = 256;
constexpr int k_shake_blocks = 6;
constexpr float k_shake_mom = 0.9f;
constexpr float k_shake_wd = 0.f;

void mix_residual(Matrix& Y, const Matrix& skip, const Matrix& pre, float alpha, const std::uint32_t* bits, bool linear) {
    expect(skip.n() == pre.n(), "resid shape");
    Y.ensure(pre.rows, pre.cols);
    for (std::size_t i = 0; i < pre.n(); ++i) {
        float v = pre.data[i];
        if (!linear && !relu_mask_bit(bits, i)) v = 0.f;
        Y.data[i] = skip.data[i] + alpha * v;
    }
}

struct ShakeVocab {
    int id[256];
    std::string chars;
    int V = 0;
};

ShakeVocab shake_vocab(const std::vector<std::uint8_t>& text) {
    bool seen[256] = {};
    std::vector<unsigned char> uniq;
    for (std::uint8_t b : text) {
        if (!seen[b]) {
            seen[b] = true;
            uniq.push_back(b);
        }
    }
    std::sort(uniq.begin(), uniq.end());
    ShakeVocab v;
    std::fill(v.id, v.id + 256, -1);
    v.chars.assign(uniq.begin(), uniq.end());
    v.V = static_cast<int>(v.chars.size());
    for (int i = 0; i < v.V; ++i) v.id[static_cast<unsigned char>(v.chars[static_cast<std::size_t>(i)])] = i;
    return v;
}

std::vector<std::uint8_t> read_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    expect(static_cast<bool>(in), "file open failed: " + path);
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    expect(n > 1000, "file too small: " + path);
    in.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> text(static_cast<std::size_t>(n));
    in.read(reinterpret_cast<char*>(text.data()), static_cast<std::streamsize>(text.size()));
    expect(static_cast<bool>(in), "file read failed: " + path);
    return text;
}

struct Shake {
    int T = k_shake_T;
    int D = k_shake_D;
    int V = 65;
    int blocks = k_shake_blocks;
    int batch = 8;
    float alpha = 0.f;
    float mom = k_shake_mom;
    float max_norm = 100.f;
    bool clip = true;
    bool linear = false;
    bool sap = false;
    int align_k = 10;
    int align_k_early = 0;
    int align_k_until = 0;
    int clusters = 1;
    int ticks = 0;
    int clip_hits = 0;
    double scale_sum = 0.0;
    int scale_n = 0;
    int sweep_calls = 0;

    std::vector<Matrix> Wch, Vch, dWch, P;
    Matrix P_in;
    std::vector<Matrix> Pins;
    std::uint32_t seed_ = 1;
    std::vector<std::vector<std::int8_t>> hq;
    std::vector<float> hscale;
    // Channel input is two ping-pong streams. Backward reads only the final one.
    Matrix xin[2];
    int x_final = 0;
    bool int8_strip = false;
    bool adam = false;
    float wd = 0.f;
    float b2 = 0.999f;
    float adam_eps = 1e-8f;
    // Frozen W0. Trainable channel update is s * A[D,r] * B[r,D], B starts at 0.
    bool lora = false;
    int r = 16;
    float lora_s = 1.f;
    // Frozen W0 packed per output. Empty until compress_w0(), which frees the FP32 matrix.
    struct W0Pack {
        std::vector<std::int8_t> q;
        std::vector<float> scale;
        std::vector<std::int32_t> row_sum;
    };
    std::vector<W0Pack> W0q;
    bool qlora = false;
    bool gather_mask = false;
    bool ema_mask = false;
    bool mean_mask = false;
    float ema_keep = 0.9f;
    float ema_blend = 0.1f;
    int m_count = 0;
    float ab_clip = 0.f;
    int ab_clip_hits = 0;
    bool log_delta = false;
    std::vector<float> delta_n;
    std::vector<std::vector<float>> m_ema;
    std::vector<std::vector<float>> m_sum;
    std::vector<float> mask_scratch;
    std::vector<Matrix> Al, Bl, VAl, VBl, dAl, dBl, SAl, SBl, xa;
    Matrix lora_y, lora_mid;
    std::vector<Matrix> Sv;
    Matrix Svout, Sve, Svp;
    bool tally = false;
    double us_fwd = 0;
    double us_out = 0;
    double us_bwd = 0;
    double us_dw = 0;
    double us_deq = 0;
    double us_upd = 0;
    int tau[16] = {};
    float lam[16] = {};
    Matrix Wout, Vout, dWout, E, Ve, dE, Pos, Vp, dPos;
    std::vector<Matrix> H;
    Matrix pre_ch;
    std::vector<std::vector<std::uint32_t>> mask_ch;
    std::vector<std::vector<float>> rate_ch;
    std::vector<std::vector<int>> rate_n;
    std::vector<std::uint8_t> cluster_of;
    std::vector<std::uint8_t> cid;
    std::vector<int> grp_order;
    int rate_fill = 0;
    Matrix logits, err, g, dpre, sap_delta, dH;
    bool capture_probe = false;
    Matrix cap_bp_top, cap_bp_bot, cap_sap_top, cap_sap_bot;
    struct AlignView {
        float cos_top = 0.f;
        float cos_bot = 0.f;
        float cos_e = 0.f;
        float nr_top = 0.f;
        float nr_bot = 0.f;
        float nr_e = 0.f;
    };

    std::size_t mask_words() const {
        const std::size_t n = static_cast<std::size_t>(batch) * static_cast<std::size_t>(T) * static_cast<std::size_t>(D);
        return (n + 31u) / 32u;
    }

    std::size_t side_bytes() const {
        std::size_t q = 0;
        for (const auto& v : hq) q += v.size();
        std::size_t m = 0;
        for (const auto& v : mask_ch) m += v.size() * sizeof(std::uint32_t);
        std::size_t r = 0;
        for (const auto& v : rate_ch) r += v.size() * sizeof(float);
        for (const auto& v : rate_n) r += v.size() * sizeof(int);
        r += cid.size() + grp_order.size() * sizeof(int) + cluster_of.size();
        std::size_t w0 = 0;
        for (const auto& p : W0q)
            w0 += p.q.size() + p.scale.size() * sizeof(float) + p.row_sum.size() * sizeof(std::int32_t);
        for (const auto& v : m_ema) r += v.size() * sizeof(float);
        for (const auto& v : m_sum) r += v.size() * sizeof(float);
        return q + m + r + w0;
    }

    std::size_t w0_pack_bytes() const {
        std::size_t w0 = 0;
        for (const auto& p : W0q)
            w0 += p.q.size() + p.scale.size() * sizeof(float) + p.row_sum.size() * sizeof(std::int32_t);
        return w0;
    }

    // Live training buffers only. Cosine probes allocate temporary copies and release them.
    std::size_t bytes() const {
        return bytes_of(Wch) + bytes_of(Vch) + bytes_of(dWch) + bytes_of(P) + P_in.bytes() + bytes_of(Pins) + xin[0].bytes() +
               xin[1].bytes() + bytes_of(H) + pre_ch.bytes() + Wout.bytes() + Vout.bytes() + dWout.bytes() + E.bytes() +
               Ve.bytes() + dE.bytes() + Pos.bytes() + Vp.bytes() + dPos.bytes() + logits.bytes() + err.bytes() +
               g.bytes() + dpre.bytes() + sap_delta.bytes() + dH.bytes() + bytes_of(Sv) + Svout.bytes() + Sve.bytes() +
               Svp.bytes() + bytes_of(Al) + bytes_of(Bl) + bytes_of(VAl) + bytes_of(VBl) + bytes_of(dAl) +
               bytes_of(dBl) + bytes_of(SAl) + bytes_of(SBl) + bytes_of(xa) + lora_y.bytes() + lora_mid.bytes() +
               side_bytes();
    }

    std::size_t matrix_bytes() const { return bytes() - side_bytes(); }

    void dump_mem(const char* name) const {
        std::size_t q = 0;
        for (const auto& v : hq) q += v.size();
        std::size_t msk = 0;
        for (const auto& v : mask_ch) msk += v.size() * sizeof(std::uint32_t);
        const double mb = 1.0 / (1024.0 * 1024.0);
        char line[640];
        std::snprintf(line, sizeof(line),
                      "memory %s  total %7.3f  stream %6.3f  H %6.3f  int8 %6.3f  P %6.3f  "
                      "weight %6.3f  moment %6.3f  dW %6.3f  scratch %6.3f  mask %6.3f\n",
                      name, static_cast<double>(bytes()) * mb, static_cast<double>(xin[0].bytes() + xin[1].bytes()) * mb,
                      static_cast<double>(bytes_of(H)) * mb, static_cast<double>(q) * mb,
                      static_cast<double>(bytes_of(P) + P_in.bytes() + bytes_of(Pins)) * mb,
                      static_cast<double>(bytes_of(Wch) + Wout.bytes() + E.bytes() + Pos.bytes() + bytes_of(Al) +
                                          bytes_of(Bl) + w0_pack_bytes()) *
                          mb,
                      static_cast<double>(bytes_of(Vch) + Vout.bytes() + Ve.bytes() + Vp.bytes() + bytes_of(Sv) +
                                          Svout.bytes() + Sve.bytes() + Svp.bytes() + bytes_of(VAl) + bytes_of(VBl) +
                                          bytes_of(SAl) + bytes_of(SBl)) *
                          mb,
                      static_cast<double>(bytes_of(dWch) + dWout.bytes() + dE.bytes() + dPos.bytes() + bytes_of(dAl) +
                                          bytes_of(dBl)) *
                          mb,
                      static_cast<double>(pre_ch.bytes() + logits.bytes() + err.bytes() + g.bytes() + dpre.bytes() +
                                          sap_delta.bytes() + dH.bytes() + bytes_of(xa) + lora_y.bytes() +
                                          lora_mid.bytes()) *
                          mb,
                      static_cast<double>(msk) * mb);
        std::cout << line << std::flush;
    }

    void add_us(double& slot, std::chrono::steady_clock::time_point t0) {
        if (!tally) return;
        slot += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    }

    void configure(int t, int d, int vocab, int nblk, int nb) {
        T = t;
        D = d;
        V = vocab;
        blocks = nblk;
        batch = nb;
        alpha = 1.f / std::sqrt(static_cast<float>(2 * blocks));
        expect(blocks >= 1 && blocks <= 16, "tau table holds 16 blocks");
        for (int l = 0; l < blocks; ++l) {
            tau[l] = 1 << l;
            lam[l] = 1.f - std::exp(-1.f / static_cast<float>(tau[l]));
        }
    }

    void init_weights(std::uint32_t seed) {
        seed_ = seed;
        std::mt19937 rng(seed);
        Wch.assign(static_cast<std::size_t>(blocks), Matrix{});
        Vch.assign(static_cast<std::size_t>(blocks), Matrix{});
        dWch.assign(static_cast<std::size_t>(blocks), Matrix{});
        const float ch_std = std::sqrt(2.f / static_cast<float>(D));
        for (int l = 0; l < blocks; ++l) {
            Wch[static_cast<std::size_t>(l)].ensure(D, D);
            fill_normal(Wch[static_cast<std::size_t>(l)], rng, ch_std);
            Vch[static_cast<std::size_t>(l)].ensure(D, D);
            Vch[static_cast<std::size_t>(l)].zero();
            dWch[static_cast<std::size_t>(l)].ensure(D, D);
        }
        Wout.ensure(D, V);
        E.ensure(V, D);
        Pos.ensure(T, D);
        fill_normal(Wout, rng, 0.02f);
        fill_normal(E, rng, 0.02f);
        fill_normal(Pos, rng, 0.02f);
        Vout.ensure(D, V);
        Ve.ensure(V, D);
        Vp.ensure(T, D);
        dWout.ensure(D, V);
        dE.ensure(V, D);
        dPos.ensure(T, D);
        Vout.zero();
        Ve.zero();
        Vp.zero();
    }

    // Drop W0 / E / Pos moments and grads. A ~ N(0, 0.01), B = 0, so step 0 matches the frozen net.
    void attach_lora(std::uint32_t seed) {
        expect(Wch.size() == static_cast<std::size_t>(blocks) && D > 0, "lora before weights");
        expect(clusters == 1, "lora keeps a single raw Jacobian");
        lora = true;
        r = 16;
        lora_s = 1.f;
        ticks = 0;
        clip_hits = 0;
        scale_sum = 0.0;
        scale_n = 0;
        sweep_calls = 0;
        Vch.clear();
        dWch.clear();
        Sv.clear();
        Ve = Matrix{};
        dE = Matrix{};
        Sve = Matrix{};
        Vp = Matrix{};
        dPos = Matrix{};
        Svp = Matrix{};
        Vout.zero();
        if (Svout.n()) Svout.zero();
        std::mt19937 rng(seed);
        Al.assign(static_cast<std::size_t>(blocks), Matrix{});
        Bl.assign(static_cast<std::size_t>(blocks), Matrix{});
        VAl.assign(static_cast<std::size_t>(blocks), Matrix{});
        VBl.assign(static_cast<std::size_t>(blocks), Matrix{});
        dAl.assign(static_cast<std::size_t>(blocks), Matrix{});
        dBl.assign(static_cast<std::size_t>(blocks), Matrix{});
        if (adam) {
            SAl.assign(static_cast<std::size_t>(blocks), Matrix{});
            SBl.assign(static_cast<std::size_t>(blocks), Matrix{});
        } else {
            SAl.clear();
            SBl.clear();
        }
        xa.assign(static_cast<std::size_t>(blocks), Matrix{});
        for (int l = 0; l < blocks; ++l) {
            Al[static_cast<std::size_t>(l)].ensure(D, r);
            fill_normal(Al[static_cast<std::size_t>(l)], rng, 0.01f);
            Bl[static_cast<std::size_t>(l)].ensure(r, D);
            Bl[static_cast<std::size_t>(l)].zero();
            VAl[static_cast<std::size_t>(l)].ensure(D, r);
            VBl[static_cast<std::size_t>(l)].ensure(r, D);
            dAl[static_cast<std::size_t>(l)].ensure(D, r);
            dBl[static_cast<std::size_t>(l)].ensure(r, D);
            VAl[static_cast<std::size_t>(l)].zero();
            VBl[static_cast<std::size_t>(l)].zero();
            if (adam) {
                SAl[static_cast<std::size_t>(l)].ensure(D, r);
                SBl[static_cast<std::size_t>(l)].ensure(r, D);
                SAl[static_cast<std::size_t>(l)].zero();
                SBl[static_cast<std::size_t>(l)].zero();
            }
        }
    }

    // Per-output symmetric INT8. FP32 Wch is released; the sweep rebuilds it for one step.
    void compress_w0() {
        expect(Wch.size() == static_cast<std::size_t>(blocks) && D > 0, "compress before weights");
        expect(!qlora, "W0 already packed");
        W0q.assign(static_cast<std::size_t>(blocks), W0Pack{});
        for (int l = 0; l < blocks; ++l) {
            auto& P = W0q[static_cast<std::size_t>(l)];
            P.q.assign(static_cast<std::size_t>(D) * static_cast<std::size_t>(D), 0);
            P.scale.assign(static_cast<std::size_t>(D), 1.f);
            P.row_sum.assign(static_cast<std::size_t>(D), 0);
            expect(Wch[static_cast<std::size_t>(l)].rows == D && Wch[static_cast<std::size_t>(l)].cols == D, "W0 shape");
            qlora_pack(Wch[static_cast<std::size_t>(l)].data, D, D, P.q.data(), P.scale.data(), P.row_sum.data());
            Wch[static_cast<std::size_t>(l)] = Matrix{};
        }
        qlora = true;
    }

    void materialize_w0(int layer, Matrix& Weff) const {
        Weff.ensure(D, D);
        const auto& P = W0q[static_cast<std::size_t>(layer)];
        for (int j = 0; j < D; ++j) {
            const float s = P.scale[static_cast<std::size_t>(j)];
            const std::int8_t* src = P.q.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(D);
            for (int i = 0; i < D; ++i)
                Weff.data[static_cast<std::size_t>(i) * static_cast<std::size_t>(D) + static_cast<std::size_t>(j)] =
                    static_cast<float>(src[i]) * s;
        }
    }

    void save_frozen(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        expect(static_cast<bool>(out), "w0 save open");
        const std::int32_t magic = 0x30514C57;
        const std::int32_t dims[4] = {D, T, V, blocks};
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(dims), sizeof(dims));
        auto write_m = [&](const Matrix& M) {
            expect(M.data != nullptr || M.n() == 0, "w0 save empty");
            out.write(reinterpret_cast<const char*>(M.data), static_cast<std::streamsize>(M.n() * sizeof(float)));
        };
        for (int l = 0; l < blocks; ++l) write_m(Wch[static_cast<std::size_t>(l)]);
        write_m(E);
        write_m(Pos);
        write_m(Wout);
        expect(static_cast<bool>(out), "w0 save write");
    }

    void load_frozen(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        expect(static_cast<bool>(in), "w0 load open");
        std::int32_t magic = 0;
        std::int32_t dims[4] = {};
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        in.read(reinterpret_cast<char*>(dims), sizeof(dims));
        expect(magic == 0x30514C57, "w0 magic");
        expect(dims[0] == D && dims[1] == T && dims[2] == V && dims[3] == blocks, "w0 shape");
        auto read_m = [&](Matrix& M) {
            expect(M.data != nullptr && M.n() > 0, "w0 load buffer");
            in.read(reinterpret_cast<char*>(M.data), static_cast<std::streamsize>(M.n() * sizeof(float)));
        };
        for (int l = 0; l < blocks; ++l) read_m(Wch[static_cast<std::size_t>(l)]);
        read_m(E);
        read_m(Pos);
        read_m(Wout);
        expect(static_cast<bool>(in), "w0 load read");
    }

    void prepare() {
        if (sap) int8_strip = true;
        const int N = batch * T;
        const std::size_t words = mask_words();
        xin[0].ensure(N, D);
        xin[1].ensure(N, D);
        x_final = 0;
        pre_ch.ensure(N, D);
        mask_ch.assign(static_cast<std::size_t>(blocks), std::vector<std::uint32_t>(words, 0u));
        const std::size_t nelt = static_cast<std::size_t>(N) * static_cast<std::size_t>(D);
        if (int8_strip) {
            H.clear();
            hq.assign(static_cast<std::size_t>(blocks), std::vector<std::int8_t>(nelt));
            hscale.assign(static_cast<std::size_t>(blocks), 1.f);
        } else {
            hq.clear();
            hscale.clear();
            H.assign(static_cast<std::size_t>(blocks), Matrix{});
            for (int l = 0; l < blocks; ++l) H[static_cast<std::size_t>(l)].ensure(N, D);
        }
        logits.ensure(N, V);
        err.ensure(N, V);
        g.ensure(N, D);
        dH.ensure(N, D);
        if (sap) {
            dpre = Matrix{};
            sap_delta.ensure(N, D);
            const int width = clusters > 1 ? clusters * D : D;
            rate_ch.assign(static_cast<std::size_t>(blocks), std::vector<float>(static_cast<std::size_t>(width), 0.f));
            if (clusters > 1) {
                rate_n.assign(static_cast<std::size_t>(blocks), std::vector<int>(static_cast<std::size_t>(clusters), 0));
                cid.assign(static_cast<std::size_t>(N), 0);
                grp_order.assign(static_cast<std::size_t>(N), 0);
            } else {
                rate_n.clear();
                cid.clear();
                grp_order.clear();
                Pins.clear();
            }
        } else {
            P.clear();
            P_in = Matrix{};
            Pins.clear();
            sap_delta = Matrix{};
            dpre.ensure(N, D);
            rate_ch.clear();
            rate_n.clear();
            cid.clear();
            grp_order.clear();
        }
        if (adam) {
            Sv.assign(static_cast<std::size_t>(blocks), Matrix{});
            for (int l = 0; l < blocks; ++l) {
                Sv[static_cast<std::size_t>(l)].ensure(D, D);
                Sv[static_cast<std::size_t>(l)].zero();
            }
            Svout.ensure(D, V);
            Sve.ensure(V, D);
            Svp.ensure(T, D);
            Svout.zero();
            Sve.zero();
            Svp.zero();
        } else {
            Sv.clear();
            Svout = Matrix{};
            Sve = Matrix{};
            Svp = Matrix{};
        }
        rate_fill = 0;
    }

    void embed(const int* tok, int nb) {
        Matrix& X = xin[0];
        X.ensure(nb * T, D);
        for (int b = 0; b < nb; ++b) {
            for (int t = 0; t < T; ++t) {
                const int id = tok[static_cast<std::size_t>(b) * T + t];
                float* row = X.data + (static_cast<std::size_t>(b) * T + t) * static_cast<std::size_t>(D);
                const float* e = E.data + static_cast<std::size_t>(id) * static_cast<std::size_t>(D);
                const float* p = Pos.data + static_cast<std::size_t>(t) * static_cast<std::size_t>(D);
                for (int d = 0; d < D; ++d) row[d] = e[d] + p[d];
            }
        }
    }

    // s_t = (1-λ) s_{t-1} + λ x_t, H_t = x_t + α s_t. One (batch, channel) scan per thread.
    // peak, when set, is the max |H| gathered in the same write, while the value is in a register.
    void ema_forward(const Matrix& X, Matrix& Hout, int nb, float lambda, float* peak) {
        Hout.ensure(nb * T, D);
        const float keep = 1.f - lambda;
        const float a = alpha;
        const int width = D;
        constexpr int k_slots = 128;
        float slot[k_slots];
        for (int i = 0; i < k_slots; ++i) slot[i] = 0.f;
#if defined(_OPENMP)
#pragma omp parallel
#endif
        {
            float lp = 0.f;
#if defined(_OPENMP)
#pragma omp for schedule(static) nowait
#endif
            for (int bd = 0; bd < nb * width; ++bd) {
                const int b = bd / width;
                const int d = bd - b * width;
                float s = 0.f;
                const float* xbase = X.data + (static_cast<std::size_t>(b * T) * static_cast<std::size_t>(width) + d);
                float* hbase = Hout.data + (static_cast<std::size_t>(b * T) * static_cast<std::size_t>(width) + d);
                for (int t = 0; t < T; ++t) {
                    const float x = xbase[static_cast<std::size_t>(t) * static_cast<std::size_t>(width)];
                    s = keep * s + lambda * x;
                    const float h = x + a * s;
                    hbase[static_cast<std::size_t>(t) * static_cast<std::size_t>(width)] = h;
                    if (peak) lp = std::max(lp, std::fabs(h));
                }
            }
#if defined(_OPENMP)
            const int tid = omp_get_thread_num();
            if (peak && tid >= 0 && tid < k_slots) slot[tid] = lp;
#else
            if (peak) slot[0] = lp;
#endif
        }
        if (peak) {
            float m = 0.f;
            int nslot = 1;
#if defined(_OPENMP)
            nslot = omp_get_max_threads();
            if (nslot > k_slots) nslot = k_slots;
#endif
            for (int i = 0; i < nslot; ++i) m = std::max(m, slot[i]);
            *peak = m;
        }
    }

    // Adjoint, t = T-1 → 0: s = α e_t + (1-λ) s, then e_t += λ s. In place, width is D or V.
    void ema_backward(Matrix& Err, int nb, int width, float lambda) {
        const float keep = 1.f - lambda;
        const float a = alpha;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int bd = 0; bd < nb * width; ++bd) {
            const int b = bd / width;
            const int d = bd - b * width;
            float s = 0.f;
            float* ebase = Err.data + (static_cast<std::size_t>(b * T) * static_cast<std::size_t>(width) + d);
            for (int t = T - 1; t >= 0; --t) {
                float& e = ebase[static_cast<std::size_t>(t) * static_cast<std::size_t>(width)];
                s = a * e + keep * s;
                e += lambda * s;
            }
        }
    }

    // Batch-mean ReLU gate, one float per channel. Count with popcnt, not a float scan.
    void gate_mean(int layer, int rows, float* dst) const {
        const std::uint32_t* words = mask_ch[static_cast<std::size_t>(layer)].data();
        const int width = D;
        const float inv = rows > 0 ? 1.f / static_cast<float>(rows) : 0.f;
        for (int c = 0; c < width; ++c) dst[c] = 0.f;
        if (width % 32 == 0) {
            const int wpr = width >> 5;
            for (int lane = 0; lane < wpr; ++lane) {
                int cnt[32] = {};
                int r = 0;
                for (; r + 32 <= rows; r += 32) {
                    std::uint32_t gathered[32] = {};
                    for (int k = 0; k < 32; ++k) {
                        std::uint32_t w = words[(r + k) * wpr + lane];
                        while (w) {
                            unsigned long b = 0;
                            _BitScanForward(&b, w);
                            gathered[b] |= 1u << k;
                            w &= w - 1u;
                        }
                    }
                    for (int b = 0; b < 32; ++b) cnt[b] += static_cast<int>(_mm_popcnt_u32(gathered[b]));
                }
                for (; r < rows; ++r) {
                    const std::uint32_t w = words[r * wpr + lane];
                    for (int b = 0; b < 32; ++b) cnt[b] += static_cast<int>((w >> b) & 1u);
                }
                for (int b = 0; b < 32; ++b) dst[lane * 32 + b] = static_cast<float>(cnt[b]) * inv;
            }
            return;
        }
        for (int c = 0; c < width; ++c) {
            int on = 0;
            int r = 0;
            for (; r + 32 <= rows; r += 32) {
                std::uint32_t pack = 0;
                for (int k = 0; k < 32; ++k) {
                    const std::size_t idx = static_cast<std::size_t>(r + k) * static_cast<std::size_t>(width) +
                                            static_cast<std::size_t>(c);
                    if (relu_mask_bit(words, idx)) pack |= 1u << k;
                }
                on += static_cast<int>(_mm_popcnt_u32(pack));
            }
            for (; r < rows; ++r) {
                const std::size_t idx = static_cast<std::size_t>(r) * static_cast<std::size_t>(width) + static_cast<std::size_t>(c);
                if (relu_mask_bit(words, idx)) ++on;
            }
            dst[c] = static_cast<float>(on) * inv;
        }
    }

    void observe_mask(int layer, int rows) {
        mask_scratch.assign(static_cast<std::size_t>(D), 0.f);
        gate_mean(layer, rows, mask_scratch.data());
        auto& acc = rate_ch[static_cast<std::size_t>(layer)];
        for (int c = 0; c < D; ++c) acc[static_cast<std::size_t>(c)] += mask_scratch[static_cast<std::size_t>(c)];
    }

    void seed_ema_from_rates() {
        expect(rate_fill > 0, "ema seed needs the opening batches");
        const float inv = 1.f / static_cast<float>(rate_fill);
        m_ema.assign(static_cast<std::size_t>(blocks), std::vector<float>(static_cast<std::size_t>(D), 0.f));
        for (int l = 0; l < blocks; ++l) {
            const auto& acc = rate_ch[static_cast<std::size_t>(l)];
            auto& dst = m_ema[static_cast<std::size_t>(l)];
            for (int c = 0; c < D; ++c) dst[static_cast<std::size_t>(c)] = acc[static_cast<std::size_t>(c)] * inv;
        }
    }

    void update_ema_mask(int layer, int rows) {
        expect(static_cast<int>(m_ema.size()) == blocks, "ema mask was not seeded");
        mask_scratch.assign(static_cast<std::size_t>(D), 0.f);
        gate_mean(layer, rows, mask_scratch.data());
        auto& dst = m_ema[static_cast<std::size_t>(layer)];
        for (int c = 0; c < D; ++c)
            dst[static_cast<std::size_t>(c)] =
                ema_keep * dst[static_cast<std::size_t>(c)] + ema_blend * mask_scratch[static_cast<std::size_t>(c)];
    }

    void accumulate_mean_mask(int layer, int rows) {
        if (static_cast<int>(m_sum.size()) != blocks)
            m_sum.assign(static_cast<std::size_t>(blocks), std::vector<float>(static_cast<std::size_t>(D), 0.f));
        mask_scratch.assign(static_cast<std::size_t>(D), 0.f);
        gate_mean(layer, rows, mask_scratch.data());
        auto& dst = m_sum[static_cast<std::size_t>(layer)];
        for (int c = 0; c < D; ++c) dst[static_cast<std::size_t>(c)] += mask_scratch[static_cast<std::size_t>(c)];
    }

    int k_for(int step_done) const {
        if (align_k_early > 0 && step_done < align_k_until) return align_k_early;
        return align_k;
    }

    void forward(const int* tok, int nb, bool record_rates) {
        const auto t_fwd = std::chrono::steady_clock::now();
        embed(tok, nb);
        const int k_next = k_for(ticks + 1);
        const bool sweep_sample =
            !mean_mask && (gather_mask || (record_rates && sap && k_next > 0 && ((ticks + 1) % k_next == 0)));
        int src = 0;
        for (int l = 0; l < blocks; ++l) {
            const int dst = src ^ 1;
            Matrix& Hcur = int8_strip ? dH : H[static_cast<std::size_t>(l)];
            float peak = 0.f;
            ema_forward(xin[src], Hcur, nb, lam[l], int8_strip ? &peak : nullptr);
            const std::size_t nelt = static_cast<std::size_t>(nb) * static_cast<std::size_t>(T) * static_cast<std::size_t>(D);
            pre_ch.ensure(nb * T, D);
            if (qlora) {
                const auto& P = W0q[static_cast<std::size_t>(l)];
                qlora_gemm(Hcur.data, nb * T, D, P.q.data(), P.scale.data(), P.row_sum.data(), D, pre_ch.data);
            } else {
                gemm(Hcur, Wch[static_cast<std::size_t>(l)], pre_ch, false, false, 1.f, 0.f);
            }
            if (lora) {
                xa[static_cast<std::size_t>(l)].ensure(nb * T, r);
                gemm(Hcur, Al[static_cast<std::size_t>(l)], xa[static_cast<std::size_t>(l)], false, false, 1.f, 0.f);
                lora_y.ensure(nb * T, D);
                gemm(xa[static_cast<std::size_t>(l)], Bl[static_cast<std::size_t>(l)], lora_y, false, false, lora_s, 0.f);
                axpy(1.f, lora_y, pre_ch);
            }
            pack_relu_mask(pre_ch.data, mask_ch[static_cast<std::size_t>(l)].data(), nelt);
            mix_residual(xin[dst], Hcur, pre_ch, alpha, mask_ch[static_cast<std::size_t>(l)].data(), linear);
            if (int8_strip)
                quantize_int8_peak(Hcur.data, hq[static_cast<std::size_t>(l)].data(), nelt, peak,
                                   hscale[static_cast<std::size_t>(l)]);
            if (mean_mask && record_rates) {
                accumulate_mean_mask(l, nb * T);
            } else if (ema_mask) {
                update_ema_mask(l, nb * T);
            } else if (sweep_sample) {
                if (clusters > 1) observe_mask_m(l, nb * T, tok);
                else observe_mask(l, nb * T);
            }
            src = dst;
        }
        x_final = src;
        if (mean_mask && record_rates) ++m_count;
        if (sweep_sample) ++rate_fill;
        add_us(us_fwd, t_fwd);
        const auto t_out = std::chrono::steady_clock::now();
        logits.ensure(nb * T, V);
        gemm(xin[x_final], Wout, logits, false, false, 1.f, 0.f);
        add_us(us_out, t_out);
    }

    void scatter_input_grad(const int* tok, int nb) {
        dE.ensure(V, D);
        dPos.ensure(T, D);
        dE.zero();
        dPos.zero();
        for (int b = 0; b < nb; ++b) {
            for (int t = 0; t < T; ++t) {
                const int id = tok[static_cast<std::size_t>(b) * T + t];
                const float* src = g.data + (static_cast<std::size_t>(b) * T + t) * static_cast<std::size_t>(D);
                float* er = dE.data + static_cast<std::size_t>(id) * static_cast<std::size_t>(D);
                float* pr = dPos.data + static_cast<std::size_t>(t) * static_cast<std::size_t>(D);
                for (int d = 0; d < D; ++d) {
                    er[d] += src[d];
                    pr[d] += src[d];
                }
            }
        }
    }

    // dB = (x A)^T delta, dA = x^T (delta B^T). delta is the masked local gradient.
    void lora_dab(int layer, const Matrix& Hact, const Matrix& delta, int nb) {
        const auto t_dw = std::chrono::steady_clock::now();
        dBl[static_cast<std::size_t>(layer)].ensure(r, D);
        gemm(xa[static_cast<std::size_t>(layer)], delta, dBl[static_cast<std::size_t>(layer)], true, false, 1.f, 0.f);
        lora_mid.ensure(nb * T, r);
        gemm(delta, Bl[static_cast<std::size_t>(layer)], lora_mid, false, true, 1.f, 0.f);
        dAl[static_cast<std::size_t>(layer)].ensure(D, r);
        gemm(Hact, lora_mid, dAl[static_cast<std::size_t>(layer)], true, false, 1.f, 0.f);
        add_us(us_dw, t_dw);
    }

    // A chains the channel branch, then the EMA adjoint on the D-wide gradient.
    // B writes the channel step from P and runs the same adjoint on the V-wide error.
    // LoRA never forms dW. BP-LoRA still multiplies by the frozen W0^T; SAP-LoRA does not.
    void backward(const int* tok, const Matrix& Y, int nb) {
        const auto t_out = std::chrono::steady_clock::now();
        err.ensure(nb * T, V);
        softmax_ce(logits, Y, &err);
        if (sap && clusters > 1) build_cid(tok, nb * T);
        gemm(xin[x_final], err, dWout, true, false, 1.f, 0.f);
        add_us(us_out, t_out);
        const auto t_bwd = std::chrono::steady_clock::now();
        g.ensure(nb * T, D);
        if (!sap) gemm(err, Wout, g, false, true, 1.f, 0.f);
        add_us(us_bwd, t_bwd);
        const std::size_t hidden = static_cast<std::size_t>(nb * T * D);
        for (int l = blocks - 1; l >= 0; --l) {
            dH.ensure(nb * T, D);
            if (sap) {
                const auto t_proj = std::chrono::steady_clock::now();
                sap_delta.ensure(nb * T, D);
                if (clusters > 1) grouped_project(err, nb * T, false, l, sap_delta);
                else gemm(err, P[static_cast<std::size_t>(l)], sap_delta, false, true, 1.f, 0.f);
                if (!linear) apply_bit_mask(sap_delta, mask_ch[static_cast<std::size_t>(l)].data());
                scale_inplace(sap_delta, alpha);
                if (log_delta) {
                    if (static_cast<int>(delta_n.size()) != blocks) delta_n.assign(static_cast<std::size_t>(blocks), 0.f);
                    delta_n[static_cast<std::size_t>(l)] = frobenius(sap_delta);
                }
                if (capture_probe) {
                    if (l == blocks - 1) cap_sap_top = sap_delta;
                    if (l == 0) cap_sap_bot = sap_delta;
                }
                add_us(us_bwd, t_proj);
                const auto t_deq = std::chrono::steady_clock::now();
                dequantize_int8(hq[static_cast<std::size_t>(l)].data(), hidden, hscale[static_cast<std::size_t>(l)], dH.data);
                add_us(us_deq, t_deq);
                if (lora) {
                    lora_dab(l, dH, sap_delta, nb);
                } else {
                    const auto t_dw = std::chrono::steady_clock::now();
                    gemm(dH, sap_delta, dWch[static_cast<std::size_t>(l)], true, false, 1.f, 0.f);
                    add_us(us_dw, t_dw);
                }
                const auto t_ema = std::chrono::steady_clock::now();
                ema_backward(err, nb, V, lam[l]);
                add_us(us_bwd, t_ema);
            } else {
                const auto t_chain = std::chrono::steady_clock::now();
                Matrix& pre = dpre.n() ? dpre : sap_delta;
                pre.ensure(nb * T, D);
                std::memcpy(pre.data, g.data, hidden * sizeof(float));
                if (!linear) apply_bit_mask(pre, mask_ch[static_cast<std::size_t>(l)].data());
                scale_inplace(pre, alpha);
                if (capture_probe) {
                    if (l == blocks - 1) cap_bp_top = pre;
                    if (l == 0) cap_bp_bot = pre;
                }
                add_us(us_bwd, t_chain);
                if (int8_strip) {
                    const auto t_deq = std::chrono::steady_clock::now();
                    dequantize_int8(hq[static_cast<std::size_t>(l)].data(), hidden,
                                    hscale[static_cast<std::size_t>(l)], dH.data);
                    add_us(us_deq, t_deq);
                    if (lora) {
                        lora_dab(l, dH, pre, nb);
                    } else {
                        const auto t_dw = std::chrono::steady_clock::now();
                        gemm(dH, pre, dWch[static_cast<std::size_t>(l)], true, false, 1.f, 0.f);
                        add_us(us_dw, t_dw);
                    }
                } else {
                    if (lora) {
                        lora_dab(l, H[static_cast<std::size_t>(l)], pre, nb);
                    } else {
                        const auto t_dw = std::chrono::steady_clock::now();
                        gemm(H[static_cast<std::size_t>(l)], pre, dWch[static_cast<std::size_t>(l)], true, false, 1.f, 0.f);
                        add_us(us_dw, t_dw);
                    }
                }
                const auto t_chain2 = std::chrono::steady_clock::now();
                if (qlora) {
                    const auto& pack = W0q[static_cast<std::size_t>(l)];
                    gemm_dequant_nt(pre.data, nb * T, D, pack.q.data(), pack.scale.data(), D, dH.data);
                } else {
                    gemm(pre, Wch[static_cast<std::size_t>(l)], dH, false, true, 1.f, 0.f);
                }
                axpy(1.f, g, dH);
                ema_backward(dH, nb, D, lam[l]);
                std::memcpy(g.data, dH.data, hidden * sizeof(float));
                add_us(us_bwd, t_chain2);
            }
        }
        const auto t_tail = std::chrono::steady_clock::now();
        if (!lora) {
            if (sap) {
                if (clusters > 1) grouped_project(err, nb * T, true, 0, g);
                else gemm(err, P_in, g, false, true, 1.f, 0.f);
            }
            scatter_input_grad(tok, nb);
        }
        add_us(us_bwd, t_tail);
    }

    void scale_energy(Matrix& M) const {
        std::vector<double> s;
        double ss = 0.0;
        column_energy(M, s, ss);
        const float gain = std::sqrt(static_cast<float>(M.cols));
        const float mul = ss > 1e-18 && std::isfinite(ss) ? static_cast<float>(gain / std::sqrt(ss)) : 1.f;
        if (std::isfinite(mul)) scale_inplace(M, mul);
    }

    static int char_cluster(unsigned char ch, int M) {
        const bool lower = ch >= 'a' && ch <= 'z';
        if (M <= 2) return lower ? 0 : 1;
        if (lower) return (ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u') ? 0 : 1;
        if (ch == ' ' || ch == '\n') return 2;
        return 3;
    }

    void bind_clusters(const ShakeVocab& vocab) {
        expect(clusters == 2 || clusters == 4, "clusters must be 2 or 4");
        expect(vocab.V == V, "cluster vocab");
        cluster_of.assign(static_cast<std::size_t>(V), 0);
        for (int i = 0; i < V; ++i) {
            const unsigned char ch = static_cast<unsigned char>(vocab.chars[static_cast<std::size_t>(i)]);
            cluster_of[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(char_cluster(ch, clusters));
        }
    }

    void build_cid(const int* tok, int n) {
        cid.resize(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            const int id = tok[i];
            expect(id >= 0 && id < static_cast<int>(cluster_of.size()), "token outside cluster table");
            cid[static_cast<std::size_t>(i)] = cluster_of[static_cast<std::size_t>(id)];
        }
    }

    void observe_mask_m(int layer, int rows, const int* tok) {
        const std::uint32_t* words = mask_ch[static_cast<std::size_t>(layer)].data();
        auto& acc = rate_ch[static_cast<std::size_t>(layer)];
        auto& nn = rate_n[static_cast<std::size_t>(layer)];
        expect(static_cast<int>(acc.size()) == clusters * D, "cluster mask shape");
        const int wpr = D >> 5;
        for (int r = 0; r < rows; ++r) {
            const int id = tok[r];
            expect(id >= 0 && id < static_cast<int>(cluster_of.size()), "observe token");
            const int m = cluster_of[static_cast<std::size_t>(id)];
            nn[static_cast<std::size_t>(m)] += 1;
            float* dest = acc.data() + static_cast<std::size_t>(m) * static_cast<std::size_t>(D);
            if (wpr > 0 && (D & 31) == 0) {
                const std::uint32_t* row = words + static_cast<std::size_t>(r) * static_cast<std::size_t>(wpr);
                for (int lane = 0; lane < wpr; ++lane) {
                    std::uint32_t w = row[lane];
                    while (w) {
                        unsigned long b = 0;
                        _BitScanForward(&b, w);
                        dest[lane * 32 + static_cast<int>(b)] += 1.f;
                        w &= w - 1u;
                    }
                }
            } else {
                for (int c = 0; c < D; ++c) {
                    const std::size_t idx = static_cast<std::size_t>(r) * static_cast<std::size_t>(D) + static_cast<std::size_t>(c);
                    if (relu_mask_bit(words, idx)) dest[c] += 1.f;
                }
            }
        }
    }

    // Pack each cluster into a contiguous slice, one GEMM per cluster, scatter back.
    // logits and dH are free here: softmax has already written err, and dequant happens after.
    void grouped_project(const Matrix& Src, int ntok, bool use_pin, int layer, Matrix& Dest) {
        expect(clusters > 1 && clusters <= 4 && ntok > 0, "grouped project");
        expect(static_cast<int>(cid.size()) >= ntok, "cid missing");
        int cnt[4] = {};
        for (int i = 0; i < ntok; ++i) {
            const int m = cid[static_cast<std::size_t>(i)];
            expect(m >= 0 && m < clusters, "cid range");
            ++cnt[m];
        }
        int off[4];
        int cursor[4];
        int sum = 0;
        for (int m = 0; m < clusters; ++m) {
            off[m] = sum;
            cursor[m] = sum;
            sum += cnt[m];
        }
        expect(sum == ntok, "cluster partition");
        grp_order.resize(static_cast<std::size_t>(ntok));
        for (int i = 0; i < ntok; ++i) grp_order[static_cast<std::size_t>(cursor[cid[static_cast<std::size_t>(i)]]++)] = i;

        const int src_cols = Src.cols;
        const int dst_cols = Dest.cols;
        expect(logits.n() >= static_cast<std::size_t>(ntok) * static_cast<std::size_t>(src_cols), "logit scratch");
        expect(dH.n() >= static_cast<std::size_t>(ntok) * static_cast<std::size_t>(dst_cols), "dH scratch");
        for (int m = 0; m < clusters; ++m) {
            if (cnt[m] == 0) continue;
            logits.reshape(cnt[m], src_cols);
            dH.reshape(cnt[m], dst_cols);
            for (int j = 0; j < cnt[m]; ++j) {
                const int src = grp_order[static_cast<std::size_t>(off[m] + j)];
                std::memcpy(logits.data + static_cast<std::size_t>(j) * static_cast<std::size_t>(src_cols),
                            Src.data + static_cast<std::size_t>(src) * static_cast<std::size_t>(src_cols),
                            static_cast<std::size_t>(src_cols) * sizeof(float));
            }
            const Matrix& Pm = use_pin ? Pins[static_cast<std::size_t>(m)] : P[static_cast<std::size_t>(layer * clusters + m)];
            expect(Pm.rows == dst_cols && Pm.cols == src_cols, "P cluster shape");
            gemm(logits, Pm, dH, false, true, 1.f, 0.f);
            for (int j = 0; j < cnt[m]; ++j) {
                const int src = grp_order[static_cast<std::size_t>(off[m] + j)];
                std::memcpy(Dest.data + static_cast<std::size_t>(src) * static_cast<std::size_t>(dst_cols),
                            dH.data + static_cast<std::size_t>(j) * static_cast<std::size_t>(dst_cols),
                            static_cast<std::size_t>(dst_cols) * sizeof(float));
            }
        }
        logits.reshape(ntok, src_cols);
        dH.reshape(ntok, dst_cols);
    }

    void sweep_m(bool use_mask) {
        expect(clusters > 1, "sweep_m");
        P.assign(static_cast<std::size_t>(blocks * clusters), Matrix{});
        Pins.assign(static_cast<std::size_t>(clusters), Matrix{});
        P_in = Matrix{};
        std::vector<float> mbar(static_cast<std::size_t>(blocks) * static_cast<std::size_t>(clusters) * static_cast<std::size_t>(D), 1.f);
        if (use_mask) {
            for (int layer = 0; layer < blocks; ++layer) {
                const auto& acc = rate_ch[static_cast<std::size_t>(layer)];
                const auto& nn = rate_n[static_cast<std::size_t>(layer)];
                int g_n = 0;
                std::vector<float> g_on(static_cast<std::size_t>(D), 0.f);
                for (int m = 0; m < clusters; ++m) {
                    g_n += nn[static_cast<std::size_t>(m)];
                    for (int j = 0; j < D; ++j) g_on[static_cast<std::size_t>(j)] += acc[static_cast<std::size_t>(m) * D + j];
                }
                const float g_inv = g_n > 0 ? 1.f / static_cast<float>(g_n) : 0.f;
                for (int m = 0; m < clusters; ++m) {
                    float* dst = mbar.data() + (static_cast<std::size_t>(layer) * clusters + m) * static_cast<std::size_t>(D);
                    const int n = nn[static_cast<std::size_t>(m)];
                    if (n > 0) {
                        const float inv = 1.f / static_cast<float>(n);
                        const float* src = acc.data() + static_cast<std::size_t>(m) * static_cast<std::size_t>(D);
                        for (int j = 0; j < D; ++j) dst[j] = src[j] * inv;
                    } else {
                        for (int j = 0; j < D; ++j) dst[j] = g_on[static_cast<std::size_t>(j)] * g_inv;
                    }
                }
            }
            for (auto& acc : rate_ch) std::fill(acc.begin(), acc.end(), 0.f);
            for (auto& nn : rate_n) std::fill(nn.begin(), nn.end(), 0);
            rate_fill = 0;
        }
        for (int m = 0; m < clusters; ++m) {
            Matrix carried = Wout;
            auto place = [&](int layer, Matrix& src) { P[static_cast<std::size_t>(layer * clusters + m)] = src; };
            place(blocks - 1, carried);
            auto residual = [&](int layer) {
                Matrix scaled = carried;
                if (use_mask) {
                    const float* mb = mbar.data() + (static_cast<std::size_t>(layer) * clusters + m) * static_cast<std::size_t>(D);
                    for (int r = 0; r < scaled.rows; ++r) {
                        const float mv = mb[r];
                        for (int c = 0; c < scaled.cols; ++c) scaled(r, c) *= mv;
                    }
                }
                Matrix branch(D, carried.cols);
                gemm(Wch[static_cast<std::size_t>(layer)], scaled, branch, false, false, alpha, 0.f);
                axpy(1.f, branch, carried);
            };
            for (int l = blocks - 2; l >= 0; --l) {
                residual(l + 1);
                place(l, carried);
            }
            residual(0);
            Pins[static_cast<std::size_t>(m)] = carried;
        }
        ++sweep_calls;
    }

    void sweep(bool use_mask) {
        expect(sap, "sweep is the SAP path");
        if (clusters > 1) {
            sweep_m(use_mask);
            return;
        }
        if (static_cast<int>(P.size()) != blocks) P.assign(static_cast<std::size_t>(blocks), Matrix{});
        Matrix carried = Wout;
        auto place = [&](int layer, Matrix& src) {
            P[static_cast<std::size_t>(layer)] = src;
        };
        place(blocks - 1, carried);
        const bool use_mean = use_mask && mean_mask;
        if (use_mean) {
            expect(m_count > 0, "running mean is empty");
            expect(static_cast<int>(m_sum.size()) == blocks, "running mean layers");
        }
        const float inv = use_mask && !use_mean && rate_fill > 0 ? 1.f / static_cast<float>(rate_fill) : 1.f;
        const float inv_mean = use_mean ? 1.f / static_cast<float>(m_count) : 1.f;
        auto residual = [&](int layer) {
            Matrix scaled = carried;
            if (use_mask) {
                const auto& acc = use_mean ? m_sum[static_cast<std::size_t>(layer)] : rate_ch[static_cast<std::size_t>(layer)];
                for (int r = 0; r < scaled.rows; ++r) {
                    float m = 0.f;
                    if (use_mean) m = acc[static_cast<std::size_t>(r)] * inv_mean;
                    else if (ema_mask) m = m_ema[static_cast<std::size_t>(layer)][static_cast<std::size_t>(r)];
                    else m = acc[static_cast<std::size_t>(r)] * inv;
                    for (int c = 0; c < scaled.cols; ++c) scaled(r, c) *= m;
                }
            }
            Matrix branch(D, carried.cols);
            if (qlora) {
                Matrix Weff;
                materialize_w0(layer, Weff);
                gemm(Al[static_cast<std::size_t>(layer)], Bl[static_cast<std::size_t>(layer)], Weff, false, false, lora_s,
                     1.f);
                gemm(Weff, scaled, branch, false, false, alpha, 0.f);
            } else if (lora) {
                Matrix Weff = Wch[static_cast<std::size_t>(layer)];
                gemm(Al[static_cast<std::size_t>(layer)], Bl[static_cast<std::size_t>(layer)], Weff, false, false, lora_s,
                     1.f);
                gemm(Weff, scaled, branch, false, false, alpha, 0.f);
            } else {
                gemm(Wch[static_cast<std::size_t>(layer)], scaled, branch, false, false, alpha, 0.f);
            }
            axpy(1.f, branch, carried);
        };
        for (int l = blocks - 2; l >= 0; --l) {
            residual(l + 1);
            place(l, carried);
        }
        residual(0);
        P_in = carried;
        if (use_mask) {
            for (auto& acc : rate_ch) std::fill(acc.begin(), acc.end(), 0.f);
            rate_fill = 0;
            if (use_mean) {
                for (auto& row : m_sum) std::fill(row.begin(), row.end(), 0.f);
                m_count = 0;
            }
        }
        ++sweep_calls;
    }

    void init_feedback() {
        sap = true;
        sweep(rate_fill > 0);
        sweep_calls = 0;
    }

    double grad_ss() const {
        double ss = 0.0;
        if (lora) {
            for (int l = 0; l < blocks; ++l) {
                const double a = frobenius(dAl[static_cast<std::size_t>(l)]);
                const double b = frobenius(dBl[static_cast<std::size_t>(l)]);
                ss += a * a + b * b;
            }
        } else {
            for (int l = 0; l < blocks; ++l) {
                const double b = frobenius(dWch[static_cast<std::size_t>(l)]);
                ss += b * b;
            }
        }
        const double c = frobenius(dWout);
        ss += c * c;
        if (dE.n()) {
            const double e = frobenius(dE);
            ss += e * e;
        }
        if (dPos.n()) {
            const double p = frobenius(dPos);
            ss += p * p;
        }
        return ss;
    }

    void apply(float lr) {
        if (lora && ab_clip > 0.f) {
            double ss = 0.0;
            for (int l = 0; l < blocks; ++l) {
                const double a = frobenius(dAl[static_cast<std::size_t>(l)]);
                const double b = frobenius(dBl[static_cast<std::size_t>(l)]);
                ss += a * a + b * b;
            }
            const float nrm = static_cast<float>(std::sqrt(ss));
            if (nrm > ab_clip && std::isfinite(nrm)) {
                const float s = ab_clip / nrm;
                for (int l = 0; l < blocks; ++l) {
                    scale_inplace(dAl[static_cast<std::size_t>(l)], s);
                    scale_inplace(dBl[static_cast<std::size_t>(l)], s);
                }
                ++ab_clip_hits;
            }
        }
        const float gnorm = static_cast<float>(std::sqrt(grad_ss()));
        float scale = 1.f;
        if (clip && max_norm > 0.f && gnorm > max_norm && std::isfinite(gnorm)) {
            scale = max_norm / gnorm;
            ++clip_hits;
        }
        last_unused_scale(scale);
        if (lora) {
            auto step_sgd_l = [&](Matrix& V, Matrix& W, const Matrix& dW) {
                for (std::size_t i = 0; i < W.n(); ++i) V.data[i] = mom * V.data[i] + scale * dW.data[i];
                for (std::size_t i = 0; i < W.n(); ++i) W.data[i] -= lr * (V.data[i] + wd * W.data[i]);
            };
            if (!adam) {
                for (int l = 0; l < blocks; ++l) {
                    step_sgd_l(VAl[static_cast<std::size_t>(l)], Al[static_cast<std::size_t>(l)], dAl[static_cast<std::size_t>(l)]);
                    step_sgd_l(VBl[static_cast<std::size_t>(l)], Bl[static_cast<std::size_t>(l)], dBl[static_cast<std::size_t>(l)]);
                }
                step_sgd_l(Vout, Wout, dWout);
                return;
            }
            const int t = ticks + 1;
            const float b1 = mom;
            const double bc1 = 1.0 - std::pow(static_cast<double>(b1), t);
            const double bc2 = 1.0 - std::pow(static_cast<double>(b2), t);
            auto step_adam_l = [&](Matrix& V, Matrix& S, Matrix& W, const Matrix& dW) {
                for (std::size_t i = 0; i < W.n(); ++i) {
                    const float graw = scale * dW.data[i];
                    V.data[i] = b1 * V.data[i] + (1.f - b1) * graw;
                    S.data[i] = b2 * S.data[i] + (1.f - b2) * graw * graw;
                    const float mhat = static_cast<float>(static_cast<double>(V.data[i]) / bc1);
                    const float vhat = static_cast<float>(static_cast<double>(S.data[i]) / bc2);
                    W.data[i] -= lr * (mhat / (std::sqrt(vhat) + adam_eps) + wd * W.data[i]);
                }
            };
            for (int l = 0; l < blocks; ++l) {
                step_adam_l(VAl[static_cast<std::size_t>(l)], SAl[static_cast<std::size_t>(l)], Al[static_cast<std::size_t>(l)],
                            dAl[static_cast<std::size_t>(l)]);
                step_adam_l(VBl[static_cast<std::size_t>(l)], SBl[static_cast<std::size_t>(l)], Bl[static_cast<std::size_t>(l)],
                            dBl[static_cast<std::size_t>(l)]);
            }
            step_adam_l(Vout, Svout, Wout, dWout);
            return;
        }
        auto step_sgd = [&](Matrix& V, Matrix& W, const Matrix& dW) {
            for (std::size_t i = 0; i < W.n(); ++i) V.data[i] = mom * V.data[i] + scale * dW.data[i];
            for (std::size_t i = 0; i < W.n(); ++i) W.data[i] -= lr * (V.data[i] + wd * W.data[i]);
        };
        if (!adam) {
            for (int l = 0; l < blocks; ++l)
                step_sgd(Vch[static_cast<std::size_t>(l)], Wch[static_cast<std::size_t>(l)], dWch[static_cast<std::size_t>(l)]);
            step_sgd(Vout, Wout, dWout);
            step_sgd(Ve, E, dE);
            step_sgd(Vp, Pos, dPos);
            return;
        }
        const int t = ticks + 1;
        const float b1 = mom;
        const double bc1 = 1.0 - std::pow(static_cast<double>(b1), t);
        const double bc2 = 1.0 - std::pow(static_cast<double>(b2), t);
        auto step_adam = [&](Matrix& V, Matrix& S, Matrix& W, const Matrix& dW) {
            for (std::size_t i = 0; i < W.n(); ++i) {
                const float graw = scale * dW.data[i];
                V.data[i] = b1 * V.data[i] + (1.f - b1) * graw;
                S.data[i] = b2 * S.data[i] + (1.f - b2) * graw * graw;
                const float mhat = static_cast<float>(static_cast<double>(V.data[i]) / bc1);
                const float vhat = static_cast<float>(static_cast<double>(S.data[i]) / bc2);
                W.data[i] -= lr * (mhat / (std::sqrt(vhat) + adam_eps) + wd * W.data[i]);
            }
        };
        for (int l = 0; l < blocks; ++l) {
            step_adam(Vch[static_cast<std::size_t>(l)], Sv[static_cast<std::size_t>(l)], Wch[static_cast<std::size_t>(l)],
                      dWch[static_cast<std::size_t>(l)]);
        }
        step_adam(Vout, Svout, Wout, dWout);
        step_adam(Ve, Sve, E, dE);
        step_adam(Vp, Svp, Pos, dPos);
    }

    float last_scale = 1.f;
    void last_unused_scale(float scale) {
        last_scale = scale;
        scale_sum += scale;
        ++scale_n;
    }

    // Same weights, same batch. Block grads are the masked, alpha-scaled local
    // gradients (dpre versus sap_delta). E is dE. Ratio is ||g_SAP|| / ||g_BP||.
    void align_cosine(const int* tok, const Matrix& Y, int nb, AlignView& view) {
        const bool was = sap;
        const bool was_tally = tally;
        tally = false;
        capture_probe = true;
        sap = false;
        backward(tok, Y, nb);
        Matrix bp_e;
        if (!lora) bp_e = dE;
        sap = true;
        backward(tok, Y, nb);
        capture_probe = false;
        auto ratio = [](const Matrix& g_sap, const Matrix& g_bp) {
            const float bpn = frobenius(g_bp);
            if (!(bpn > 1e-12f) || !std::isfinite(bpn)) return 0.f;
            return frobenius(g_sap) / bpn;
        };
        view.cos_top = cosine_matrix(cap_bp_top, cap_sap_top);
        view.cos_bot = cosine_matrix(cap_bp_bot, cap_sap_bot);
        view.nr_top = ratio(cap_sap_top, cap_bp_top);
        view.nr_bot = ratio(cap_sap_bot, cap_bp_bot);
        if (!lora) {
            view.cos_e = cosine_matrix(bp_e, dE);
            view.nr_e = ratio(dE, bp_e);
        } else {
            view.cos_e = 0.f;
            view.nr_e = 0.f;
        }
        cap_bp_top = Matrix{};
        cap_bp_bot = Matrix{};
        cap_sap_top = Matrix{};
        cap_sap_bot = Matrix{};
        sap = was;
        tally = was_tally;
    }

    float step(const int* tok, const Matrix& Y, float lr, bool do_apply, AlignView* align = nullptr) {
        tally = true;
        forward(tok, batch, true);
        const auto t_loss = std::chrono::steady_clock::now();
        const float loss = softmax_ce(logits, Y, nullptr);
        add_us(us_out, t_loss);
        if (align && sap) align_cosine(tok, Y, batch, *align);
        else backward(tok, Y, batch);
        if (do_apply) {
            const auto t_upd = std::chrono::steady_clock::now();
            apply(lr);
            ++ticks;
            const int k_now = k_for(ticks);
            if (sap && k_now > 0 && ticks % k_now == 0) sweep(true);
            add_us(us_upd, t_upd);
        }
        tally = false;
        return loss;
    }

    void reset_profile() {
        us_fwd = us_out = us_bwd = us_dw = us_deq = us_upd = 0;
    }
};

void shake_fill_batch(const std::vector<std::uint8_t>& text, const ShakeVocab& vocab, const int* starts, int nb, int T,
                      std::vector<int>& tok, Matrix& Y) {
    tok.resize(static_cast<std::size_t>(nb) * static_cast<std::size_t>(T));
    Y.ensure(nb * T, vocab.V);
    Y.zero();
    for (int b = 0; b < nb; ++b) {
        const int s = starts[b];
        for (int t = 0; t < T; ++t) {
            const int id = vocab.id[text[static_cast<std::size_t>(s + t)]];
            const int nxt = vocab.id[text[static_cast<std::size_t>(s + t + 1)]];
            tok[static_cast<std::size_t>(b) * T + t] = id;
            Y(b * T + t, nxt) = 1.f;
        }
    }
}

float shake_bpc(Shake& net, const std::vector<std::uint8_t>& text, const ShakeVocab& vocab, int begin, int end) {
    const int T = net.T;
    const int nb = net.batch;
    int nwin = 0;
    for (int s = begin; s + T < end; s += T) ++nwin;
    if (nwin <= 0) return 0.f;
    std::vector<int> tok;
    Matrix Y;
    double sum = 0.0;
    int seen = 0;
    std::vector<int> starts(static_cast<std::size_t>(nb), begin);
    int cursor = begin;
    while (cursor + T < end) {
        int b = 0;
        for (; b < nb && cursor + T < end; ++b, cursor += T) starts[static_cast<std::size_t>(b)] = cursor;
        shake_fill_batch(text, vocab, starts.data(), b, T, tok, Y);
        const int keep = net.batch;
        net.batch = b;
        net.forward(tok.data(), b, false);
        sum += static_cast<double>(softmax_ce(net.logits, Y, nullptr)) * static_cast<double>(b);
        seen += b;
        net.batch = keep;
    }
    expect(seen > 0, "empty shake val");
    return static_cast<float>((sum / static_cast<double>(seen)) / std::log(2.0));
}

std::string shake_generate(Shake& net, const ShakeVocab& vocab, const std::string& prompt, int nchar, float temperature,
                           int top_k, std::uint32_t seed) {
    expect(temperature > 0.f && top_k > 0, "sample params");
    std::vector<int> win(static_cast<std::size_t>(net.T), 0);
    int filled = 0;
    std::string out = prompt;
    for (unsigned char ch : prompt) {
        const int id = vocab.id[ch];
        expect(id >= 0, "prompt character missing from vocab");
        if (filled < net.T) win[static_cast<std::size_t>(filled++)] = id;
        else {
            std::memmove(win.data(), win.data() + 1, static_cast<std::size_t>(net.T - 1) * sizeof(int));
            win[static_cast<std::size_t>(net.T - 1)] = id;
        }
    }
    expect(filled > 0, "empty prompt");
    std::mt19937 rng(seed);
    std::vector<int> order(static_cast<std::size_t>(net.V));
    for (int i = 0; i < nchar; ++i) {
        const int keep = net.batch;
        net.batch = 1;
        net.forward(win.data(), 1, false);
        net.batch = keep;
        const int pos = std::min(filled, net.T) - 1;
        const float* row = net.logits.data + static_cast<std::size_t>(pos) * static_cast<std::size_t>(net.V);
        for (int k = 0; k < net.V; ++k) order[static_cast<std::size_t>(k)] = k;
        const int keep_k = std::min(top_k, net.V);
        std::partial_sort(order.begin(), order.begin() + keep_k, order.end(), [&](int a, int b) {
            return row[a] > row[b];
        });
        float m = row[order[0]] / temperature;
        for (int k = 1; k < keep_k; ++k) m = std::max(m, row[order[static_cast<std::size_t>(k)]] / temperature);
        double sum = 0.0;
        std::vector<double> p(static_cast<std::size_t>(keep_k));
        for (int k = 0; k < keep_k; ++k) {
            p[static_cast<std::size_t>(k)] = std::exp(static_cast<double>(row[order[static_cast<std::size_t>(k)]] / temperature - m));
            sum += p[static_cast<std::size_t>(k)];
        }
        std::uniform_real_distribution<double> pick(0.0, sum);
        double r = pick(rng);
        int choice = order[static_cast<std::size_t>(keep_k - 1)];
        for (int k = 0; k < keep_k; ++k) {
            r -= p[static_cast<std::size_t>(k)];
            if (r <= 0.0) {
                choice = order[static_cast<std::size_t>(k)];
                break;
            }
        }
        out.push_back(vocab.chars[static_cast<std::size_t>(choice)]);
        if (filled < net.T) win[static_cast<std::size_t>(filled++)] = choice;
        else {
            std::memmove(win.data(), win.data() + 1, static_cast<std::size_t>(net.T - 1) * sizeof(int));
            win[static_cast<std::size_t>(net.T - 1)] = choice;
        }
    }
    return out;
}

void shake_self_test() {
    {
        Shake net;
        net.configure(8, 16, 10, 3, 2);
        net.init_weights(7);
        net.prepare();
        std::vector<int> tok(static_cast<std::size_t>(2 * 8), 1);
        tok[3] = 4;
        net.forward(tok.data(), 2, false);
        Matrix before = net.logits;
        tok[3] = 9;
        net.forward(tok.data(), 2, false);
        const int Dpos = 16;
        const int V = 10;
        const int T = 8;
        for (int pos = 0; pos < 3; ++pos) {
            for (int k = 0; k < V; ++k) {
                const float a = before(pos, k);
                const float b = net.logits(pos, k);
                expect(a == b, "causal leak: a future token changed the past");
            }
        }
        bool changed = false;
        for (int k = 0; k < V; ++k) {
            if (before(3, k) != net.logits(3, k)) changed = true;
        }
        expect(changed, "causal mix ignored the present token");
        (void)Dpos;
        (void)T;
    }
    {
        Shake net;
        net.linear = true;
        net.configure(4, 4, 3, 2, 2);
        net.clip = false;
        net.init_weights(3);
        net.prepare();
        std::vector<int> tok = {0, 1, 2, 0, 1, 2, 0, 1};
        Matrix Y(8, 3);
        Y.zero();
        for (int i = 0; i < 8; ++i) Y(i, (tok[static_cast<std::size_t>(i)] + 1) % 3) = 1.f;
        net.forward(tok.data(), 2, false);
        const float base = softmax_ce(net.logits, Y, nullptr);
        net.backward(tok.data(), Y, 2);
        const float analytic = net.dWch[0](1, 2);
        const float eps = 1e-3f;
        net.Wch[0](1, 2) += eps;
        net.forward(tok.data(), 2, false);
        const float lp = softmax_ce(net.logits, Y, nullptr);
        net.Wch[0](1, 2) -= 2.f * eps;
        net.forward(tok.data(), 2, false);
        const float lm = softmax_ce(net.logits, Y, nullptr);
        net.Wch[0](1, 2) += eps;
        const float numeric = (lp - lm) / (2.f * eps);
        expect(std::fabs(numeric - analytic) <= 1e-3f + 0.05f * std::fabs(analytic), "shake channel grad");
        (void)base;
    }
    {
        Shake A;
        A.configure(4, 8, 5, 2, 4);
        A.clip = false;
        A.init_weights(11);
        A.prepare();
        Shake B = A;
        B.sap = true;
        B.align_k = 4;
        B.prepare();
        B.init_feedback();
        std::vector<int> tok(static_cast<std::size_t>(4 * 4));
        Matrix Y(16, 5);
        for (int i = 0; i < 16; ++i) tok[static_cast<std::size_t>(i)] = i % 5;
        Y.zero();
        for (int i = 0; i < 16; ++i) Y(i, (tok[static_cast<std::size_t>(i)] + 1) % 5) = 1.f;
        const float a0 = A.step(tok.data(), Y, 0.05f, true);
        const float b0 = B.step(tok.data(), Y, 0.05f, true);
        float a1 = a0;
        float b1 = b0;
        for (int s = 0; s < 40; ++s) {
            a1 = A.step(tok.data(), Y, 0.05f, true);
            b1 = B.step(tok.data(), Y, 0.05f, true);
        }
        expect(std::isfinite(a1) && a1 < a0 * 0.7f, "shake backprop did not learn " + std::to_string(a0) + " -> " + std::to_string(a1));
        expect(std::isfinite(b1) && b1 < b0 * 0.85f, "shake SAP did not learn " + std::to_string(b0) + " -> " + std::to_string(b1));
    }
    {
        Shake B;
        B.configure(4, 8, 5, 2, 4);
        B.sap = true;
        B.clusters = 2;
        B.clip = false;
        B.align_k = 2;
        B.cluster_of = {0, 1, 0, 1, 0};
        B.init_weights(5);
        B.prepare();
        B.init_feedback();
        expect(static_cast<int>(B.P.size()) == 4 && static_cast<int>(B.Pins.size()) == 2, "cluster feedback count");
        std::vector<int> tok(static_cast<std::size_t>(4 * 4));
        Matrix Y(16, 5);
        for (int i = 0; i < 16; ++i) tok[static_cast<std::size_t>(i)] = i % 5;
        Y.zero();
        for (int i = 0; i < 16; ++i) Y(i, (tok[static_cast<std::size_t>(i)] + 1) % 5) = 1.f;
        const float b0 = B.step(tok.data(), Y, 0.05f, true);
        float b1 = b0;
        for (int s = 0; s < 30; ++s) b1 = B.step(tok.data(), Y, 0.05f, true);
        expect(std::isfinite(b1) && b1 < b0, "clustered SAP did not learn " + std::to_string(b0) + " -> " + std::to_string(b1));
        expect(B.Pins[0].n() > 0 && B.Pins[1].n() > 0, "both cluster inputs missing");
    }
    {
        Shake net;
        net.linear = true;
        net.clip = false;
        net.configure(4, 8, 5, 2, 2);
        net.init_weights(3);
        net.prepare();
        net.attach_lora(9);
        expect(net.dWch.empty() && net.Vch.empty() && net.dE.n() == 0, "lora still holds frozen grads");
        net.Bl[0](2, 4) = 0.1f;
        std::vector<int> tok = {0, 1, 2, 0, 1, 2, 0, 1};
        Matrix Y(8, 5);
        Y.zero();
        for (int i = 0; i < 8; ++i) Y(i, (tok[static_cast<std::size_t>(i)] + 1) % 5) = 1.f;
        const Matrix w0 = net.Wch[0];
        const Matrix e0 = net.E;
        const Matrix p0 = net.Pos;
        net.forward(tok.data(), 2, false);
        net.backward(tok.data(), Y, 2);
        const float aA = net.dAl[0](3, 5);
        const float aB = net.dBl[0](2, 4);
        expect(std::fabs(aA) + std::fabs(aB) > 1e-8f, "lora grad vanished");
        auto fd = [&](Matrix& M, int i, int j, float analytic, const char* what) {
            const float eps = 1e-3f;
            M(i, j) += eps;
            net.forward(tok.data(), 2, false);
            const float lp = softmax_ce(net.logits, Y, nullptr);
            M(i, j) -= 2.f * eps;
            net.forward(tok.data(), 2, false);
            const float lm = softmax_ce(net.logits, Y, nullptr);
            M(i, j) += eps;
            const float numeric = (lp - lm) / (2.f * eps);
            expect(std::fabs(numeric - analytic) <= 1e-3f + 0.05f * std::fabs(analytic),
                   std::string(what) + " " + std::to_string(numeric) + " vs " + std::to_string(analytic));
        };
        fd(net.Al[0], 3, 5, aA, "lora dA");
        fd(net.Bl[0], 2, 4, aB, "lora dB");
        net.step(tok.data(), Y, 0.01f, true);
        for (std::size_t i = 0; i < w0.n(); ++i) expect(w0.data[i] == net.Wch[0].data[i], "lora thawed W0");
        for (std::size_t i = 0; i < e0.n(); ++i) expect(e0.data[i] == net.E.data[i], "lora thawed E");
        for (std::size_t i = 0; i < p0.n(); ++i) expect(p0.data[i] == net.Pos.data[i], "lora thawed Pos");
    }
    {
        Shake B;
        B.sap = true;
        B.align_k = 4;
        B.clip = false;
        B.configure(4, 8, 5, 2, 4);
        B.init_weights(11);
        B.prepare();
        B.attach_lora(4);
        B.init_feedback();
        std::vector<int> tok(static_cast<std::size_t>(4 * 4));
        Matrix Y(16, 5);
        for (int i = 0; i < 16; ++i) tok[static_cast<std::size_t>(i)] = i % 5;
        Y.zero();
        for (int i = 0; i < 16; ++i) Y(i, (tok[static_cast<std::size_t>(i)] + 1) % 5) = 1.f;
        const Matrix w0 = B.Wch[0];
        const float b0 = B.step(tok.data(), Y, 0.05f, true);
        float b1 = b0;
        for (int s = 0; s < 40; ++s) b1 = B.step(tok.data(), Y, 0.05f, true);
        expect(std::isfinite(b1) && b1 < b0, "SAP-LoRA did not learn " + std::to_string(b0) + " -> " + std::to_string(b1));
        for (std::size_t i = 0; i < w0.n(); ++i) expect(w0.data[i] == B.Wch[0].data[i], "SAP-LoRA thawed W0");
    }
    {
        const int din = 96;
        const int dout = 3;
        const int rows = 4;
        std::mt19937 rng(5);
        std::uniform_int_distribution<int> iq(-126, 126);
        std::vector<int> xq(static_cast<std::size_t>(rows * din));
        std::vector<int> wq(static_cast<std::size_t>(din * dout));
        for (int& v : xq) v = iq(rng);
        for (int& v : wq) v = iq(rng);
        for (int n = 0; n < rows; ++n) xq[static_cast<std::size_t>(n * din)] = 127;
        for (int j = 0; j < dout; ++j) {
            wq[static_cast<std::size_t>(j)] = 127;
            wq[static_cast<std::size_t>(dout + j)] = -127;
        }
        Matrix X(rows, din);
        Matrix W(din, dout);
        for (int n = 0; n < rows; ++n)
            for (int i = 0; i < din; ++i) X(n, i) = static_cast<float>(xq[static_cast<std::size_t>(n * din + i)]);
        for (int i = 0; i < din; ++i)
            for (int j = 0; j < dout; ++j) W(i, j) = static_cast<float>(wq[static_cast<std::size_t>(i * dout + j)]);
        std::vector<std::int8_t> q(static_cast<std::size_t>(dout * din));
        std::vector<float> scale(static_cast<std::size_t>(dout));
        std::vector<std::int32_t> rsum(static_cast<std::size_t>(dout));
        qlora_pack(W.data, din, dout, q.data(), scale.data(), rsum.data());
        Matrix Ys(rows, dout);
        Matrix Ya(rows, dout);
        Matrix Yv(rows, dout);
        qlora_gemm_scalar(X.data, rows, din, q.data(), scale.data(), rsum.data(), dout, Ys.data);
        qlora_gemm_avx2(X.data, rows, din, q.data(), scale.data(), rsum.data(), dout, Ya.data);
        const bool vnni = qlora_vnni_ok();
        if (vnni) qlora_gemm_vnni(X.data, rows, din, q.data(), scale.data(), rsum.data(), dout, Yv.data);
        for (int n = 0; n < rows; ++n) {
            for (int j = 0; j < dout; ++j) {
                std::int32_t acc = 0;
                for (int i = 0; i < din; ++i)
                    acc += xq[static_cast<std::size_t>(n * din + i)] * wq[static_cast<std::size_t>(i * dout + j)];
                const float want = static_cast<float>(acc);
                expect(Ys(n, j) == want, "qlora scalar missed the integer dot");
                expect(Ya(n, j) == Ys(n, j), "qlora avx2 diverged from scalar");
                if (vnni) expect(Yv(n, j) == Ys(n, j), "qlora vnni diverged from scalar");
            }
        }
        Matrix Xr(4, 48);
        Matrix Wr(48, 16);
        Matrix Yref(4, 16);
        Matrix Yhat(4, 16);
        Matrix Yhat2(4, 16);
        fill_normal(Xr, rng, 0.4f);
        fill_normal(Wr, rng, 0.15f);
        gemm(Xr, Wr, Yref, false, false, 1.f, 0.f);
        std::vector<std::int8_t> q2(static_cast<std::size_t>(16 * 48));
        std::vector<float> sc2(16);
        std::vector<std::int32_t> rs2(16);
        qlora_pack(Wr.data, 48, 16, q2.data(), sc2.data(), rs2.data());
        qlora_gemm_scalar(Xr.data, 4, 48, q2.data(), sc2.data(), rs2.data(), 16, Yhat.data);
        qlora_gemm_avx2(Xr.data, 4, 48, q2.data(), sc2.data(), rs2.data(), 16, Yhat2.data);
        float px = 0.f;
        float pw = 0.f;
        for (std::size_t i = 0; i < Xr.n(); ++i) px = std::max(px, std::fabs(Xr.data[i]));
        for (std::size_t i = 0; i < Wr.n(); ++i) pw = std::max(pw, std::fabs(Wr.data[i]));
        const float bound = 48.f * px * pw * (2.f / 127.f) + 1e-3f;
        for (std::size_t i = 0; i < Yref.n(); ++i) {
            expect(Yhat2.data[i] == Yhat.data[i], "qlora avx2 float path diverged");
            expect(std::fabs(Yhat.data[i] - Yref.data[i]) <= bound, "qlora far from fp32");
        }
        if (vnni) {
            Matrix Yhat3(4, 16);
            qlora_gemm_vnni(Xr.data, 4, 48, q2.data(), sc2.data(), rs2.data(), 16, Yhat3.data);
            for (std::size_t i = 0; i < Yref.n(); ++i)
                expect(Yhat3.data[i] == Yhat.data[i], "qlora vnni float path diverged");
        }
        std::cout << "[ok] qlora int8 dot  " << qlora_kernel() << "\n";
        {
            const int rows = 9;
            const int din = 200;
            const int dout = 40;
            std::mt19937 rngd(9);
            Matrix dy(rows, dout);
            Matrix Wd(din, dout);
            fill_normal(dy, rngd, 0.3f);
            fill_normal(Wd, rngd, 0.2f);
            std::vector<std::int8_t> qd(static_cast<std::size_t>(dout) * static_cast<std::size_t>(din));
            std::vector<float> scd(static_cast<std::size_t>(dout));
            std::vector<std::int32_t> rsd(static_cast<std::size_t>(dout));
            qlora_pack(Wd.data, din, dout, qd.data(), scd.data(), rsd.data());
            Matrix Weff(din, dout);
            for (int j = 0; j < dout; ++j) {
                const float s = scd[static_cast<std::size_t>(j)];
                const std::int8_t* src = qd.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(din);
                for (int i = 0; i < din; ++i) Weff(i, j) = static_cast<float>(src[i]) * s;
            }
            Matrix ref(rows, din);
            Matrix hat(rows, din);
            gemm(dy, Weff, ref, false, true, 1.f, 0.f);
            gemm_dequant_nt(dy.data, rows, dout, qd.data(), scd.data(), din, hat.data);
            for (std::size_t i = 0; i < ref.n(); ++i)
                expect(hat.data[i] == ref.data[i], "dequant dgrad diverged from W^T");
            std::cout << "[ok] qlora dequant dgrad\n";
        }
    }
    std::cout << "[ok] causal ema mixer and SAP step\n";
}

void run_shake_ablation(const Config& cfg) {
    if ((cfg.trio && cfg.adam) || (cfg.phase2 && (cfg.trio || cfg.adam)))
        throw std::runtime_error("trio, adam, and phase2 are separate runs");
    const std::vector<std::uint8_t> text = read_bytes(cfg.shake);
    const ShakeVocab vocab = shake_vocab(text);
    expect(vocab.V == 65, "TinyShakespeare vocab is not 65");
    const int train_end = static_cast<int>(text.size() * 9 / 10);
    const int val_begin = train_end;
    const int val_end = static_cast<int>(text.size());
    expect(train_end > k_shake_T + 2 && val_end - val_begin > k_shake_T + 2, "split shorter than the window");

    struct Spec {
        const char* name;
        bool sap;
        bool i8;
        int clusters;
    };
    Spec specs[3];
    int n_net = 0;
    if (cfg.trio) {
        specs[n_net++] = {"A1", false, false, 1};
        specs[n_net++] = {"A2", false, true, 1};
        specs[n_net++] = {"B", true, true, 1};
    } else if (cfg.phase2) {
        specs[n_net++] = {"A2", false, true, 1};
        specs[n_net++] = {"M2", true, true, 2};
        specs[n_net++] = {"M4", true, true, 4};
    } else {
        specs[n_net++] = {"A2", false, true, 1};
        specs[n_net++] = {"B", true, true, 1};
    }
    const bool use_adam = cfg.adam;
    const float wd = use_adam && cfg.wd == 0.f ? 1e-4f : cfg.wd;
    const int K = cfg.align_k > 0 ? cfg.align_k : 10;

    Shake nets[3];
    for (int i = 0; i < n_net; ++i) {
        Shake& net = nets[i];
        net.configure(k_shake_T, k_shake_D, vocab.V, k_shake_blocks, cfg.batch);
        net.max_norm = cfg.max_norm;
        net.clip = cfg.clip;
        net.align_k = K;
        net.mom = k_shake_mom;
        net.sap = specs[i].sap;
        net.int8_strip = specs[i].i8;
        net.adam = use_adam;
        net.wd = wd;
        net.clusters = specs[i].clusters;
        const std::size_t b0 = g_mem.live;
        net.init_weights(cfg.seed);
        if (net.clusters > 1) net.bind_clusters(vocab);
        net.prepare();
        if (net.sap) net.init_feedback();
        const std::size_t mats = g_mem.live - b0;
        expect(mats == net.matrix_bytes(),
               std::string(specs[i].name) + " memory counter " + std::to_string(mats) + " != " +
                   std::to_string(net.matrix_bytes()));
        net.dump_mem(specs[i].name);
    }
    for (int i = 1; i < n_net; ++i)
        expect(nets[0].Wch[0](0, 0) == nets[i].Wch[0](0, 0), "models did not share the seed");

    std::mt19937 rng(cfg.seed + 99u);
    std::uniform_int_distribution<int> pick(0, train_end - k_shake_T - 2);
    std::vector<int> starts(static_cast<std::size_t>(cfg.batch));
    std::vector<int> tok;
    Matrix Y;
    const double ln2 = std::log(2.0);
    double used[3] = {};
    double sum[3] = {};
    int seen[3] = {};
    int steps_n[3] = {};
    bool alive[3] = {true, true, true};
    float last_val[3] = {};

    std::cout << "\nshakespeare " << cfg.shake << "  bytes " << text.size() << "  train " << train_end << "  val "
              << (val_end - val_begin) << "  V " << vocab.V << "\n";
    std::cout << "mixer     T " << nets[0].T << "  D " << nets[0].D << "  block " << nets[0].blocks << "  batch "
              << nets[0].batch << "  alpha " << nets[0].alpha << "  " << (use_adam ? "adam" : "sgd") << "  wd " << wd
              << "  lr " << cfg.lr << "  K " << K << "  max-norm " << nets[0].max_norm << "  tau";
    for (int l = 0; l < nets[0].blocks; ++l) std::cout << " " << nets[0].tau[l];
    std::cout << "\nmod       ";
    for (int i = 0; i < n_net; ++i) {
        std::cout << specs[i].name << " ";
        if (specs[i].clusters > 1) std::cout << "SAP-M" << specs[i].clusters;
        else if (specs[i].sap) std::cout << "SAP-INT8";
        else if (specs[i].i8) std::cout << "BP-INT8";
        else std::cout << "BP-FP32";
        std::cout << "   ";
    }
    std::cout << "\n";

    using clock = std::chrono::steady_clock;
    for (int step = 1; step <= cfg.steps; ++step) {
        for (int b = 0; b < cfg.batch; ++b) starts[static_cast<std::size_t>(b)] = pick(rng);
        shake_fill_batch(text, vocab, starts.data(), cfg.batch, k_shake_T, tok, Y);
        const bool probe = step == 1 || step == 200 || step == 500 || step % 1000 == 0;
        for (int i = 0; i < n_net; ++i) {
            if (!alive[i]) continue;
            Shake::AlignView av;
            const bool want = probe && nets[i].sap;
            const auto t0 = clock::now();
            const float loss = nets[i].step(tok.data(), Y, cfg.lr, true, want ? &av : nullptr);
            used[i] += std::chrono::duration<double, std::micro>(clock::now() - t0).count();
            ++steps_n[i];
            if (!std::isfinite(loss)) alive[i] = false;
            else {
                sum[i] += loss;
                ++seen[i];
            }
            if (want && alive[i]) {
                char cline[256];
                std::snprintf(cline, sizeof(cline),
                              "cos %s step %5d block%d %6.3f ratio %6.3f | block0 %6.3f ratio %6.3f | E %6.3f ratio %6.3f\n",
                              specs[i].name, step, nets[i].blocks - 1, av.cos_top, av.nr_top, av.cos_bot, av.nr_bot,
                              av.cos_e, av.nr_e);
                std::cout << cline << std::flush;
            }
        }
        if (step % 1000 != 0 && step != cfg.steps) continue;
        char line[720];
        int pos = std::snprintf(line, sizeof(line), "step %5d", step);
        for (int i = 0; i < n_net; ++i) {
            const float val = alive[i] ? shake_bpc(nets[i], text, vocab, val_begin, val_end)
                                        : std::numeric_limits<float>::quiet_NaN();
            last_val[i] = val;
            const float tr = seen[i] ? static_cast<float>((sum[i] / seen[i]) / ln2) : std::numeric_limits<float>::quiet_NaN();
            pos += std::snprintf(line + pos, sizeof(line) - static_cast<std::size_t>(pos),
                                 " | %s train %6.3f val %6.3f  %6.2f ms  %6.3f MB  clip %d", specs[i].name, tr, val,
                                 steps_n[i] ? (used[i] / steps_n[i]) / 1000.0 : 0.0,
                                 static_cast<double>(nets[i].bytes()) / (1024.0 * 1024.0), nets[i].clip_hits);
            sum[i] = 0.0;
            seen[i] = 0;
        }
        std::snprintf(line + pos, sizeof(line) - static_cast<std::size_t>(pos), "\n");
        std::cout << line << std::flush;
    }

    std::cout << "\nablation  step " << cfg.steps << "  " << (use_adam ? "adam" : "sgd") << "  lr " << cfg.lr << "  wd "
              << wd << "\n";
    for (int i = 0; i < n_net; ++i) {
        char line[256];
        std::snprintf(line, sizeof(line), "  %-3s val %7.3f   ms %6.2f   MB %7.3f   clip %d   %s\n", specs[i].name,
                      last_val[i], steps_n[i] ? (used[i] / steps_n[i]) / 1000.0 : 0.0,
                      static_cast<double>(nets[i].bytes()) / (1024.0 * 1024.0), nets[i].clip_hits, alive[i] ? "ayakta" : "patladi");
        std::cout << line;
    }
    if (cfg.phase2) {
        char line[160];
        std::snprintf(line, sizeof(line), "  makas M2-A2 %+.3f   M4-A2 %+.3f\n", last_val[1] - last_val[0],
                      last_val[2] - last_val[0]);
        std::cout << line;
    } else if (n_net == 3) {
        char line[128];
        std::snprintf(line, sizeof(line), "  makas B-A1 %+.3f   B-A2 %+.3f\n", last_val[2] - last_val[0],
                      last_val[2] - last_val[1]);
        std::cout << line;
    } else if (n_net == 2) {
        char line[128];
        std::snprintf(line, sizeof(line), "  makas B-A2 %+.3f\n", last_val[1] - last_val[0]);
        std::cout << line;
    }
}

void run_shake_scale(const Config& cfg) {
#if defined(__AVX2__)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
    const std::vector<std::uint8_t> text = read_bytes(cfg.shake);
    const ShakeVocab vocab = shake_vocab(text);
    expect(vocab.V == 65, "TinyShakespeare vocab is not 65");
    const int T = 256;
    const int batch = 8;
    const int blocks = 6;
    const int warmup = 10;
    const int measure = 100;
    const int total = warmup + measure;
    const int train_end = static_cast<int>(text.size() * 9 / 10);
    expect(train_end > T + 2, "split shorter than the window");
    const int dims[3] = {256, 512, 1024};
    const int K = 10;

    std::mt19937 rng(cfg.seed + 99u);
    std::uniform_int_distribution<int> pick(0, train_end - T - 2);
    std::vector<int> starts(static_cast<std::size_t>(total) * static_cast<std::size_t>(batch));
    for (int s = 0; s < total * batch; ++s) starts[static_cast<std::size_t>(s)] = pick(rng);

    std::cout << "\nscale     T " << T << "  batch " << batch << "  token " << (batch * T) << "  block " << blocks
              << "  V " << vocab.V << "  K " << K << "  M 1  ReLU  raw P\n";
    std::cout << "profile    warmup " << warmup << "  measure " << measure
              << "  val off  same minibatch sequence replayed in every mode\n";

    struct Spec {
        const char* name;
        bool sap;
        bool i8;
    };
    const Spec specs[3] = {{"A1", false, false}, {"A2", false, true}, {"B", true, true}};

    for (int d : dims) {
        double ms[3] = {};
        double bucket[3][6] = {};
        double meg[3] = {};
        double strip[3] = {};
        for (int i = 0; i < 3; ++i) {
            Shake net;
            net.configure(T, d, vocab.V, blocks, batch);
            net.max_norm = 100.f;
            net.clip = true;
            net.align_k = K;
            net.mom = k_shake_mom;
            net.sap = specs[i].sap;
            net.int8_strip = specs[i].i8;
            net.wd = 0.f;
            net.clusters = 1;
            const std::size_t b0 = g_mem.live;
            net.init_weights(cfg.seed);
            net.prepare();
            if (net.sap) net.init_feedback();
            const std::size_t mats = g_mem.live - b0;
            expect(mats == net.matrix_bytes(), std::string(specs[i].name) + " memory counter");
            meg[i] = static_cast<double>(net.bytes()) / (1024.0 * 1024.0);
            std::size_t strip_b = 0;
            if (net.int8_strip) {
                for (const auto& q : net.hq) strip_b += q.size();
            } else {
                strip_b = bytes_of(net.H);
            }
            strip[i] = static_cast<double>(strip_b) / (1024.0 * 1024.0);

            std::vector<int> tok;
            Matrix Y;
            auto replay = [&](int begin, int n) {
                for (int s = 0; s < n; ++s) {
                    const int* st = starts.data() + static_cast<std::size_t>(begin + s) * batch;
                    shake_fill_batch(text, vocab, st, batch, T, tok, Y);
                    net.step(tok.data(), Y, 0.15f, true, nullptr);
                }
            };
            replay(0, warmup);
            net.reset_profile();
            const auto t0 = std::chrono::steady_clock::now();
            replay(warmup, measure);
            ms[i] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / measure;
            const double s = 1.0 / (static_cast<double>(measure) * 1000.0);
            bucket[i][0] = net.us_fwd * s;
            bucket[i][1] = net.us_out * s;
            bucket[i][2] = net.us_bwd * s;
            bucket[i][3] = net.us_dw * s;
            bucket[i][4] = net.us_deq * s;
            bucket[i][5] = net.us_upd * s;
        }
        std::cout << "\nD " << d << "  strip A1 " << strip[0] << " MB  A2 " << strip[1] << " MB  B " << strip[2]
                  << " MB\n";
        std::cout << "mod   fwd  out   bwd    dW  dequant update  total    MB\n";
        for (int i = 0; i < 3; ++i) {
            const double sum = bucket[i][0] + bucket[i][1] + bucket[i][2] + bucket[i][3] + bucket[i][4] + bucket[i][5];
            char line[256];
            std::snprintf(line, sizeof(line),
                          "%-4s %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %7.2f %7.2f  wall %7.2f\n", specs[i].name,
                          bucket[i][0], bucket[i][1], bucket[i][2], bucket[i][3], bucket[i][4], bucket[i][5], sum,
                          meg[i], ms[i]);
            std::cout << line;
        }
        auto pct = [](double ref, double got) { return ref > 0.0 ? (ref - got) / ref * 100.0 : 0.0; };
        char line[192];
        std::snprintf(line, sizeof(line), "speed B vs A1 %5.1f%%   B vs A2 %5.1f%%   (wall time)\n", pct(ms[0], ms[2]),
                      pct(ms[1], ms[2]));
        std::cout << line << std::flush;
    }
    std::cout << "scale done\n" << std::flush;
}

void run_shake_lora(const Config& cfg) {
#if defined(__AVX2__)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
    const std::vector<std::uint8_t> text = read_bytes(cfg.shake);
    const ShakeVocab vocab = shake_vocab(text);
    expect(vocab.V == 65, "TinyShakespeare vocab is not 65");
    const int T = 256;
    const int batch = 8;
    const int blocks = 6;
    const int n = static_cast<int>(text.size());
    const int base_end = n * 4 / 5;
    const int tgt_begin = base_end;
    const int tgt_end = n;
    const int tgt_train_end = tgt_begin + (tgt_end - tgt_begin) * 9 / 10;
    const int tgt_val_begin = tgt_train_end;
    expect(base_end > T + 2 && tgt_train_end - T - 2 > tgt_begin && tgt_val_begin + T < tgt_end,
           "lora split shorter than the window");
    const int base_steps = 5000;
    const int lora_steps = 2000;
    const float base_lr = 3e-3f;
    const float lora_lr = 3e-3f;
    const float wd = 1e-4f;
    const std::uint32_t lora_seed = cfg.seed + 17u;

    std::mt19937 rng(cfg.seed + 23u);
    std::uniform_int_distribution<int> pick_base(0, base_end - T - 2);
    std::vector<int> base_starts(static_cast<std::size_t>(base_steps) * static_cast<std::size_t>(batch));
    for (int s = 0; s < base_steps * batch; ++s) base_starts[static_cast<std::size_t>(s)] = pick_base(rng);
    std::uniform_int_distribution<int> pick_tgt(tgt_begin, tgt_train_end - T - 2);
    std::vector<int> tgt_starts(static_cast<std::size_t>(lora_steps) * static_cast<std::size_t>(batch));
    for (int s = 0; s < lora_steps * batch; ++s) tgt_starts[static_cast<std::size_t>(s)] = pick_tgt(rng);

    std::cout << "\nlora      T " << T << "  batch " << batch << "  block " << blocks << "  V " << vocab.V
              << "  r 16  s 1  ReLU  raw P  K 100\n";
    std::cout << "split     base 0.." << base_end << "  target train " << tgt_begin << ".." << tgt_train_end
              << "  target val " << tgt_val_begin << ".." << tgt_end << "\n";
    std::cout << "base     A2 INT8 adam  lr " << base_lr << "  wd " << wd << "  step " << base_steps
              << "  W0 E Pos Wout egitilir\n"
              << std::flush;

    Shake bp;
    Shake sapnet;
    float frozen_bpc = 0.f;
    {
        Shake base;
        base.configure(T, 256, vocab.V, blocks, batch);
        base.int8_strip = true;
        base.sap = false;
        base.adam = true;
        base.wd = wd;
        base.mom = k_shake_mom;
        base.max_norm = 100.f;
        base.clip = true;
        base.align_k = 0;
        base.clusters = 1;
        base.init_weights(cfg.seed);
        base.prepare();
        std::vector<int> tok;
        Matrix Y;
        double loss_sum = 0.0;
        int loss_n = 0;
        const auto t_base = std::chrono::steady_clock::now();
        for (int s = 1; s <= base_steps; ++s) {
            const int* st = base_starts.data() + static_cast<std::size_t>(s - 1) * batch;
            shake_fill_batch(text, vocab, st, batch, T, tok, Y);
            loss_sum += static_cast<double>(base.step(tok.data(), Y, base_lr, true, nullptr));
            ++loss_n;
            if (s % 1000 == 0) {
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_base).count();
                const double bpc = (loss_sum / loss_n) / std::log(2.0);
                std::cout << "base step " << s << "  train BPC " << bpc << "  " << sec << " s\n" << std::flush;
                loss_sum = 0.0;
                loss_n = 0;
            }
        }
        frozen_bpc = shake_bpc(base, text, vocab, tgt_val_begin, tgt_end);
        std::cout << "base target val BPC " << frozen_bpc << "  clip " << base.clip_hits << "\n" << std::flush;
        bp = base;
        sapnet = base;
    }
    bp.attach_lora(lora_seed);
    const float step0 = shake_bpc(bp, text, vocab, tgt_val_begin, tgt_end);
    expect(std::fabs(frozen_bpc - step0) < 1e-3f, "lora step 0 changed the frozen net");
    std::cout << "lora step 0 val BPC " << step0 << "  (B=0, must match the base)\n" << std::flush;

    sapnet.sap = true;
    sapnet.align_k = 100;
    sapnet.clusters = 1;
    sapnet.prepare();
    sapnet.attach_lora(lora_seed);
    sapnet.init_feedback();

    auto fit = [&](Shake& net, const char* name, bool want_cos) {
        std::vector<int> tok;
        Matrix Y;
        net.reset_profile();
        double wall_ms = 0.0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int s = 1; s <= lora_steps; ++s) {
            const int* st = tgt_starts.data() + static_cast<std::size_t>(s - 1) * batch;
            shake_fill_batch(text, vocab, st, batch, T, tok, Y);
            Shake::AlignView view;
            Shake::AlignView* probe = nullptr;
            if (want_cos && s == 1) probe = &view;
            const auto ts = std::chrono::steady_clock::now();
            net.step(tok.data(), Y, lora_lr, true, probe);
            wall_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
            if (probe) {
                std::cout << name << " cos step " << s << "  block-5 " << view.cos_top << "/" << view.nr_top
                          << "  block-0 " << view.cos_bot << "/" << view.nr_bot << "\n"
                          << std::flush;
            }
            if (s % 1000 == 0 || s == lora_steps) {
                const float bpc = shake_bpc(net, text, vocab, tgt_val_begin, tgt_end);
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::cout << name << " step " << s << "  val BPC " << bpc << "  " << sec << " s\n" << std::flush;
            }
        }
        const double ms = wall_ms / lora_steps;
        const double inv = 1.0 / (static_cast<double>(lora_steps) * 1000.0);
        const double mb = static_cast<double>(net.bytes()) / (1024.0 * 1024.0);
        char line[320];
        std::snprintf(line, sizeof(line),
                      "%s  ms/step %6.2f  MB %6.2f  clip %d  | fwd %5.2f out %5.2f bwd %5.2f dAB %5.2f dequant "
                      "%5.2f update %5.2f\n",
                      name, ms, mb, net.clip_hits, net.us_fwd * inv, net.us_out * inv, net.us_bwd * inv, net.us_dw * inv,
                      net.us_deq * inv, net.us_upd * inv);
        std::cout << line << std::flush;
        if (want_cos) {
            const int* st = tgt_starts.data() + static_cast<std::size_t>(lora_steps - 1) * batch;
            shake_fill_batch(text, vocab, st, batch, T, tok, Y);
            Shake::AlignView view;
            net.step(tok.data(), Y, lora_lr, false, &view);
            std::cout << name << " cos step " << lora_steps << "  block-5 " << view.cos_top << "/" << view.nr_top
                      << "  block-0 " << view.cos_bot << "/" << view.nr_bot << "\n"
                      << std::flush;
        }
        return ms;
    };

    std::cout << "finetune  D 256  step " << lora_steps << "  adam lr " << lora_lr << "  wd " << wd
              << "  W0 E Pos frozen  A B Wout trainable\n"
              << std::flush;
    const double ms_bp = fit(bp, "BP-LoRA", false);
    const double ms_sap = fit(sapnet, "SAP-LoRA", true);
    std::cout << "D 256 speed SAP/BP " << (ms_sap > 0.0 ? ms_bp / ms_sap : 0.0) << "x\n" << std::flush;

    const int speed_d = 1024;
    const int warmup = 10;
    const int measure = 100;
    const int total = warmup + measure;
    std::uniform_int_distribution<int> pick_spd(0, base_end - T - 2);
    std::vector<int> spd_starts(static_cast<std::size_t>(total) * static_cast<std::size_t>(batch));
    for (int s = 0; s < total * batch; ++s) spd_starts[static_cast<std::size_t>(s)] = pick_spd(rng);
    std::cout << "\nspeed      D " << speed_d << "  warmup " << warmup << "  measure " << measure
              << "  val off  K 100  same minibatch\n";
    std::cout << "mod   fwd  out   bwd    dAB dequant update  total    MB\n";
    double speed_ms[2] = {};
    for (int i = 0; i < 2; ++i) {
        const bool is_sap = i == 1;
        const char* name = is_sap ? "SAP" : "BP";
        Shake net;
        net.configure(T, speed_d, vocab.V, blocks, batch);
        net.int8_strip = true;
        net.sap = is_sap;
        net.adam = true;
        net.wd = wd;
        net.mom = k_shake_mom;
        net.max_norm = 100.f;
        net.clip = true;
        net.align_k = is_sap ? 100 : 0;
        net.clusters = 1;
        const std::size_t b0 = g_mem.live;
        net.init_weights(cfg.seed + 3u);
        net.prepare();
        net.attach_lora(lora_seed);
        if (is_sap) net.init_feedback();
        const std::size_t mats = g_mem.live - b0;
        expect(mats == net.matrix_bytes(), std::string(name) + " lora memory counter");
        std::vector<int> tok;
        Matrix Y;
        auto replay = [&](int begin, int nstep) {
            for (int s = 0; s < nstep; ++s) {
                const int* st = spd_starts.data() + static_cast<std::size_t>(begin + s) * batch;
                shake_fill_batch(text, vocab, st, batch, T, tok, Y);
                net.step(tok.data(), Y, lora_lr, true, nullptr);
            }
        };
        replay(0, warmup);
        net.reset_profile();
        const auto t0 = std::chrono::steady_clock::now();
        replay(warmup, measure);
        speed_ms[i] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / measure;
        const double inv = 1.0 / (static_cast<double>(measure) * 1000.0);
        const double bucket[6] = {net.us_fwd * inv, net.us_out * inv, net.us_bwd * inv,
                                  net.us_dw * inv,  net.us_deq * inv, net.us_upd * inv};
        const double sum = bucket[0] + bucket[1] + bucket[2] + bucket[3] + bucket[4] + bucket[5];
        const double meg = static_cast<double>(net.bytes()) / (1024.0 * 1024.0);
        char line[256];
        std::snprintf(line, sizeof(line), "%-4s %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %7.2f %7.2f  wall %7.2f\n", name,
                      bucket[0], bucket[1], bucket[2], bucket[3], bucket[4], bucket[5], sum, meg, speed_ms[i]);
        std::cout << line << std::flush;
    }
    std::cout << "D 1024 speed SAP/BP " << (speed_ms[1] > 0.0 ? speed_ms[0] / speed_ms[1] : 0.0) << "x   ("
              << (speed_ms[0] > 0.0 ? (speed_ms[0] - speed_ms[1]) / speed_ms[0] * 100.0 : 0.0) << "%)\n";
    std::cout << "lora done\n" << std::flush;
}

void run_shake_qlora(const Config& cfg) {
#if defined(__AVX2__)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
    const std::vector<std::uint8_t> text = read_bytes(cfg.shake);
    const ShakeVocab vocab = shake_vocab(text);
    expect(vocab.V == 65, "TinyShakespeare vocab is not 65");
    const int T = 256;
    const int batch = 8;
    const int blocks = 6;
    const int n = static_cast<int>(text.size());
    const int base_end = n * 4 / 5;
    const int tgt_begin = base_end;
    const int tgt_end = n;
    const int tgt_train_end = tgt_begin + (tgt_end - tgt_begin) * 9 / 10;
    const int tgt_val_begin = tgt_train_end;
    expect(base_end > T + 2 && tgt_train_end - T - 2 > tgt_begin && tgt_val_begin + T < tgt_end,
           "qlora split shorter than the window");
    const int base_steps = 5000;
    const int lora_steps = 2000;
    const float base_lr = 3e-3f;
    const float lora_lr = 3e-3f;
    const float wd = 1e-4f;
    const std::uint32_t lora_seed = cfg.seed + 17u;
    const double mb = 1.0 / (1024.0 * 1024.0);

    std::mt19937 rng(cfg.seed + 23u);
    std::uniform_int_distribution<int> pick_base(0, base_end - T - 2);
    std::vector<int> base_starts(static_cast<std::size_t>(base_steps) * static_cast<std::size_t>(batch));
    for (int s = 0; s < base_steps * batch; ++s) base_starts[static_cast<std::size_t>(s)] = pick_base(rng);
    std::uniform_int_distribution<int> pick_tgt(tgt_begin, tgt_train_end - T - 2);
    std::vector<int> tgt_starts(static_cast<std::size_t>(lora_steps) * static_cast<std::size_t>(batch));
    for (int s = 0; s < lora_steps * batch; ++s) tgt_starts[static_cast<std::size_t>(s)] = pick_tgt(rng);

    std::cout << "\nqlora     kernel " << qlora_kernel() << "  T " << T << "  batch " << batch << "  block " << blocks
              << "  r 16  s 1  K 100\n";
    std::cout << "base     A2 INT8 adam  lr " << base_lr << "  wd " << wd << "  step " << base_steps << "\n" << std::flush;

    Shake fp;
    Shake ql;
    {
        Shake base;
        base.configure(T, 256, vocab.V, blocks, batch);
        base.int8_strip = true;
        base.sap = false;
        base.adam = true;
        base.wd = wd;
        base.mom = k_shake_mom;
        base.max_norm = 100.f;
        base.clip = true;
        base.align_k = 0;
        base.clusters = 1;
        base.init_weights(cfg.seed);
        base.prepare();
        std::vector<int> tok;
        Matrix Y;
        double loss_sum = 0.0;
        int loss_n = 0;
        const auto t_base = std::chrono::steady_clock::now();
        for (int s = 1; s <= base_steps; ++s) {
            const int* st = base_starts.data() + static_cast<std::size_t>(s - 1) * batch;
            shake_fill_batch(text, vocab, st, batch, T, tok, Y);
            loss_sum += static_cast<double>(base.step(tok.data(), Y, base_lr, true, nullptr));
            ++loss_n;
            if (s % 1000 == 0) {
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_base).count();
                std::cout << "base step " << s << "  train BPC " << (loss_sum / loss_n) / std::log(2.0) << "  " << sec
                          << " s\n"
                          << std::flush;
                loss_sum = 0.0;
                loss_n = 0;
            }
        }
        const float frozen_bpc = shake_bpc(base, text, vocab, tgt_val_begin, tgt_end);
        std::cout << "base target val BPC " << frozen_bpc << "\n" << std::flush;
        fp = base;
        ql = base;
    }
    fp.sap = true;
    fp.align_k = 100;
    fp.clusters = 1;
    fp.prepare();
    fp.attach_lora(lora_seed);
    ql = fp;
    ql.compress_w0();
    expect(ql.Wch[0].n() == 0 && ql.w0_pack_bytes() > 0, "qlora did not free FP32 W0");

    {
        std::vector<int> tok;
        Matrix Y;
        shake_fill_batch(text, vocab, tgt_starts.data(), batch, T, tok, Y);
        fp.forward(tok.data(), batch, false);
        const Matrix logits_fp = fp.logits;
        ql.forward(tok.data(), batch, false);
        float diff = 0.f;
        for (std::size_t i = 0; i < logits_fp.n(); ++i)
            diff = std::max(diff, std::fabs(logits_fp.data[i] - ql.logits.data[i]));
        std::cout << "paket     max |logit fp32 - int8| " << diff << "  W0 fp32 "
                  << static_cast<double>(bytes_of(fp.Wch)) * mb << " MB  W0 int8 "
                  << static_cast<double>(ql.w0_pack_bytes()) * mb << " MB\n"
                  << std::flush;
        expect(std::isfinite(diff) && diff < 2.f, "qlora forward left the fp32 logits");
    }
    fp.init_feedback();
    ql.init_feedback();

    {
        Shake primed = ql;
        primed.gather_mask = true;
        std::vector<int> tok;
        Matrix Y;
        for (int s = 0; s < 8; ++s) {
            const int* st = tgt_starts.data() + static_cast<std::size_t>(s) * batch;
            shake_fill_batch(text, vocab, st, batch, T, tok, Y);
            primed.forward(tok.data(), batch, false);
        }
        primed.gather_mask = false;
        expect(primed.rate_fill == 8, "prime did not record masks");
        primed.sweep(true);
        primed.ticks = 0;
        shake_fill_batch(text, vocab, tgt_starts.data(), batch, T, tok, Y);
        Shake::AlignView view;
        primed.step(tok.data(), Y, lora_lr, false, &view);
        std::cout << "prime 8   cos step 0  block-5 " << view.cos_top << "/" << view.nr_top << "  block-0 " << view.cos_bot
                  << "/" << view.nr_bot << "\n"
                  << std::flush;
    }

    auto fit = [&](Shake& net, const char* name) {
        std::vector<int> tok;
        Matrix Y;
        net.reset_profile();
        double wall_ms = 0.0;
        for (int s = 1; s <= lora_steps; ++s) {
            const int* st = tgt_starts.data() + static_cast<std::size_t>(s - 1) * batch;
            shake_fill_batch(text, vocab, st, batch, T, tok, Y);
            const auto ts = std::chrono::steady_clock::now();
            net.step(tok.data(), Y, lora_lr, true, nullptr);
            wall_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
            if (s % 1000 == 0 || s == lora_steps) {
                const float bpc = shake_bpc(net, text, vocab, tgt_val_begin, tgt_end);
                std::cout << name << " step " << s << "  val BPC " << bpc << "\n" << std::flush;
            }
        }
        const double ms = wall_ms / lora_steps;
        const double inv = 1.0 / (static_cast<double>(lora_steps) * 1000.0);
        char line[320];
        std::snprintf(line, sizeof(line),
                      "%s  ms/step %6.2f  MB %6.2f  W0 %5.2f  | fwd %5.2f out %5.2f bwd %5.2f dAB %5.2f\n", name, ms,
                      static_cast<double>(net.bytes()) * mb, static_cast<double>(net.qlora ? net.w0_pack_bytes() : bytes_of(net.Wch)) * mb,
                      net.us_fwd * inv, net.us_out * inv, net.us_bwd * inv, net.us_dw * inv);
        std::cout << line << std::flush;
        return ms;
    };

    const float bpc0 = shake_bpc(fp, text, vocab, tgt_val_begin, tgt_end);
    std::cout << "step 0    val BPC " << bpc0 << "\nfinetune D 256  step " << lora_steps << "\n" << std::flush;
    const double ms_fp = fit(fp, "SAP-LoRA");
    const double ms_ql = fit(ql, "SAP-QLoRA");
    std::cout << "D 256 speed Q/SAP " << (ms_ql > 0.0 ? ms_fp / ms_ql : 0.0) << "x\n" << std::flush;

    const int dims[3] = {256, 512, 1024};
    const int warmup = 10;
    const int measure = 100;
    const int total = warmup + measure;
    std::uniform_int_distribution<int> pick_spd(0, base_end - T - 2);
    std::vector<int> spd_starts(static_cast<std::size_t>(total) * static_cast<std::size_t>(batch));
    for (int s = 0; s < total * batch; ++s) spd_starts[static_cast<std::size_t>(s)] = pick_spd(rng);
    std::cout << "\nspeed      warmup " << warmup << "  measure " << measure << "  K 100  kernel " << qlora_kernel()
              << "\n";
    for (int d : dims) {
        double speed_ms[3] = {};
        std::cout << "\nD " << d << "\n";
        std::cout << "mod    fwd  out   bwd    dAB dequant update  total     MB    W0\n";
        for (int i = 0; i < 3; ++i) {
            const char* name = i == 0 ? "BP" : (i == 1 ? "SAP" : "QL");
            Shake net;
            net.configure(T, d, vocab.V, blocks, batch);
            net.int8_strip = true;
            net.sap = i != 0;
            net.adam = true;
            net.wd = wd;
            net.mom = k_shake_mom;
            net.max_norm = 100.f;
            net.clip = true;
            net.align_k = i == 0 ? 0 : 100;
            net.clusters = 1;
            const std::size_t b0 = g_mem.live;
            net.init_weights(cfg.seed + 3u);
            net.prepare();
            net.attach_lora(lora_seed);
            if (i == 2) net.compress_w0();
            if (net.sap) net.init_feedback();
            expect(g_mem.live - b0 == net.matrix_bytes(), std::string(name) + " qlora memory counter");
            std::vector<int> tok;
            Matrix Y;
            auto replay = [&](int begin, int nstep) {
                for (int s = 0; s < nstep; ++s) {
                    const int* st = spd_starts.data() + static_cast<std::size_t>(begin + s) * batch;
                    shake_fill_batch(text, vocab, st, batch, T, tok, Y);
                    net.step(tok.data(), Y, lora_lr, true, nullptr);
                }
            };
            replay(0, warmup);
            net.reset_profile();
            const auto t0 = std::chrono::steady_clock::now();
            replay(warmup, measure);
            speed_ms[i] =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / measure;
            const double inv = 1.0 / (static_cast<double>(measure) * 1000.0);
            const double bucket[6] = {net.us_fwd * inv, net.us_out * inv, net.us_bwd * inv,
                                      net.us_dw * inv,  net.us_deq * inv, net.us_upd * inv};
            const double sum = bucket[0] + bucket[1] + bucket[2] + bucket[3] + bucket[4] + bucket[5];
            const double w0 = static_cast<double>(net.qlora ? net.w0_pack_bytes() : bytes_of(net.Wch)) * mb;
            char line[256];
            std::snprintf(line, sizeof(line), "%-4s %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %7.2f %7.2f %6.2f  wall %7.2f\n",
                          name, bucket[0], bucket[1], bucket[2], bucket[3], bucket[4], bucket[5], sum,
                          static_cast<double>(net.bytes()) * mb, w0, speed_ms[i]);
            std::cout << line << std::flush;
        }
        std::cout << "D " << d << " speed QL/BP " << (speed_ms[2] > 0.0 ? speed_ms[0] / speed_ms[2] : 0.0) << "x   QL/SAP "
                  << (speed_ms[2] > 0.0 ? speed_ms[1] / speed_ms[2] : 0.0) << "x\n"
                  << std::flush;
    }
    std::cout << "qlora done\n" << std::flush;
}

void run_shake_qsuite(const Config& cfg) {
#if defined(__AVX2__)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
    expect(cfg.qsuite >= 1 && cfg.qsuite <= 7, "qsuite is 1..7");
    char brand[49] = {};
    int regs[4] = {};
    for (int i = 0; i < 3; ++i) {
        __cpuid(regs, 0x80000002 + i);
        std::memcpy(brand + i * 16, regs, 16);
    }
    const std::vector<std::uint8_t> text = read_bytes(cfg.shake);
    const ShakeVocab vocab = shake_vocab(text);
    expect(vocab.V == 65, "TinyShakespeare vocab is not 65");
    const int T = 256;
    const int batch = 8;
    const int blocks = 6;
    const int n = static_cast<int>(text.size());
    const int base_end = n * 4 / 5;
    const int tgt_begin = base_end;
    const int tgt_end = n;
    const int tgt_train_end = tgt_begin + (tgt_end - tgt_begin) * 9 / 10;
    const int tgt_val_begin = tgt_train_end;
    const float lr = 3e-3f;
    const float wd = 1e-4f;
    const std::uint32_t lora_seed = cfg.seed + 17u;
    const char* choice_path = "C:\\yz\\build\\mask_choice.txt";
    const char* w0_path = "C:\\yz\\build\\w0_d1024.bin";
    std::cout << "\nsuite " << cfg.qsuite << "  cpu " << brand << "\n" << std::flush;

    auto prime8 = [&](Shake& net, const int* starts) {
        net.gather_mask = true;
        net.ema_mask = false;
        std::vector<int> tok;
        Matrix Y;
        for (int s = 0; s < 8; ++s) {
            shake_fill_batch(text, vocab, starts + static_cast<std::size_t>(s) * batch, batch, T, tok, Y);
            net.forward(tok.data(), batch, false);
        }
        net.gather_mask = false;
        expect(net.rate_fill == 8, "prime batches");
    };

    if (cfg.qsuite == 4) {
        const int d = 1024;
        const int lora_steps = 1000;
        const float base_lr = lr;
        const float peak = base_lr * 0.5f;
        const float min_lr = base_lr * 0.05f;
        const float pi = std::acos(-1.f);
        auto sched = [&](int step) {
            if (step < 100) return peak * (static_cast<float>(step + 1) / 100.f);
            const float t = static_cast<float>(step - 100) / 900.f;
            return min_lr + 0.5f * (peak - min_lr) * (1.f + std::cos(pi * t));
        };
        std::mt19937 rng_lora(cfg.seed + 101u);
        std::uniform_int_distribution<int> pick_tgt(tgt_begin, tgt_train_end - T - 2);
        std::vector<int> tgt_starts(static_cast<std::size_t>(lora_steps) * batch);
        for (int s = 0; s < lora_steps * batch; ++s) tgt_starts[static_cast<std::size_t>(s)] = pick_tgt(rng_lora);

        Shake net;
        net.configure(T, d, vocab.V, blocks, batch);
        net.int8_strip = true;
        net.adam = true;
        net.wd = wd;
        net.mom = k_shake_mom;
        net.max_norm = 100.f;
        net.clip = true;
        net.sap = true;
        net.align_k = 100;
        net.clusters = 1;
        net.ab_clip = 1.f;
        net.init_weights(cfg.seed);
        net.prepare();
        net.load_frozen("C:\\yz\\build\\w0_d1024.bin");
        net.attach_lora(lora_seed);
        net.compress_w0();
        prime8(net, tgt_starts.data());
        net.init_feedback();
        const float b0 = shake_bpc(net, text, vocab, tgt_val_begin, tgt_end);
        std::cout << "run1  prime  half-lr  warmup 100  cosine  ab-clip 1\n";
        std::cout << "step 0  val BPC " << b0 << "  lr1 " << sched(1) << "  lr100 " << sched(100) << "  lr1000 " << sched(1000)
                  << "\n"
                  << std::flush;
        std::vector<int> tok;
        Matrix Y;
        float best = b0;
        int best_at = 0;
        for (int s = 1; s <= lora_steps; ++s) {
            net.log_delta = (s == 1 || s == 500);
            shake_fill_batch(text, vocab, tgt_starts.data() + static_cast<std::size_t>(s - 1) * batch, batch, T, tok, Y);
            net.step(tok.data(), Y, sched(s), true, nullptr);
            net.log_delta = false;
            if (s == 1 || s == 500) {
                std::cout << "delta step " << s;
                for (int l = 0; l < blocks; ++l) std::cout << "  b" << l << " " << net.delta_n[static_cast<std::size_t>(l)];
                const float top = net.delta_n[static_cast<std::size_t>(blocks - 1)];
                const float bot = net.delta_n[0];
                std::cout << "  top/bot " << (bot > 0.f ? top / bot : 0.f) << "\n" << std::flush;
            }
            if (s % 100 == 0) {
                const float bpc = shake_bpc(net, text, vocab, tgt_val_begin, tgt_end);
                if (bpc < best) {
                    best = bpc;
                    best_at = s;
                }
                std::cout << "step " << s << "  val BPC " << bpc << "  lr " << sched(s) << "\n" << std::flush;
            }
        }
        std::cout << "best " << best << " @ " << best_at << "  ab-clip " << net.ab_clip_hits << "\n";
        std::cout << "run1 done\n" << std::flush;
        return;
    }

    if (cfg.qsuite == 5) {
        const int d = 1024;
        const int lora_steps = 1000;
        const float base_lr = lr;
        const float peak = base_lr * 0.5f;
        const float min_lr = base_lr * 0.05f;
        const float pi = std::acos(-1.f);
        auto sched = [&](int step) {
            if (step < 100) return peak * (static_cast<float>(step + 1) / 100.f);
            const float t = static_cast<float>(step - 100) / 900.f;
            return min_lr + 0.5f * (peak - min_lr) * (1.f + std::cos(pi * t));
        };
        std::mt19937 rng_lora(cfg.seed + 101u);
        std::uniform_int_distribution<int> pick_tgt(tgt_begin, tgt_train_end - T - 2);
        std::vector<int> tgt_starts(static_cast<std::size_t>(lora_steps) * batch);
        for (int s = 0; s < lora_steps * batch; ++s) tgt_starts[static_cast<std::size_t>(s)] = pick_tgt(rng_lora);

        auto train = [&](Shake& net, const char* title) {
            const float b0 = shake_bpc(net, text, vocab, tgt_val_begin, tgt_end);
            std::cout << title << "\n";
            std::cout << "step 0  val BPC " << b0 << "  lr1 " << sched(1) << "  lr100 " << sched(100)
                      << "  lr1000 " << sched(1000) << "\n"
                      << std::flush;
            std::vector<int> tok;
            Matrix Y;
            float best = b0;
            int best_at = 0;
            for (int s = 1; s <= lora_steps; ++s) {
                shake_fill_batch(text, vocab, tgt_starts.data() + static_cast<std::size_t>(s - 1) * batch, batch, T, tok, Y);
                net.step(tok.data(), Y, sched(s), true, nullptr);
                if (s % 100 == 0) {
                    const float bpc = shake_bpc(net, text, vocab, tgt_val_begin, tgt_end);
                    if (bpc < best) {
                        best = bpc;
                        best_at = s;
                    }
                    std::cout << "step " << s << "  val BPC " << bpc << "  lr " << sched(s) << "\n" << std::flush;
                }
            }
            std::cout << "best " << best << " @ " << best_at << "\n" << std::flush;
        };

        Shake sap;
        sap.configure(T, d, vocab.V, blocks, batch);
        sap.int8_strip = true;
        sap.adam = true;
        sap.wd = wd;
        sap.mom = k_shake_mom;
        sap.max_norm = 100.f;
        sap.clip = true;
        sap.sap = true;
        sap.align_k = 100;
        sap.align_k_early = 20;
        sap.align_k_until = 200;
        sap.clusters = 1;
        sap.ab_clip = 0.f;
        sap.init_weights(cfg.seed);
        sap.prepare();
        sap.load_frozen("C:\\yz\\build\\w0_d1024.bin");
        sap.attach_lora(lora_seed);
        sap.compress_w0();
        prime8(sap, tgt_starts.data());
        sap.init_feedback();
        sap.mean_mask = true;
        sap.m_count = 0;
        sap.m_sum.clear();
        train(sap, "run2  prime  half-lr  warmup 100  cosine  K20-200  running-mean");
        std::cout << "run2 done  sweeps " << sap.sweep_calls << "\n" << std::flush;

        Shake bp;
        bp.configure(T, d, vocab.V, blocks, batch);
        bp.int8_strip = true;
        bp.adam = true;
        bp.wd = wd;
        bp.mom = k_shake_mom;
        bp.max_norm = 100.f;
        bp.clip = true;
        bp.sap = false;
        bp.align_k = 0;
        bp.clusters = 1;
        bp.ab_clip = 0.f;
        bp.init_weights(cfg.seed);
        bp.prepare();
        bp.load_frozen("C:\\yz\\build\\w0_d1024.bin");
        bp.attach_lora(lora_seed);
        train(bp, "bp-lora  same takvim  warmup 100  cosine");
        std::cout << "bp-lora done\n" << std::flush;
        return;
    }

    if (cfg.qsuite == 6) {
        const int dims[4] = {256, 512, 1024, 2048};
        const int warmup = 10;
        const int measure = 100;
        const int total = warmup + measure;
        std::mt19937 rng(cfg.seed + 23u);
        std::uniform_int_distribution<int> pick(0, base_end - T - 2);
        std::vector<int> starts(static_cast<std::size_t>(total) * batch);
        for (int s = 0; s < total * batch; ++s) starts[static_cast<std::size_t>(s)] = pick(rng);
        const double mb = 1.0 / (1024.0 * 1024.0);
        std::cout << "testA  warmup " << warmup << "  measure " << measure
                  << "  K 100  val off  " << qlora_kernel() << "\n"
                  << std::flush;
        for (int d : dims) {
            double speed_ms[4] = {};
            std::cout << "D " << d << "\n";
            std::cout << "mod    fwd  out   bwd    dAB dequant update  total     MB    W0\n";
            for (int i = 0; i < 4; ++i) {
                const char* name = i == 0 ? "BP" : (i == 1 ? "BPQ" : (i == 2 ? "SAP" : "QL"));
                const bool sap = i >= 2;
                const bool quant = i == 1 || i == 3;
                Shake net;
                net.configure(T, d, vocab.V, blocks, batch);
                net.int8_strip = true;
                net.sap = sap;
                net.adam = true;
                net.wd = wd;
                net.mom = k_shake_mom;
                net.max_norm = 100.f;
                net.clip = true;
                net.align_k = sap ? 100 : 0;
                net.clusters = 1;
                const std::size_t b0 = g_mem.live;
                net.init_weights(cfg.seed + 3u);
                net.prepare();
                net.attach_lora(lora_seed);
                if (quant) net.compress_w0();
                if (net.sap) net.init_feedback();
                expect(g_mem.live - b0 == net.matrix_bytes(), std::string(name) + " memory counter");
                std::vector<int> tok;
                Matrix Y;
                auto replay = [&](int begin, int nstep) {
                    for (int s = 0; s < nstep; ++s) {
                        shake_fill_batch(text, vocab, starts.data() + static_cast<std::size_t>(begin + s) * batch, batch, T,
                                         tok, Y);
                        net.step(tok.data(), Y, lr, true, nullptr);
                    }
                };
                replay(0, warmup);
                net.reset_profile();
                const auto t0 = std::chrono::steady_clock::now();
                replay(warmup, measure);
                speed_ms[i] =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / measure;
                const double inv = 1.0 / (static_cast<double>(measure) * 1000.0);
                const double bucket[6] = {net.us_fwd * inv, net.us_out * inv, net.us_bwd * inv,
                                          net.us_dw * inv,  net.us_deq * inv, net.us_upd * inv};
                const double sum = bucket[0] + bucket[1] + bucket[2] + bucket[3] + bucket[4] + bucket[5];
                const double w0 = static_cast<double>(net.qlora ? net.w0_pack_bytes() : bytes_of(net.Wch)) * mb;
                char line[256];
                std::snprintf(line, sizeof(line),
                              "%-4s %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %7.2f %7.2f %6.2f  wall %7.2f\n", name,
                              bucket[0], bucket[1], bucket[2], bucket[3], bucket[4], bucket[5], sum,
                              static_cast<double>(net.bytes()) * mb, w0, speed_ms[i]);
                std::cout << line << std::flush;
            }
            auto ratio = [](double a, double b) { return b > 0.0 ? a / b : 0.0; };
            std::cout << "D " << d << "  VNNI BP/BPQ " << ratio(speed_ms[0], speed_ms[1]) << "  SAP BPQ/QL "
                      << ratio(speed_ms[1], speed_ms[3]) << "  total BP/QL " << ratio(speed_ms[0], speed_ms[3])
                      << "  QL/SAP " << ratio(speed_ms[2], speed_ms[3]) << "\n"
                      << std::flush;
        }
        std::cout << "testA done\n" << std::flush;
        return;
    }

    if (cfg.qsuite == 7) {
        const int d = 256;
        const int depth = 12;
        const int cap = 4000;
        const char* path = "C:\\yz\\build\\w0_d256_l12.bin";
        std::mt19937 rng(cfg.seed + 23u);
        std::uniform_int_distribution<int> pick_base(0, base_end - T - 2);
        std::vector<int> base_starts(static_cast<std::size_t>(cap) * batch);
        for (int s = 0; s < cap * batch; ++s) base_starts[static_cast<std::size_t>(s)] = pick_base(rng);
        int trained = 0;
        float frozen = 0.f;
        float base_lr_run = lr;
        {
            Shake base;
            base.configure(T, d, vocab.V, depth, batch);
            base.int8_strip = true;
            base.adam = true;
            base.wd = wd;
            base.mom = k_shake_mom;
            base.max_norm = 100.f;
            base.clip = true;
            base.align_k = 0;
            base.clusters = 1;
            base.init_weights(cfg.seed);
            base.prepare();
            frozen = shake_bpc(base, text, vocab, tgt_val_begin, tgt_end);
            std::cout << "base L12 step 0  val BPC " << frozen << "  lr " << base_lr_run << "\n" << std::flush;
            float prev = frozen;
            int rises = 0;
            std::vector<int> tok;
            Matrix Y;
            for (int s = 1; s <= cap; ++s) {
                shake_fill_batch(text, vocab, base_starts.data() + static_cast<std::size_t>(s - 1) * batch, batch, T, tok, Y);
                base.step(tok.data(), Y, base_lr_run, true, nullptr);
                if (s % 100 == 0) {
                    frozen = shake_bpc(base, text, vocab, tgt_val_begin, tgt_end);
                    trained = s;
                    std::cout << "base L12 step " << s << "  val BPC " << frozen << "  lr " << base_lr_run << "\n"
                              << std::flush;
                    if (!std::isfinite(frozen)) throw std::runtime_error("base val is not a number");
                    if (frozen > prev + 0.02f) ++rises;
                    else rises = 0;
                    if (rises >= 2 && frozen > 3.2f) {
                        base_lr_run *= 0.5f;
                        rises = 0;
                        std::cout << "base L12 lr halved  " << base_lr_run << "\n" << std::flush;
                    }
                    prev = frozen;
                    if (s >= 200 && frozen <= 3.10f) break;
                }
            }
            base.save_frozen(path);
            std::cout << "W0 saved  " << path << "  step " << trained << "  val " << frozen << "\n" << std::flush;
        }

        const int lora_steps = 1000;
        const float base_lr = lr;
        const float peak = base_lr * 0.5f;
        const float min_lr = base_lr * 0.05f;
        const float pi = std::acos(-1.f);
        auto sched = [&](int step) {
            if (step < 100) return peak * (static_cast<float>(step + 1) / 100.f);
            const float t = static_cast<float>(step - 100) / 900.f;
            return min_lr + 0.5f * (peak - min_lr) * (1.f + std::cos(pi * t));
        };
        std::mt19937 rng_lora(cfg.seed + 101u);
        std::uniform_int_distribution<int> pick_tgt(tgt_begin, tgt_train_end - T - 2);
        std::vector<int> tgt_starts(static_cast<std::size_t>(lora_steps) * batch);
        for (int s = 0; s < lora_steps * batch; ++s) tgt_starts[static_cast<std::size_t>(s)] = pick_tgt(rng_lora);

        auto arm = [&](bool sap) {
            Shake net;
            net.configure(T, d, vocab.V, depth, batch);
            net.int8_strip = true;
            net.adam = true;
            net.wd = wd;
            net.mom = k_shake_mom;
            net.max_norm = 100.f;
            net.clip = true;
            net.sap = sap;
            net.align_k = sap ? 100 : 0;
            net.clusters = 1;
            net.ab_clip = 0.f;
            net.init_weights(cfg.seed);
            net.prepare();
            net.load_frozen(path);
            net.attach_lora(lora_seed);
            if (sap) {
                net.compress_w0();
                prime8(net, tgt_starts.data());
                net.init_feedback();
            }
            return net;
        };

        float curve[2][10] = {};
        int best_at[2] = {};
        float best_bpc[2] = {1e9f, 1e9f};
        for (int i = 0; i < 2; ++i) {
            Shake net = arm(i == 1);
            const char* name = i == 0 ? "BP-LoRA" : "SAP-QLoRA";
            const float b0 = shake_bpc(net, text, vocab, tgt_val_begin, tgt_end);
            std::cout << name << " step 0  val BPC " << b0 << "\n" << std::flush;
            std::vector<int> tok;
            Matrix Y;
            if (i == 1) {
                shake_fill_batch(text, vocab, tgt_starts.data(), batch, T, tok, Y);
                Shake::AlignView view;
                net.step(tok.data(), Y, sched(1), false, &view);
                std::cout << "step 0 cos  block-11 " << view.cos_top << "  block-0 " << view.cos_bot << "\n" << std::flush;
            }
            for (int s = 1; s <= lora_steps; ++s) {
                if (i == 1) net.log_delta = (s == 1 || s == 500);
                shake_fill_batch(text, vocab, tgt_starts.data() + static_cast<std::size_t>(s - 1) * batch, batch, T, tok, Y);
                net.step(tok.data(), Y, sched(s), true, nullptr);
                net.log_delta = false;
                if (i == 1 && (s == 1 || s == 500)) {
                    const float d0 = net.delta_n[0];
                    const float d5 = net.delta_n[5];
                    const float d11 = net.delta_n[static_cast<std::size_t>(depth - 1)];
                    std::cout << "delta step " << s << "  d0 " << d0 << "  d5 " << d5 << "  d11 " << d11 << "  d0/d11 "
                              << (d11 > 0.f ? d0 / d11 : 0.f) << "\n"
                              << std::flush;
                }
                if (s % 100 == 0) {
                    const int k = s / 100 - 1;
                    curve[i][k] = shake_bpc(net, text, vocab, tgt_val_begin, tgt_end);
                    if (curve[i][k] < best_bpc[i]) {
                        best_bpc[i] = curve[i][k];
                        best_at[i] = s;
                    }
                    std::cout << name << " step " << s << "  val BPC " << curve[i][k] << "  lr " << sched(s) << "\n"
                              << std::flush;
                }
            }
        }
        std::cout << "\ntestB  L 12  D 256  base val " << frozen << "  base step " << trained << "\n";
        std::cout << "step    BP-LoRA   SAP-QLoRA\n";
        for (int k = 0; k < 10; ++k)
            std::cout << (k + 1) * 100 << "  " << curve[0][k] << "  " << curve[1][k] << "\n";
        std::cout << "best BP " << best_bpc[0] << " @ " << best_at[0] << "   SAP " << best_bpc[1] << " @ " << best_at[1]
                  << "   gap " << (best_bpc[1] - best_bpc[0]) << "\n";
        std::cout << "testB done\n" << std::flush;
        return;
    }

    if (cfg.qsuite == 1) {
        const int base_steps = 5000;
        const int lora_steps = 2000;
        std::mt19937 rng(cfg.seed + 23u);
        std::uniform_int_distribution<int> pick_base(0, base_end - T - 2);
        std::vector<int> base_starts(static_cast<std::size_t>(base_steps) * batch);
        for (int s = 0; s < base_steps * batch; ++s) base_starts[static_cast<std::size_t>(s)] = pick_base(rng);
        std::uniform_int_distribution<int> pick_tgt(tgt_begin, tgt_train_end - T - 2);
        std::vector<int> tgt_starts(static_cast<std::size_t>(lora_steps) * batch);
        for (int s = 0; s < lora_steps * batch; ++s) tgt_starts[static_cast<std::size_t>(s)] = pick_tgt(rng);

        Shake src;
        float frozen = 0.f;
        {
            Shake base;
            base.configure(T, 256, vocab.V, blocks, batch);
            base.int8_strip = true;
            base.adam = true;
            base.wd = wd;
            base.mom = k_shake_mom;
            base.max_norm = 100.f;
            base.clip = true;
            base.align_k = 0;
            base.clusters = 1;
            base.init_weights(cfg.seed);
            base.prepare();
            std::vector<int> tok;
            Matrix Y;
            for (int s = 1; s <= base_steps; ++s) {
                shake_fill_batch(text, vocab, base_starts.data() + static_cast<std::size_t>(s - 1) * batch, batch, T, tok, Y);
                base.step(tok.data(), Y, lr, true, nullptr);
                if (s % 1000 == 0) std::cout << "base step " << s << "\n" << std::flush;
            }
            frozen = shake_bpc(base, text, vocab, tgt_val_begin, tgt_end);
            std::cout << "base target val BPC " << frozen << "\n" << std::flush;
            expect(std::fabs(frozen - 3.01591f) < 1e-3f, "base is not 3.016, minibatch sequence drifted");
            src = base;
        }
        src.sap = true;
        src.align_k = 100;
        src.clusters = 1;
        src.prepare();
        src.attach_lora(lora_seed);
        src.compress_w0();
        Shake primed = src;
        Shake ema = src;

        auto fit = [&](Shake& net, const char* name, float& b1000, float& b2000) {
            std::vector<int> tok;
            Matrix Y;
            for (int s = 1; s <= lora_steps; ++s) {
                shake_fill_batch(text, vocab, tgt_starts.data() + static_cast<std::size_t>(s - 1) * batch, batch, T, tok, Y);
                net.step(tok.data(), Y, lr, true, nullptr);
                if (s == 1000 || s == 2000) {
                    const float bpc = shake_bpc(net, text, vocab, tgt_val_begin, tgt_end);
                    std::cout << name << " step " << s << "  val BPC " << bpc << "\n" << std::flush;
                    if (s == 1000) b1000 = bpc;
                    else b2000 = bpc;
                }
            }
        };

        float p1000 = 0.f, p2000 = 0.f, e1000 = 0.f, e2000 = 0.f;
        prime8(primed, tgt_starts.data());
        primed.init_feedback();
        fit(primed, "prime", p1000, p2000);
        prime8(ema, tgt_starts.data());
        ema.seed_ema_from_rates();
        ema.ema_mask = true;
        ema.init_feedback();
        fit(ema, "ema", e1000, e2000);

        const float z1000 = 2.64071f;
        const float z2000 = 2.674f;
        std::cout << "\ntest1\n";
        std::cout << "mod            1000     2000\n";
        std::cout << "SAP-QLoRA     " << z1000 << "  " << z2000 << "\n";
        std::cout << "+prime        " << p1000 << "  " << p2000 << "\n";
        std::cout << "+ema          " << e1000 << "  " << e2000 << "\n" << std::flush;
        const char* choice = "none";
        float best = z2000;
        if (p2000 < best) {
            best = p2000;
            choice = "prime";
        }
        if (e2000 < best) {
            best = e2000;
            choice = "ema";
        }
        std::cout << "choice " << choice << "  2000 BPC " << best << "\n" << std::flush;
        std::ofstream choice_out(choice_path, std::ios::binary);
        expect(static_cast<bool>(choice_out), "choice file");
        choice_out << choice;
        std::cout << "test1 done\n" << std::flush;
        return;
    }

    if (cfg.qsuite == 2) {
        std::ifstream choice_in(choice_path);
        std::string choice;
        expect(static_cast<bool>(choice_in), "mask choice missing, run suite 1 first");
        choice_in >> choice;
        expect(choice == "none" || choice == "prime" || choice == "ema", "mask choice");
        const int cap = 800;
        const int d = 1024;
        std::mt19937 rng(cfg.seed + 23u);
        std::uniform_int_distribution<int> pick_base(0, base_end - T - 2);
        std::vector<int> base_starts(static_cast<std::size_t>(cap) * batch);
        for (int s = 0; s < cap * batch; ++s) base_starts[static_cast<std::size_t>(s)] = pick_base(rng);
        int trained = 0;
        float frozen = 0.f;
        {
            Shake base;
            base.configure(T, d, vocab.V, blocks, batch);
            base.int8_strip = true;
            base.adam = true;
            base.wd = wd;
            base.mom = k_shake_mom;
            base.max_norm = 100.f;
            base.clip = true;
            base.align_k = 0;
            base.clusters = 1;
            base.init_weights(cfg.seed);
            base.prepare();
            frozen = shake_bpc(base, text, vocab, tgt_val_begin, tgt_end);
            std::cout << "base step 0  val BPC " << frozen << "\n" << std::flush;
            std::vector<int> tok;
            Matrix Y;
            for (int s = 1; s <= cap; ++s) {
                shake_fill_batch(text, vocab, base_starts.data() + static_cast<std::size_t>(s - 1) * batch, batch, T, tok, Y);
                base.step(tok.data(), Y, lr, true, nullptr);
                if (s % 100 == 0) {
                    frozen = shake_bpc(base, text, vocab, tgt_val_begin, tgt_end);
                    trained = s;
                    std::cout << "base step " << s << "  val BPC " << frozen << "\n" << std::flush;
                    if (!std::isfinite(frozen)) throw std::runtime_error("base val is not a number");
                    if (s >= 500 && frozen <= 3.3f) break;
                }
            }
            base.save_frozen(w0_path);
            std::cout << "W0 saved  " << w0_path << "  step " << trained << "  val " << frozen << "\n" << std::flush;
        }

        const int lora_steps = 1000;
        std::mt19937 rng_lora(cfg.seed + 101u);
        std::uniform_int_distribution<int> pick_tgt(tgt_begin, tgt_train_end - T - 2);
        std::vector<int> tgt_starts(static_cast<std::size_t>(lora_steps) * batch);
        for (int s = 0; s < lora_steps * batch; ++s) tgt_starts[static_cast<std::size_t>(s)] = pick_tgt(rng_lora);

        auto arm = [&](bool sap) {
            Shake net;
            net.configure(T, d, vocab.V, blocks, batch);
            net.int8_strip = true;
            net.adam = true;
            net.wd = wd;
            net.mom = k_shake_mom;
            net.max_norm = 100.f;
            net.clip = true;
            net.sap = sap;
            net.align_k = sap ? 100 : 0;
            net.clusters = 1;
            net.init_weights(cfg.seed);
            net.prepare();
            net.load_frozen(w0_path);
            net.attach_lora(lora_seed);
            if (sap) {
                net.compress_w0();
                if (choice == "prime" || choice == "ema") prime8(net, tgt_starts.data());
                if (choice == "ema") {
                    net.seed_ema_from_rates();
                    net.ema_mask = true;
                }
                net.init_feedback();
            }
            return net;
        };

        float curve[2][10] = {};
        int best_at[2] = {};
        float best_bpc[2] = {1e9f, 1e9f};
        for (int i = 0; i < 2; ++i) {
            Shake net = arm(i == 1);
            const char* name = i == 0 ? "BP-LoRA" : "SAP-QLoRA";
            std::vector<int> tok;
            Matrix Y;
            if (i == 0) {
                const float b0 = shake_bpc(net, text, vocab, tgt_val_begin, tgt_end);
                std::cout << "lora step 0  val BPC " << b0 << "  mask " << choice << "\n" << std::flush;
            }
            for (int s = 1; s <= lora_steps; ++s) {
                shake_fill_batch(text, vocab, tgt_starts.data() + static_cast<std::size_t>(s - 1) * batch, batch, T, tok, Y);
                net.step(tok.data(), Y, lr, true, nullptr);
                if (s % 100 == 0) {
                    const int k = s / 100 - 1;
                    curve[i][k] = shake_bpc(net, text, vocab, tgt_val_begin, tgt_end);
                    std::cout << name << " step " << s << "  val BPC " << curve[i][k] << "\n" << std::flush;
                    if (curve[i][k] < best_bpc[i]) {
                        best_bpc[i] = curve[i][k];
                        best_at[i] = s;
                    }
                }
            }
        }
        std::cout << "\ntest2  mask " << choice << "  base val " << frozen << "  base step " << trained << "\n";
        std::cout << "step    BP-LoRA   SAP-QLoRA\n";
        for (int k = 0; k < 10; ++k)
            std::cout << (k + 1) * 100 << "  " << curve[0][k] << "  " << curve[1][k] << "\n";
        std::cout << "best BP " << best_bpc[0] << " @ " << best_at[0] << "   SAP " << best_bpc[1] << " @ " << best_at[1]
                  << "   gap " << (best_bpc[1] - best_bpc[0]) << "\n";
        std::cout << "test2 done\n" << std::flush;
        return;
    }

    const int d = 2048;
    const int warmup = 10;
    const int measure = 100;
    const int total = warmup + measure;
    std::mt19937 rng(cfg.seed + 23u);
    std::uniform_int_distribution<int> pick(0, base_end - T - 2);
    std::vector<int> starts(static_cast<std::size_t>(total) * batch);
    for (int s = 0; s < total * batch; ++s) starts[static_cast<std::size_t>(s)] = pick(rng);
    const double mb = 1.0 / (1024.0 * 1024.0);
    std::cout << "speed D " << d << "  warmup " << warmup << "  measure " << measure << "  K 100  val off  " << qlora_kernel()
              << "\n";
    std::cout << "mod    fwd  out   bwd    dAB dequant update  total     MB    W0\n";
    double speed_ms[3] = {};
    for (int i = 0; i < 3; ++i) {
        const char* name = i == 0 ? "BP" : (i == 1 ? "SAP" : "QL");
        Shake net;
        net.configure(T, d, vocab.V, blocks, batch);
        net.int8_strip = true;
        net.sap = i != 0;
        net.adam = true;
        net.wd = wd;
        net.mom = k_shake_mom;
        net.max_norm = 100.f;
        net.clip = true;
        net.align_k = i == 0 ? 0 : 100;
        net.clusters = 1;
        const std::size_t b0 = g_mem.live;
        net.init_weights(cfg.seed + 3u);
        net.prepare();
        net.attach_lora(lora_seed);
        if (i == 2) net.compress_w0();
        if (net.sap) net.init_feedback();
        expect(g_mem.live - b0 == net.matrix_bytes(), std::string(name) + " memory counter");
        std::vector<int> tok;
        Matrix Y;
        auto replay = [&](int begin, int nstep) {
            for (int s = 0; s < nstep; ++s) {
                shake_fill_batch(text, vocab, starts.data() + static_cast<std::size_t>(begin + s) * batch, batch, T, tok, Y);
                net.step(tok.data(), Y, lr, true, nullptr);
            }
        };
        replay(0, warmup);
        net.reset_profile();
        const auto t0 = std::chrono::steady_clock::now();
        replay(warmup, measure);
        speed_ms[i] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / measure;
        const double inv = 1.0 / (static_cast<double>(measure) * 1000.0);
        const double bucket[6] = {net.us_fwd * inv, net.us_out * inv, net.us_bwd * inv,
                                  net.us_dw * inv,  net.us_deq * inv, net.us_upd * inv};
        const double sum = bucket[0] + bucket[1] + bucket[2] + bucket[3] + bucket[4] + bucket[5];
        const double w0 = static_cast<double>(net.qlora ? net.w0_pack_bytes() : bytes_of(net.Wch)) * mb;
        char line[256];
        std::snprintf(line, sizeof(line), "%-4s %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %7.2f %7.2f %6.2f  wall %7.2f\n",
                      name, bucket[0], bucket[1], bucket[2], bucket[3], bucket[4], bucket[5], sum,
                      static_cast<double>(net.bytes()) * mb, w0, speed_ms[i]);
        std::cout << line << std::flush;
    }
    std::cout << "D " << d << " speed QL/BP " << (speed_ms[2] > 0.0 ? speed_ms[0] / speed_ms[2] : 0.0) << "x   QL/SAP "
              << (speed_ms[2] > 0.0 ? speed_ms[1] / speed_ms[2] : 0.0) << "x\n";
    std::cout << "test3 done\n" << std::flush;
}

void run_shake(const Config& cfg) {
    if (cfg.qsuite) {
        if (cfg.scale || cfg.trio || cfg.adam || cfg.phase2 || cfg.lora || cfg.qlora)
            throw std::runtime_error("qsuite is a separate run");
        run_shake_qsuite(cfg);
        return;
    }
    if (cfg.qlora) {
        if (cfg.scale || cfg.trio || cfg.adam || cfg.phase2 || cfg.lora)
            throw std::runtime_error("qlora is a separate run");
        run_shake_qlora(cfg);
        return;
    }
    if (cfg.lora) {
        if (cfg.scale || cfg.trio || cfg.adam || cfg.phase2)
            throw std::runtime_error("lora is a separate run");
        run_shake_lora(cfg);
        return;
    }
    if (cfg.scale) {
        run_shake_scale(cfg);
        return;
    }
    if (cfg.trio || cfg.adam || cfg.phase2) {
        run_shake_ablation(cfg);
        return;
    }
    const std::vector<std::uint8_t> text = read_bytes(cfg.shake);
    const ShakeVocab vocab = shake_vocab(text);
    expect(vocab.V == 65, "TinyShakespeare vocab is not 65");
    const int train_end = static_cast<int>(text.size() * 9 / 10);
    const int val_begin = train_end;
    const int val_end = static_cast<int>(text.size());
    expect(train_end > k_shake_T + 2 && val_end - val_begin > k_shake_T + 2, "split shorter than the window");

    Shake A;
    A.configure(k_shake_T, k_shake_D, vocab.V, k_shake_blocks, cfg.batch);
    A.max_norm = cfg.max_norm;
    A.clip = cfg.clip;
    A.align_k = cfg.align_k > 0 ? cfg.align_k : 10;
    A.init_weights(cfg.seed);
    A.prepare();
    Shake B = A;
    B.sap = true;
    B.prepare();
    B.init_feedback();
    const bool diag_k = cfg.k2 > 0;
    Shake B2;
    if (diag_k) {
        B2 = A;
        B2.sap = true;
        B2.align_k = cfg.k2;
        B2.prepare();
        B2.init_feedback();
    }

    std::mt19937 rng(cfg.seed + 99u);
    std::uniform_int_distribution<int> pick(0, train_end - k_shake_T - 2);
    std::vector<int> starts(static_cast<std::size_t>(cfg.batch));
    std::vector<int> tok;
    Matrix Y;
    const double ln2 = std::log(2.0);

    std::cout << "\nshakespeare " << cfg.shake << "  bytes " << text.size() << "  train " << train_end << "  val "
              << (val_end - val_begin) << "  V " << vocab.V << "\n";
    std::cout << "mixer     T " << A.T << "  D " << A.D << "  block " << A.blocks << "  batch " << A.batch
              << "  alpha " << A.alpha << "  momentum " << A.mom << "  wd " << k_shake_wd << "  lr " << cfg.lr
              << "  K " << A.align_k;
    if (diag_k) std::cout << "  K2 " << B2.align_k;
    std::cout << "  max-norm " << A.max_norm << "  tau";
    for (int l = 0; l < A.blocks; ++l) std::cout << " " << A.tau[l];
    std::cout << "\n";
    std::cout << "A memory  " << (static_cast<double>(A.bytes()) / (1024.0 * 1024.0)) << " MB   B memory "
              << (static_cast<double>(B.bytes()) / (1024.0 * 1024.0)) << " MB\n";

    using clock = std::chrono::steady_clock;
    double used_a = 0.0;
    double used_b = 0.0;
    double used_b2 = 0.0;
    double sum_a = 0.0;
    double sum_b = 0.0;
    double sum_b2 = 0.0;
    int n_a = 0;
    int n_b = 0;
    int n_b2 = 0;
    int steps_a = 0;
    int steps_b = 0;
    int steps_b2 = 0;
    bool a_alive = true;
    bool b_alive = true;
    bool b2_alive = diag_k;
    const std::string prompt = "ROMEO:\n";
    for (int step = 1; step <= cfg.steps; ++step) {
        for (int b = 0; b < cfg.batch; ++b) starts[static_cast<std::size_t>(b)] = pick(rng);
        float lr_now = cfg.lr;
        if (step > 15000) lr_now = 0.015f;
        else if (step > 10000) lr_now = 0.05f;
        if (step == 10001 || step == 15001) std::cout << "lr " << lr_now << "\n" << std::flush;
        shake_fill_batch(text, vocab, starts.data(), cfg.batch, k_shake_T, tok, Y);
        const bool probe = cfg.steps >= 5000 && (step == 1 || step == 200 || step == 500 || step % 1000 == 0);
        Shake::AlignView av;
        Shake::AlignView av2;
        if (a_alive) {
            const auto t0 = clock::now();
            const float la = A.step(tok.data(), Y, lr_now, true);
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
            const float lb = B.step(tok.data(), Y, lr_now, true, probe ? &av : nullptr);
            used_b += std::chrono::duration<double, std::micro>(clock::now() - t0).count();
            ++steps_b;
            if (!std::isfinite(lb)) b_alive = false;
            else {
                sum_b += lb;
                ++n_b;
            }
        }
        if (b2_alive) {
            const auto t0 = clock::now();
            const float lb2 = B2.step(tok.data(), Y, lr_now, true, probe ? &av2 : nullptr);
            used_b2 += std::chrono::duration<double, std::micro>(clock::now() - t0).count();
            ++steps_b2;
            if (!std::isfinite(lb2)) b2_alive = false;
            else {
                sum_b2 += lb2;
                ++n_b2;
            }
        }
        if (probe && b_alive) {
            char cline[512];
            if (diag_k && b2_alive) {
                std::snprintf(cline, sizeof(cline),
                              "cos K=%d step %5d block%d %6.3f ratio %6.3f | block0 %6.3f ratio %6.3f | E %6.3f ratio %6.3f"
                              " | K=%d block%d %6.3f ratio %6.3f | block0 %6.3f ratio %6.3f | E %6.3f ratio %6.3f\n",
                              B.align_k, step, B.blocks - 1, av.cos_top, av.nr_top, av.cos_bot, av.nr_bot, av.cos_e,
                              av.nr_e, B2.align_k, B2.blocks - 1, av2.cos_top, av2.nr_top, av2.cos_bot, av2.nr_bot,
                              av2.cos_e, av2.nr_e);
            } else {
                std::snprintf(cline, sizeof(cline),
                              "cos K=%d step %5d block%d %6.3f ratio %6.3f | block0 %6.3f ratio %6.3f | E %6.3f ratio %6.3f\n",
                              B.align_k, step, B.blocks - 1, av.cos_top, av.nr_top, av.cos_bot, av.nr_bot, av.cos_e,
                              av.nr_e);
            }
            std::cout << cline << std::flush;
        }
        const bool report = step % 1000 == 0 || step == cfg.steps;
        if (!report) continue;
        const float ba = a_alive ? shake_bpc(A, text, vocab, val_begin, val_end) : std::numeric_limits<float>::quiet_NaN();
        const float bb = b_alive ? shake_bpc(B, text, vocab, val_begin, val_end) : std::numeric_limits<float>::quiet_NaN();
        const float bb2 = b2_alive ? shake_bpc(B2, text, vocab, val_begin, val_end) : std::numeric_limits<float>::quiet_NaN();
        const float tra = n_a ? static_cast<float>((sum_a / n_a) / ln2) : std::numeric_limits<float>::quiet_NaN();
        const float trb = n_b ? static_cast<float>((sum_b / n_b) / ln2) : std::numeric_limits<float>::quiet_NaN();
        const float trb2 = n_b2 ? static_cast<float>((sum_b2 / n_b2) / ln2) : std::numeric_limits<float>::quiet_NaN();
        char line[640];
        if (diag_k) {
            std::snprintf(line, sizeof(line),
                          "step %5d | A val %6.3f %6.2f ms clip %d | K=%d val %6.3f %6.2f ms clip %d | "
                          "K=%d val %6.3f %6.2f ms clip %d\n",
                          step, ba, steps_a ? (used_a / steps_a) / 1000.0 : 0.0, A.clip_hits, B.align_k, bb,
                          steps_b ? (used_b / steps_b) / 1000.0 : 0.0, B.clip_hits, B2.align_k, bb2,
                          steps_b2 ? (used_b2 / steps_b2) / 1000.0 : 0.0, B2.clip_hits);
        } else {
            std::snprintf(line, sizeof(line),
                          "step %5d | A train %6.3f val %6.3f bpc  time %6.2f ms mem %6.2f MB clip %d | "
                          "B train %6.3f val %6.3f bpc  time %6.2f ms mem %6.2f MB clip %d\n",
                          step, tra, ba, steps_a ? (used_a / steps_a) / 1000.0 : 0.0,
                          static_cast<double>(A.bytes()) / (1024.0 * 1024.0), A.clip_hits, trb, bb,
                          steps_b ? (used_b / steps_b) / 1000.0 : 0.0, static_cast<double>(B.bytes()) / (1024.0 * 1024.0),
                          B.clip_hits);
        }
        (void)trb2;
        std::cout << line << std::flush;
        sum_a = 0.0;
        sum_b = 0.0;
        sum_b2 = 0.0;
        n_a = 0;
        n_b = 0;
        n_b2 = 0;
        if (!diag_k && step == cfg.steps) {
            if (a_alive) {
                std::cout << "A sample T=0.7 k=15\n"
                          << shake_generate(A, vocab, prompt, 400, 0.7f, 15, cfg.seed + 17u) << "\n----\n";
            }
            if (b_alive) {
                std::cout << "B sample T=0.7 k=15\n"
                          << shake_generate(B, vocab, prompt, 400, 0.7f, 15, cfg.seed + 17u) << "\n----\n";
            }
        }
    }
    std::cout << "clip A " << A.clip_hits << " scale " << (A.scale_n ? A.scale_sum / A.scale_n : 1.0) << "  B "
              << B.clip_hits << " scale " << (B.scale_n ? B.scale_sum / B.scale_n : 1.0);
    if (diag_k) std::cout << "  B2 " << B2.clip_hits << " sweep " << B2.sweep_calls;
    std::cout << "\n";
    std::cout << "sweep B " << B.sweep_calls << "\n";
    auto prof = [](const char* name, const Shake& net, int n) {
        if (n <= 0) return;
        const double s = 1.0 / (static_cast<double>(n) * 1000.0);
        const double fwd = net.us_fwd * s;
        const double out = net.us_out * s;
        const double bwd = net.us_bwd * s;
        const double dw = net.us_dw * s;
        const double deq = net.us_deq * s;
        const double upd = net.us_upd * s;
        char line[320];
        std::snprintf(line, sizeof(line),
                      "profile %s  fwd %5.2f  out %5.2f  bwd %5.2f  dW %5.2f (dequant %5.2f)  update %5.2f  total %5.2f ms\n",
                      name, fwd, out, bwd, dw, deq, upd, fwd + out + bwd + dw + deq + upd);
        std::cout << line;
    };
    prof("A", A, steps_a);
    prof("B", B, steps_b);
}
