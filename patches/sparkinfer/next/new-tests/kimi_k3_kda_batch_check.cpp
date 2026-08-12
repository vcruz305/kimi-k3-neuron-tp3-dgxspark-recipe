// KDA batched Attn equivalence: kimi_k3_forward_layer_phase(Attn, n_tok=K) on a KDA
// layer must be BIT-IDENTICAL to K sequential n_tok=1 calls over the same K rows.
//
//   kimi_k3_kda_batch_check <first-shard.gguf> [Kmax] [layer]
//
// WHY THIS EXISTS. SPARKINFER_K3_KDA_QKVG_BATCH is the gate that lets a KDA layer take
// n_tok > 1 for its PROJECTION half (q/k/v/g and attn_output; the conv + delta-rule scan
// stays a per-token loop by construction). It ships OFF because turning it on faulted --
//   [k3] LAUNCH FAILED at layer 0, phase Attn: invalid argument
// -- for every chunk >= 2 tokens, and nothing in the tree reproduced that at a size a
// person could iterate on. This does: ONE layer, one GPU, seconds per run.
//
// It is also the equivalence proof the gate needs before it can default on. 69 of 93
// layers are KDA and their attention projections are 52.1% of the dense bytes a decode
// token streams, so this path is the difference between a batched verify amortising most
// of the model and half of it.
//
// THE COMPARISON IS BITWISE, NOT APPROXIMATE, AND THAT IS THE POINT. The batched arm is
// supposed to change WHEN the projections are issued, never what they compute: same
// weights, same activation bytes, same accumulation order per output element. Any
// tolerance at all would let a real reordering through, and this engine has already
// shipped one silent-wrong-answer bug that every approximate check passed.
//
// Both arms drive the SAME recurrent state forward across the K rows -- reset once, then
// tokens 0..K-1 in order -- so this tests the scan's carry as well as the projections.
//
// THE `full` MODE IS THE ONE THAT MATTERS FOR THE GATE, and the reason is in the
// CHANGELOG. #148 shipped TWO defects together: this gate being read opt-out, and a
// residual-bank stride swap in which `(n_rows, act_row_stride, bank_row_stride)` was
// called as `(n_rows, bank_row, H)`. The second is invisible until the model is deep
// enough to bank TWICE -- max_ckpt is ceil(n_layers/12), so at <= 12 layers the two
// strides are equal and each wrong argument lands on the right value. It was bisected
// as "bit-identical through 12 layers, KLD 1.64 at 14".
//
// Both were fixed in #148. Only the second was fixed by CORRECTING it; the first was
// "fixed" by disabling the feature, and its comment says so ("left opt-in until the
// fault is understood"). So the `LAUNCH FAILED` narrative in that comment describes a
// tree that no longer exists, and nothing re-tested the gate afterwards.
//
// `full` runs >= 13 layers through phase All so the bank is written twice. That is the
// exact condition the stride bug needed, and it is the arm that can tell "the fault was
// collateral damage from the other defect" from "the fault is still there".

#include "sparkinfer/gguf.h"
#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/models/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_config.h"
#include "sparkinfer/models/kimi_k3_gguf_manifest.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace sparkinfer;
namespace k3k = sparkinfer::kernels::k3;

namespace {

struct Ctx {
    KimiK3Weights      w;
    KimiK3RuntimeState st;
    KimiK3Forward      fwd;
    // What alloc_state handed out, so the per-token repointing below can be undone
    // before free. state.d_pos and state.res_bank are OWNED by kimi_k3_alloc_state and
    // freed through it; leaving them pointing at this harness's chunk buffers would
    // free the wrong allocation.
    int*   base_d_pos = nullptr;
    float* base_bank  = nullptr;
    cudaStream_t stream = nullptr;
};

bool setup(Ctx& c, const GGUF& g, const KimiK3Config& cfg, const K3PlanOptions& opt,
           int n_layers, int max_ctx, int tp_size, int rank, int layer) {
    // ExpertsAndKda at tp > 1 is what the fleet actually runs: the KDA heads band, so
    // qkv narrows from n_q_heads*head_dim to this rank's share and the recurrent state
    // divides with it. Testing only the tp=1 geometry would leave the shipped shape
    // unproven -- a batched kernel that is fine at qkv 12288 and wrong at 4096 is
    // exactly the kind of shape dependence this is looking for.
    c.w.policy = (tp_size > 1) ? KimiK3Weights::ShardPolicy::ExpertsAndKda
                               : KimiK3Weights::ShardPolicy::ExpertsOnly;
    c.w.shard.tp_size = tp_size;
    c.w.shard.rank = rank;
    c.w.shard.hidden = cfg.hidden;
    c.w.shard.n_experts_total = cfg.n_experts;
    c.w.shard.n_experts = cfg.n_experts;
    c.w.shard.expert_band = tp::Band{0, cfg.n_experts};
    c.w.shard.experts_sharded = false;
    c.w.shard.moe_ffn_total = cfg.moe_ffn;
    c.w.shard.moe_ffn = cfg.moe_ffn;
    c.w.shard.moe_shard_mode = tp::MoeShardMode::None;

    // A MoE layer's expert tensors need a legal mode at tp>1 or the loader refuses
    // them. Irrelevant to this comparison -- only the Attn phase runs -- but it has
    // to load.
    if (tp_size > 1 && layer >= cfg.leading_dense) {
        const tp::ShardError e = tp::moe_all_expert_width_dims(
            cfg.n_experts, cfg.moe_ffn, tp_size, rank, 256, &c.w.shard);
        if (!e.ok()) { std::printf("  width dims: %s\n", e.message.c_str()); return false; }
        c.w.shard.hidden = cfg.hidden;
    }

    if (!kimi_k3_load_weights(g, cfg, opt, c.w, 0, n_layers - 1)) return false;
    // The KDA state must be sized to THIS RANK's heads, or the forward indexes past
    // the end of the allocation.
    if (!kimi_k3_alloc_state(cfg, max_ctx, c.st, 0, n_layers - 1,
                             tp_size > 1 ? &c.w.kda : nullptr)) return false;
    c.fwd.cfg = &cfg; c.fwd.w = &c.w; c.fwd.state = &c.st;
    c.fwd.opt = opt;
    // A REAL STREAM, not the legacy default one, because the chunk driver captures its
    // chunk into a CUDA graph and the legacy stream cannot be captured. Capture is the
    // one structural difference left between this harness and the driver that faulted,
    // so the harness has to be able to enter it.
    if (cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking) != cudaSuccess)
        return false;
    c.fwd.stream = c.stream;
    if (!kimi_k3_forward_alloc_scratch(cfg, c.fwd)) return false;
    c.base_d_pos = c.st.d_pos;
    c.base_bank  = c.st.res_bank;
    return true;
}

void teardown(Ctx& c) {
    c.st.d_pos    = c.base_d_pos;
    c.st.res_bank = c.base_bank;
    if (c.stream) cudaStreamDestroy(c.stream);
    kimi_k3_forward_free_scratch(c.fwd);
    kimi_k3_free_state(c.st);
    kimi_k3_free_weights(c.w);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <first-shard.gguf> [Kmax] [layer] [tp_size] [rank] "
                     "[full]\n",
                     argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int Kmax    = argc > 2 ? std::atoi(argv[2]) : 8;
    const int tp_size = argc > 4 ? std::atoi(argv[4]) : 1;
    const int rank    = argc > 5 ? std::atoi(argv[5]) : 0;
    // "full": run layers 0..layer through phase All instead of one layer's Attn.
    const bool full   = argc > 6 && std::string(argv[6]) == "full";

    GGUF g; KimiK3Config cfg; KimiK3LayerCoverage cov;
    if (!kimi_k3_load_partial(path.c_str(), cfg, g, &cov)) {
        std::printf("load failed\n"); return 1;
    }
    int n_avail = 0;
    while (n_avail < cfg.n_layers && cov.layer_complete[n_avail]) ++n_avail;

    // An explicit layer may be MLA: the batched Attn phase takes both layer types and
    // the MLA half has been on by default all along, so it is the control arm -- if MLA
    // batching also faults in this harness the fault is not KDA-specific.
    int layer = argc > 3 ? std::atoi(argv[3]) : -1;
    if (layer < 0)
        for (int i = 0; i < n_avail; ++i) if (cfg.is_kda_layer(i)) { layer = i; break; }
    if (layer < 0 || layer >= n_avail) {
        std::printf("layer %d not available (n_avail=%d)\n", layer, n_avail);
        return 1;
    }
    const int n_layers = layer + 1;

    K3PlanOptions opt;
    {
        const std::string p = "blk." + std::to_string(layer) + ".";
        opt.has_q_lora         = g.tensor((p + "attn_q_a.weight").c_str()) != nullptr;
        opt.has_attn_gate      = g.tensor((p + "attn_gate.weight").c_str()) != nullptr;
        opt.has_shared_experts = g.tensor((p + "ffn_gate_shexp.weight").c_str()) != nullptr;
        opt.has_routed_norm    = g.tensor((p + "ffn_routed_norm.weight").c_str()) != nullptr;
    }

    const int H = cfg.hidden;
    const int max_ctx = Kmax + 8;

    Ctx c;
    if (!setup(c, g, cfg, opt, n_layers, max_ctx, tp_size, rank, layer)) {
        std::printf("setup failed\n"); return 1;
    }

    const int cap = kimi_k3_ffn_batch_cap(c.fwd);
    std::printf("batched-Attn equivalence: layer %d (%s), hidden %d, n_q_heads %d, "
                "kda_head_dim %d, tp %d rank %d, scratch cap %d\n",
                layer, cfg.is_kda_layer(layer) ? "KDA" : "MLA", H, cfg.n_q_heads,
                cfg.kda_head_dim, tp_size, rank, cap);
    if (cfg.is_kda_layer(layer))
        std::printf("  kda heads=%d qkv_band=[%d,%d) head_band=[%d,%d)\n",
                    c.w.kda.n_heads, c.w.kda.qkv_band.offset, c.w.kda.qkv_band.end(),
                    c.w.kda.head_band.offset, c.w.kda.head_band.end());
    if (cap < Kmax) {
        std::printf("  scratch cap %d < Kmax %d -- raise SPARKINFER_K3_PREFILL_CHUNK\n",
                    cap, Kmax);
        teardown(c); return 1;
    }
    std::printf("  attn_batch_ok(layer %d, n_tok 2) = %s\n", layer,
                kimi_k3_attn_batch_ok(c.fwd, layer, 2) ? "true"
                    : "FALSE (set SPARKINFER_K3_KDA_QKVG_BATCH=1)");

    // ---- inputs: Kmax distinct hidden rows, token-major and CONTIGUOUS at exactly H.
    // Contiguity is load-bearing rather than convenient: the batched activation
    // quantise is flat over 32-element blocks and has no row index to apply a stride
    // to, so a padded leading dimension would quantise the padding as data.
    std::mt19937 rng(20260811);
    std::normal_distribution<float> N01(0.f, 1.f);
    std::vector<float> host_in((size_t)Kmax * H);
    for (auto& v : host_in) v = N01(rng);

    float* d_in = nullptr; float* d_alt = nullptr;
    int*   d_pos_vec = nullptr; float* d_bank = nullptr;
    const int    max_ckpt   = c.st.max_ckpt;
    const size_t bank_elems = (size_t)max_ckpt * (size_t)c.st.res_bank_row_elems;
    cudaMalloc(&d_in,  (size_t)Kmax * H * sizeof(float));
    cudaMalloc(&d_alt, (size_t)Kmax * H * sizeof(float));
    cudaMalloc(&d_pos_vec, (size_t)Kmax * sizeof(int));
    if (bank_elems) cudaMalloc(&d_bank, (size_t)Kmax * bank_elems * sizeof(float));

    if (full)
        std::printf("  FULL mode: layers 0..%d, phase All, max_ckpt %d (banks %s)\n",
                    n_layers - 1, max_ckpt, max_ckpt > 1 ? "TWICE - stride exposed"
                                                         : "once - stride NOT exposed");

    // Walk `x` through every loaded layer, ping-ponging into `xn`. Returns the buffer
    // the result landed in, or null on failure. Both arms run the same layer count so
    // the parity -- and therefore which buffer holds the answer -- matches.
    auto run_stack = [&](float* x, float* xn, int n_tok) -> float* {
        for (int l = 0; l < n_layers; ++l) {
            if (!kimi_k3_forward_layer_phase(c.fwd, l, K3LayerPhase::All, x, xn, n_tok))
                return nullptr;
            float* t = x; x = xn; xn = t;
        }
        return x;
    };

    int fails = 0;
    for (int K = 2; K <= Kmax; ++K) {
        std::vector<float> seq((size_t)K * H, 0.f), bat((size_t)K * H, 0.f);

        // ---- arm A: K sequential single-token calls, state carried across them ----
        kimi_k3_reset_state(c.st);
        bool ok_seq = true;
        for (int b = 0; b < K && ok_seq; ++b) {
            kimi_k3_set_position(c.st, b);
            if (d_bank) c.st.res_bank = d_bank + (size_t)b * bank_elems;
            // n_ckpt is reset at the top of every forward call -- the bank is
            // intra-token scratch, not cross-step state -- so token b starts at 0
            // exactly as kimi_k3_dist_forward_token does.
            c.st.n_ckpt = 0;
            cudaMemcpy(d_in + (size_t)b * H, host_in.data() + (size_t)b * H,
                       (size_t)H * sizeof(float), cudaMemcpyHostToDevice);
            if (full) {
                float* got = run_stack(d_in + (size_t)b * H, d_alt + (size_t)b * H, 1);
                ok_seq = got != nullptr;
                if (ok_seq) cudaStreamSynchronize(c.stream);
                if (ok_seq)
                    cudaMemcpy(seq.data() + (size_t)b * H, got, (size_t)H * sizeof(float),
                               cudaMemcpyDeviceToHost);
            } else {
                ok_seq = kimi_k3_forward_layer_phase(c.fwd, layer, K3LayerPhase::Attn,
                                                     d_in + (size_t)b * H,
                                                     d_alt + (size_t)b * H, 1);
                if (ok_seq) {
                    cudaStreamSynchronize(c.stream);
                    int n = 0;
                    float* part = kimi_k3_partial_buffer(c.fwd, layer,
                                                         K3LayerPhase::Attn, &n);
                    cudaMemcpy(seq.data() + (size_t)b * H, part,
                               (size_t)H * sizeof(float), cudaMemcpyDeviceToHost);
                }
            }
        }
        c.st.res_bank = c.base_bank;
        if (!ok_seq) {
            std::printf("K=%d: SEQUENTIAL arm FAILED\n", K);
            ++fails; continue;
        }

        // ---- arm B: one batched call over the same K rows ----
        kimi_k3_reset_state(c.st);
        kimi_k3_set_position(c.st, 0);
        // Token b's position is d_pos[b]; state.position is TOKEN 0's. Handing the
        // batched phase a single shared position is the "every token attends over the
        // same prefix" bug -- wrong AND faster, so a timing run cannot see it.
        k3k::k3_fill_pos_vec(d_pos_vec, c.st.d_pos, K, c.stream);
        cudaStreamSynchronize(c.stream);
        c.st.d_pos = d_pos_vec;
        if (d_bank) c.st.res_bank = d_bank;
        c.st.n_ckpt = 0;
        cudaMemcpy(d_in, host_in.data(), (size_t)K * H * sizeof(float),
                   cudaMemcpyHostToDevice);
        bool ok_bat = true;
        if (full) {
            float* got = run_stack(d_in, d_alt, K);
            ok_bat = got != nullptr;
            if (ok_bat) cudaStreamSynchronize(c.stream);
            if (ok_bat)
                cudaMemcpy(bat.data(), got, (size_t)K * H * sizeof(float),
                           cudaMemcpyDeviceToHost);
        } else {
            ok_bat = kimi_k3_forward_layer_phase(c.fwd, layer, K3LayerPhase::Attn,
                                                 d_in, d_alt, K);
            if (ok_bat) {
                cudaStreamSynchronize(c.stream);
                // The Attn partial strides by `hidden` per token; kimi_k3_partial_buffer
                // reports ONE token's width, so the K-row read is computed here rather
                // than taken from it. That asymmetry is the exact trap a distributed
                // batched port has to avoid when it sizes an all-reduce.
                int n = 0;
                float* part = kimi_k3_partial_buffer(c.fwd, layer,
                                                     K3LayerPhase::Attn, &n);
                cudaMemcpy(bat.data(), part, (size_t)K * H * sizeof(float),
                           cudaMemcpyDeviceToHost);
            }
        }
        c.st.d_pos    = c.base_d_pos;
        c.st.res_bank = c.base_bank;
        if (!ok_bat) {
            std::printf("K=%d: BATCHED arm FAILED (see the LAUNCH FAILED line above)\n", K);
            ++fails; continue;
        }

        // A DIGEST OF THE RAW BYTES, printed for both arms. Within one binary the diff
        // below is the real check; the digest is what makes an ACROSS-BINARY comparison
        // possible, which is the only way to prove a pure memory-layout refactor (the
        // KDA state arenas) changed no computed value. FNV-1a over the float bytes, so
        // a one-ULP move shows up.
        auto digest = [](const std::vector<float>& v) {
            uint64_t h = 1469598103934665603ULL;
            const unsigned char* p = (const unsigned char*)v.data();
            for (size_t i = 0; i < v.size() * sizeof(float); ++i)
                { h ^= p[i]; h *= 1099511628211ULL; }
            return h;
        };
        std::printf("K=%d: digest seq=%016llx bat=%016llx\n", K,
                    (unsigned long long)digest(seq), (unsigned long long)digest(bat));

        size_t n_diff = 0; double worst = 0; int worst_i = -1;
        for (size_t i = 0; i < seq.size(); ++i) {
            if (std::memcmp(&seq[i], &bat[i], sizeof(float)) != 0) {
                ++n_diff;
                const double d = std::fabs((double)seq[i] - (double)bat[i]);
                const double rel = d / (std::fabs((double)seq[i]) + 1e-9);
                if (rel > worst) { worst = rel; worst_i = (int)i; }
            }
        }
        if (n_diff == 0) {
            std::printf("K=%d: PASS  %zu elems bit-identical\n", K, seq.size());
        } else {
            std::printf("K=%d: FAIL  %zu/%zu elems differ, worst_rel %.3e @ %d "
                        "(seq=%.9g bat=%.9g)\n",
                        K, n_diff, seq.size(), worst, worst_i,
                        worst_i >= 0 ? seq[(size_t)worst_i] : 0.f,
                        worst_i >= 0 ? bat[(size_t)worst_i] : 0.f);
            ++fails;
        }
    }

    cudaFree(d_in); cudaFree(d_alt); cudaFree(d_pos_vec);
    if (d_bank) cudaFree(d_bank);
    teardown(c);

    std::printf("\n%s: batched %s %s K sequential calls for K=2..%d\n",
                fails == 0 ? "PASS" : "FAIL",
                full ? "full-stack forward" : "Attn phase",
                fails == 0 ? "reproduces" : "DOES NOT reproduce", Kmax);
    return fails == 0 ? 0 : 1;
}
