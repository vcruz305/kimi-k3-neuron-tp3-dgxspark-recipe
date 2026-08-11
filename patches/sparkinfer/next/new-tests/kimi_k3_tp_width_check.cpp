// AllExpertsFfnWidth: tp_size disjoint FFN-WIDTH bands, summed, must equal the
// unsharded MoE.
//
//   kimi_k3_tp_width_check <first-shard.gguf> [tp_size] [layer]
//
// THIS IS THE ONE MoE SHARDING MODE TP3 ACTUALLY USES, AND IT HAD NO NUMERICAL
// TEST. kimi_k3_tp_moe_check covers the OTHER mode (disjoint expert bands) and
// refuses TP3 outright -- 896 experts do not divide 3 -- so on a 3-Spark fleet it
// reports nothing at all. tp3_all_expert_width_cpu_test and tp_shard_cpu_test
// check the PLAN (dims, bands, receipts, divisibility), never the arithmetic. So
// the actual math of "gate/up column-split to moe_ffn/tp, situ elementwise, down
// row-split over the same band, shared expert banded the same way, everything
// fused into one expert_latent+hidden payload" was unvalidated on the exact
// configuration the fleet runs.
//
// Everything is simulated on ONE device, exactly as kimi_k3_tp_moe_check does it:
// rank r's weights are loaded at its band, its FfnPartial partial is read back to
// the host, and the host sums them. That is the same arithmetic the collective
// does, without needing tp_size GPUs.
//
// WHAT A FAILURE HERE WOULD MEAN, and why nothing else catches it: every rank
// computes the same wrong thing, so a cross-rank consistency probe sees three
// bit-identical ranks and reports success. In particular, a tensor that is loaded
// FULL WIDTH but still included in the fused all-reduce is summed tp_size times --
// the exact failure kimi_k3.cpp:243 documents having shipped once for the shared
// expert stack. That bug is invisible to every rank-comparison test by
// construction, and visible here immediately as a ~tp_size ratio.
//
// TOLERANCE, NOT BITWISE. The reference accumulates each expert's full moe_ffn
// width in one pass; the banded run accumulates tp_size sub-widths and then sums
// across ranks. Same addends, different association. What must NOT differ is the
// magnitude.

#include "sparkinfer/gguf.h"
#include "sparkinfer/models/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_config.h"
#include "sparkinfer/models/kimi_k3_gguf_manifest.h"
#include "sparkinfer/tp/shard.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace sparkinfer;

namespace {

// Run Attn + FfnPartial for `layer` and return this rank's fused MoE partial.
bool run_partial(const GGUF& g, const KimiK3Config& cfg, const K3PlanOptions& opt,
                 int layer, int n_layers, int tp_size, int rank,
                 const std::vector<float>& host_in, std::vector<float>& out_partial) {
    KimiK3Weights w;
    // Attention deliberately unsharded: this test is about the MoE axis only.
    w.policy = KimiK3Weights::ShardPolicy::ExpertsOnly;
    w.shard.tp_size = tp_size;
    w.shard.rank = rank;
    w.shard.hidden = cfg.hidden;
    w.shard.n_experts_total = cfg.n_experts;
    w.shard.moe_ffn_total = cfg.moe_ffn;

    if (tp_size > 1) {
        // 256 = the quant super-block the production callers pass
        // (rank_protocol.cpp, kimi_k3_tp.cpp).
        const tp::ShardError e = tp::moe_all_expert_width_dims(
            cfg.n_experts, cfg.moe_ffn, tp_size, rank, 256, &w.shard);
        if (!e.ok()) {
            std::printf("  moe_all_expert_width_dims: %s\n", e.message.c_str());
            return false;
        }
        w.shard.hidden = cfg.hidden;
    } else {
        // tp=1 identity: every expert, full width, no mode.
        w.shard.n_experts = cfg.n_experts;
        w.shard.expert_band = tp::Band{0, cfg.n_experts};
        w.shard.experts_sharded = false;
        w.shard.moe_ffn = cfg.moe_ffn;
        w.shard.moe_shard_mode = tp::MoeShardMode::None;
    }

    if (!kimi_k3_load_weights(g, cfg, opt, w, 0, n_layers - 1)) return false;

    KimiK3RuntimeState st;
    if (!kimi_k3_alloc_state(cfg, 8, st, 0, n_layers - 1)) return false;
    KimiK3Forward fwd;
    fwd.cfg = &cfg; fwd.w = &w; fwd.state = &st; fwd.opt = opt; fwd.stream = nullptr;
    if (!kimi_k3_forward_alloc_scratch(cfg, fwd)) return false;

    const int H = cfg.hidden;
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, (size_t)H * sizeof(float));
    cudaMalloc(&d_out, (size_t)H * sizeof(float));
    cudaMemcpy(d_in, host_in.data(), (size_t)H * sizeof(float), cudaMemcpyHostToDevice);

    kimi_k3_reset_state(st);
    bool ok = kimi_k3_forward_layer_phase(fwd, layer, K3LayerPhase::Attn, d_in, d_out) &&
              kimi_k3_forward_layer_phase(fwd, layer, K3LayerPhase::FfnPartial, d_in, d_out);
    cudaDeviceSynchronize();

    int n = 0;
    float* partial = kimi_k3_partial_buffer(fwd, layer, K3LayerPhase::FfnPartial, &n);
    if (ok && partial && n > 0) {
        out_partial.assign((size_t)n, 0.0f);
        cudaMemcpy(out_partial.data(), partial, (size_t)n * sizeof(float),
                   cudaMemcpyDeviceToHost);
    } else {
        ok = false;
    }

    cudaFree(d_in); cudaFree(d_out);
    kimi_k3_forward_free_scratch(fwd);
    kimi_k3_free_state(st);
    kimi_k3_free_weights(w);
    return ok;
}

void report(const char* name, const std::vector<float>& ref,
            const std::vector<float>& got, int lo, int hi) {
    double num = 0, den = 0, worst = 0;
    int worst_i = -1;
    for (int i = lo; i < hi; ++i) {
        const double d = (double)got[i] - (double)ref[i];
        num += d * d; den += (double)ref[i] * (double)ref[i];
        const double rel = std::fabs(d) / (std::fabs((double)ref[i]) + 1e-6);
        if (rel > worst) { worst = rel; worst_i = i; }
    }
    const double rl2 = std::sqrt(num / (den + 1e-30));
    double sref = 0, sgot = 0;
    for (int i = lo; i < hi; ++i) { sref += std::fabs(ref[i]); sgot += std::fabs(got[i]); }
    std::printf("  %-18s relL2 = %.3e   |ref|_1 = %.5g  |sum|_1 = %.5g  ratio = %.5f\n",
                name, rl2, sref, sgot, sref > 0 ? sgot / sref : 0.0);
    if (worst_i >= 0)
        std::printf("  %-18s worst_rel = %.3e @ %d (ref=%.6g sum=%.6g)\n",
                    "", worst, worst_i, ref[worst_i], got[worst_i]);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <first-shard.gguf> [tp_size] [layer]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int tp_size = argc > 2 ? std::atoi(argv[2]) : 3;

    GGUF g; KimiK3Config cfg; KimiK3LayerCoverage cov;
    if (!kimi_k3_load_partial(path.c_str(), cfg, g, &cov)) { std::printf("load failed\n"); return 1; }
    int n_avail = 0;
    while (n_avail < cfg.n_layers && cov.layer_complete[n_avail]) ++n_avail;

    int layer = argc > 3 ? std::atoi(argv[3]) : -1;
    if (layer < 0) {
        for (int i = cfg.leading_dense; i < n_avail; ++i) { layer = i; break; }
    }
    if (layer < 0 || layer >= n_avail) { std::printf("no complete MoE layer available\n"); return 1; }
    const int n_layers = layer + 1;

    K3PlanOptions opt;
    {
        const std::string p = "blk." + std::to_string(layer) + ".";
        opt.has_q_lora         = g.tensor((p + "attn_q_a.weight").c_str()) != nullptr;
        opt.has_attn_gate      = g.tensor((p + "attn_gate.weight").c_str()) != nullptr;
        opt.has_fused_kv_b     = g.tensor((p + "attn_kv_b.weight").c_str()) != nullptr;
        opt.has_shared_experts = g.tensor((p + "ffn_gate_shexp.weight").c_str()) != nullptr;
        opt.has_routed_norm    = g.tensor((p + "ffn_routed_norm.weight").c_str()) != nullptr;
    }

    std::printf("AllExpertsFfnWidth check: layer %d, %d experts, moe_ffn %d, top_k %d, "
                "tp_size %d (width %d/rank), shexp=%d\n",
                layer, cfg.n_experts, cfg.moe_ffn, cfg.top_k, tp_size,
                cfg.moe_ffn / tp_size, (int)opt.has_shared_experts);

    std::mt19937 rng(20260811);
    std::normal_distribution<float> N01(0.f, 1.f);
    std::vector<float> host_in(cfg.hidden);
    for (auto& v : host_in) v = N01(rng);

    std::vector<float> ref;
    if (!run_partial(g, cfg, opt, layer, n_layers, 1, 0, host_in, ref)) {
        std::printf("reference (tp=1) run failed\n"); return 1;
    }
    std::printf("reference (tp=1): %zu elems  (expert_latent=%d + hidden=%d)\n",
                ref.size(), cfg.expert_latent, cfg.hidden);

    std::vector<double> acc(ref.size(), 0.0);
    for (int r = 0; r < tp_size; ++r) {
        std::vector<float> part;
        if (!run_partial(g, cfg, opt, layer, n_layers, tp_size, r, host_in, part)) {
            std::printf("rank %d run failed\n", r); return 1;
        }
        if (part.size() != ref.size()) {
            std::printf("rank %d produced %zu elems, want %zu\n", r, part.size(), ref.size());
            return 1;
        }
        double n1 = 0;
        for (size_t i = 0; i < part.size(); ++i) { acc[i] += part[i]; n1 += std::fabs(part[i]); }
        std::printf("  rank %d: ffn band [%4d,%4d)  |partial|_1 = %.5g\n", r,
                    r * (cfg.moe_ffn / tp_size), (r + 1) * (cfg.moe_ffn / tp_size), n1);
    }

    std::vector<float> sum(ref.size());
    for (size_t i = 0; i < ref.size(); ++i) sum[i] = (float)acc[i];

    // The payload is TWO views: [0, expert_latent) is the routed-expert
    // accumulator, [expert_latent, expert_latent+hidden) is the shared expert.
    // Reported separately because they shard by the same rule but through
    // different code, and a 3x on only one of them is the signature that matters.
    std::printf("\nsum(width bands) vs unsharded:\n");
    report("ROUTED (latent)", ref, sum, 0, cfg.expert_latent);
    if (opt.has_shared_experts)
        report("SHARED EXPERT", ref, sum, cfg.expert_latent, (int)ref.size());
    report("WHOLE PAYLOAD", ref, sum, 0, (int)ref.size());

    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double)sum[i] - (double)ref[i];
        num += d * d; den += (double)ref[i] * (double)ref[i];
    }
    const double rl2 = std::sqrt(num / (den + 1e-30));
    const double thresh = 5e-3;
    std::printf("\n%s: FFN-width bands %s the unsharded MoE output (relL2 %.3e, bar %.0e)\n",
                rl2 < thresh ? "PASS" : "FAIL",
                rl2 < thresh ? "reproduce" : "DO NOT reproduce", rl2, thresh);
    return rl2 < thresh ? 0 : 1;
}
