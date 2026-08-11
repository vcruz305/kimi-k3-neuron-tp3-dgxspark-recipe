// KDA head parallelism: tp_size disjoint head bands, summed, must equal the
// unsharded KDA attention output.
//
//   kimi_k3_tp_kda_check <first-shard.gguf> [tp_size] [layer]
//
// THE LAST UNVALIDATED SHARDING AXIS, AND IT COVERS 69 OF 93 LAYERS. MLA head
// sharding was proven correct on the live fleet (8e-09 against an unsharded run);
// AllExpertsFfnWidth was proven correct by kimi_k3_tp_width_check; expert-band
// parallelism has kimi_k3_tp_moe_check. KDA head sharding had nothing, and on the
// fleet it cannot be A/B'd by disabling: replicating 69 layers of KDA attention at
// full width OOMs a 121 GB GB10 during load (blk.84.attn_k.weight).
//
// But that OOM is an artifact of loading all 93 layers. THIS test loads ONE, so
// the same comparison costs nothing -- rank r's weights are loaded at its head
// band, its Attn-phase partial is read back to the host, and the host sums them,
// which is exactly what the all-reduce does.
//
// WHAT IS BEING TESTED, specifically. A KDA head owns a private recurrent state
// (conv window + delta matrix) and never reads another head's, so the branch is
// head-parallel: attn_q/k/v and ssm_g/ssm_f_b row-shard by qkv, attn_output
// col-shards, and the state divides with them. On top of that, four small
// per-channel tensors are REPLICATE-AND-OFFSET -- ssm_a, ssm_dt.bias,
// ssm_conv1d_{q,k,v} and ssm_beta stay full width and the forward is trusted to
// index them at this rank's band. That trust is the fragile part: an offset that
// is silently dropped is exactly right on rank 0 (offset 0) and wrong on every
// other rank, and because the all-reduce forces agreement afterwards, no
// cross-rank comparison can see it. This test can.

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

// Run the Attn phase for `layer` and return this rank's attention partial.
bool run_attn_partial(const GGUF& g, const KimiK3Config& cfg, const K3PlanOptions& opt,
                      int layer, int n_layers, int tp_size, int rank,
                      const std::vector<float>& host_in, std::vector<float>& out_partial) {
    KimiK3Weights w;
    // MoE deliberately left at the tp=1 identity in every arm, so the only thing
    // that varies between reference and bands is the KDA head axis.
    w.policy = (tp_size > 1) ? KimiK3Weights::ShardPolicy::ExpertsAndKda
                             : KimiK3Weights::ShardPolicy::ExpertsOnly;
    w.shard.tp_size = tp_size;
    w.shard.rank = rank;
    w.shard.hidden = cfg.hidden;
    w.shard.n_experts_total = cfg.n_experts;
    w.shard.n_experts = cfg.n_experts;
    w.shard.expert_band = tp::Band{0, cfg.n_experts};
    w.shard.experts_sharded = false;
    w.shard.moe_ffn_total = cfg.moe_ffn;
    w.shard.moe_ffn = cfg.moe_ffn;
    w.shard.moe_shard_mode = tp::MoeShardMode::None;

    if (!kimi_k3_load_weights(g, cfg, opt, w, 0, n_layers - 1)) return false;
    std::printf("    [rank %d] kda heads=%d qkv_band=[%d,%d) head_band=[%d,%d)\n",
                rank, w.kda.n_heads, w.kda.qkv_band.offset, w.kda.qkv_band.end(),
                w.kda.head_band.offset, w.kda.head_band.end());

    KimiK3RuntimeState st;
    if (!kimi_k3_alloc_state(cfg, 8, st, 0, n_layers - 1, &w.kda)) return false;
    KimiK3Forward fwd;
    fwd.cfg = &cfg; fwd.w = &w; fwd.state = &st; fwd.opt = opt; fwd.stream = nullptr;
    if (!kimi_k3_forward_alloc_scratch(cfg, fwd)) return false;

    const int H = cfg.hidden;
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, (size_t)H * sizeof(float));
    cudaMalloc(&d_out, (size_t)H * sizeof(float));
    cudaMemcpy(d_in, host_in.data(), (size_t)H * sizeof(float), cudaMemcpyHostToDevice);

    kimi_k3_reset_state(st);
    bool ok = kimi_k3_forward_layer_phase(fwd, layer, K3LayerPhase::Attn, d_in, d_out);
    cudaDeviceSynchronize();

    int n = 0;
    float* partial = kimi_k3_partial_buffer(fwd, layer, K3LayerPhase::Attn, &n);
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
        for (int i = 0; i < n_avail; ++i) if (cfg.is_kda_layer(i)) { layer = i; break; }
    }
    if (layer < 0 || layer >= n_avail || !cfg.is_kda_layer(layer)) {
        std::printf("no complete KDA layer available (layer=%d, n_avail=%d)\n", layer, n_avail);
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

    std::printf("KDA head-shard check: layer %d, n_q_heads %d, kda_head_dim %d, tp_size %d "
                "(%d heads/rank)\n",
                layer, cfg.n_q_heads, cfg.kda_head_dim, tp_size, cfg.n_q_heads / tp_size);
    if (cfg.n_q_heads % tp_size != 0) {
        std::printf("  %d heads do not divide %d ranks\n", cfg.n_q_heads, tp_size);
        return 1;
    }

    std::mt19937 rng(20260811);
    std::normal_distribution<float> N01(0.f, 1.f);
    std::vector<float> host_in(cfg.hidden);
    for (auto& v : host_in) v = N01(rng);

    std::printf("  reference (tp=1):\n");
    std::vector<float> ref;
    if (!run_attn_partial(g, cfg, opt, layer, n_layers, 1, 0, host_in, ref)) {
        std::printf("reference (tp=1) run failed\n"); return 1;
    }

    std::printf("  head bands:\n");
    std::vector<double> acc(ref.size(), 0.0);
    for (int r = 0; r < tp_size; ++r) {
        std::vector<float> part;
        if (!run_attn_partial(g, cfg, opt, layer, n_layers, tp_size, r, host_in, part)) {
            std::printf("rank %d run failed\n", r); return 1;
        }
        if (part.size() != ref.size()) {
            std::printf("rank %d produced %zu elems, want %zu\n", r, part.size(), ref.size());
            return 1;
        }
        double n1 = 0;
        for (size_t i = 0; i < part.size(); ++i) { acc[i] += part[i]; n1 += std::fabs(part[i]); }
        std::printf("    [rank %d] |partial|_1 = %.6g\n", r, n1);
    }

    double num = 0, den = 0, worst = 0, sref = 0, sgot = 0;
    int worst_i = -1;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double s = acc[i];
        const double d = s - (double)ref[i];
        num += d * d; den += (double)ref[i] * (double)ref[i];
        sref += std::fabs((double)ref[i]); sgot += std::fabs(s);
        const double rel = std::fabs(d) / (std::fabs((double)ref[i]) + 1e-6);
        if (rel > worst) { worst = rel; worst_i = (int)i; }
    }
    const double rl2 = std::sqrt(num / (den + 1e-30));
    std::printf("\nsum(head bands) vs unsharded attention:\n");
    std::printf("  relL2 = %.3e   |ref|_1 = %.6g  |sum|_1 = %.6g  ratio = %.5f\n",
                rl2, sref, sgot, sref > 0 ? sgot / sref : 0.0);
    if (worst_i >= 0)
        std::printf("  worst_rel = %.3e @ %d (ref=%.6g sum=%.6g)\n",
                    worst, worst_i, ref[worst_i], acc[worst_i]);

    const double thresh = 5e-3;
    std::printf("\n%s: KDA head bands %s the unsharded attention output (relL2 %.3e, bar %.0e)\n",
                rl2 < thresh ? "PASS" : "FAIL",
                rl2 < thresh ? "reproduce" : "DO NOT reproduce", rl2, thresh);
    return rl2 < thresh ? 0 : 1;
}
