// Stage 3 equivalence check: KDA state rollback after a partially-accepted verify pass.
//
// ---------------------------------------------------------------------------
// WHAT CORRECTNESS MEANS HERE, AND WHY IT IS NOT "MATCHES SEQUENTIAL DECODE"
// ---------------------------------------------------------------------------
//
// The obvious gate — roll back to j, continue, and demand the floats match what a pure
// sequential decode would have produced — CANNOT PASS ON THIS ENGINE, and failing it
// would not mean the rollback is broken.
//
// ncclAllReduce is deterministic for a fixed element count, but its reduce-scatter
// partitioning, and therefore the order the three ranks' partials are summed in, depends
// on that count. The count is width * n_tok. So a K-token verify pass and a sequential
// run legitimately differ in the low bits of every activation — and because the KDA state
// is updated from those same activations, they differ in the STATE too, not just the
// logits. A snapshot taken inside a K=5 batch is not the state a sequential run would
// have reached at that position, and no amount of correct copying will make it one.
//
// So what does greedy accept/reject actually need? Not losslessness. CAUSALITY:
//
//   (C1) SUFFIX INDEPENDENCE — logits row j and the state after token j must not depend
//        on the drafts at positions > j. They may depend on K.
//   (C2) SNAPSHOT CONSISTENCY — rollback(j) must restore the state THAT PASS computed
//        after token j: the same state that produced the row-j logits the caller used to
//        make its accept decision.
//
// Given C1 and C2, every emitted token is the argmax of logits computed from its true
// accepted prefix, so the emitted text is a valid greedy trajectory. It is simply not the
// same trajectory the non-batched path takes — in exactly the way, and for exactly the
// reason, this engine already diverges from the reference implementation. Restoring a
// hypothetical "pure sequential" state would in fact be WORSE than the snapshot: it would
// splice a state that never produced the logits the accept decision was made from.
//
// Tests A and C below are hard gates on C1 and C2 and are compared BITWISE, because under
// C1 an exact answer exists and a tolerance would throw the whole result away. Test D
// measures the gap to sequential decode and is reported, never asserted — it quantifies
// what NOT having a count-invariant collective costs, which is a Stage 4 planning input
// rather than a Stage 3 defect.

#include "sparkinfer/models/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_dist_forward.h"
#include "sparkinfer/models/kimi_k3_dist_rank.h"
#include "sparkinfer/tp/rank_collective.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Deterministic on every rank from data every rank already has. The check sends no
// control messages and derives nothing from logits (which only rank 0 holds), so all
// ranks issue an identical sequence of collectives and stay in lockstep without the
// K-token protocol change that belongs to Stage 4.
int cyc(const std::vector<int>& p, int i) {
    return p[(size_t)(i % (int)p.size())];
}

struct Cmp {
    double cos = 0, maxabs = 0;
    double exact_pct = 0;
    int top_a = 0, top_b = 0;
    bool bitwise = false;
};

Cmp compare_rows(const float* a, const float* b, int n) {
    Cmp c;
    double dot = 0, na = 0, nb = 0;
    size_t n_exact = 0;
    for (int i = 0; i < n; ++i) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
        const double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > c.maxabs) c.maxabs = d;
        if (std::memcmp(&a[i], &b[i], sizeof(float)) == 0) ++n_exact;
        if (a[i] > a[c.top_a]) c.top_a = i;
        if (b[i] > b[c.top_b]) c.top_b = i;
    }
    c.cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    c.exact_pct = 100.0 * (double)n_exact / (double)n;
    c.bitwise = (n_exact == (size_t)n);
    return c;
}

}  // namespace

// Returns 0 on pass, 1 on a hard-gate failure.
int run_spec_rollback_check(sparkinfer::KimiK3DistRank& dist,
                            sparkinfer::tp::dist::RankCollective& coll,
                            const std::vector<int>& prompt, int Kmax, int n_cont) {
    using namespace sparkinfer;
    const int rank_id = dist.plan.spec.rank;
    const int vocab = dist.cfg.vocab;
    const bool r0 = (rank_id == 0);
    std::string err;
    int failures = 0;

    auto replay = [&]() -> bool {
        kimi_k3_reset_state(dist.state);
        if (cudaStreamSynchronize(dist.stream) != cudaSuccess) return false;
        for (size_t i = 0; i < prompt.size(); ++i)
            if (!kimi_k3_dist_forward_token(dist, coll, prompt[i], nullptr, &err))
                return false;
        return true;
    };

    // The forced continuation. FIXED rather than greedy-fed, and that is what keeps the
    // arms comparable: feeding back an argmax would need rank 0 to tell the workers which
    // token it picked, which is the protocol change Stage 4 owns. With the inputs pinned,
    // any difference between two arms is a difference in STATE, which is the thing under
    // test. The ids are far from the prompt's own so the continuation exercises different
    // experts than the drafts did.
    auto cont_tok = [&](int s) { return 4096 + 101 * (s + 1); };

    // Run `n_cont` forced continuation steps and collect their logits on rank 0.
    auto continue_and_capture = [&](std::vector<float>& out) -> bool {
        out.assign(r0 ? (size_t)n_cont * vocab : 0, 0.0f);
        for (int s = 0; s < n_cont; ++s)
            if (!kimi_k3_dist_forward_token(dist, coll, cont_tok(s),
                                            r0 ? out.data() + (size_t)s * vocab : nullptr,
                                            &err))
                return false;
        return true;
    };

    if (r0) {
        std::printf("\n[spec-rollback] prompt=%zu tokens, Kmax=%d, continuation=%d steps,"
                    " vocab=%d\n", prompt.size(), Kmax, n_cont, vocab);
        std::fflush(stdout);
    }

    // -----------------------------------------------------------------------
    // TEST C — THE RING IS INERT
    // -----------------------------------------------------------------------
    // Enabling snapshots must not change the pass's own output by one bit. The save is a
    // read-only copy OUT of the live state, so any difference here means a destination
    // pointer is landing somewhere it should not — and a scribble into the live arena
    // would be indistinguishable from a fidelity gap in every other test.
    if (r0) {
        std::printf("[spec-rollback] --- C: ring inert (batch output unchanged by "
                    "snapshotting) ---\n");
        std::fflush(stdout);
    }
    for (int K = 2; K <= Kmax; ++K) {
        std::vector<int> ids((size_t)K);
        for (int b = 0; b < K; ++b) ids[(size_t)b] = cyc(prompt, b);
        std::vector<float> off(r0 ? (size_t)K * vocab : 0), on(r0 ? (size_t)K * vocab : 0);

        kimi_k3_dist_set_rollback(dist, false);
        if (!replay() ||
            !kimi_k3_dist_forward_batch(dist, coll, ids.data(), K,
                                        r0 ? off.data() : nullptr, &err)) {
            std::fprintf(stderr, "[spec-rollback] C ring-off K=%d failed: %s\n", K,
                         err.c_str());
            return 1;
        }
        kimi_k3_dist_set_rollback(dist, true);
        if (!replay() ||
            !kimi_k3_dist_forward_batch(dist, coll, ids.data(), K,
                                        r0 ? on.data() : nullptr, &err)) {
            std::fprintf(stderr, "[spec-rollback] C ring-on K=%d failed: %s\n", K,
                         err.c_str());
            return 1;
        }
        if (!r0) continue;
        const bool same = std::memcmp(off.data(), on.data(),
                                      (size_t)K * vocab * sizeof(float)) == 0;
        if (!same) ++failures;
        std::printf("[spec-rollback] C K=%d all %d rows bitwise-equal ring off vs on: "
                    "%s\n", K, K, same ? "YES" : "**NO — RING PERTURBS THE PASS**");
        std::fflush(stdout);
    }

    // -----------------------------------------------------------------------
    // TEST A — SUFFIX INDEPENDENCE OF THE ROLLED-BACK STATE  (the Stage 3 gate)
    // -----------------------------------------------------------------------
    // Two batches sharing tokens 0..j and disagreeing on everything after. Roll both back
    // to j, run the SAME forced continuation, and the logits must be BITWISE identical.
    //
    // This is one test but it closes four holes at once:
    //   * the snapshot carries nothing from the rejected drafts (C1 for the state);
    //   * the restore is complete — miss any one of the four buffers and the arms diverge,
    //     which is the conv-window-ahead-of-delta failure the design doc calls out;
    //   * the MLA KV rollback holds. The two arms wrote DIFFERENT values into the cache
    //     rows the rejected tokens occupied, so if the continuation's attention ever read
    //     past its own position the arms could not match;
    //   * the residual bank really is intra-token scratch, since a leak across the batch
    //       boundary would carry the fillers with it.
    if (r0) {
        std::printf("[spec-rollback] --- A: suffix independence after rollback "
                    "(HARD GATE, bitwise) ---\n");
        std::fflush(stdout);
    }
    kimi_k3_dist_set_rollback(dist, true);
    for (int K = 2; K <= Kmax; ++K) {
        for (int j = 0; j < K; ++j) {
            std::vector<int> ia((size_t)K), ib((size_t)K);
            for (int b = 0; b < K; ++b) ia[(size_t)b] = cyc(prompt, b);
            for (int b = 0; b <= j; ++b) ib[(size_t)b] = ia[(size_t)b];
            // Out-of-distribution and far apart: different experts, different KDA state,
            // the loudest perturbation available without changing the shape by one byte.
            for (int b = j + 1; b < K; ++b) ib[(size_t)b] = 1000 + 37 * b;

            std::vector<float> ca, cb;
            if (!replay() ||
                !kimi_k3_dist_forward_batch(dist, coll, ia.data(), K, nullptr, &err) ||
                !kimi_k3_dist_forward_rollback(dist, j, &err) ||
                !continue_and_capture(ca)) {
                std::fprintf(stderr, "[spec-rollback] A arm-A K=%d j=%d failed: %s\n", K, j,
                             err.c_str());
                return 1;
            }
            if (!replay() ||
                !kimi_k3_dist_forward_batch(dist, coll, ib.data(), K, nullptr, &err) ||
                !kimi_k3_dist_forward_rollback(dist, j, &err) ||
                !continue_and_capture(cb)) {
                std::fprintf(stderr, "[spec-rollback] A arm-B K=%d j=%d failed: %s\n", K, j,
                             err.c_str());
                return 1;
            }
            if (!r0) continue;

            // j == K-1 shares every token with arm B by construction, so it is not a
            // suffix test — it is the no-op arm of the rollback, and what it proves is
            // that "accept everything" leaves the live state untouched and usable.
            const char* kind = (j == K - 1) ? "no-op" : "suffix";
            bool all_bit = true;
            double worst_cos = 1.0;
            for (int s = 0; s < n_cont; ++s) {
                const Cmp c = compare_rows(ca.data() + (size_t)s * vocab,
                                           cb.data() + (size_t)s * vocab, vocab);
                if (!c.bitwise) all_bit = false;
                if (c.cos < worst_cos) worst_cos = c.cos;
            }
            if (!all_bit) ++failures;
            std::printf("[spec-rollback] A K=%d j=%d (%s) continuation bitwise-identical: "
                        "%s  (worst cos=%.9f)\n", K, j, kind,
                        all_bit ? "YES" : "**NO — SUFFIX LEAKED THROUGH ROLLBACK**",
                        worst_cos);
            std::fflush(stdout);
        }
    }

    // -----------------------------------------------------------------------
    // TEST D — DISTANCE TO PURE SEQUENTIAL DECODE  (MEASUREMENT, NOT A GATE)
    // -----------------------------------------------------------------------
    // Roll back to j and continue, versus consuming the same accepted tokens one at a time
    // and continuing. Under a count-invariant collective these would be identical; here
    // they are not, and the size of the gap is the number that decides whether Stage 4 can
    // still claim "token-identical to non-speculative greedy" or has to claim the weaker
    // and true "a valid greedy trajectory". Top-1 agreement is the figure that matters —
    // it is the rate at which the difference would actually change an emitted token.
    if (r0) {
        std::printf("[spec-rollback] --- D: rollback+continue vs pure sequential "
                    "(MEASUREMENT, not a gate) ---\n");
        std::fflush(stdout);
    }
    for (int K = 2; K <= Kmax; ++K) {
        for (int j = 0; j < K; ++j) {
            std::vector<int> ids((size_t)K);
            for (int b = 0; b < K; ++b) ids[(size_t)b] = cyc(prompt, b);

            std::vector<float> spec, seq;
            if (!replay() ||
                !kimi_k3_dist_forward_batch(dist, coll, ids.data(), K, nullptr, &err) ||
                !kimi_k3_dist_forward_rollback(dist, j, &err) ||
                !continue_and_capture(spec)) {
                std::fprintf(stderr, "[spec-rollback] D spec K=%d j=%d failed: %s\n", K, j,
                             err.c_str());
                return 1;
            }
            if (!replay()) {
                std::fprintf(stderr, "[spec-rollback] D replay failed: %s\n", err.c_str());
                return 1;
            }
            bool ok = true;
            for (int b = 0; b <= j && ok; ++b)
                ok = kimi_k3_dist_forward_token(dist, coll, ids[(size_t)b], nullptr, &err);
            if (!ok || !continue_and_capture(seq)) {
                std::fprintf(stderr, "[spec-rollback] D seq K=%d j=%d failed: %s\n", K, j,
                             err.c_str());
                return 1;
            }
            if (!r0) continue;
            for (int s = 0; s < n_cont; ++s) {
                const Cmp c = compare_rows(spec.data() + (size_t)s * vocab,
                                           seq.data() + (size_t)s * vocab, vocab);
                std::printf("[spec-rollback] D K=%d j=%d step=%d cos=%.9f top1 spec=%d "
                            "seq=%d %s exact=%.4f%% max|d|=%.3e\n",
                            K, j, s, c.cos, c.top_a, c.top_b,
                            c.top_a == c.top_b ? "AGREE" : "DIFFER", c.exact_pct,
                            c.maxabs);
            }
            std::fflush(stdout);
        }
    }

    kimi_k3_dist_set_rollback(dist, false);
    if (r0) {
        std::printf("[spec-rollback] ===== %s ===== (%d hard-gate failure%s)\n",
                    failures == 0 ? "STAGE 3 GATES PASS" : "STAGE 3 GATES FAIL",
                    failures, failures == 1 ? "" : "s");
        std::fflush(stdout);
    }
    return failures == 0 ? 0 : 1;
}
