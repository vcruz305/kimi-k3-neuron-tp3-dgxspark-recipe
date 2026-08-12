// Stage 4 host-side gates: the n-gram drafter, the accept/reject rule, the decode-loop
// algebra that ties them together, and the K-token rank protocol frame.
//
// All CPU, no CUDA, no NCCL, no model. Everything tested here is logic that decides which
// tokens get emitted and which state gets restored, and every one of its failure modes is
// silent — an off-by-one in the accepted-prefix length emits a token the model never
// verified, and a drafter that proposes the token AT the match instead of the one AFTER it
// shifts the whole proposal one position and merely lowers acceptance. Neither shows up as
// a crash on the fleet, so both are gated here where the answer is checkable by hand.
//
// The load-bearing test is D: a full speculative decode loop run against a deterministic
// oracle must emit EXACTLY the sequence a plain greedy loop emits from the same oracle.
// That is the property Stage 4 exists to deliver, stated in the one place it can be
// checked exactly.

#include "sparkinfer/models/kimi_k3_ngram_draft.h"
#include "sparkinfer/tp/rank_protocol.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace sparkinfer;
using namespace sparkinfer::tp::dist;

namespace {

int g_fail = 0;
int g_pass = 0;

void check(bool ok, const std::string& what) {
    if (ok) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("  FAIL: %s\n", what.c_str());
    }
}

std::string vec_str(const std::vector<int>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(v[i]);
    }
    return s + "]";
}

std::vector<int> draft_of(const std::vector<int>& ctx, int max_draft,
                          const K3NgramDraftConfig& cfg) {
    std::vector<int> out;
    kimi_k3_ngram_draft(ctx, max_draft, cfg, &out);
    return out;
}

// Independent restatement of the contract, written from the specification rather than
// from the implementation: longest n first, most recent occurrence, the tokens AFTER it.
// Deliberately the slow obvious version so a randomized comparison means something.
std::vector<int> reference_draft(const std::vector<int>& ctx, int max_draft,
                                 const K3NgramDraftConfig& cfg) {
    std::vector<int> out;
    if (max_draft <= 0 || ctx.empty()) return out;
    const int len = (int)ctx.size();
    for (int n = (cfg.n_max < len ? cfg.n_max : len); n >= cfg.n_min; --n) {
        int best = -1;
        for (int s = 0; s + n < len; ++s) {
            bool eq = true;
            for (int k = 0; k < n; ++k)
                if (ctx[s + k] != ctx[len - n + k]) { eq = false; break; }
            if (eq) best = s;  // keep scanning: the LAST hit is the most recent
        }
        if (best < 0) continue;
        for (int t = best + n; t < len && (int)out.size() < max_draft; ++t)
            out.push_back(ctx[t]);
        if (!out.empty()) return out;
    }
    return out;
}

// ---------------------------------------------------------------------------
// A — the drafter
// ---------------------------------------------------------------------------
void test_drafter() {
    std::printf("[A] n-gram drafter\n");
    // The SHIPPED default, gated here so a change to it is a deliberate act with a
    // measurement behind it rather than something that silently alters every caller.
    // n_min = 2 was measured on TP3: n_min = 1 cost freeform prose 12.5%.
    {
        K3NgramDraftConfig def;
        check(def.n_max == 3 && def.n_min == 2, "A0 shipped default is n=[2,3]");
    }
    // Every case below pins the config it means instead of inheriting the default.
    K3NgramDraftConfig c3;
    c3.n_max = 3;
    c3.n_min = 1;

    {   // Nothing repeats: no match at any n, so no draft and the caller pays nothing.
        const std::vector<int> ctx = {5, 6, 7, 8};
        check(draft_of(ctx, 4, c3).empty(), "A1 distinct ctx must not draft");
    }
    {   // "1,2,3" recurs; what followed it last time was 4.
        const std::vector<int> ctx = {1, 2, 3, 4, 9, 1, 2, 3};
        const std::vector<int> d = draft_of(ctx, 3, c3);
        check(d == std::vector<int>({4, 9, 1}), "A2 3-gram hit, got " + vec_str(d));
    }
    {   // No 3-gram, but "2,3" recurs. Must fall back rather than give up.
        const std::vector<int> ctx = {7, 2, 3, 4, 5, 8, 2, 3};
        const std::vector<int> d = draft_of(ctx, 2, c3);
        check(d == std::vector<int>({4, 5}), "A3 fallback to n=2, got " + vec_str(d));
    }
    {   // Only a single token recurs.
        const std::vector<int> ctx = {4, 9, 9, 9, 1, 2, 4};
        const std::vector<int> d = draft_of(ctx, 1, c3);
        check(d == std::vector<int>({9}), "A4 fallback to n=1, got " + vec_str(d));
    }
    {   // Two occurrences of "1,2": the RECENT one is followed by 7, the old one by 3.
        // Picking the old one is the classic wrong-direction scan and is invisible except
        // as lower acceptance, so it gets its own case.
        const std::vector<int> ctx = {1, 2, 3, 4, 1, 2, 7, 8, 1, 2};
        const std::vector<int> d = draft_of(ctx, 2, c3);
        check(d == std::vector<int>({7, 8}), "A5 most recent occurrence, got " + vec_str(d));
    }
    {   // max_draft truncates; the tail is simply not proposed.
        const std::vector<int> ctx = {1, 2, 3, 4, 5, 6, 1, 2, 3};
        const std::vector<int> d = draft_of(ctx, 2, c3);
        check(d == std::vector<int>({4, 5}), "A6 truncation to max_draft, got " + vec_str(d));
    }
    {   // The tail matching ITSELF is not a match: nothing follows it to propose.
        const std::vector<int> ctx = {1, 2, 3};
        check(draft_of(ctx, 3, c3).empty(), "A7 tail must not match itself");
    }
    {   // n_min excludes the short, low-quality matches.
        const std::vector<int> ctx = {4, 9, 9, 9, 1, 2, 4};
        K3NgramDraftConfig c2 = c3;
        c2.n_min = 2;
        check(draft_of(ctx, 2, c2).empty(), "A8 n_min=2 must reject a 1-gram match");
    }
    {   // Zero budget must not touch the output at all.
        const std::vector<int> ctx = {1, 2, 1, 2};
        check(draft_of(ctx, 0, c3).empty(), "A9 max_draft=0 draws no draft");
    }
    {   // ctx shorter than n_max must not read before the buffer.
        const std::vector<int> ctx = {1};
        check(draft_of(ctx, 4, c3).empty(), "A10 single-token ctx is safe");
    }

    // Randomized agreement with the independent reference, over a small alphabet so
    // matches at every n actually occur.
    unsigned s = 12345u;
    auto rnd = [&]() { s = s * 1664525u + 1013904223u; return (int)((s >> 16) & 0x7fff); };
    int mismatch = 0;
    for (int trial = 0; trial < 4000; ++trial) {
        const int len = 1 + rnd() % 40;
        const int alpha = 2 + rnd() % 5;
        std::vector<int> ctx((size_t)len);
        for (int i = 0; i < len; ++i) ctx[(size_t)i] = rnd() % alpha;
        K3NgramDraftConfig cfg;
        cfg.n_max = 1 + rnd() % 4;
        cfg.n_min = 1 + rnd() % cfg.n_max;
        const int md = rnd() % 6;
        if (draft_of(ctx, md, cfg) != reference_draft(ctx, md, cfg)) ++mismatch;
    }
    check(mismatch == 0,
          "A11 randomized vs reference: " + std::to_string(mismatch) + " mismatches");
    std::printf("     randomized cross-check: 4000 trials, %d mismatches\n", mismatch);
}

// ---------------------------------------------------------------------------
// B — the accept/reject rule
// ---------------------------------------------------------------------------
void test_accept() {
    std::printf("[B] accepted-prefix rule\n");
    // rows[i] is the argmax the verify pass produced at row i.
    auto run = [](const std::vector<int>& draft, const std::vector<int>& rows, int* bonus,
                  int* calls) {
        *calls = 0;
        return kimi_k3_spec_accept_prefix(draft.data(), (int)draft.size(),
                                          [&](int r) { ++(*calls); return rows[(size_t)r]; },
                                          bonus);
    };
    int bonus = -1, calls = 0;

    {   // Every draft right: j == d, and the bonus is the extra token the pass verified
        // past the end of the drafts — the source of the "+1" in the speedup.
        const int j = run({11, 12, 13}, {11, 12, 13, 14}, &bonus, &calls);
        check(j == 3 && bonus == 14, "B1 full accept j=3 bonus=14");
        check(calls == 4, "B1 evaluates exactly d+1 rows");
    }
    {   // First draft wrong: nothing is accepted but the verified token still lands, so
        // a total miss is never worse than not drafting.
        const int j = run({11, 12, 13}, {99, 12, 13, 14}, &bonus, &calls);
        check(j == 0 && bonus == 99, "B2 total reject j=0 bonus=99");
        check(calls == 1, "B2 must not evaluate rows past the rejection");
    }
    {   // Partial: rows after the mismatch describe a prefix that did not happen and are
        // never consulted, which is also why they cost nothing.
        const int j = run({11, 12, 13}, {11, 77, 13, 14}, &bonus, &calls);
        check(j == 1 && bonus == 77, "B3 partial accept j=1 bonus=77");
        check(calls == 2, "B3 stops at the mismatch");
    }
    {   // No draft at all still emits the verified token.
        const int j = run({}, {42}, &bonus, &calls);
        check(j == 0 && bonus == 42 && calls == 1, "B4 d=0 emits the verified token");
    }
    {   // A later row coincidentally matching does NOT resurrect the run.
        const int j = run({11, 12, 13}, {11, 77, 13, 14}, &bonus, &calls);
        check(j == 1, "B5 acceptance is a prefix, not a set");
    }
}

// ---------------------------------------------------------------------------
// C — the K-token rank protocol frame
// ---------------------------------------------------------------------------
RankSpec make_spec(int rank) {
    RankSpec s;
    s.session_id = 7;
    s.world_size = 3;
    s.rank = rank;
    s.local_device = 0;
    s.n_experts = 896;
    s.moe_ffn = 1536;
    s.expert_block_elems = 256;
    s.vocab = 163840;
    s.model_digest[0] = 0xA5;
    s.plan_digest[0] = 0x5A;
    return s;
}

void test_protocol() {
    std::printf("[C] K-token rank protocol\n");
    {   // Round trip: ids and the deferred rollback index must survive the wire exactly.
        Message m;
        m.type = MessageType::Token;
        m.session_id = 7;
        m.sequence = 4;
        m.rank = 1;
        m.world_size = 3;
        m.vocab = 163840;
        m.phase = Phase::Step;
        m.token_id = 101;
        m.rollback_j = 2;
        m.token_ids = {101, 202, 303, 404};
        std::vector<std::uint8_t> frame;
        std::string e;
        check(encode_message(m, &frame, &e), "C1 encode K-token: " + e);
        Message got;
        check(decode_message(frame.data(), frame.size(), &got, &e), "C1 decode: " + e);
        check(got.token_ids == m.token_ids, "C1 ids round-trip " + vec_str(got.token_ids));
        check(got.rollback_j == 2, "C1 rollback_j round-trip");
        check(got.token_id == 101, "C1 token_id round-trip");
    }
    {   // A v1-shaped frame (no ids) still round-trips and reads as a single-token step.
        Message m;
        m.type = MessageType::Token;
        m.session_id = 7;
        m.vocab = 163840;
        m.phase = Phase::Step;
        m.token_id = 55;
        std::vector<std::uint8_t> frame;
        std::string e;
        check(encode_message(m, &frame, &e), "C2 encode scalar Token: " + e);
        check(frame.size() == kWireHeaderBytes + kWireFixedPayloadBytes,
              "C2 empty batch is exactly the fixed payload");
        Message got;
        check(decode_message(frame.data(), frame.size(), &got, &e), "C2 decode: " + e);
        check(got.token_ids.empty() && got.token_id == 55 && got.rollback_j == -1,
              "C2 scalar Token stays scalar");
    }
    {   // The leading id and the scalar must agree, or a reader that only understands the
        // scalar would run a different token than the batch says.
        Message m;
        m.type = MessageType::Token;
        m.session_id = 7;
        m.vocab = 163840;
        m.phase = Phase::Step;
        m.token_id = 5;
        m.token_ids = {6, 7};
        std::vector<std::uint8_t> frame;
        check(!encode_message(m, &frame, nullptr), "C3 encode rejects id/scalar mismatch");
    }
    {   // Bound on the batch width.
        Message m;
        m.type = MessageType::Token;
        m.session_id = 7;
        m.vocab = 163840;
        m.phase = Phase::Step;
        m.token_id = 1;
        m.token_ids.assign((size_t)kMaxStepTokens + 1, 1);
        std::vector<std::uint8_t> frame;
        check(!encode_message(m, &frame, nullptr), "C4 encode rejects oversize batch");
    }
    {   // Corruption must not decode. A truncated tail is the shape a partial read takes.
        Message m;
        m.type = MessageType::Token;
        m.session_id = 7;
        m.vocab = 163840;
        m.phase = Phase::Step;
        m.token_id = 1;
        m.token_ids = {1, 2, 3};
        std::vector<std::uint8_t> frame;
        check(encode_message(m, &frame, nullptr), "C5 encode ok");
        std::vector<std::uint8_t> cut(frame.begin(), frame.end() - 4);
        Message got;
        check(!decode_message(cut.data(), cut.size(), &got, nullptr),
              "C5 truncated frame rejected");
        std::vector<std::uint8_t> grown = frame;
        grown.push_back(0);
        check(!decode_message(grown.data(), grown.size(), &got, nullptr),
              "C5 extended frame rejected");
        std::vector<std::uint8_t> flip = frame;
        flip[flip.size() - 1] ^= 0xFF;
        check(!decode_message(flip.data(), flip.size(), &got, nullptr),
              "C5 corrupt payload rejected by CRC");
    }
    {   // Coordinator -> ranks, end to end, including the sequence gate.
        Coordinator coord(make_spec(0));
        std::vector<Message> out;
        std::string e;
        // Drive bootstrap to Ready the same way the transport does.
        for (int r = 0; r < 3; ++r) {
            Message hello;
            RankSession s(make_spec(r));
            check(s.hello(&hello, &e), "C6 hello");
            AuthenticatedPeer p{r, (std::uint64_t)(r + 1)};
            check(coord.accept(p, hello, &out, &e), "C6 accept hello: " + e);
        }
        NcclUniqueId id{};
        id[0] = 9;
        out.clear();
        check(coord.install_nccl_unique_id(id, &out, &e), "C6 install id: " + e);
        std::vector<RankSession> ranks;
        for (int r = 0; r < 3; ++r) ranks.push_back(RankSession(make_spec(r)));
        for (int r = 0; r < 3; ++r) {
            Message hello;
            check(ranks[(size_t)r].hello(&hello, &e), "C6 rank hello");
        }
        for (const Message& m : out) {
            check(ranks[(size_t)m.rank].accept_control(m, nullptr, &e), "C6 nccl id: " + e);
        }
        out.clear();
        for (int r = 0; r < 3; ++r) {
            Message rdy;
            check(ranks[(size_t)r].report_nccl_ready(true, 0, "", &rdy, &e), "C6 ready");
            AuthenticatedPeer p{r, (std::uint64_t)(r + 1)};
            check(coord.accept(p, rdy, &out, &e), "C6 accept ready: " + e);
        }
        for (const Message& m : out) {
            check(ranks[(size_t)m.rank].accept_control(m, nullptr, &e), "C6 beginload: " + e);
        }
        out.clear();
        for (int r = 0; r < 3; ++r) {
            Message rdy;
            check(ranks[(size_t)r].report_load_ready(true, 0, "", &rdy, &e), "C6 loadready");
            AuthenticatedPeer p{r, (std::uint64_t)(r + 1)};
            check(coord.accept(p, rdy, &out, &e), "C6 accept loadready: " + e);
        }
        check(coord.state() == CoordinatorState::Ready, "C6 coordinator Ready");

        // The actual thing under test: one batch, one Token per rank, one sequence.
        const int ids[4] = {10, 20, 30, 40};
        out.clear();
        check(coord.begin_step_multi(ids, 4, 2, &out, &e), "C6 begin_step_multi: " + e);
        check(out.size() == 3, "C6 one Token per rank, not one per token");
        for (const Message& m : out) {
            std::vector<std::uint8_t> frame;
            Message wire_m;
            check(encode_message(m, &frame, &e) &&
                      decode_message(frame.data(), frame.size(), &wire_m, &e),
                  "C6 wire: " + e);
            int tok = -1, rb = -1;
            std::vector<int> got_ids;
            check(ranks[(size_t)wire_m.rank].accept_control(wire_m, &tok, &e, &got_ids, &rb),
                  "C6 accept Token: " + e);
            check(tok == 10 && rb == 2 &&
                      got_ids == std::vector<int>({10, 20, 30, 40}),
                  "C6 rank sees the whole batch");
        }
        // A second batch before StepDone must still be refused — widening the frame must
        // not have widened what the sequence gate lets through.
        std::vector<Message> out2;
        check(!coord.begin_step_multi(ids, 4, -1, &out2, nullptr),
              "C6 sequence gate still closed before StepDone");
    }
    {   // The desync guard: a batch delivered to a caller that cannot receive it must be
        // refused, not silently reduced to its first token.
        RankSession s(make_spec(1));
        Message hello;
        std::string e;
        s.hello(&hello, &e);
        Message m;
        m.type = MessageType::Token;
        m.session_id = 7;
        m.rank = 1;
        m.world_size = 3;
        m.local_device = 0;
        m.n_experts = 896;
        m.moe_ffn = 1536;
        m.expert_block_elems = 256;
        m.vocab = 163840;
        m.model_digest[0] = 0xA5;
        m.plan_digest[0] = 0x5A;
        m.phase = Phase::Step;
        m.token_id = 3;
        m.token_ids = {3, 4};
        int tok = -1;
        check(!s.accept_control(m, &tok, nullptr, nullptr, nullptr),
              "C7 multi-token Token with no batch sink is rejected");
    }
}

// ---------------------------------------------------------------------------
// D — the whole loop, against a deterministic oracle
// ---------------------------------------------------------------------------
//
// The engine guarantees causality (C1 in the Stage 3 receipt): verify row i depends only
// on batch positions 0..i. Model that exactly with an oracle over the accepted prefix,
// then assert the speculative loop emits the same tokens as plain greedy. Any error in
// ctx maintenance, in the emitted count, or in the rollback index shows up here as a
// divergent sequence.
void test_loop_equivalence() {
    std::printf("[D] speculative loop == greedy, against a fixed oracle\n");

    // Two alphabets, and the pair is the point. A LAST-3 oracle over a WIDE alphabet
    // rarely repeats a 3-gram, so almost every draft comes from a short uninformative
    // match and is rejected — that exercises j == 0. Over a NARROW alphabet 3-grams recur
    // constantly and, because the oracle is a function of exactly those 3 tokens, a
    // 3-gram match predicts perfectly and whole chains are accepted — that exercises
    // j == d. Running only the wide case would let this test pass while never once
    // taking the accept branch, which is the branch that can emit an unverified token.
    const int alphabets[2] = {4, 17};

    long long saw_full = 0, saw_partial = 0, saw_none = 0, saw_accepted = 0;

    for (int ai = 0; ai < 2; ++ai) {
      const unsigned alpha = (unsigned)alphabets[ai];
      auto oracle = [alpha](const std::vector<int>& consumed) {
        unsigned h = 2166136261u;
        const size_t n = consumed.size();
        for (size_t i = (n >= 3 ? n - 3 : 0); i < n; ++i) {
            h ^= (unsigned)consumed[i];
            h *= 16777619u;
        }
        return (int)(h % alpha);
      };

      for (int K = 2; K <= 8; ++K) {
        for (int nmin = 1; nmin <= 3; ++nmin) {
            for (int seed = 0; seed < 6; ++seed) {
                std::vector<int> prompt;
                unsigned s = 1000u + (unsigned)seed * 77u;
                const int plen = 3 + seed;
                for (int i = 0; i < plen; ++i) {
                    s = s * 1664525u + 1013904223u;
                    prompt.push_back((int)((s >> 16) % 17u));
                }
                const int n_predict = 40;

                // --- plain greedy reference ---
                std::vector<int> ref, consumed = prompt;
                {
                    int next = oracle(consumed);
                    ref.push_back(next);
                    while ((int)ref.size() < n_predict) {
                        consumed.push_back(next);
                        next = oracle(consumed);
                        ref.push_back(next);
                    }
                }

                // --- speculative loop, same structure as the decode loop ---
                K3NgramDraftConfig cfg;
                cfg.n_max = 3;
                cfg.n_min = nmin;
                std::vector<int> got, ctx = prompt, spec_consumed = prompt;
                long long steps = 0, sum_d = 0, sum_j = 0;
                int next = oracle(spec_consumed);
                got.push_back(next);
                ctx.push_back(next);
                while ((int)got.size() < n_predict) {
                    const int budget = n_predict - (int)got.size();
                    int max_draft = K - 1;
                    if (max_draft > budget - 1) max_draft = budget - 1;
                    std::vector<int> draft;
                    const int d = max_draft > 0
                        ? kimi_k3_ngram_draft(ctx, max_draft, cfg, &draft) : 0;

                    // The verify pass: row i is the oracle over the accepted prefix
                    // through batch position i. This IS the causality guarantee.
                    std::vector<int> rows;
                    {
                        std::vector<int> walk = spec_consumed;
                        walk.push_back(next);
                        rows.push_back(oracle(walk));
                        for (int i = 0; i < d; ++i) {
                            walk.push_back(draft[(size_t)i]);
                            rows.push_back(oracle(walk));
                        }
                    }
                    int bonus = -1;
                    const int j = kimi_k3_spec_accept_prefix(
                        draft.data(), d, [&](int r) { return rows[(size_t)r]; }, &bonus);

                    spec_consumed.push_back(next);
                    for (int i = 0; i < j; ++i) {
                        spec_consumed.push_back(draft[(size_t)i]);
                        ctx.push_back(draft[(size_t)i]);
                        got.push_back(draft[(size_t)i]);
                    }
                    got.push_back(bonus);
                    next = bonus;
                    ctx.push_back(next);

                    ++steps;
                    sum_d += d;
                    sum_j += j;
                    if (d > 0) {
                        if (j == d) ++saw_full;
                        else if (j > 0) ++saw_partial;
                        else ++saw_none;
                    }
                    saw_accepted += j;
                    if (steps > 4 * n_predict) break;  // livelock guard
                }

                const std::string tag = "D a=" + std::to_string(alpha) + " K=" +
                                        std::to_string(K) + " nmin=" +
                                        std::to_string(nmin) + " seed=" +
                                        std::to_string(seed);
                check(got.size() == ref.size(), tag + " emitted count " +
                                                    std::to_string(got.size()) + " vs " +
                                                    std::to_string(ref.size()));
                check(got == ref, tag + " sequence differs from greedy");
                // ctx must equal everything consumed plus the one pending token, or the
                // drafter is being fed a history the model never saw.
                check(ctx.size() == spec_consumed.size() + 1, tag + " ctx/consumed skew");
                check(steps <= (long long)n_predict, tag + " speculation must not add steps");
                if (K == 8 && nmin == 1 && seed == 0) {
                    std::printf("     alphabet=%u K=8 nmin=1: %lld steps for %d tokens, "
                                "drafted=%lld accepted=%lld (%.1f%%)\n",
                                alpha, steps, n_predict, sum_d, sum_j,
                                sum_d ? 100.0 * (double)sum_j / (double)sum_d : 0.0);
                }
            }
        }
      }
    }

    // Equivalence alone is satisfiable by a loop that accepts nothing and falls back to
    // one token per step every time. These pin that all three arms actually ran, so a
    // regression that silently disables acceptance fails here instead of quietly costing
    // the entire speedup while every correctness gate stays green.
    std::printf("     arms exercised: full=%lld partial=%lld none=%lld accepted=%lld\n",
                saw_full, saw_partial, saw_none, saw_accepted);
    check(saw_accepted > 0, "D accept path never taken");
    check(saw_full > 0, "D full-accept arm (rollback no-op) never taken");
    check(saw_partial > 0, "D partial-accept arm (real rollback) never taken");
    check(saw_none > 0, "D total-reject arm never taken");
}

}  // namespace

int main() {
    std::printf("kimi_k3_spec_draft_check — Stage 4 host-side gates\n\n");
    test_drafter();
    test_accept();
    test_protocol();
    test_loop_equivalence();
    std::printf("\n%s  passed=%d failed=%d\n", g_fail == 0 ? "ALL PASS" : "**FAILURES**",
                g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
