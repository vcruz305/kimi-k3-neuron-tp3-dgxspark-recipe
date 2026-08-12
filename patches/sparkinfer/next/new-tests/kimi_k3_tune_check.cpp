// Gates for the two tuning changes that sit on top of Stage 4:
//
//   E. Match-confidence gating and adaptive draft width in the n-gram drafter.
//   F. Uneven head-band arithmetic for the distributed path.
//
// All CPU, no CUDA, no NCCL, no model.
//
// E is not a correctness gate in the sense D was — the drafter cannot change which tokens
// are emitted, only how many verifies are wasted proposing them. What E must prove is that
// the DEFAULT config still behaves exactly as Stage 4 did, because the shipped default is
// the thing every prior measurement was taken against, and that the gate's predicate is
// the one the measurements were modelled on.
//
// F IS a correctness gate, and a silent one. The three bands are summed into a zero-filled
// buffer, so a gap leaves a row of logits at zero (a token that can never be argmax) and an
// overlap doubles a row (a token that wins when it should not). Neither crashes; both would
// read as a quality regression somewhere far away. The tiling property is therefore checked
// exhaustively rather than at the one size TP3 happens to use.

#include "sparkinfer/models/k3_head_band.h"
#include "sparkinfer/models/kimi_k3_ngram_draft.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace sparkinfer;

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
                          const K3NgramDraftConfig& cfg, K3NgramDraftInfo* info = nullptr) {
    std::vector<int> out;
    kimi_k3_ngram_draft(ctx, max_draft, cfg, &out, info);
    return out;
}

// ---------------------------------------------------------------------------
// E. drafter confidence gating
// ---------------------------------------------------------------------------
void test_gating() {
    std::printf("E. match-confidence gating and adaptive width\n");

    K3NgramDraftConfig def;   // shipped default
    check(def.min_occurrences == 1 && !def.require_agreement && def.weak_max_draft == 0,
          "E0 default config is ungated (Stage 4 behaviour)");

    // A tail that occurs exactly once earlier. Ungated this drafts; gated it must not,
    // because a single occurrence carries no evidence the continuation repeats — the
    // bucket that produced 0 accepted tokens out of 12 drafted across code and prose.
    {
        const std::vector<int> ctx = {1, 2, 7, 8, 9, 5, 6, 1, 2};
        K3NgramDraftConfig c = def; c.n_min = 2; c.n_max = 2;
        check(draft_of(ctx, 3, c) == std::vector<int>({7, 8, 9}),
              "E1 ungated drafts a single-occurrence match");
        K3NgramDraftInfo info;
        c.min_occurrences = 2;
        check(draft_of(ctx, 3, c, &info).empty(),
              "E2 min_occurrences=2 declines a single-occurrence match");
        check(info.n == 0, "E2 info reports nothing proposed");
    }

    // Two occurrences continuing the same way: the bucket that carried every accepted
    // token. Must survive both gates, and must still draft from the MOST RECENT one —
    // recurrence decides whether, recency still decides what.
    {
        const std::vector<int> ctx = {4, 4, 9, 1, 2, 3, 4, 4, 9, 7, 7, 4, 4};
        K3NgramDraftConfig c = def; c.n_min = 2; c.n_max = 2;
        c.min_occurrences = 2; c.require_agreement = true;
        K3NgramDraftInfo info;
        const std::vector<int> d = draft_of(ctx, 2, c, &info);
        check(d == std::vector<int>({9, 7}), "E3 recurring+agreeing match drafts, got " + vec_str(d));
        check(info.occurrences == 2 && info.agreed && !info.weak,
              "E3 info: occ=2 agreed weak=0, got occ=" + std::to_string(info.occurrences));
    }

    // Two occurrences continuing DIFFERENTLY. Counting alone accepts it; agreement
    // rejects it. This is the split that separated code's 0.85 bucket from its 0.00 one,
    // so the two knobs are checked to be genuinely independent and not the same test.
    {
        const std::vector<int> ctx = {5, 5, 1, 0, 0, 5, 5, 2, 0, 0, 5, 5};
        K3NgramDraftConfig c = def; c.n_min = 2; c.n_max = 2;
        c.min_occurrences = 2;
        K3NgramDraftInfo info;
        const std::vector<int> d = draft_of(ctx, 1, c, &info);
        check(d == std::vector<int>({2}), "E4 count-only gate admits a disagreeing match, got " + vec_str(d));
        check(info.occurrences == 2 && !info.agreed, "E4 info records the disagreement");
        c.require_agreement = true;
        check(draft_of(ctx, 1, c).empty(), "E5 agreement gate rejects the same match");
    }

    // A declined long tail must fall through to a shorter one that does clear the bar,
    // not abandon the step. Best-first still holds; the gate only removes candidates.
    {
        //          n=3 tail (8,1,2) occurs exactly once -> declined by the count bar;
        //          n=2 tail (1,2) occurs twice, both continuing with 3 -> admitted.
        const std::vector<int> ctx = {8, 1, 2, 3, 9, 9, 1, 2, 3, 4, 8, 1, 2};
        K3NgramDraftConfig c = def; c.n_min = 2; c.n_max = 3;
        c.min_occurrences = 2; c.require_agreement = true;
        K3NgramDraftInfo info;
        const std::vector<int> d = draft_of(ctx, 1, c, &info);
        check(d == std::vector<int>({3}) && info.n == 2,
              "E6 declined n=3 falls through to an admitted n=2, got " + vec_str(d) +
                  " n=" + std::to_string(info.n));
    }

    // Adaptive width: a weak match is not declined, it is capped. The point is that a
    // rejected 1-token draft costs one marginal position instead of three.
    {
        const std::vector<int> ctx = {1, 2, 7, 8, 9, 5, 6, 1, 2};
        K3NgramDraftConfig c = def; c.n_min = 2; c.n_max = 2;
        c.min_occurrences = 2; c.require_agreement = true; c.weak_max_draft = 1;
        K3NgramDraftInfo info;
        const std::vector<int> d = draft_of(ctx, 3, c, &info);
        check(d == std::vector<int>({7}), "E7 weak match is width-capped, not declined, got " + vec_str(d));
        check(info.weak, "E7 info flags the match as weak");
    }

    // A strong match under the same config must still get the full width, or the cap is
    // not adaptive, it is just a smaller K.
    {
        const std::vector<int> ctx = {4, 4, 9, 7, 1, 4, 4, 9, 7, 2, 4, 4};
        K3NgramDraftConfig c = def; c.n_min = 2; c.n_max = 2;
        c.min_occurrences = 2; c.require_agreement = true; c.weak_max_draft = 1;
        K3NgramDraftInfo info;
        const std::vector<int> d = draft_of(ctx, 2, c, &info);
        check(d == std::vector<int>({9, 7}) && !info.weak,
              "E8 strong match keeps full width under a weak cap, got " + vec_str(d));
    }

    // The gated scan walks the whole history where the ungated one returns early. Both
    // must pick the same occurrence when the gate admits everything, or the two arms are
    // not the same drafter and every Stage 4 number stops applying to the default.
    {
        const std::vector<int> ctx = {3, 1, 5, 0, 3, 1, 6, 0, 3, 1};
        K3NgramDraftConfig a = def; a.n_min = 2; a.n_max = 2;
        K3NgramDraftConfig b = a; b.min_occurrences = 1; b.weak_max_draft = 1;
        // b takes the gated arm (weak_max_draft > 0) but admits every match, so its
        // "strong" verdict is the same and the width cap never binds.
        check(draft_of(ctx, 1, a) == draft_of(ctx, 1, b),
              "E9 gated and ungated arms choose the same occurrence when nothing is gated");
    }
}

// ---------------------------------------------------------------------------
// F. uneven head band
// ---------------------------------------------------------------------------
void test_head_band() {
    std::printf("F. uneven head-band arithmetic (distributed path)\n");

    // The even helper is what the dist path could not use, and the reason this variant
    // exists. Stated as a test so the motivation cannot quietly stop being true.
    {
        K3HeadBand hb;
        check(!k3_head_band(163840, 163840ull * 14336, 3, 0, &hb),
              "F0 even helper declines TP3 (163840 %% 3 != 0)");
    }

    // Exhaustive tiling. For every band the union must be exactly [0, vocab) with no gap
    // and no overlap, and byte_off must equal offset * row_bytes.
    {
        const int vocabs[] = {163840, 32000, 7, 8, 9, 100, 101, 102, 1000003};
        const int tps[] = {2, 3, 4, 5, 7, 8};
        bool tiling_ok = true, bytes_ok = true, spread_ok = true;
        int cases = 0;
        for (int vocab : vocabs) {
            for (int tp : tps) {
                if (vocab < tp) continue;
                const size_t row_bytes = 14336;
                const size_t wb = (size_t)vocab * row_bytes;
                int next = 0, min_rows = vocab, max_rows = 0;
                for (int r = 0; r < tp; ++r) {
                    K3HeadBand hb;
                    if (!k3_head_band_uneven(vocab, wb, tp, r, &hb)) { tiling_ok = false; break; }
                    if (hb.offset != next) tiling_ok = false;
                    if (hb.byte_off != (size_t)hb.offset * row_bytes) bytes_ok = false;
                    if (hb.rows < min_rows) min_rows = hb.rows;
                    if (hb.rows > max_rows) max_rows = hb.rows;
                    next = hb.offset + hb.rows;
                }
                if (next != vocab) tiling_ok = false;
                // Bands may differ by at most one row, or the load is not balanced and
                // the slowest rank sets the step time.
                if (max_rows - min_rows > 1) spread_ok = false;
                ++cases;
            }
        }
        check(tiling_ok, "F1 bands tile [0,vocab) exactly, no gap or overlap (" +
                             std::to_string(cases) + " cases)");
        check(bytes_ok, "F2 byte_off == offset * row_bytes in every band");
        check(spread_ok, "F3 band sizes differ by at most one row");
    }

    // The TP3 case by hand, because it is the one that actually ships.
    {
        const size_t wb = 163840ull * 14336;   // F16 head, 7168 cols
        K3HeadBand a, b, c;
        const bool ok = k3_head_band_uneven(163840, wb, 3, 0, &a) &&
                        k3_head_band_uneven(163840, wb, 3, 1, &b) &&
                        k3_head_band_uneven(163840, wb, 3, 2, &c);
        check(ok, "F4 TP3 bands resolve");
        check(a.rows == 54614 && b.rows == 54613 && c.rows == 54613,
              "F5 TP3 rows are 54614/54613/54613, got " + std::to_string(a.rows) + "/" +
                  std::to_string(b.rows) + "/" + std::to_string(c.rows));
        check(a.offset == 0 && b.offset == 54614 && c.offset == 109227,
              "F6 TP3 offsets are 0/54614/109227");
        check(a.rows + b.rows + c.rows == 163840, "F7 TP3 bands sum to vocab");
    }

    // Declines. Each of these would otherwise build a pointer from bad arithmetic and
    // read real weights belonging to other rows at full speed.
    {
        const size_t wb = 163840ull * 14336;
        K3HeadBand hb;
        check(!k3_head_band_uneven(163840, wb, 1, 0, &hb), "F8 declines tp_size=1");
        check(!k3_head_band_uneven(163840, wb, 3, 3, &hb), "F9 declines rank out of range");
        check(!k3_head_band_uneven(163840, wb, 3, -1, &hb), "F10 declines negative rank");
        check(!k3_head_band_uneven(163840, wb + 1, 3, 0, &hb),
              "F11 declines a weight whose bytes do not divide into rows");
        check(!k3_head_band_uneven(163840, 0, 3, 0, &hb), "F12 declines zero-byte weight");
        check(!k3_head_band_uneven(2, wb, 3, 0, &hb), "F13 declines vocab < tp_size");
        check(!k3_head_band_uneven(163840, wb, 3, 0, nullptr), "F14 declines null out");
    }
}

}  // namespace

int main() {
    // The toggle is read once into a function-local static, so it has to be set before the
    // first call rather than between cases.
    ::setenv("SPARKINFER_K3_DIST_HEAD_BAND", "1", 1);

    std::printf("kimi_k3_tune_check — drafter gating + uneven head band\n\n");
    test_gating();
    std::printf("\n");
    test_head_band();
    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
