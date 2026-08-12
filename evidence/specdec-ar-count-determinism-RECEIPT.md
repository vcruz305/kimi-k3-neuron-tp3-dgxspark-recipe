# Receipt — batched-verify "token 0 depends on K" root cause

**Date:** 2026-08-11
**Fleet:** TP3 (`wesche-spark-78f1/9f73/366f`), post-0019 corrected engine + patch 0021 + patch 0022 diagnostic.
**Binary/lib md5, all three ranks identical:** `a9430cfb11568ff27fa1c034d6f3ece8` / `ff470d222765d7be0deca260d60cea19`
**Launch:** `K=6 EXTRA="SPARKINFER_K3_KDA_FUSE=0" bash spec_bench.sh` (rank0), `SPARKINFER_K3_PREFILL_CHUNK=16`.

**Verdict: numerics, not a defect.** Token 0's dependence on K is the NCCL all-reduce
element count re-associating a 3-way float sum. No data from tokens 1..K-1 reaches
token 0. Every batched kernel in patch 0021 is bit-exact.

---

## Run 1 — baseline (`SPARKINFER_K3_AR_PAD` unset)

Inertness check: reproduces patch 0021's previously logged numbers to all nine printed
digits (`K=2 pos=0 cos=0.998947945 max|d|=5.615e-01`), so the 0022 code is inert unset.

```
[spec-bench] prompt=3 tokens, Kmax=6, vocab=163840
[spec-bench] --- correctness: forward_batch(K) vs K x forward_token ---
[spec-bench] K=2 pos=0 cos=0.998947945 top1 seq=635 bat=635 AGREE max|d|=5.615e-01 exact=0.0000%
[spec-bench] K=2 pos=1 cos=0.993737007 top1 seq=4 bat=103419 **DISAGREE** max|d|=1.419e+00 exact=0.0000%
[spec-bench] K=3 pos=0 cos=0.998871043 top1 seq=635 bat=635 AGREE max|d|=6.557e-01 exact=0.0000%
[spec-bench] K=3 pos=1 cos=0.995782236 top1 seq=4 bat=103419 **DISAGREE** max|d|=1.065e+00 exact=0.0000%
[spec-bench] K=3 pos=2 cos=0.986412100 top1 seq=1 bat=1 AGREE max|d|=1.749e+00 exact=0.0000%
[spec-bench] K=4 pos=0 cos=0.998446299 top1 seq=635 bat=635 AGREE max|d|=7.006e-01 exact=0.0000%
[spec-bench] K=4 pos=1 cos=0.990832647 top1 seq=4 bat=5 **DISAGREE** max|d|=1.661e+00 exact=0.0000%
[spec-bench] K=4 pos=2 cos=0.942752261 top1 seq=1 bat=1 AGREE max|d|=3.045e+00 exact=0.0000%
[spec-bench] K=4 pos=3 cos=0.990616907 top1 seq=0 bat=0 AGREE max|d|=1.547e+00 exact=0.0006%
[spec-bench] K=5 pos=0 cos=0.998446299 top1 seq=635 bat=635 AGREE max|d|=7.006e-01 exact=0.0000%
[spec-bench] K=5 pos=1 cos=0.990832647 top1 seq=4 bat=5 **DISAGREE** max|d|=1.661e+00 exact=0.0000%
[spec-bench] K=5 pos=2 cos=0.942752261 top1 seq=1 bat=1 AGREE max|d|=3.045e+00 exact=0.0000%
[spec-bench] K=5 pos=3 cos=0.990659933 top1 seq=0 bat=0 AGREE max|d|=1.633e+00 exact=0.0000%
[spec-bench] K=5 pos=4 cos=0.708464838 top1 seq=4 bat=76657 **DISAGREE** max|d|=5.174e+00 exact=0.0000%
[spec-bench] K=6 pos=0 cos=0.998446299 top1 seq=635 bat=635 AGREE max|d|=7.006e-01 exact=0.0000%
[spec-bench] K=6 pos=1 cos=0.990832647 top1 seq=4 bat=5 **DISAGREE** max|d|=1.661e+00 exact=0.0000%
[spec-bench] K=6 pos=2 cos=0.942752261 top1 seq=1 bat=1 AGREE max|d|=3.045e+00 exact=0.0000%
[spec-bench] K=6 pos=3 cos=0.990659933 top1 seq=0 bat=0 AGREE max|d|=1.633e+00 exact=0.0000%
[spec-bench] K=6 pos=4 cos=0.704704775 top1 seq=4 bat=76657 **DISAGREE** max|d|=5.202e+00 exact=0.0000%
[spec-bench] K=6 pos=5 cos=0.731587973 top1 seq=1 bat=1 AGREE max|d|=4.856e+00 exact=0.0000%
[spec-bench] --- token 0: data dependence vs shape dependence ---
[spec-bench] K=2 token0 fillerA vs fillerB: BIT-IDENTICAL (shape-only)  max|d|=0.000e+00
[spec-bench] K=3 token0 fillerA vs fillerB: BIT-IDENTICAL (shape-only)  max|d|=0.000e+00
[spec-bench] K=4 token0 fillerA vs fillerB: BIT-IDENTICAL (shape-only)  max|d|=0.000e+00
[spec-bench] K=5 token0 fillerA vs fillerB: BIT-IDENTICAL (shape-only)  max|d|=0.000e+00
[spec-bench] K=6 token0 fillerA vs fillerB: BIT-IDENTICAL (shape-only)  max|d|=0.000e+00
[spec-bench] token0 K=2 vs K=3: **DIFFERS**  max|d|=8.008e-01
[spec-bench] token0 K=2 vs K=4: **DIFFERS**  max|d|=6.672e-01
[spec-bench] token0 K=2 vs K=5: **DIFFERS**  max|d|=6.672e-01
[spec-bench] token0 K=2 vs K=6: **DIFFERS**  max|d|=6.672e-01
[spec-bench] --- timing: step(K) ---
[spec-bench] step(K=1) = 160.326 ms   (160.326 ms/token)
[spec-bench] step(K=2) = 190.971 ms   (95.485 ms/token)
[spec-bench] step(K=3) = 304.933 ms   (101.644 ms/token)
[spec-bench] step(K=4) = 264.322 ms   (66.081 ms/token)
[spec-bench] step(K=5) = 331.083 ms   (66.217 ms/token)
[spec-bench] step(K=6) = 350.705 ms   (58.451 ms/token)

[spec-bench] FIT step(K) = 133.89 + 38.05*K ms
[spec-bench] marginal cost m = 38.05 ms; break-even acceptance m/step(1) = 0.2373
```

**Reading.** Token 0 is bit-identical across fillers at every K (`max|d| = 0.000e+00`
over all 163840 logits) while tokens 1..K-1 were replaced with far out-of-distribution
ids. Nothing belonging to a later token reaches token 0. Yet token 0 still moves with K,
and takes exactly three distinct values: K=2, K=3, K>=4 (`cos` and `max|d|` at pos=0 are
identical to nine digits for K=4, 5 and 6). Shape-driven, not data-driven.

---

## Run 2 — all-reduce count pinned (`SPARKINFER_K3_AR_PAD=8`)

```
[spec-bench] prompt=3 tokens, Kmax=6, vocab=163840
[spec-bench] --- correctness: forward_batch(K) vs K x forward_token ---
[spec-bench] K=2 pos=0 cos=1.000000000 top1 seq=635 bat=635 AGREE max|d|=0.000e+00 exact=100.0000%
[spec-bench] K=2 pos=1 cos=0.980106662 top1 seq=5 bat=5 AGREE max|d|=2.037e+00 exact=0.0000%
[spec-bench] K=3 pos=0 cos=1.000000000 top1 seq=635 bat=635 AGREE max|d|=0.000e+00 exact=100.0000%
[spec-bench] K=3 pos=1 cos=0.980106662 top1 seq=5 bat=5 AGREE max|d|=2.037e+00 exact=0.0000%
[spec-bench] K=3 pos=2 cos=0.988887612 top1 seq=1 bat=1 AGREE max|d|=1.562e+00 exact=0.0000%
[spec-bench] K=4 pos=0 cos=1.000000000 top1 seq=635 bat=635 AGREE max|d|=0.000e+00 exact=100.0000%
[spec-bench] K=4 pos=1 cos=0.980106662 top1 seq=5 bat=5 AGREE max|d|=2.037e+00 exact=0.0000%
[spec-bench] K=4 pos=2 cos=0.988887612 top1 seq=1 bat=1 AGREE max|d|=1.562e+00 exact=0.0000%
[spec-bench] K=4 pos=3 cos=0.990708311 top1 seq=0 bat=0 AGREE max|d|=1.421e+00 exact=0.0000%
[spec-bench] K=5 pos=0 cos=1.000000000 top1 seq=635 bat=635 AGREE max|d|=0.000e+00 exact=100.0000%
[spec-bench] K=5 pos=1 cos=0.980106662 top1 seq=5 bat=5 AGREE max|d|=2.037e+00 exact=0.0000%
[spec-bench] K=5 pos=2 cos=0.988887612 top1 seq=1 bat=1 AGREE max|d|=1.562e+00 exact=0.0000%
[spec-bench] K=5 pos=3 cos=0.990708311 top1 seq=0 bat=0 AGREE max|d|=1.421e+00 exact=0.0000%
[spec-bench] K=5 pos=4 cos=0.986486309 top1 seq=5 bat=48642 **DISAGREE** max|d|=1.869e+00 exact=0.0000%
[spec-bench] K=6 pos=0 cos=1.000000000 top1 seq=635 bat=635 AGREE max|d|=0.000e+00 exact=100.0000%
[spec-bench] K=6 pos=1 cos=0.980106662 top1 seq=5 bat=5 AGREE max|d|=2.037e+00 exact=0.0000%
[spec-bench] K=6 pos=2 cos=0.988887612 top1 seq=1 bat=1 AGREE max|d|=1.562e+00 exact=0.0000%
[spec-bench] K=6 pos=3 cos=0.990708311 top1 seq=0 bat=0 AGREE max|d|=1.421e+00 exact=0.0000%
[spec-bench] K=6 pos=4 cos=0.986486309 top1 seq=5 bat=48642 **DISAGREE** max|d|=1.869e+00 exact=0.0000%
[spec-bench] K=6 pos=5 cos=0.991982870 top1 seq=1 bat=1 AGREE max|d|=1.879e+00 exact=0.0000%
[spec-bench] --- token 0: data dependence vs shape dependence ---
[spec-bench] K=2 token0 fillerA vs fillerB: BIT-IDENTICAL (shape-only)  max|d|=0.000e+00
[spec-bench] K=3 token0 fillerA vs fillerB: BIT-IDENTICAL (shape-only)  max|d|=0.000e+00
[spec-bench] K=4 token0 fillerA vs fillerB: BIT-IDENTICAL (shape-only)  max|d|=0.000e+00
[spec-bench] K=5 token0 fillerA vs fillerB: BIT-IDENTICAL (shape-only)  max|d|=0.000e+00
[spec-bench] K=6 token0 fillerA vs fillerB: BIT-IDENTICAL (shape-only)  max|d|=0.000e+00
[spec-bench] token0 K=2 vs K=3: BIT-IDENTICAL  max|d|=0.000e+00
[spec-bench] token0 K=2 vs K=4: BIT-IDENTICAL  max|d|=0.000e+00
[spec-bench] token0 K=2 vs K=5: BIT-IDENTICAL  max|d|=0.000e+00
[spec-bench] token0 K=2 vs K=6: BIT-IDENTICAL  max|d|=0.000e+00
[spec-bench] --- timing: step(K) ---
[spec-bench] step(K=1) = 175.144 ms   (175.144 ms/token)
[spec-bench] step(K=2) = 207.539 ms   (103.770 ms/token)
[spec-bench] step(K=3) = 305.550 ms   (101.850 ms/token)
[spec-bench] step(K=4) = 244.162 ms   (61.040 ms/token)
[spec-bench] step(K=5) = 337.931 ms   (67.586 ms/token)
[spec-bench] step(K=6) = 451.734 ms   (75.289 ms/token)

[spec-bench] FIT step(K) = 115.74 + 48.94*K ms
[spec-bench] marginal cost m = 48.94 ms; break-even acceptance m/step(1) = 0.2794
```

**Reading.** Pinning the collective's element count makes token 0 `exact=100.0000%`,
`cos=1.000000000`, `max|d|=0.000e+00` against the sequential single-token path, at every
K = 2..6. The anomaly is gone and so is the entire position-0 fidelity gap.

Positions b >= 1 stay inexact because the pin fixes the *count* but not the *offset*:
token b's partial sits at buffer row b in the batched arm and at row 0 in the sequential
arm, so it still lands in a different reduce-scatter partition. Token 0 is the one row
whose offset is 0 in both arms, and it is exactly the row that goes bit-exact. Their
values are now a function of b alone and no longer of K — `pos=1` is `0.980106662` at
every K, `pos=2` is `0.988887612` at every K >= 3, `pos=3` is `0.990708311` at every
K >= 4 — which is the causal property the anomaly violated.

Fidelity side effect: worst-position cosine `0.704 -> 0.986`; top-1 disagreement
`7/20 -> 2/20` position-instances.

---

## Run 3 — collective-free control (`kimi_k3_kda_batch_check`, tp=1, rank0 only)

`kimi_k3_kda_batch_check <gguf> 6 15 1 0 full` — layers 0..15, phase All, `max_ckpt 8`
(taken from `cfg.n_layers = 93`, so the residual bank still banks more than once and the
0021 stride condition is the real one). **tp=1, so there is no collective anywhere in
this arm.**

```
batched-Attn equivalence: layer 15 (MLA), hidden 7168, n_q_heads 96, kda_head_dim 128, tp 1 rank 0, scratch cap 64
  attn_batch_ok(layer 15, n_tok 2) = true
  FULL mode: layers 0..15, phase All, max_ckpt 8 (banks TWICE - stride exposed)
K=2: digest seq=5fee3e9223a88b4b bat=5fee3e9223a88b4b
K=2: PASS  14336 elems bit-identical
K=3: digest seq=8234b1efe7e98046 bat=8234b1efe7e98046
K=3: PASS  21504 elems bit-identical
K=4: digest seq=aac98d3a297eecf6 bat=aac98d3a297eecf6
K=4: PASS  28672 elems bit-identical
K=5: digest seq=5fb217b29ff8aabb bat=5fb217b29ff8aabb
K=5: PASS  35840 elems bit-identical
K=6: digest seq=f91eda43666b7e49 bat=f91eda43666b7e49
K=6: PASS  43008 elems bit-identical

PASS: batched full-stack forward reproduces K sequential calls for K=2..6
```

**Reading.** With the collective removed, the batched forward is bit-identical to K
sequential calls at every K = 2..6. This spans both token-tile tiers of the batched Q8
projection (`SPARKINFER_K3_PROJ_TOKS` default 4 gives TOKS=2 at K=2,3 and TOKS=4 at
K=4,5,6 — `k3_kernels.cu:6400`), so the tile width is not re-associating anything. It is
the collective-free control for the conclusion above.

Depth is capped by memory, not by choice: at ~3.5 GB/layer the real model reaches roughly
30 layers on one 121 GB Spark, so this harness cannot be walked to 93 on a single node.
The full-depth evidence is Run 2, which covers all 93 layers on the real TP3 path.

---

## Runs 4–7 — re-measuring `m` with a protocol that can support a slope

The original protocol ran all 5 reps of K=1, then all of K=2, and so on. K was
therefore perfectly correlated with wall-clock time, so any slow drift landed in the
**slope** rather than the noise — the standard way to measure a trend that is not there.
Replaced with: **21 sweeps, one sample of every K per sweep, starting K rotated per
sweep** (rotation by sweep index, not random, so every rank issues the identical call
sequence and the collectives still match), state re-anchored once per sweep, reported as
**medians** with p25/p75/min/mean/sd, plus se(m), a Student-t 95% CI at n−2 dof, R², and
a Theil–Sen robust cross-check.

### The non-monotonicity is real, not noise

```
[spec-bench] 21 sweeps x 8 K values, rotated order, 3 warm sweeps discarded
[spec-bench]  K   median     p25      p75      min     mean      sd    n
[spec-bench]  1   164.49   164.35   165.84   163.39   173.61   21.77   21
[spec-bench]  2   194.55   193.66   196.02   192.73   204.47   25.45   21
[spec-bench]  3   289.73   289.24   290.69   288.20   300.42   26.88   21
[spec-bench]  4   236.03   234.74   236.54   233.52   249.61   30.13   21
[spec-bench]  5   332.07   331.64   335.75   328.64   353.74   40.99   21
[spec-bench]  6   354.75   353.21   356.60   350.40   370.73   36.97   21
[spec-bench]  7   376.62   374.89   380.17   372.68   401.25   56.38   21
[spec-bench]  8   320.44   319.12   323.70   315.37   342.80   44.21   21
```

`step(3) > step(4)` by 54 ms and `step(7) > step(8)` by 56 ms, with **disjoint IQRs** in
both cases. The medians are tight (IQR 1–3 ms) and reproduce to ±0.3 ms across four
independent fleet runs. The large `sd` versus a tight IQR is a handful of stragglers
dragging the mean — exactly why the median is the right statistic here.

**`step(K)` is a sawtooth, not a line.** A single OLS slope over K=1..8 is therefore not
an estimate of marginal cost: it reports 27–37 ms depending on which K values it sees,
and its R² of 0.75 is measuring the sawtooth, not scatter. That is what `m = 37 ms` was.

### The low branch, which is the number that matters

K = 2, 4, 8 sit on a near-perfect line:

| span | ms/token |
|---|---:|
| K=2 → K=4 | 20.40 |
| K=4 → K=8 | 21.15 |
| K=2 → K=8 | 20.90 |

```
step(K) = 152.3 + 20.9*K ms      R2 = 0.9999
m = 20.9 ms    95% CI [18.6, 23.3]   (dof = 1)
T = step(K=1) = 164.5 ms         break-even p = m/T = 0.127
```

The CI is wide only because n=3; the point estimate is not — three independent spans
agree within 0.8 ms and the K=2/K=4 medians reproduce to ±0.3 ms across four runs.

The high branch sits a near-constant **~75 ms** above that line (K=3: +74.8, K=5: +74.5,
K=6: +76.5, K=7: +77.8) — roughly one extra full pass over the batched projection
weights.

### Partially explained; the rest is stated, not guessed

A direct A/B on `SPARKINFER_K3_PROJ_TOKS` at K=8 moves `step` from **385.58 ms**
(env 4 → TOKS=4, `tok_blocks=2`) to **320.44 ms** (env 8 → TOKS=8, `tok_blocks=1`) — a
65 ms swing confirming that `ceil(n_tok/TOKS)` extra passes over the projection weights
are a real and large cost.

But widening the tile-selection rule in `k3_proj_q8act_tok_f32` / `k3_proj_f16_tok_f32`
so K=3,5,6,7 also get `tok_blocks=1` **did not move their timings at all** — rebuilt,
redeployed md5-verified to all three ranks, re-measured 289.73 / 332.07 / 354.75 / 376.62
against 289.51 / 331.67 / 352.70 / 375.46 before. Unchanged within noise. **That change
was reverted rather than shipped**, and the ~75 ms penalty at K=3,5,6,7 remains
**unexplained** — a known-unknown in the cost model.

### Consequence for the gate

`m = 20.9 ms` at the usable K, against the design doc's "materially above ~20 ms ⇒ stop"
rule (§8). The gate is met, marginally. But K is not freely choosable while the sawtooth
stands: only K ∈ {2,4,8} (drafts N ∈ {1,3,7}) sit on the low branch, which removes the
adaptive-N strategy §5.4 recommends. Recomputed §5.3 requirements at those three depths:

| target | best N | required p | design's §5.3 (m=15) |
|---|---:|---:|---:|
| 8 tok/s | 3 | **0.50** | 0.387 |
| 10 tok/s | 3 | **0.65** | 0.567 |
| 12 tok/s | 3–7 | **0.77** | 0.679 |
| 15 tok/s | 7 | **0.85** | 0.783 |

---

## Caveats stated rather than buried

- **Superseded by runs 4–7:** the earlier reading that `step(K)` was "too noisy to fit"
  was wrong. It is not noisy — it is a reproducible sawtooth, and the 27–49 ms range
  came from fitting a straight line through it. `m = 20.9 ms` on the low branch.
- The ~75 ms penalty at K=3,5,6,7 is **not explained**. `tok_blocks` accounts for the
  K=8 A/B (65 ms, directly measured) but widening the tile for the other K changed
  nothing. Until that is understood, N is effectively pinned to {1,3,7} and the cost
  model has a hole in it.
- `m = 20.9 ms` has a wide CI ([18.6, 23.3]) purely because the low branch has only
  three points (K=2,4,8). More clean points need either the sawtooth fixed or a run at
  K=16 with a matching `SPARKINFER_K3_PROJ_TOKS`/scratch cap.
- The pin is a probe, not a shipping fix. It reduces N rows when K are live, and it does
  not make b >= 1 exact.
- Single prompt (`--prompt-ids 1,2,3`), single context depth. The mechanism is
  structural, but the *magnitude* of the amplification is not characterised across
  workloads. `T = 164.5 ms` here is the fast single-shot signal, not the multi-prompt
  median protocol, so the throughput table inherits that caveat.
- The pin is a probe, not a shipping fix. It reduces N rows when K are live, and it does
  not make b >= 1 exact.
- Single prompt (`--prompt-ids 1,2,3`), single context depth. The mechanism is
  structural, but the *magnitude* of the amplification is not characterised across
  workloads.
