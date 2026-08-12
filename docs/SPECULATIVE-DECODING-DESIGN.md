# Speculative decoding for SparkInfer TP3 — design

**Status:** implemented (Stages 0-4, patches 0021-0024) and measured on live TP3 hardware — see the results table in [`APPLY.md`](../APPLY.md#0021-0024---speculative-decoding-n-gram--prompt-lookup) and usage in [`README.md`](../README.md#speculative-decoding-experimental-opt-in). This document is the original design rationale and break-even analysis; treat the measured results elsewhere as authoritative where they differ from the projections below (notably: real acceptance binds `N` more than the marginal-cost sawtooth this doc focuses on, and the 10-12 tok/s projection in §5.3 is only reached on literal repetition — realistic code lands around 7.7 tok/s, and prose measurably regresses).
**Date:** 2026-08-11 (design), measured 2026-08-12
**Engine baseline:** post-patch-0019 (corrected) SparkInfer, TP3 on 3× DGX Spark GB10.
**Source read:** live fleet tree `/home/victor/work/k3-tp3-0012/sparkinfer-k3` (rank0) for the dist path, local `\.scratch\sparkinfer-phase3\` for the model/kernel internals. Byte figures parsed from the real GGUF headers on rank0.

---

## 1. Summary and recommendation

**Recommendation: build it, and build it in this order.** The economics on this hardware are substantially *better* than the Track D vLLM/H200 precedent, and the biggest engineering risk is not the one that was expected.

Four findings drive the design:

1. **Decode is almost pure weight-streaming, and 92.9% of the bytes are dense (non-expert).** Re-derived from the real GGUF against the corrected engine: **66.67 GB/token** fleet-wide, of which **61.92 GB is read unconditionally every token** and only **4.75 GB is routed-expert traffic**. Dense weights amortise perfectly across a batched verify; expert weights do not. §4.
2. **The break-even acceptance is ~0.09**, not the 0.2075 measured on H200/vLLM. Marginal cost of one extra verified token is **~14 ms against a 162 ms step (8.6%)**. Speculation is cheap here — cheaper than anywhere in the Track D data. §5.
3. **The feared KDA rollback problem is real but tractable, and the residual-bank problem does not exist.** The `attn_res` bank is *intra-token* scratch (`n_ckpt` is reset to 0 at the top of every forward call, `kimi_k3_dist_forward.cpp:251`) and carries nothing across steps — no rollback needed. KDA state is 147.7 MB/rank and needs a snapshot ring; costed at ~1.0 ms per saved position. §6.
4. **The real risk is a known, undiagnosed defect that gates 52% of the amortisable bytes.** `SPARKINFER_K3_KDA_QKVG_BATCH` — the flag that lets a KDA layer accept `n_tok > 1` — is **disabled by default because it crashes** (`[k3] LAUNCH FAILED at layer 0, phase Attn: invalid argument` for every chunk ≥ 2 tokens, `kimi_k3.cpp:1341-1359`). 69 of 93 layers are KDA, and their attention projections are **32.29 GB of the 61.92 GB dense budget (52.1%)**. Without this fix, most of the speculative win does not exist. **This is Stage 0.** §7.

**Drafting strategy: prompt-lookup / n-gram self-speculation. Confirmed, with a stronger argument than "it's simpler."** Two hard constraints decide it: (a) each Spark holds ~110 GB of a 121 GB budget, leaving no room for a draft model's weights; (b) at a break-even of 0.09, a *free* drafter that fires only when it has a match is net-positive at almost any hit rate, while a draft model would have to pay its cost on every step including misses. DSpark is not adaptable here. §3.

**Honest expectation:** prompt-lookup should reach **8–11 tok/s on structured/agentic/code output** and give **near-zero benefit on freeform prose**. Reaching **15 tok/s single-stream requires ~0.78–0.80 per-position acceptance**, which n-gram lookup will hit only on highly repetitive generation. 15 tok/s across all workloads needs a trained draft, which is a separate project. §5.3.

---

## 2. What the engine does today

`kimi_k3_dist_forward_token()` (`runtime/src/models/kimi_k3_dist_forward.cpp:220-514`) processes **exactly one token**:

- embed one row, then loop 93 layers (`:294`), each running `Attn` → optional all-reduce → `FfnPartial` → optional all-reduce → `FfnFinish`;
- **185 NCCL all-reduces per token** (93 attn + 92 moe, confirmed by the profiler's `mean_n_attn_ar=93 mean_n_moe_ar=92`);
- rank 0 alone runs the head (`:412-449`) and produces `vocab` logits for the single position;
- position advances by exactly 1 (`:490-504`).

Prefill is the same function in a loop — `kimi_k3_dist_forward_prompt()` is literally `for (i...) kimi_k3_dist_forward_token(...)` (`:526-529`). **There is no batched path in the distributed engine at all**, for decode or prefill.

The generation loop (`runtime/examples/kimi_k3_dist_generate.cpp:404-414`) is a plain greedy `argmax → one_step` cycle, with a KV-reset sentinel (`token id = -2`, `:106`) broadcast between prompts.

The control protocol carries **one token per step**: `Message` has a single `int token_id` (`runtime/include/sparkinfer/tp/rank_protocol.h:97`), and `RankSession::accept_control` takes `int* token_id` (`:220`).

**But the single-process/TP model code is much further along than the distributed path.** `kimi_k3_forward_layer_phase()` already takes an `n_tok` parameter (`kimi_k3.h:516-518`, default 1) with a fully documented multi-token contract (`kimi_k3.h:454-534`), and `kimi_k3_tp_prefill_chunk()` (`kimi_k3_tp.cpp:2692`) already drives it at B=64 for chunked prefill. That machinery is the foundation this design builds on — it is not new work, it is *unused* work.

**There is also a working speculative-decoding precedent inside this same engine, for a different model.** `Qwen35Model` has `save_spec_snapshot()` / `restore_spec_snapshot()` (`qwen35.h:185-186`, impl `qwen35.cpp:1752-1777`), a `dflash_generate()` greedy speculative loop reporting mean acceptance τ (`qwen35.h:96-103`), and a `DFlashDraftModel` with a `crop(int keep)` rollback primitive (`dflash_draft.cpp:339`). The K3 design should follow this precedent rather than invent one.

---

## 3. Drafting strategy

### 3.1 Recommendation: prompt-lookup (n-gram self-speculation)

At each step, on rank 0 only, host-side:

1. take the last `n` emitted tokens (try `n = 3`, then 2, then 1);
2. scan backwards through the full prompt+generation sequence for the most recent earlier occurrence of that n-gram;
3. the `N` tokens following that occurrence become the draft;
4. **no match at any `n` ⇒ `N = 0` ⇒ take the ordinary single-token path with zero added cost.**

Cost: a backwards `memcmp` over a few thousand ints, microseconds, entirely off the GPU and off the critical path. No weights, no NCCL, no memory.

### 3.2 Why this and not a draft model

- **Memory is the binding constraint.** The model occupies ~110 GB of each Spark's 121 GB unified memory. There is no room for draft weights, and any draft small enough to fit in the remainder would be too weak to clear the acceptance bar that matters.
- **The break-even is 0.09 (§5.2).** A drafter that costs nothing when it declines to draft is nearly free to attempt. A draft *model* pays its forward cost on every step, including the ones where it is wrong — that fixed cost is exactly the `~8.5 ms just to engage the speculative path` that Track D measured (`current.md:252-254`) and is what stops small N from running away with the win there. Prompt-lookup has no such fixed term.
- **The workload argues for it.** Track D's own strongest result was workload-driven, not model-driven: coding prompts gave 0.744 acceptance vs 0.547 for prose, and that alone moved the optimum from N=2 to N=3 and cleared the 50 t/s gate (`current.md:427-441`). Repetitive, structured, low-entropy output is precisely where n-gram lookup is strongest, and it is the workload this fleet is most likely serving.

### 3.3 Secondary option: a trained draft (do not block on it)

`Inferact/Kimi-K3-DSpark` is **not adaptable to SparkInfer**. It is a different architecture (`Qwen3DSparkModel` → `dflash`), it expects to share the target's embedding and LM head, and — per Track D's own conclusion — it was distilled against full-precision Kimi-K3 and needs a **+20.5% relative acceptance lift** that "no runtime flag fixes… it needs a draft distilled against the compressed target" (`current.md:342-348`, `:376-382`).

That said, SparkInfer already has `DFlashDraftModel` (`dflash_draft.cpp`) with a safetensors loader and shared-weight plumbing. If a draft is ever distilled against this IQ1_S target, the plumbing to run it exists. Note it as a future lever, not a dependency.

### 3.4 Sampling contract

**Greedy only, at least initially.** The engine's decode is already `kimi_k3_dist_argmax` (`kimi_k3_dist_forward.cpp:533`). Under greedy decoding, exact-match acceptance is lossless by construction and needs no rejection-sampling machinery. Track D found the probabilistic/block sampling contract to be untestable at temperature 0 anyway (`current.md:275-280`).

---

## 4. Re-derived byte accounting (corrected engine)

All figures parsed directly from the GGUF tensor headers on rank0 (`/home/victor/models/kimi-k3-neuron-iq1s-local/`, 9 shards, 2573 tensors, **330,160,710,016 B = 330.161 GB**). Quantisation: IQ1_S = 50 B / 256 weights; Q8_0 = 34 B / 32 weights.

Confirmed GGUF metadata: `expert_count 896`, `expert_used_count 16`, `expert_shared_count 4`, `expert_feed_forward_length 1536`, `block_count 93`, `leading_dense_block_count 1`, `head_count_kv` = period-4 array `[0,0,0,1,...]` ⇒ **69 KDA / 24 MLA layers**.

### 4.1 What one decode token actually reads

| class | bytes | GB | share |
|---|---:|---:|---:|
| **Dense — read unconditionally every token** | 61,919,240,576 | **61.919** | **92.88%** |
| **Routed experts — 16 of 896 per MoE layer** | 4,748,083,200 | 4.748 | 7.12% |
| **Total per token (fleet-wide)** | **66,667,323,776** | **66.667** | 100% |

Routed-expert arithmetic: `3584 × 1536` per matrix ⇒ 5,505,024 weights ⇒ 1,075,200 B at IQ1_S; × 3 matrices = 3,225,600 B/expert; × 16 experts = 51,609,600 B/layer; × 92 MoE layers = 4.748 GB. The full expert tensors are 265.89 GB (80.5% of the file) but only 1.8% of them are touched per token.

### 4.2 Dense breakdown — where the 61.92 GB goes

| tensor role | type | GB (all layers) | amortises under batched verify? |
|---|---|---:|---|
| `attn_output` (93 layers) | Q8_0 | 8.703 | yes |
| `attn_q` (69 KDA) | Q8_0 | 6.457 | **yes — but gated on the KDA batch defect** |
| `attn_k` (69 KDA) | Q8_0 | 6.457 | **gated** |
| `attn_v` (69 KDA) | Q8_0 | 6.457 | **gated** |
| `ssm_g` (69 KDA) | Q8_0 | 6.457 | **gated** |
| `ffn_gate_shexp` | Q8_0 | 4.305 | yes |
| `ffn_up_shexp` | Q8_0 | 4.305 | yes |
| `ffn_down_shexp` | Q8_0 | 4.305 | yes |
| `ffn_routed_down` | Q8_0 | 2.511 | yes |
| `ffn_routed_up` | Q8_0 | 2.511 | yes |
| `ffn_gate_inp` (router) | F32 | 2.364 | yes |
| `output.weight` (LM head) | F16 | 2.349 | yes — **only if the head is batched (§7.2)** |
| `attn_gate` (24 MLA) | Q8_0 | 2.246 | yes |
| remainder (q_a/q_b/k_b/v_b/kv_a_mqa, ssm_f_a/f_b/beta/conv1d, layer-0 dense FFN, norms) | mixed | 2.492 | yes |

`token_embd` (2.349 GB, F16) is excluded — decode reads one row, 14 KB.

**The KDA-gated group** — `attn_q + attn_k + attn_v + ssm_g` (25.829 GB) plus the KDA layers' share of `attn_output` (69/93 × 8.703 = 6.457 GB) — totals **32.286 GB = 52.1% of all dense bytes.** This is the single most important number in the document: it is the amount of amortisable traffic that is currently behind a crashing feature flag.

### 4.3 Per-rank load and effective bandwidth

Under the dist load policy only rank 0 holds `output.weight` (`kimi_k3_dist_forward.cpp:198` comment, `:412`), so the head is unsharded:

| | GB / token |
|---|---:|
| sharded portion (66.667 − 2.349) ÷ 3 | 21.439 |
| **rank 0** (+ unsharded head) | **23.788** |
| workers (rank 1, 2) | 21.439 |

Rank 0 is the critical path. Against measured post-fix speeds:

| config | tok/s | ms/token | effective B/W on rank0 | vs GB10 ~273 GB/s peak |
|---|---:|---:|---:|---:|
| baseline | 5.1534 | 194.0 | 122.6 GB/s | 45% |
| `KDA_FUSE_off` (best) | 6.1713 | 162.0 | **146.8 GB/s** | 54% |

A 45–54% memory-bandwidth utilisation for a mixed IQ1_S/Q8_0 GEMV workload with 185 collectives per token is a coherent, believable number. **Decode is bandwidth-bound, and the flag sweep was moving bandwidth efficiency, not compute.**

### 4.4 Independent validation: this byte model predicts the pre/post-fix speed drop

Patch 0019 restored four tensor families the loader had been silently skipping. Their exact byte cost:

| restored by 0019 | GB/token |
|---|---:|
| shared experts (`ffn_{gate,up,down}_shexp`) | 12.915 |
| MLA attention gate (`attn_gate`) | 2.246 |
| MLA q-lora (`attn_q_a` + norm) | 0.281 |
| routed RMSNorm (`ffn_routed_norm`) | 0.001 |
| **total** | **15.443** |

Pre-fix the engine was therefore streaming 66.667 − 15.443 = **51.224 GB/token**.

- **Predicted** pre-fix ÷ post-fix speed ratio: 66.667 / 51.224 = **1.301×**
- **Measured**: 6.84 (pre-fix median) ÷ 5.1534 (post-fix baseline) = **1.327×**

**Agreement within 2.0%.** Three things fall out of this: the byte model is correct; decode really is dominated by weight streaming (a compute-bound or latency-bound step would not track byte count this tightly); and the pre-fix numbers were inflated by precisely the weight traffic the loader was skipping.

*Caveat, stated rather than buried:* the two speeds come from different measurement protocols — 6.84 is the multi-prompt syncfix median, 5.1534 is the flag sweep's single-shot fast signal. The agreement is strong enough to be meaningful but should not be quoted to three digits. Re-measuring the post-fix baseline under the multi-prompt median protocol (open TODO #4 in `current-tp3-sparks.md`) would tighten it.

### 4.5 A free ~6% that falls out of this analysis

Rank 0 carries the entire 2.349 GB F16 head while workers idle. The TP path already has head-band sharding (`k3_head_band`, `kimi_k3_tp.cpp:474-483`, `:1189-1201`); the dist path does not use it. Sharding the head three ways cuts rank0's per-token bytes from 23.788 to 22.222 GB — **a 6.6% step-time reduction, independent of speculative decoding**. Worth doing regardless; it also removes the head from the per-pass cost under speculation.

---

## 5. Break-even math

### 5.1 Cost model

For a verify pass over `N` draft tokens on rank 0:

```
step(N) = T + N·m
```

`T` = 162.0 ms (current best, `KDA_FUSE_off`). `m` = marginal cost of one additional verified position. Components, at the measured 146.8 GB/s:

| component | per extra token | ms | amortises? |
|---|---:|---:|---|
| routed experts (4.748 GB ÷ 3 ranks) | 1.583 GB | 10.78 | no — different tokens route to different experts |
| KDA `delta_state` read+write (414.11 MB ÷ 3, ×2) | 276.1 MB | 1.88 | no — recurrence is strictly sequential |
| KDA conv state read+write (29.10 MB ÷ 3, ×2) | 19.4 MB | 0.13 | no |
| MLA KV-cache walk @ 4k ctx | ~37.7 MB | 0.26 | no — each position attends its own prefix |
| residual per-token launches (~440 × 1.7 µs) | — | 0.75 | no |
| dense weights (19.857 GB/rank) | 0 | 0 | **yes** |
| LM head (2.349 GB, batched) | 0 | 0 | **yes** |
| 185 NCCL all-reduces | 0 | ~0 | **yes** — 28–43 KB payloads are latency-bound, so K× the bytes costs ~the same |
| **total m** | | **≈13.8** | use **15 ms** with margin |

The expert-growth assumption is deliberately pessimistic. Under independent uniform routing the distinct-expert multiplier is `u(K) = 56·(1 − (55/56)^K)`: u(2)=1.98, u(4)=3.89, u(8)=7.52 — i.e. essentially linear. Real routing correlation between adjacent tokens will make this sublinear, so `m` is an upper bound.

The collectives point deserves emphasis: the profiler puts both all-reduces at **26% of decode time** (`docs/TP3-DECODE-PROFILE.md`) for payloads of 28.7 KB (attn) and 43.0 KB (moe). At those sizes NCCL is pure latency. Batching N tokens multiplies the payload by N and the time by ~1. **A quarter of the decode step amortises to nothing.**

### 5.2 Break-even acceptance

```
p_break-even = m / T = 15 / 162 = 0.093
```

**Compare Track D's H200/vLLM figure of 0.2075** (`current.md:350-356`). Ours is **2.2× lower**, for a structural reason: the dense fraction here is 92.9%, versus a stack where MoE routing and a much larger per-position marginal cost dominate. Speculation is a better bet on this hardware than it was on the H200 fleet, despite the hardware being far slower.

Practical consequence: a speculative position is worth drafting if it has better than a **1-in-11** chance of being accepted.

### 5.3 What acceptance is required to hit the targets

With `E(N,p) = Σ_{i=0..N} p^i` (chain acceptance, greedy) and `step(N) = 162 + 15N` ms:

**Achieved throughput (tok/s):**

| p ↓ / N → | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|
| 0.3 | **7.24** | 6.85 | 6.42 | 6.02 | 5.67 |
| 0.4 | **8.12** | 7.85 | 7.43 | 7.00 | 6.60 |
| 0.5 | **9.11** | 9.06 | 8.73 | 8.31 | 7.87 |
| 0.6 | 10.21 | **10.51** | 10.39 | 10.06 | 9.64 |
| 0.7 | 11.41 | 12.24 | **12.49** | 12.41 | 12.14 |
| 0.8 | 12.71 | 14.26 | 15.14 | 15.57 | **15.68** |
| 0.9 | 14.11 | 16.61 | 18.45 | 19.77 | **20.70** |

Bold marks the best N for each acceptance rate. **The optimal N rises with acceptance** — the same behaviour Track D measured when moving from prose to code (`current.md:439-441`), and the reason a single pinned N is the wrong shipping decision.

**Acceptance required per target:**

| target | best N | required p |
|---|---:|---:|
| 8 tok/s (+30%) | 2 | **0.387** |
| **10 tok/s (+62%)** | **3** | **0.567** |
| 12 tok/s (+94%) | 4 | 0.679 |
| **15 tok/s (+143%)** | **6** | **0.783** |
| 20 tok/s | 9 | 0.877 |

**Ceiling check** (perfect acceptance, `E = N+1`): N=4 → 22.5 tok/s, N=8 → 31.9 tok/s, N→∞ → 66.7 tok/s. **The architecture is not the limit; acceptance is.** This mirrors Track D's conclusion exactly — "headroom is acceptance-bound, not overhead-bound" (`current.md:256`).

**Reading of these numbers:** 10 tok/s at p≈0.57 is a realistic target for structured/agentic output with n-gram lookup. 15 tok/s at p≈0.78 is at or beyond what prompt-lookup typically delivers outside highly repetitive generation. The honest framing for the 15 tok/s goal is: **speculative decoding plus the existing flag win plus head sharding gets to roughly 10–12 tok/s on favourable workloads; the last leg to 15 needs either a trained draft or a lower `T` (TP4, or further kernel work).**

### 5.4 Optimal N is workload-dependent — do not pin one value

Prompt-lookup acceptance is *bimodal*, not geometric: either the n-gram hits and a long run is accepted, or it misses immediately. This makes deeper N cheaper than the geometric model suggests when a hit occurs, and free when there is no match at all (`N=0`). The `p_i > m/T` rule applies per position and should be evaluated against **measured per-position acceptance**, exactly as Track D did (`current.md:357-369`) — that methodology transfers even though its numbers do not.

---

## 6. Rollback design

### 6.1 The residual bank: no rollback needed

This was flagged as a likely nasty surprise. It is not one.

`state.n_ckpt` is **reset to 0 at the top of every forward call** (`kimi_k3_dist_forward.cpp:251`, `rank.state.n_ckpt = 0;`) and the bank is rebuilt across the 93-layer pass, at most `ceil(93/12) = 8` entries (`kimi_k3.cpp:664-665`). It is **intra-token scratch, not cross-step state.** Nothing survives a forward call, so nothing can need rolling back.

What it *does* require for a K-token batch is one bank **per token**: the contract states `res_bank` must point at token 0's bank with token b at `res_bank + b·res_bank_row_elems·max_ckpt` (`kimi_k3.h:501-503`), and the chunk driver already allocates exactly this (`kimi_k3_tp.cpp:3129-3134`). At `7168 × 8 × 4 B = 229 KB` per token, K=4 costs 917 KB. Trivial.

One real subtlety to preserve: `n_ckpt` is a function of the *layer*, not the token, so one scalar serves the whole chunk (`kimi_k3.h:514-515`, enforced `kimi_k3.cpp:2215-2246`). The bank pointer moves per token; the slot index does not. Getting this backwards is a documented silent-corruption mode — writing all rows at `res_bank + n_ckpt·H` is "invisible at short context and wrong at depth" (`kimi_k3.cpp:2236-2238`).

### 6.2 The MLA KV cache: rollback is free

Every MLA attention kernel lengths its walk from its own `*d_pos + 1` (`kimi_k3.h:477-480`). Rows written by rejected tokens sit beyond `position + j` and are never read; the next pass overwrites them. **Rollback = set the position back.** No buffer work.

### 6.3 The KDA recurrent state: this is the real one

The state is `delta_state` (`[128,128,96]` per KDA layer) plus three conv windows. Fleet-wide totals, and per-rank at TP3:

| buffer | fleet | per rank (TP3) |
|---|---:|---:|
| `delta_state` × 69 layers | 414.11 MB | 138.04 MB |
| `conv_state_{q,k,v}` × 69 layers | 29.10 MB | 9.70 MB |
| **total** | **443.21 MB** | **147.74 MB** |

The crux question was whether the state advances per-token (clean rollback points) or in internal chunks (no clean points). **The answer is favourable, but with a trap:**

- **The path the runtime actually runs is strictly sequential per token.** `for (int b = 0; b < a_tok; ++b)` at `kimi_k3.cpp:2585`, closing at `:2768`. The engine's own comment is explicit: "st.delta_state carries the delta rule from token b-1 into token b; both are updated IN PLACE… the stream order IS the recurrence order" (`kimi_k3.cpp:2524-2533`). **A well-defined per-token state exists at every boundary** — it simply is not saved anywhere.
- **The trap:** a true chunkwise-parallel WY/UT-transform kernel exists — `k3_kda_chunk_prefill` (`kernels/csrc/cuda/kimi_k3/k3_kda_chunk_prefill.cu`, `CHUNK = 16` and load-bearing at `:80`) — which materialises **no** per-token intermediate states and holds the state in registers across the whole scan, writing back once (`:390-397`, `:481-483`). It is also explicitly *not* bit-identical to sequential decode steps (`kimi_k3.h:320-326`). **It has zero callers in `runtime/`.** It must stay that way. If anyone ever wires it into the decode path, speculative rollback becomes impossible. Worth an explicit comment at the call site.

Both state buffers are read-modify-write in place (`k3_kda_step_ip.cu:163`, `:210`; `k3_kernels.cu:414`, `:430`; conv shift at `:742-745`). There is **no** existing snapshot/restore for K3 state, and `kimi_k3_reset_state` zeroes rather than restores (`kimi_k3.cpp:772-795`).

**The conv state must be rolled back too.** It is only 6.6% of the bytes, but restoring `delta_state` alone leaves the 4-deep conv window ahead of the delta state, and the next token convolves over the wrong window — fluent, silently wrong output. `kimi_k3_reset_state` treats all four buffers as one unit for exactly this reason.

### 6.4 Proposed mechanism

After a verify pass over N drafts we accept `j ∈ [0, N]` and must restore the state to "having consumed exactly j drafts". `j = N` needs nothing.

**Stage-1 mechanism (N ≤ 2–3): per-token state history ring.**

Save the state after each token *inside* the existing scan loop, into a depth-N ring `delta_hist[kda_ord][b]` / `conv_hist[kda_ord][b]`. On acceptance of `j`, one `cudaMemcpyDeviceToDevice` per layer per buffer restores slot `j`. The save is a natural extension of the loop that already exists at `kimi_k3.cpp:2585-2768`.

| | cost |
|---|---|
| memory | N × 147.74 MB/rank → N=3: 443 MB (fits; ~10 GB free per node) |
| bandwidth, save | N × 147.74 MB → N=3: 443 MB ≈ **3.0 ms/pass ≈ 1.0 ms per position** |
| bandwidth, restore | 147.74 MB ≈ 1.0 ms, only on partial rejection |
| launches | 4 buffers × 69 layers = 276 memcpys per save — **consolidate first, see below** |

This adds ~1.0 ms to the 15 ms marginal cost — a 7% increase in `m`, moving break-even from 0.093 to 0.099. Negligible.

**Consolidation prerequisite.** K3 allocates `delta_state` as **69 separate `cudaMalloc`s** (`kimi_k3.cpp:721-724`). 276 memcpys at ~1.7 µs launch floor is ~0.47 ms of pure launch overhead per save. Allocating the recurrent state as **one contiguous arena per kind** (as `Qwen35Model` already does, `qwen35.cpp:282`, which is why its snapshot is 2 memcpys — `qwen35.cpp:1758-1761`) reduces this to 4 memcpys. **Do the arena consolidation before the ring.** It is a contained change to `kimi_k3_alloc_state` that preserves the existing per-layer pointer vector as offsets into the arena.

**Stage-4 optimisation (N ≥ 4): snapshot-once + scan replay.**

Cache the *scan inputs* (post-conv/L2 q,k,v, decay g, beta ≈ 196 KB per token per layer) instead of the states (2 MB per token per layer) — 10× cheaper — plus one snapshot of state@0. On partial rejection, restore state@0 and replay `j` delta-rule steps across 69 layers using the cached inputs. No projections, no MoE, no collectives.

| | cost |
|---|---|
| always: cache inputs, N=4 | 69 × 4 × 196 KB = 54 MB ≈ 0.37 ms |
| always: snapshot state@0 | 147.74 MB ≈ 1.0 ms |
| on partial rejection only | restore 1.0 ms + j × 1.88 ms |

Cheaper in expectation than the ring for N ≥ 4, but more code. Defer it until measurement shows the ring is a real cost.

**Rejected: all-or-nothing acceptance.** Accepting only the full block reduces rollback to a single restore, and the in-repo Qwen35 path takes exactly this shortcut (`qwen35.cpp:1789-1794`, "Full-block accept… skips rollback when accept==B-1"). But `E = 1 + N·p^N` is far worse than the chain: at p=0.8, N=4 it gives 2.64 vs 3.36 emitted/step — enough to miss every target. Acceptable only as a Stage-3 debugging control, not as the shipped design.

---

## 7. The core engine change

Concretely, what is new work versus what is a generalisation of existing code.

### 7.1 Stage 0 — fix `SPARKINFER_K3_KDA_QKVG_BATCH` (**critical path, do first**)

`kimi_k3_attn_batch_ok()` returns false for every KDA layer unless this env is set (`kimi_k3.cpp:1417`), and when it returns false the Attn phase **refuses** `n_tok > 1` (`kimi_k3.cpp:1718-1720`), forcing the driver into a per-token loop that re-reads the layer's projection weights for every token.

The flag is off by default **because it crashes**. The in-source note (`kimi_k3.cpp:1341-1359`) records that on-by-default produced `[k3] LAUNCH FAILED at layer 0, phase Attn: invalid argument` for every chunk ≥ 2 tokens, and it was left opt-in "until the fault in the batched KDA projection path is understood". `CHANGELOG.md:28-32` records the same. There is also a stale comment at `kimi_k3.cpp:1329` claiming it "now defaults ON" — the code disagrees; trust the code.

`invalid argument` on a launch is almost always a grid-dimension or scratch-extent problem. Two concrete suspects visible in the code:

- the four batched `proj_hb` calls at `kimi_k3.cpp:2475-2480` are reached via a `||` short-circuit at `:2493` whose ordering is explicitly load-bearing ("the right operand launches kernels", `:2485-2489`) — a candidate for the projections being issued in a state the scratch is not sized for;
- the single-row scratch buffers `conv_q/k/v` (`kimi_k3.cpp:1115-1117`), `beta_out` (96 wide, `:1132`) and `delta_out` (`:1133`) are **deliberately one row** because they live inside the per-token loop. If any batched projection writes into them at `n_tok` rows, that is both the crash and, worse, a silent corruption if the geometry happens to be legal.

**Stakes: 32.29 GB of 61.92 GB dense bytes (52.1%).** Without this fix the achievable speedup roughly halves — the amortisable fraction drops from 92.9% to ~40% of the byte budget, and the break-even rises correspondingly.

### 7.2 Stage 1 — a batched LM head (**genuinely new work**)

A K-token verify needs logits at all K positions. **Nothing in the tree computes logits for more than one position.** `k3_proj_f32` takes one activation vector and has no row axis (`kernels/include/sparkinfer/kernels/kimi_k3.h:986-987`); `rank.logits` is allocated for exactly one row (`kimi_k3_tp.cpp:474-483`); the chunk driver produces no logits at all (`kimi_k3_tp.h:315`).

A K-iteration loop over `k3_proj_f32` would re-stream the 2.349 GB F16 head **per position** — 16 ms per extra token, doubling `m` and pushing break-even from 0.093 to 0.19. **That is the difference between this design working and not working.** It must be a real batched projection.

The layer code already has a batched projection helper (`k3_proj_q8act_tok_f32`, used as `proj_tok` at `kimi_k3.cpp:2063-2064`), but the head is **F16**, not Q8_0. Either extend the batched path to F16 weights or write a small F16 GEMM for the head. Also: `rank.logits` must grow to `K × vocab` (K × 640 KB), and the D2H at `kimi_k3_dist_forward.cpp:450-473` must transfer K rows.

Do this together with head-band sharding (§4.5) — same code, and it converts a 2.349 GB rank0-only cost into a 0.783 GB three-way-shared one.

### 7.3 Stage 2 — `kimi_k3_dist_forward_batch()`

A new entry point beside `kimi_k3_dist_forward_token`, structurally the dist-path twin of `kimi_k3_tp_prefill_chunk` (`kimi_k3_tp.cpp:2692`). **Most of this is transcription, not invention.** Per layer:

| requirement | status | reference |
|---|---|---|
| `Attn` over K positions, MLA | **exists** — all projections/norms/absorb/gate batched; the attention kernel itself stays per-token by design (split geometry follows each token's own `n_ctx`) | `kimi_k3.cpp:2822-2992`, rationale `:2927-2942` |
| `Attn` over K positions, KDA | **exists, gated** — projections batch, scan stays per-token | §7.1 |
| `FfnPartial` / MoE routing for K tokens | **exists** — router takes `n_tok` (`:3216-3219`); expert-major regroup groups tokens by expert (`:3306-3317`); per-token loop fallback (`:3321-3333`) | — |
| batched MoE supports IQ1_S | **yes** — `ggml_type == 19` accepted; `latent 3584 % 256 == 0`, `ffn 1536 % 256 == 0` both pass | `k3_kernels.cu:5189-5190` |
| `FfnFinish` for K tokens | **exists**, with two deliberate per-token loops (`k3_add3_f32` fused tail, stride mismatch) | `kimi_k3.cpp:3471-3545`, `:3502-3510` |
| **one all-reduce per phase per layer over K× payload** | **new in the dist path** | see below |
| contiguous K-long position vector | **new in the dist path** | pattern at `kimi_k3_tp.cpp:3110`, `:3129-3134` |
| per-token residual banks | **new in the dist path** | §6.1 |

**The all-reduce is the one place a wrong port is silently wrong rather than broken.** `kimi_k3_partial_buffer()` takes no token axis and returns **one token's width** (`kimi_k3.cpp:1584-1612`). The current dist code passes that count straight to `coll.allreduce_f32` (`kimi_k3_dist_forward.cpp:83-98`). A K-token batch using it unchanged would **reduce 1/K of the payload and return success.** The chunk driver instead computes the payload itself — `attn_payload = nb·H`, `moe_payload = nb·(expert_latent + hidden)` (`kimi_k3_tp.cpp:2999-3000`) — and points the phase at a chunk-wide base via `kimi_k3_swap_partial_buffer` (`:3207`, `:3279-3284`). **Copy that pattern exactly.**

This is where the amortisation is realised: **185 collectives per pass instead of 185 per token.**

### 7.4 Stage 3 — rank protocol for K tokens

`Message` carries one `int token_id` (`rank_protocol.h:97`). Two options:

- **Simplest:** send K `Token` messages, then one batched step. Control-plane TCP only, no data path; K small. Requires relaxing the "no later sequence before StepDone" gate (`rank_protocol.h:218-219`).
- **Cleaner:** add a `Token` variant carrying a count and K ids. The protocol already has precedent for out-of-band signalling through `token_id` — the KV-reset sentinel is `-2` (`kimi_k3_dist_generate.cpp:106`).

Prefer the second; it is a small message-format change and avoids weakening the sequence gate.

### 7.5 Stage 4 — drafting and accept/reject on rank 0

Host-side in `kimi_k3_dist_generate.cpp`'s decode loop (`:404-414`): maintain the token history, run the n-gram lookup, call `forward_batch`, compare `argmax` per position against the drafts, commit the accepted prefix plus one bonus token, roll back state and position, continue. Workers need no drafting logic — they follow the protocol.

### 7.6 Memory budget

| item | per rank |
|---|---:|
| FFN scratch (already allocated, `SPARKINFER_K3_PREFILL_CHUNK` default 64) | ~101 MB |
| KDA state history ring, N=3 | 443 MB |
| K× residual banks, K=4 | 0.9 MB |
| K× logits, K=4 | 2.6 MB |
| **total added** | **< 0.5 GB** |

Against ~10 GB free per node. Not a constraint. (Scratch is already sized for `cap=64` regardless of the `n_tok` any call passes — `kimi_k3.cpp:1074-1075` — so no growth there.)

---

## 8. Staged implementation plan

Ordered to put the riskiest unknown first and to get the **measurement that validates the whole model** before any drafting code is written.

Note on prototyping: **there is no single-rank prototype path for the real model** — 330 GB does not fit on one 121 GB Spark. Stages 0–1 should be validated on a **depth-capped config** (`n_layers` reduced; the code explicitly supports a caller lowering it, `kimi_k3_config.h:61-70`) on one node, or against the existing chunked-prefill driver which already exercises the batched code at B=64.

| stage | work | gate / evidence required |
|---|---|---|
| **0** | Diagnose and fix the `KDA_QKVG_BATCH` launch failure. Add an equivalence test in the style of `patches/.../new-tests/kimi_k3_tp_kda_check.cpp`. | `kimi_k3_forward_layer_phase(n_tok=K)` on a KDA layer is **bit-identical** to K sequential calls, K=2..8. Flag flipped on by default. |
| **0b** | Consolidate `delta_state` + conv state into one contiguous arena per kind (Qwen35 pattern, `qwen35.cpp:282`). | Existing decode unchanged, bit-identical. 4 memcpys instead of 276. |
| **1** | Batched LM head (K logit rows, one weight pass) + head-band sharding in the dist path. | K-row logits bit-identical to K single-row calls. **Standalone win: ~6% step-time** — bank it. |
| **2** | `kimi_k3_dist_forward_batch()` — per-token pos vector, per-token res banks, chunk-wide partial buffers, one AR per phase per layer. **No drafting yet.** | (a) logits from `forward_batch(K)` match K sequential `forward_token` calls; (b) **measure `step(K)` for K=1..8 and fit `m` empirically.** |
| **3** | KDA state history ring + rollback; position/KV rollback. | `forward_batch(K)` → rollback to j → `forward_token` produces the same state and logits as the pure sequential path, for every j ∈ [0,K]. |
| **4** | Rank-protocol K-token step; n-gram drafter and greedy accept/reject on rank 0. | End-to-end generation **token-identical to non-speculative greedy** on a fixed prompt set. |
| **5** | Measure per-position acceptance on real workloads (code/agentic vs prose); tune N per workload against `p_i > m/T`. | A speed table under the multi-prompt median protocol, not the sweep's fast signal. |

**Stage 2's measurement is the decision point.** If the fitted `m` is materially above ~20 ms, the acceptance requirements in §5.3 move out of reach for n-gram drafting and the project should stop and re-plan rather than build Stages 3–5.

---

## 9. Riskiest unknowns

Stated plainly rather than glossed.

1. **The `KDA_QKVG_BATCH` defect is undiagnosed.** It gates 52.1% of the amortisable bytes. If the root cause turns out to be structural rather than a launch-geometry bug, the whole design's ceiling roughly halves. **Highest risk, and it is Stage 0 precisely for that reason.**

2. **Batched verify may not be numerically identical to sequential decode — and Track D already hit exactly this.** On the H200 stack the M=1 and M=N+1 paths did not produce identical logits, and with ~5% of positions inside 0.125 nats (plus a literal exact tie within 60 tokens) that flipped tokens in every 256-token generation (`current.md:295-340`). This engine is **more** exposed, not less: the known open issue in `current-tp3-sparks.md` is that top-16 expert-set overlap against the oracle is only 6–9/16 even where hidden states match at 0.99998, because the rank-4-to-16 routing tail is a near-tie lottery under IQ1_S. Any numerical difference between the K=1 and K>1 paths will flip experts and therefore tokens. **Consequences:** the "lossless under greedy" claim may not survive; measured acceptance may be depressed by the engine disagreeing with *itself*. This must be **measured at Stage 2**, and the acceptance criterion may need to be distributional rather than exact-match — the same gate defect Track D had to retire.

3. **`m` is modelled, not measured.** The 14 ms figure rests on the byte model (which §4.4 validates to 2%) plus assumptions about launch overhead and NCCL latency-boundedness. Stage 2 measures it directly. Everything downstream depends on this number.

4. **Prompt-lookup acceptance on this workload is unknown.** No measurement exists for n-gram acceptance on K3 output. The §5.3 table says what is *needed*; nothing here says what will be *achieved*. This is the largest source of uncertainty in the final speed number, and it is cheap to bound early — an offline study over logged generations, no GPU required, would de-risk the whole project before Stage 2.

5. **The expert-union model may be optimistic or pessimistic.** `u(K)` assumes independent routing; correlated routing between adjacent tokens makes it sublinear (good). But the batched expert-major regroup declines below a token floor and for unsupported types (`k3_kernels.cu:5186-5199`), falling back to a per-token loop — correct, but with more launches than modelled.

6. **A latent trap for a future caller.** The prefill-tile path in the FFN phases has **no `n_tok == 1` exclusion**, unlike the Attn phase which guards it explicitly at `kimi_k3.cpp:2159`. `normed2_dst = ffn_pt->normed2 + prefill_tok*H` is a single row that `rms_norm_f32(..., n_tok, H, H)` at `:3099-3100` would write `n_tok` rows into, clobbering neighbouring tile rows. Unreachable today (only `kimi_k3_tp_prefill` sets `prefill_tile`, and it drives one token at a time). A speculative driver that ever sets both would get fluent, wrong output. **Add the guard while in the area.**

7. **`k3_kda_chunk_prefill` must never reach the decode path.** It has no runtime callers today. It materialises no per-token states and would make rollback impossible (§6.3).

---

## 10. Alternatives considered

| option | verdict |
|---|---|
| **Batching independent requests** | Already near-linear on aggregate throughput (dense amortises identically) and Track D measured 3.02× at batch 8. **Does nothing for single-stream latency**, which is the 15 tok/s goal. Worth shipping separately if the goal is ever re-framed as server throughput. |
| **TP4 (the idle `spark-b610`)** | Lowers `T` for everyone: rank0 bytes fall to 18.43 GB → ~125 ms → ~8.0 tok/s before speculation, and it lowers the break-even further. But the TP4 profile shows collectives rising to ~40% of the step vs ~26% on TP3, and TP4 measured worse in practice. Re-test *after* Stage 2, since a batched verify amortises collectives and changes TP4's arithmetic in its favour. |
| **More kernel-flag tuning** | Closed. 31/31 configs run; best is +20%; stacking the winners is worse than the single best. Tops out ~6.5–7.5 tok/s. |
| **Adapting `Inferact/Kimi-K3-DSpark`** | Not adaptable — different architecture, distilled against full-precision K3, and no memory headroom on a Spark. Its *methodology* (break-even rule, per-position acceptance curves, workload-entropy dependence) is used throughout this document; its *numbers* are not imported. |
| **Distilling a draft against the IQ1_S target** | The only route to 15 tok/s on general prose. Large separate project. `DFlashDraftModel` plumbing already exists in-engine if it is ever undertaken. |

---

## Appendix — figures and their sources

| figure | value | source |
|---|---|---|
| GGUF total | 330,160,710,016 B | header parse, 9 shards, rank0 |
| routed experts, total | 265.893 GB | `ffn_{gate,up,down}_exps` × 92 layers, IQ1_S |
| dense read per token | 61.919 GB | total − experts − `token_embd` |
| active experts per token | 4.748 GB | 16/896 × 3 matrices × 92 layers |
| **dense share of decode bytes** | **92.88%** | derived |
| rank0 bytes per token | 23.788 GB | (total − head)/3 + head |
| effective B/W @ 6.1713 tok/s | 146.8 GB/s | derived |
| bytes restored by patch 0019 | 15.443 GB | shexp + attn_gate + q_a + routed_norm |
| predicted vs measured pre/post-fix ratio | 1.301 vs 1.327 | §4.4 |
| KDA state, per rank | 147.74 MB | 69 layers × (6.29 MB + 0.44 MB) ÷ 3 |
| marginal verify cost `m` | ~13.8 ms (use 15) | §5.1 |
| **break-even acceptance** | **0.093** | `m / T` |
| KDA-gated dense bytes | 32.286 GB (52.1%) | §4.2 |
