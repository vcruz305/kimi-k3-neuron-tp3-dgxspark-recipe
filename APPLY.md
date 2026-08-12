# APPLY.md — SparkInfer K3 multi-Spark patch chain

**Base:** `7a9b77a043596157d74e4af376cf9f29f68ce368`  
**Tip:** `main` — **git am 0001–0026** (28 patches including 0015/0018a/0019a)

> **If you apply only one patch, make it 0019.** Everything before it runs an
> engine that silently skips the shared experts, the routed norm and the MLA
> q-lora/attn-gate weights, and therefore produces confident wrong output on
> every prompt. Any tok/s number measured without 0019 is measuring an engine
> doing less work than the model actually requires.

## Apply

```bash
git clone https://github.com/gittensor-ai-lab/sparkinfer-k3.git && cd sparkinfer-k3
git checkout -B k3-tp3 7a9b77a043596157d74e4af376cf9f29f68ce368
git clone --depth 1 --branch main \
  https://github.com/vcruz305/kimi-k3-neuron-tp3-dgxspark-recipe /tmp/k3-recipe

for p in \
  /tmp/k3-recipe/patches/sparkinfer/0001-tp-plan-K3-all-expert-FFN-width-shards-for-TP3.patch \
  /tmp/k3-recipe/patches/sparkinfer/0002-wire-tp3-all-expert-width-init-ffn-prefill.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0003-tp-three-host-rank-bootstrap-protocol.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0004-tp-distributed-rank-transport-and-tp3-tp4-plans.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0005-tp-rank-local-plan-and-nccl-microbench.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0006-tp-rank-local-gguf-load-api-and-moe-budget.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0007-tp-distributed-eager-forward-and-dist-generate.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0008-tp-fix-rank0-loadready-oneshot-and-load-before-ready.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0009-tp-f16-token-embd-output-and-dist-embed.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0010-tp-first-forward-stall-instrumentation.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0011-tp-finish-deadlock-and-decode-tps.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0012-tp-finish-ack-after-wait-token.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0013-tp-pin-per-tensor-host-memory-for-fast-load.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0015-tp-multi-prompt-file-and-kv-reset-sentinel.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0014-tp-serve-mode-for-dynamic-prompt-requests.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0016-tp-repetition-guard-for-greedy-decode.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0017-test-fix-three-bugs-in-kimi-k3-state-check.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0018-tp-attention-shard-kill-switches-for-dist-path.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0018a-tp-attention-shard-equivalence-tests.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0019-tp-probe-gguf-capability-flags-in-dist-rank-load.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0019a-tp-dist-phase-profiler.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0020-tp-per-layer-dump-instrumentation-for-dist-path.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0021-tp-specdec-stages-0-2-batched-verify.diff \
  /tmp/k3-recipe/patches/sparkinfer/next/0022-tp-specdec-ar-count-determinism-diagnostic.diff \
  /tmp/k3-recipe/patches/sparkinfer/next/0023-tp-specdec-stage3-kda-state-rollback.diff \
  /tmp/k3-recipe/patches/sparkinfer/next/0024-tp-specdec-stage4-ngram-drafter-and-ktoken-protocol.diff \
  /tmp/k3-recipe/patches/sparkinfer/next/0025-tp-specdec-tuning-confidence-gate-and-dist-head-band.diff \
  /tmp/k3-recipe/patches/sparkinfer/next/0026-tp-specdec-recursive-majority-drafter.diff
do git am "$p"; done

cmake -S runtime -B build -DSPARKINFER_TP=ON
cmake --build build -j"$(nproc)" --target kimi_k3_dist_generate \
  tp_rank_local_loader_cpu_test tp_dist_generate_protocol_cpu_test \
  kimi_k3_tp_kda_check kimi_k3_tp_width_check kimi_k3_kda_batch_check \
  kimi_k3_spec_draft_check kimi_k3_tune_check
```

## Series status — what actually applies

A clean-room audit (fresh clone of the pinned base, `git am` in order, nothing
reused from a working tree) established the following. Run it yourself before
trusting any of it.

**Verified applying cleanly: the complete 28-patch chain through 0026.** The chain
through 0025 was fresh-clone audited with zero dirty files and a byte-identical fleet
tree. Patch 0026 was then plain-`git am` applied to that exact clean tip, built on GB10,
correctness-gated, and fleet-benchmarked.

1. 0015 now includes the two protocol-side allowances for the `-2` distributed
   KV-reset sentinel. Without them 0024 could not apply and workers would reject a
   reset as an out-of-vocabulary token.
2. 0018a installs and registers the KDA/width equivalence harnesses required by the
   later CMake hunks.
3. 0021–0025 were rebased onto the real 0014/0016 serve/guard line. The repetition
   guard remains active on ordinary decode, but speculative verification deliberately
   bypasses it: 0019 fixed the collapse that the guard originally treated, while
   applying it during speculation would erase literal repetition, prompt lookup's
   measured best case.

The merged fleet build completed with CUDA architecture 121 and passed: loader
38/38, rank protocol 17/17, drafter/oracle **1091/1091**, tuning/head-band 29/29,
KDA head-shard relL2 9.956e-08, MoE width-shard relL2 1.135e-07, and batched KDA
bit-identical for K=2..8. An end-to-end `--serve` + protocol-v2 test sent two
requests (streaming first, distributed KV reset before the second) and shut down
cleanly. See `evidence/specdec-serve-merge-RECEIPT.md`.

**Root cause of this whole class of defect**: patches were captured as diffs from
a live working tree that always contained more than the series recorded — the
fleet's source tree is a git repo whose last commit is 0012, with everything from
0013 onward uncommitted. 0015 and 0019a are two such features, now recovered.
When adding a patch, capture it against a tree built by applying the series, not
against the working tree.

## Series

| # | Purpose |
|---|---|
| 0001–0007 | planner → dist generate |
| 0008 | LoadReady one-shot (GB10 verified) |
| 0009 | F16 embed (GB10 verified) |
| 0010 | first-forward breadcrumbs → **GENERATE_PASS** |
| 0011 | finish deadlock fix + decode tok/s lines |
| 0012 | FinishAck after wait_token (teardown hang) |
| **0013** | **pin per-tensor host memory — load time fix (primary path)** |
| 0014 | rank0 `--serve HOST:PORT` mode — dynamic prompt requests for the API wrapper (see `api-server/`) |
| **0015** | **rank0 `--prompts-file F` — one CSV token-id line per prompt, run back to back against a single weight load, with a reserved token id `-2` (`kKvResetSentinel`) broadcast between prompts so every rank resets KV + KDA state in lockstep through the existing step barrier (no new frame type, no protocol bump). Per-prompt timing lines are tagged `prompt=N` and a `multi_summary` line reports mean decode tok/s. Single-prompt behavior is byte-for-byte unchanged. Applies between 0013 and 0014 — 0014 was diffed against a tree that already contained this, which is why 0014 carries `kKvResetSentinel` as unchanged *context* and could not apply without it** |
| 0016 | repetition guard for ordinary greedy decode — applies on top of 0014. **Its commit message's root-cause theory is WRONG** (see 0019): it blamed collapse on IQ1_S + greedy decoding; the real cause was the 0019 capability-flags bug. The merged 0024 path keeps this safety net on ordinary decode and deliberately excludes it from speculative verification so literal repetition remains draftable |
| 0017 | test fix — three bugs in `kimi_k3_state_check` |
| 0018 | attention shard kill switches (`SPARKINFER_K3_SHARD_KDA` / `_MLA`) for the dist path. **The stored patch was structurally corrupt until repaired** — its second hunk header declared 27 new-side lines over a 26-line body, so git rejected it as `corrupt patch` against *any* tree, not as a context mismatch. The old side (7 lines) matches the real file exactly, so exactly one `+` line was missing; it was restored as the blank line separating the new block from `out->weights.shard = ...`, matching 0019's identical idiom in the same function. That reconstruction is an inference, not a recovered original — it is the only edit in the series not verified against a surviving tree, because **0018 was never applied to the production tree**: rank0's live `kimi_k3_dist_rank.cpp` contains no shard-policy block and still has the `out->weights.policy = opt.policy;` line 0018 deletes. It compiles and links — the repaired hunk is in the built `libsparkinfer_runtime.so`. The corruption shipped in `c0237e2` and had been live in the published recipe ever since; it went unnoticed because nothing re-applied the series from the pinned base until this audit |
| **0019** | **capability-flag probe in dist rank load — CORRECTNESS FIX, the one that made the engine produce right answers (details below). Does NOT require 0018 — an earlier note here claimed it did, but 0019's hunk is offset against the *pre*-0018 file and the live fleet tree carries 0019 without 0018, so it applies directly on 0017** |
| **0018a** | **the two attention-shard equivalence harnesses from the 0019 investigation — `kimi_k3_tp_kda_check` (KDA/MLA head sharding) and `kimi_k3_tp_width_check` (MoE FFN width sharding), each asking whether splitting work across ranks and recombining reproduces the unsharded result to float-reassociation tolerance. This is what 0018's `SHARD_KDA`/`_MLA` switches exist to A/B, and what measured relL2 8.2e-09 at the first MLA layer. Their sources sat in `patches/sparkinfer/next/new-tests/` with nothing to install them and no `add_executable` lines anywhere, so they were unbuildable from a clean checkout — and 0021/0024/0025 carry both target names as unchanged *context*, which is why 0021 failed on `CMakeLists.txt`. Both binaries build from the clean-room tree** |
| **0019a** | **`SPARKINFER_K3_DIST_PROFILE=1` — per-phase CUDA-event timings for the dist decode step (embed / attn / attn_ar / ffn_partial / moe_ar / ffn_finish / head / sync_d2h), each with a percentage and, for the collectives, their element counts. Skips a warmup window then samples a bounded number of steady tokens (`_SKIP`/`_EVERY`/`_MAX`). This is the instrumentation the TP3 decode-time breakdown came from (attention ~36.5%, collectives ~26.2%, MoE ffn_partial ~23.5%). Inert when unset — `timed()` degrades to a plain call with no events and no syncs. Note an ACTIVE profile run serializes each phase, so its absolute tok/s is not a valid benchmark number; read the percentages. Applies between 0019 and 0020 — like 0015, it was used on the fleet but never captured, and 0020 was diffed against a tree that already had it (0020 carries `run_embed` as unchanged *context*)** |
| 0020 | per-layer dump instrumentation for the dist path (`K3_DUMP_DIR` / `K3_SUBTAP_LAYER`) — debug tooling only, inert unless the env vars are set |
| 0021–0025 | **Rebased onto 0014/0016, verified via clean `git am`, fleet build, equivalence tests, five-run benchmark, and serve-mode integration test** |
| 0021 | speculative decoding Stages 0-2 — fixes a crashing batched-KDA flag (unlocks 52% of dense bytes for any batched path), consolidates KDA state into contiguous arenas, adds a batched F16 LM head, and a new `kimi_k3_dist_forward_batch()` entry point. No user-facing effect on its own (details below) |
| 0022 | NCCL all-reduce count-determinism diagnostic + the clean marginal-cost measurement protocol used to validate 0021-2's economics. Inert unless `SPARKINFER_K3_AR_PAD` is set. Requires 0021 |
| 0023 | Stage 3 — KDA recurrent-state rollback (a history ring on 0021's arenas), so a rejected speculative draft can be cleanly unwound. Requires 0021 (shares `kimi_k3_dist_forward.cpp`/`kimi_k3.cpp` hunks) |
| **0024** | **speculative decoding Stage 4 — the actual n-gram drafter, accept/reject loop, and rank-protocol v2 (carries a whole verify batch + deferred rollback per step). USER-FACING: `--spec-draft K`. Requires 0021+0023 (shares `kimi_k3_dist_forward.cpp`, `kimi_k3_dist_generate.cpp`, `rank_protocol.h`)** |
| **0025** | **match-confidence gating plus distributed LM-head banding. `--spec-min-occur 2 --spec-require-agree` filters false-positive n-grams; `SPARKINFER_K3_DIST_HEAD_BAND=1` gives each rank an uneven vocab band (54614/54613/54613 for TP3), zero-fills the rest, then reuses all-reduce. 129/129 dumped logits were bitwise identical to the unbanded path. Combined measured 8.5447 tok/s, +40.61% over the average of bracketing baselines** |
| **0026** | **opt-in recursive-majority continuation: `--spec-majority 2/3` re-predicts each draft position from the highest-order recurring tail with a two-thirds continuation vote instead of copying one stale span. Real TP3 candidate -> baseline -> candidate bracket: structured generation 10.3002 / 6.4272 / 11.3231 tok/s (candidate average 10.8117, +68.2%); 95/95 drafts accepted on both candidate sides. Default remains off, preserving 0025 behavior** |

### 0011
- `finish()` no longer holds `mu_` while waiting (unblocks rx FinishAcks)
- injects rank0 `FinishAck` once
- prints `decode_tok_s` / `prefill_tok_s` + `generated_ids` (fflush)

### 0013
- Root cause (via `nsys`): `cudaMemcpy`/`cudaMemcpy2D` from the plain `mmap` in
  `gguf.cpp` forced the driver through a pageable-memory bounce-buffer copy for
  every tensor upload — 99.5% of load-phase CUDA API time, individual calls up
  to 5s each. Not disk I/O (raw sequential shard read measured 2.2–18 GB/s).
- Fix: `ScopedHostRegister` (RAII `cudaHostRegister`/`cudaHostUnregister`)
  wraps each tensor's H2D copy, scoped **per tensor** — not per shard, since all
  9 GGUF split shards stay mmapped for the life of the load and pinning whole
  shards tries to lock the ~330 GB model at once (measured: 6/9 shards failed
  to register with ENOMEM before this fix was narrowed to per-tensor).
- Measured on the real 3-Spark fleet: full load+generate **~30–60 min/rank →
  ~5m45s**. Decode unaffected (5.99 tok/s) — this only touches the load-time
  copy path, never the forward pass.

### 0019 — correctness fix (read this before trusting any earlier tok/s number)

- Root cause: `KimiK3DistRank::opt` (a `K3PlanOptions`) was never assigned in
  `kimi_k3_dist_rank.cpp`. Every flag in it defaults to false and the file only
  ever read it. The function's own `opt` PARAMETER is a different type
  (`KimiK3DistRankLoadOptions`), which is what hid the omission.
- Effect: `kimi_k3_load_weights_scoped` gates uploads on those flags, so tensors
  that are present in the GGUF were never loaded and the ops consuming them were
  skipped — shared experts, routed RMSNorm, MLA q-lora, MLA attn gate. Nothing
  faults; the affected layers just compute a wrong answer.
- Why no test caught it: all 19 `runtime/examples/*` harnesses probe the GGUF and
  set these flags themselves. Only the production distributed path did not, so
  no SparkInfer-vs-SparkInfer comparison could ever expose it.
- Localized by bisecting the residual stream layer-by-layer against an external
  oracle — llama.cpp PR #26185 (`4158193ed`, build b10381), the reference Kimi-K3
  implementation — on byte-identical token ids. Layer 0 matched at cos 0.999999;
  layer 1 was the first divergence at cos 0.874, at prompt position 0 (one token,
  no KV, no recurrent state), which ruled out any state/indexing explanation.
  Layer 0 is the only dense-FFN layer, so layer 0 vs 1 differ in exactly one
  thing: dense FFN vs latent MoE.
- After the fix, same oracle and ids: all 93 layers match at cos >= 0.99987 at
  position 0, the induction test returns the oracle's exact next token, and
  `The capital of France is` decodes to ` Paris. Paris is the capital of France.`
- Cost: the corrected engine does real work it previously skipped, so decode
  tok/s DROPS. Measured on the 3-Spark fleet: **6.89 → ~5.0-6.2 tok/s** depending
  on run shape (32-token single prompt vs 3-prompt). That drop is the fix
  working. Re-baseline any speed target against a post-0019 engine.
- Known gap, NOT fixed here: at prompt positions > 0 the match against the
  reference is looser (cosine dipping to ~0.79-0.86 around layers 51-80,
  recovering by ~90-92). Suspected near-tied top-16 expert selection flipping
  under IQ1_S quantization noise — unconfirmed. Output is coherent and
  first-token-exact; it is **not** verified byte-identical beyond that.

### 0021-0026 — speculative decoding, confidence gating, majority drafting, and head banding

- **What it is**: rank0 drafts tokens by matching the current sequence against its own
  generation history (no second model), verifies the whole draft in one batched forward
  pass, and rolls back the recurrent state on partial rejection. Off by default —
  `--spec-draft K` (K in `[2,16]`) opts in; omitting it leaves behavior and performance
  unchanged.
- **Why 4 patches**: built and verified in stages, same discipline as 0017-0019, because
  this touches the same distributed-execution code that produced the 0019 incident.
  0021 = the infrastructure (batched forward pass, no drafting logic yet). 0022 = a
  diagnostic that found and explained a real numerical-fidelity gap between batched and
  sequential decode (root cause: NCCL's reduce-scatter partitioning depends on the
  element count, so summation order shifts with batch size K — not a bug, confirmed via
  a count-pinning diagnostic and a filler-token invariance test). 0023 = the state
  rollback, with a formal correctness argument (not just "seems to work"): rollback only
  needs to restore the state that produced the accepted logits (suffix independence +
  snapshot consistency), never a hypothetical bit-identical sequential run — which is
  good, because that would be uncomputable without redoing the sequential pass anyway.
  0024 = the actual drafter and the rank-protocol change to carry it.
- **Measured on real TP3 hardware** (64-token generations, 3 workload types). The
  original Stage 4 result used `--spec-draft 4 --spec-ngram-min 2`; the current
  candidate adds `--spec-ngram-max 3 --spec-min-occur 2 --spec-require-agree` and
  `SPARKINFER_K3_DIST_HEAD_BAND=1`:

  | workload | baseline tok/s | with speculative decoding | delta |
  |---|---:|---:|---:|
  | code / structured | 6.07 | 7.71 | +27% |
  | literal repetition | 5.73 | 10.51 | +83% |
  | freeform prose | 6.16 | 5.93 | **−3.7% (real regression, not noise)** |
  | mean | 5.99 | 8.05 | +34.5% |

  `K=8` measured *worse* than `K=4` — per-position acceptance collapses past match-depth
  3 regardless of batch size, so a bigger speculative batch just costs more with nothing
  left to fill it. Do not assume larger K is better.

  Current merged-tree five-run sequence, with baselines bracketing the candidates:

  | config | code | repetition | prose | mean | vs avg baseline |
  |---|---:|---:|---:|---:|---:|
  | baseline before | 6.1612 | 5.7251 | 6.1622 | 6.0162 | — |
  | confidence-gated K=4 | — | — | — | 8.0591 | +32.61% |
  | head-band only | 6.4718 | 6.3173 | 6.3888 | 6.3926 | +5.19% |
  | **confidence K=4 + head-band** | **8.1651** | **10.9018** | **6.5672** | **8.5447** | **+40.61%** |
  | baseline after | 6.1477 | 6.1166 | 6.1496 | 6.1379 | — |

  Average bracketing baseline: 6.07705 tok/s. Draft acceptance in both gated
  configurations was 59/64 = 0.9219; the serve/guard merge did not change it.
- **0026 result**: with K=4, n=[1,8], at least 2 occurrences, recursive 2/3 majority,
  and head banding, structured generation measured 10.3002 and 11.3231 tok/s on the two
  candidate sides of a matched 6.4272 baseline (10.8117 candidate average, +68.2%). Both
  candidate runs accepted 95/95 drafts. The bracketed four-request mean was 9.2345, so
  do not turn this into a claim that every workload or the general mean exceeds 10.
- **Correctness**: verified as "always a valid greedy decoding trajectory," not
  byte-identical to non-speculative decoding (see the NCCL finding above — closing that
  gap needs a count-*and*-offset-invariant collective, not yet built, tracked as a
  follow-up). On a literal-repetition prompt, output is bit-for-bit identical to
  non-speculative decoding. On code/prose it diverges only where the model was already
  near-tied between two choices, and stays coherent throughout. 1000+ automated checks:
  a drafter-vs-independent-reference test over 4000 randomized trials, a full
  protocol round-trip test including rejection cases, a full speculative-loop-vs-oracle
  proof with all three accept arms proven to fire, plus the pre-existing 568-check
  rank-protocol test updated for the v2 wire format.
- **Protocol version bump, on purpose**: 0024 bumps the rank-to-rank protocol from 1 to
  2. A rank running an older binary refuses a v2 frame loudly ("unsupported protocol
  version") instead of silently running a mismatched step — deploying a stale binary to
  one rank is exactly the failure class 0019 taught this project to guard against.
- **What we chose not to build (yet)**: a trained draft model (the official
  `Inferact/Kimi-K3-DSpark` is distilled against full-precision Kimi-K3, not this
  IQ1_S-compressed target, and there's no spare memory on a Spark for a second model's
  weights anyway) and TP4. Full design rationale, break-even math, and the staged
  verification process: [`docs/SPECULATIVE-DECODING-DESIGN.md`](docs/SPECULATIVE-DECODING-DESIGN.md).
- **Apply note**: the rebased 0021–0026 patches are ordinary mail patches. Apply them
  with `git am` exactly as shown above; no whitespace override is required.

## TPS measurement run

```bash
# rank0 — prefer longer decode for stable tok/s
./kimi_k3_dist_generate --rank 0 --world 3 --listen 0.0.0.0:29500 \
  --model .../Kimi-K3-UD-IQ1_S-00001-of-00009.gguf \
  --prompt-ids 1,2,3 --n-predict 32 --max-ctx 8192 2> r0.stderr | tee r0.stdout

# ranks 1/2
./kimi_k3_dist_generate --rank {1|2} --world 3 --coord HOST:29500 \
  --model ... --max-ctx 8192 2> rN.stderr
```

Env: `SPARKINFER_K3_MOE_WEPS=0 SPARKINFER_K3_GRAPH=0 NCCL_NVLS_ENABLE=0` auto GID.

Report lines containing `[k3-dist][tps]` and `OK finished clean`.

## Claims
No multi-Spark tok/s until 0011 receipt. Load time is not decode tok/s.
No correctness claim without 0019, and no tok/s comparison across the 0019
boundary — a pre-0019 engine is faster only because it is skipping work.
Confirm from the run's own logs: rank0 stderr must show
`[k3-dist] caps probe(blk.3): q_lora=1 attn_gate=1 shexp=1 routed_norm=1`.

The clean-room audit covers the chain through **0025**, and 0026 was separately
plain-`git am` applied to that exact clean tip, built, correctness-gated, and measured on
the fleet. Do not extend that claim to later unverified working-tree changes.
