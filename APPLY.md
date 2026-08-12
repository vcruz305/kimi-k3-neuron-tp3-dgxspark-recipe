# APPLY.md — SparkInfer K3 multi-Spark patch chain

**Base:** `7a9b77a043596157d74e4af376cf9f29f68ce368`  
**Tip:** `main` — **git am 0001–0024**

> **If you apply only one patch, make it 0019.** Everything before it runs an
> engine that silently skips the shared experts, the routed norm and the MLA
> q-lora/attn-gate weights, and therefore produces confident wrong output on
> every prompt. Any tok/s number measured without 0019 is measuring an engine
> doing less work than the model actually requires.

## Apply

```bash
git clone https://github.com/gittensor-ai-lab/sparkinfer-k3.git && cd sparkinfer-k3
git checkout -B k3-tp3 7a9b77a043596157d74e4af376cf9f29f68ce368
git clone --depth 1 --branch sparkinfer-tp3-phase3-loadready-fix \
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
  /tmp/k3-recipe/patches/sparkinfer/next/0014-tp-serve-mode-for-dynamic-prompt-requests.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0016-tp-repetition-guard-for-greedy-decode.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0017-test-fix-three-bugs-in-kimi-k3-state-check.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0018-tp-attention-shard-kill-switches-for-dist-path.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0019-tp-probe-gguf-capability-flags-in-dist-rank-load.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0020-tp-per-layer-dump-instrumentation-for-dist-path.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0021-tp-specdec-stages-0-2-batched-verify.diff \
  /tmp/k3-recipe/patches/sparkinfer/next/0022-tp-specdec-ar-count-determinism-diagnostic.diff \
  /tmp/k3-recipe/patches/sparkinfer/next/0023-tp-specdec-stage3-kda-state-rollback.diff \
  /tmp/k3-recipe/patches/sparkinfer/next/0024-tp-specdec-stage4-ngram-drafter-and-ktoken-protocol.diff
do git am "$p"; done

cmake -S runtime -B build -DSPARKINFER_TP=ON
cmake --build build -j"$(nproc)" --target kimi_k3_dist_generate \
  tp_rank_local_loader_cpu_test tp_dist_generate_protocol_cpu_test
```

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
| 0015 | *(reserved)* multi-prompt/`--prompts-file` + KV-reset — currently only exists as an uncommitted diff on rank0; needs to land between 0013 and 0014 once captured (tracked separately) |
| 0016 | repetition guard for greedy decode — applies on top of 0014. **Its commit message's root-cause theory is WRONG** (see 0019): it blamed the collapse-to-one-repeated-token failure on IQ1_S quantization + greedy decoding. The real cause was the 0019 capability-flags bug. The guard itself is still worth keeping as a decode-loop safety net, but it was treating a symptom |
| 0017 | test fix — three bugs in `kimi_k3_state_check` |
| 0018 | attention shard kill switches (`SPARKINFER_K3_SHARD_KDA` / `_MLA`) for the dist path |
| **0019** | **capability-flag probe in dist rank load — CORRECTNESS FIX, the one that made the engine produce right answers (details below). Requires 0018 (shares `kimi_k3_dist_rank.cpp` hunks)** |
| 0020 | per-layer dump instrumentation for the dist path (`K3_DUMP_DIR` / `K3_SUBTAP_LAYER`) — debug tooling only, inert unless the env vars are set |
| 0021 | speculative decoding Stages 0-2 — fixes a crashing batched-KDA flag (unlocks 52% of dense bytes for any batched path), consolidates KDA state into contiguous arenas, adds a batched F16 LM head, and a new `kimi_k3_dist_forward_batch()` entry point. No user-facing effect on its own (details below) |
| 0022 | NCCL all-reduce count-determinism diagnostic + the clean marginal-cost measurement protocol used to validate 0021-2's economics. Inert unless `SPARKINFER_K3_AR_PAD` is set. Requires 0021 |
| 0023 | Stage 3 — KDA recurrent-state rollback (a history ring on 0021's arenas), so a rejected speculative draft can be cleanly unwound. Requires 0021 (shares `kimi_k3_dist_forward.cpp`/`kimi_k3.cpp` hunks) |
| **0024** | **speculative decoding Stage 4 — the actual n-gram drafter, accept/reject loop, and rank-protocol v2 (carries a whole verify batch + deferred rollback per step). USER-FACING: `--spec-draft K`. Requires 0021+0023 (shares `kimi_k3_dist_forward.cpp`, `kimi_k3_dist_generate.cpp`, `rank_protocol.h`)** |

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

### 0021-0024 — speculative decoding (n-gram / prompt-lookup)

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
- **Measured on real TP3 hardware** (64-token generations, 3 workload types, best config
  `--spec-draft 4 --spec-ngram-min 2`):

  | workload | baseline tok/s | with speculative decoding | delta |
  |---|---:|---:|---:|
  | code / structured | 6.07 | 7.71 | +27% |
  | literal repetition | 5.73 | 10.51 | +83% |
  | freeform prose | 6.16 | 5.93 | **−3.7% (real regression, not noise)** |
  | mean | 5.99 | 8.05 | +34.5% |

  `K=8` measured *worse* than `K=4` — per-position acceptance collapses past match-depth
  3 regardless of batch size, so a bigger speculative batch just costs more with nothing
  left to fill it. Do not assume larger K is better.
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
- **CRLF note**: `runtime/src/models/kimi_k3_dist_forward.cpp` has pre-existing CRLF line
  endings inconsistent with the rest of the LF tree (contamination from an earlier
  Windows-side editor round-trip, not this series' doing). Patches 0021 and 0023 touch
  that file and carry the mixed endings through. `git apply` may warn or refuse on
  strict whitespace checking — use `git apply --ignore-whitespace`, which applies these
  cleanly (verified: reproduces the exact tree that was built, deployed, and tested on
  the live fleet).

## TPS measurement run

```bash
# rank0 — prefer longer decode for stable tok/s
./kimi_k3_dist_generate --rank 0 --world 3 --listen 0.0.0.0:29500 \
  --model .../k3-neuron-iq1s-00001-of-00009.gguf \
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
