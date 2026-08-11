# APPLY.md — SparkInfer K3 multi-Spark patch chain

**Base:** `7a9b77a043596157d74e4af376cf9f29f68ce368`  
**Tip:** `main` — **git am 0001–0020**

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
  /tmp/k3-recipe/patches/sparkinfer/next/0020-tp-per-layer-dump-instrumentation-for-dist-path.patch
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
