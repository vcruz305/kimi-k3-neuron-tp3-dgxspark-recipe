# Details — benchmarks, geometry, full run steps, non-claims

Full supporting data for the top-level [`README.md`](README.md). Everything here is
measured on real hardware, not modeled.

> **Historical-results warning:** the TP3/TP4 throughput tables below predate the patch
> 0019 correctness fix and were measured while the distributed engine silently skipped
> real computation. Keep them for investigation history only; do not cite them as current
> performance. The current corrected-engine bracketed measurement is in README's
> [Speculative decoding](README.md#speculative-decoding-experimental-opt-in) section and
> `evidence/specdec-serve-merge-RECEIPT.md`.

---

## Benchmarks (measured on real Sparks)

Hardware: NVIDIA **DGX Spark** (GB10, SM121, 48 SMs). Weights on **local NVMe** (not NFS/sshfs).
Flags: `max_ctx=8192`, `SPARKINFER_K3_MOE_WEPS=0`, `SPARKINFER_K3_GRAPH=0`, `NCCL_NVLS_ENABLE=0`.
Binary: `kimi_k3_dist_generate` (multi-prompt + KV reset `-2`).

### Load time (patch 0013)

| | Before 0013 | After 0013 |
|---|---:|---:|
| Per-rank load (3-Spark TP3) | ~30–60 min | **~5m45s** |

Root cause (confirmed via `nsys` CUDA API trace, not guesswork): every tensor upload in
`gguf.cpp`/`kimi_k3.cpp` did `cudaMemcpy`/`cudaMemcpy2D` straight out of a plain `mmap`.
Pageable (unregistered) host memory forces the CUDA driver through a staged bounce-buffer
copy instead of DMA'ing directly — on GB10 this was **99.5% of load-phase CUDA API time**,
individual calls up to 5s each for tensors that should transfer in milliseconds. Disk I/O
was ruled out first (raw sequential shard read measured 2.2–18 GB/s — nowhere near the
bottleneck). Fix pins each tensor's host memory immediately before its H2D copy
(`cudaHostRegister`/`cudaHostUnregister`, RAII, best-effort), scoped **per tensor** rather
than per shard — see `### 0013` in [`APPLY.md`](APPLY.md) for why per-shard registration
was tried first and rejected (ENOMEM: GGUF split shards all stay mmapped concurrently for
the life of the load, so pinning whole shards tries to lock the ~330 GB model at once).
Decode throughput is unaffected (fix only touches the load-time copy path).

### 3× Spark — TP3 `AllExpertsFfnWidth` (FFN 512/512/512 · ~113 GiB/rank)

Full write-up: [`docs/TP3-SPEED-RESULTS.md`](docs/TP3-SPEED-RESULTS.md).

| Run | decode tok/s | notes | vs RPC 2.85 |
|-----|-------------:|-------|------------:|
| Clean e2e | **5.67** | n=32 | **1.99×** |
| Longer single window | **6.21** | n=128 single-shot | **2.18×** |
| **Multi-prompt + syncfix median** | **6.84** | drop p0 · p95 **7.08** · mean **6.85** | **2.40×** |
| llama.cpp RPC layer-split (baseline) | **2.85** | — | 1.0× |

Syncfix multi-prompt (n=128, 6 prompts): p0 6.74 · p1 **7.08** · p2 6.75 · p3 6.92 · p4 6.79 · p5 6.84
Delta vs 6.21: **+0.63 t/s (~+10%)**. Finish clean · `moe_ffn_local=512`.

Decode profile (steady, n=10): **attn 36.5%** · MoE ffn_partial **23.5%** · coll **~26%** — see [`docs/TP3-DECODE-PROFILE.md`](docs/TP3-DECODE-PROFILE.md).

CUDA graphs (`SPARKINFER_K3_GRAPH=1`) A/B'd same-session on TP3 (6-prompt median): 6.98 eager
vs 7.01 graph — **no meaningful win** (within noise), same conclusion as TP4 below. Kernel
launch overhead is not the bottleneck at either TP width.

### 4× Spark — TP4 `ExpertFfn2D` eg=2/fs=2 (FFN 768/768 · ~84 GiB/rank)

Multi-prompt · n-predict=128 · 6 prompts · **drop prompt0** for median. Full tables:
[`docs/TP4-SPEED-RESULTS.md`](docs/TP4-SPEED-RESULTS.md).

| Config | Median (ex p0) | Mean (all) | Peak | notes |
|--------|---------------:|-----------:|-----:|-------|
| Eager seal | **7.90** | 7.60 | 8.02 | first clean 4-rank |
| GRAPH=1 | 7.84 | 7.85 | 7.89 | **no win** — keep GRAPH=0 |
| **+ stream syncfix** | **7.945** | **7.894** | **8.040** | worker no per-token sync; async `d_pos` / logits D2H |

Syncfix per-prompt (best so far): p0 7.90 · p1 **8.04** · p2 7.70 · p3 8.03 · p4 7.75 · p5 7.95
**p95_ex0 ≈ 8.04** · vs RPC 2.85 **~2.79×** · vs TP3 6.21 **~1.28×** · delta vs 7.90 seal **+0.045 t/s**.

**Finish:** OK finished clean on all four ranks (78f1 / 9f73 / 366f / b610).

### Fabric NCCL receipt (4 Sparks, 2026-08-08)

| | |
|--|--|
| Transport | **RoCE/IB** (`Using network IB`) — not TCP |
| p50 allreduce @ 28 672 / 43 008 B | **~50 / ~49 µs** (10k iters, all ranks PASS) |
| ×185 coll/token | **~9.1 ms/token** collective-only (~**7%** of 7.90 t/s wall) |
| GDR | **Unsupported on DGX Spark GB10** — `nvidia_peermem` → EINVAL; NVIDIA docs: no GPUDirect RDMA / peermem on unified-memory Spark. **GDR 0 is expected.** |
| Env A/B (GDR_LEVEL, few channels, PROTO=LL) | **No meaningful win** — keep defaults |

Blind NCCL tuning and GDR enablement are **not** productive levers on Spark. The ~185
attn/moe allreduce calls per token each have a strict data dependency on the previous one
via the transformer residual stream, so they cannot be merged/fused across layers without
a different TP sharding strategy.

### Forecast / ceiling

| Band | tok/s |
|------|------:|
| Forecast likely (TP3) | 4.5–6.0 |
| Forecast stretch | 6.5–7.5 |
| TP3 measured median | **6.84** (p95 7.08; single-shot was 6.21) |
| TP4 measured median | **7.945** best (seal 7.90; above prior stretch) |

Honest: single-stream exact TP still has a hard ceiling. **~10+ t/s** needs more than flag polish
(batching, different engine path, graphs with verified parity, fabric wins). See
[`docs/SPARK-TP3-PERFORMANCE-FORECAST.md`](docs/SPARK-TP3-PERFORMANCE-FORECAST.md).

**Context:** TP3 ~8K practical. TP4 has more headroom (~84 GiB/rank) but still not 1M.
**Quality:** raw `--prompt-ids` speed benches can look degenerate — use chat-template tokenization for quality.

---

## Geometry

| World | Plan | Experts / FFN | ~weights/rank |
|------:|------|---------------|--------------:|
| **3** | `AllExpertsFfnWidth` | 896 · 512/512/512 | ~113 GiB |
| **4** | `ExpertFfn2D` eg=2 fs=2 | 448/group · 768/768 | ~84 GiB |

Do **not** hot-add a 4th rank to a running TP3 job — relaunch with `--world 4`.

---

## Full run steps

### 0. Prerequisites

- 3 or 4 Sparks, SSH, fabric (RoCE preferred), CUDA stack aligned
- **≥320 GB local NVMe free per node** for the GGUF
- HF access to the gated model

### 1. Local model copy (each Spark — critical)

**Do not** load weights over NFS/sshfs for production benches. Map is fast; remote page faults kill load time.

```bash
pip install -U "huggingface_hub[hf_xet]"
hf auth login
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p $HOME/models/kimi-k3-neuron-iq1s-local
hf download vcruz305/Kimi-K3-Neuron-IQ1S-GGUF \
  --local-dir $HOME/models/kimi-k3-neuron-iq1s-local \
  --include "*.gguf" --include "k3_chat_template.jinja"
```

Entry shard:

```text
$HOME/models/kimi-k3-neuron-iq1s-local/k3-neuron-iq1s-00001-of-00009.gguf
```

### 2. Build + apply the verified chain through 0025

See **[`APPLY.md`](APPLY.md)** and use its exact 27-patch order from the pinned base.
Build targets should include `kimi_k3_dist_generate` and ship:

```text
kimi_k3_dist_generate
libsparkinfer_runtime.so
libsparkinfer_moe.so
libnccl.so*   # if not on system path
```

Copy the full `dist/` tree to **every** rank (missing `.so` on one node aborts that rank).

### 3. Topology example (4 Sparks)

| Rank | Role | Example fabric |
|-----:|------|----------------|
| 0 | `--listen 0.0.0.0:29500` | 10.10.10.2 |
| 1–3 | `--coord 10.10.10.2:29500` | .4 / .6 / .8 |

Use the **fabric** IP that workers can reach (not only the management NIC).

### 4. Launch (TP4 multi-prompt)

Env on every rank:

```bash
export LD_LIBRARY_PATH=$HOME/k3-tp3/dist:$LD_LIBRARY_PATH
export SPARKINFER_K3_MOE_WEPS=0
export SPARKINFER_K3_GRAPH=0
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=0
MODEL=$HOME/models/kimi-k3-neuron-iq1s-local/k3-neuron-iq1s-00001-of-00009.gguf
```

Rank 0:

```bash
./kimi_k3_dist_generate \
  --rank 0 --world 4 --listen 0.0.0.0:29500 \
  --model "$MODEL" \
  --prompts-file ./prompts_ids.txt \
  --n-predict 128 --max-ctx 8192
```

Ranks 1–3:

```bash
./kimi_k3_dist_generate \
  --rank R --world 4 --coord 10.10.10.2:29500 \
  --model "$MODEL" --max-ctx 8192
```

**TP3:** same with `--world 3` and three hosts.

### 5. Multi-prompt file + KV reset

`--prompts-file`: one CSV token-id line per prompt. Model loads **once**.
Between prompts the binary broadcasts KV reset sentinel **token id = -2** (protocol allowlist).

Report **median tok/s dropping prompt0** (cold window after load).

### 6. Speed knobs (after eager is sealed)

1. Local NVMe weights (done)
2. Multi-prompt / keep ranks warm
3. NCCL microbench (`k3_dist_nccl_microbench`) — RDMA vs TCP
4. `SPARKINFER_K3_GRAPH=1` A/B — already measured **no win** on either TP3 or TP4, kept for reference only
5. Do not thrash NCCL env blindly
6. ~50 `SPARKINFER_K3_*` env-gated kernel micro-opts exist in the runtime source (grep `std::getenv("SPARKINFER_K3_`) — most were measured and promoted to default-ON on the reference hardware; on a new GPU generation, sweep disabling each individually before assuming they still help

---

## Docs

| File | Topic |
|------|--------|
| [`APPLY.md`](APPLY.md) | Patch apply order |
| [`docs/OPERATOR-3-AND-4-SPARK.md`](docs/OPERATOR-3-AND-4-SPARK.md) | 3- and 4-Spark geometry |
| [`docs/SPARK-TP3-PERFORMANCE-FORECAST.md`](docs/SPARK-TP3-PERFORMANCE-FORECAST.md) | tok/s bands / ceilings |
| [`docs/TP3-SPEED-RESULTS.md`](docs/TP3-SPEED-RESULTS.md) | TP3 multi-prompt + syncfix median **6.84** |
| [`docs/TP3-DECODE-PROFILE.md`](docs/TP3-DECODE-PROFILE.md) | TP3 profile: attn **36.5%** · MoE **23.5%** · coll **~26%** |
| [`docs/TP4-SPEED-RESULTS.md`](docs/TP4-SPEED-RESULTS.md) | TP4 profile, NCCL, GRAPH A/B, syncfix median **7.945** |
| [`docs/TP3-EXPLAINED-AND-FIXED.md`](docs/TP3-EXPLAINED-AND-FIXED.md) | TP3 design |
| [`docs/TP3-KERNEL-FLAG-SWEEP.md`](docs/TP3-KERNEL-FLAG-SWEEP.md) | All 27 `SPARKINFER_K3_*` kernel flags tested on GB10 — what helped (`KDA_FUSE=0`, +20%), what didn't, what not to re-try |

---

## Related: vLLM path

Separate recipe: [`vcruz305/kimi-k3-neuron-tp3-vllm-recipe`](https://github.com/vcruz305/kimi-k3-neuron-tp3-vllm-recipe) — qualified on **3×H200** (~34 t/s graph target-only).

Tested directly on the physical 3-Spark fleet (Ray-orchestrated vLLM + GGUF plugin, with the
zero-copy `weight_utils.py` fix ported in): measured to be at or beyond the genuine physical
memory ceiling at the current quantization width (k=1536) — confirmed by an independent
watchdog tripping at literal 0 bytes free, Ray's own OOM monitor firing separately, and a real
host wedge at the most aggressive settings. Not a config/tuning gap. An intermediate prune
width (k=1280) is the next thing to try there, not further memory-margin tuning.

**This SparkInfer path does not have that ceiling** (~113 GiB/rank at TP3, comfortably under
GB10's 121 GiB) and is proven working end-to-end — use this recipe unless/until the vLLM path
clears its memory deficit.

## Non-claims

- Not a multi-user OpenAI-compatible server
- Not proven long-context parity vs llama.cpp
- Not 40 t/s on Spark from this path
- Not 1M context on 3–4 Sparks full resident
- Speed benches ≠ quality benches without chat-template prompts

## License

Patches follow upstream SparkInfer / project license terms in the base repo.
