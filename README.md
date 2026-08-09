# Kimi-K3 Neuron — multi-Spark TP3 / TP4 bring-up package

**Status: working e2e on 3- and 4-Spark fleets — not a drop-in production API server.**

Patches **0001–0012** deliver load → multi-prompt generate → finish on **DGX Spark** with
rank-local GGUF residency and NCCL collectives.

| | |
|---|---|
| **This recipe** | [`github.com/vcruz305/kimi-k3-neuron-tp3-dgxspark-recipe`](https://github.com/vcruz305/kimi-k3-neuron-tp3-dgxspark-recipe) |
| **Model (GGUF)** | [`huggingface.co/vcruz305/Kimi-K3-Neuron-IQ1S-GGUF`](https://huggingface.co/vcruz305/Kimi-K3-Neuron-IQ1S-GGUF) (~330 GB / 307.49 GiB) |
| Upstream base | `gittensor-ai-lab/sparkinfer-k3` @ `7a9b77a043596157d74e4af376cf9f29f68ce368` |
| Patch tip | **0001–0012** — see [`APPLY.md`](APPLY.md) |

---

## Benchmarks (measured on real Sparks)

Hardware: NVIDIA **DGX Spark** (GB10). Weights on **local NVMe** (not NFS/sshfs).  
Flags: `max_ctx=8192`, `SPARKINFER_K3_MOE_WEPS=0`, `SPARKINFER_K3_GRAPH=0`, `NCCL_NVLS_ENABLE=0`.  
Binary: `kimi_k3_dist_generate` (multi-prompt + KV reset `-2`).

### 3× Spark — TP3 `AllExpertsFfnWidth` (FFN 512/512/512 · ~113 GiB/rank)

| Run | decode tok/s | tokens | vs RPC 2.85 |
|-----|-------------:|-------:|------------:|
| Clean e2e | **5.67** | 32 | **1.99×** |
| Longer window | **6.21** | 128 | **2.18×** |
| llama.cpp RPC layer-split (baseline) | **2.85** | — | 1.0× |

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
| p50 allreduce @ 28 672 / 43 008 B | **~50 / ~49 µs** (10k iters, all ranks PASS) |
| ×185 coll/token | **~9.1 ms/token** collective-only (~**7%** of 7.90 t/s wall) |
| GDR | **Unsupported on DGX Spark GB10** — `nvidia_peermem` → EINVAL; NVIDIA docs: no GPUDirect RDMA / peermem on unified-memory Spark. **GDR 0 is expected.** |
| Env A/B (GDR_LEVEL, few channels, PROTO=LL) | **No meaningful win** — keep defaults |

Blind NCCL tuning and GDR enablement are **not** next levers on Spark. Prefer **reduce/fuse collectives, overlap compute↔comm, kernels**, or supported TP layout changes.


### Forecast / ceiling

| Band | tok/s |
|------|------:|
| Forecast likely (TP3) | 4.5–6.0 |
| Forecast stretch | 6.5–7.5 |
| TP3 measured | **6.21** |
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

## How to run

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

### 2. Build + apply patches 0001–0012

See **[`APPLY.md`](APPLY.md)**. Build targets should include `kimi_k3_dist_generate` and ship:

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
4. `SPARKINFER_K3_GRAPH=1` A/B (same prompts; only after eager parity)  
5. Do not thrash NCCL env blindly  

---

## Docs

| File | Topic |
|------|--------|
| [`APPLY.md`](APPLY.md) | Patch apply order |
| [`docs/OPERATOR-3-AND-4-SPARK.md`](docs/OPERATOR-3-AND-4-SPARK.md) | 3- and 4-Spark geometry |
| [`docs/SPARK-TP3-PERFORMANCE-FORECAST.md`](docs/SPARK-TP3-PERFORMANCE-FORECAST.md) | tok/s bands / ceilings |
| [`docs/TP4-SPEED-RESULTS.md`](docs/TP4-SPEED-RESULTS.md) | TP4 profile, NCCL, GRAPH A/B, syncfix median **7.945** |
| [`docs/TP3-EXPLAINED-AND-FIXED.md`](docs/TP3-EXPLAINED-AND-FIXED.md) | TP3 design |

---

## Non-claims

- Not a multi-user OpenAI-compatible server  
- Not proven long-context parity vs llama.cpp  
- Not 40 t/s on Spark from this path  
- Not 1M context on 3–4 Sparks full resident  
- Speed benches ≠ quality benches without chat-template prompts  

## License

Patches follow upstream SparkInfer / project license terms in the base repo.
