# Three-Spark TP3 recipe (Kimi-K3 Neuron IQ1_S)

> **⚠️ Superseded — historical, kept for reference only.** This describes the project's
> early bring-up state, before correctness was verified, before speed tuning, and before
> speculative decoding. It is no longer accurate: multi-Spark generation is not
> incomplete, it works, and the geometry/topology below is still a useful reference but
> the "Proven vs Missing" table is out of date. **Use [`README.md`](README.md) and
> [`APPLY.md`](APPLY.md) for the current, maintained recipe.**

**Preview bring-up recipe.** Full-model multi-Spark generation is incomplete.
Use this to align geometry, collectives, and agent ops — not to promise production tok/s.

## Goal

Run **one process per DGX Spark**, global ranks `0,1,2`, each on **local CUDA device 0**,
with **AllExpertsFfnWidth TP3**:

- all **896** routed expert identities on every rank;
- expert FFN width **1536 → 512/512/512** (256-block aligned);
- gate/up shard **ne1**, down shards **ne0**;
- NCCL via **`ncclCommInitRank`** (never `ncclCommInitAll` across machines);
- rank 0 holds full LM head + sampling for the first correct implementation;
- `SPARKINFER_K3_MOE_WEPS=0` until long-context parity exists.

## Topology

```text
Spark A  rank 0  local_device=0  coordinator + sampler + full LM head
Spark B  rank 1  local_device=0  worker
Spark C  rank 2  local_device=0  worker
         └── NCCL over RoCE / cluster fabric ──┘
         └── TCP control plane (Hello, NCCL id, Load, Token, Finish) ──┘
```

Memory plan (release 330 GB GGUF): **~112.85 GiB/rank**. On ~119 GiB usable Spark
HBM, margin is only ~6 GiB before OS/KV — start at **8K context**, watch swap.

## What is proven vs missing

| Proven | Missing for “supported product” |
|---|---|
| TP3 geometry + planner/executor wire-up (same-host) | Rank-local GGUF slice load on 3 hosts |
| Same-host TP3 ~42 tok/s on 3×H200, first-token parity | Distributed eager forward + token broadcast |
| Control-plane protocol + TCP framing (CPU tests) | RoCE microbench on **your** Sparks |
| `ncclCommInitRank` multi-process microbench | End-to-end generate CLI |
| Fail-closed quant preflight for release IQ1_S | Long-context KLD/PPL vs llama.cpp |

**Production today on 3 Sparks:** llama.cpp RPC **layer-split** (~2.85 tok/s historical).

**Forecast after full TP3 port:** **4.5–6.0 tok/s likely** at real 8K (not 40+).  
**Stop rule:** abandon single-stream TP3 if finished RDMA+graphs still **≤3.5 tok/s**.

## Agent / human checklist (in order)

1. **Inventory** — 3 Sparks SSH, RoCE/NCCL visible, clocks, driver, free HBM.  
2. **Health gate** every GPU — `scripts/gpu_health_gate.py health` (matmul + ECC).  
3. **NCCL 3-rank microbench** over fabric — payloads 28672 and 43008 bytes f32.  
4. **Apply SparkInfer patch chain** (see README).  
5. **Rank-local load smoke** — no full-model mirror; zero swap; ≥8 GiB MemAvailable.  
6. **One-token parity** vs llama.cpp (top-1).  
7. **Multi-token + long prompt** metrics.  
8. **Only then** decode tok/s vs layer-split baseline.

## Environment lore

```bash
export SPARKINFER_K3_MOE_WEPS=0          # full experts for exact gates
export SPARKINFER_TP_BACKEND=nccl
export NCCL_NVLS_ENABLE=0               # often required; harmless if unused
export SPARKINFER_K3_GRAPH=0            # off until eager parity
# Prefer per-token prefill when collective has no owned buffers (NCCL default)
```

## 4 Sparks (optional later)

Prefer **ExpertFfn2D** eg=2/fs=2 (448 experts × 768 FFN) when a fourth Spark +
supported switch exists — ~84 GiB/rank, more headroom. Same missing forward stack.

## Model download

```bash
pip install -U "huggingface_hub[hf_xet]"
hf auth login
export HF_XET_HIGH_PERFORMANCE=1
hf download vcruz305/Kimi-K3-GGUF --local-dir ./Kimi-K3-UD-IQ1_S \
  --include "*.gguf" --include "k3_chat_template.jinja"
```

Repo is **manual-gated** on Hugging Face by design.

## References in this package

- `SPARK-AGENT.md` — full agent operating contract  
- `docs/*` — design, forecast, RCA  
- `patches/sparkinfer/**` — apply chain  
- `scripts/*` — health + NCCL benches  
- `evidence/*` — measured receipts  
