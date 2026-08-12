# SparkInfer same-host TP3 — what it is, what broke, what works (2026-08-07)

## Explain TP3 like a systems diagram

**Tensor Parallelism (TP)** means: one model forward is split across GPUs **within a layer**.
Every rank sees the same tokens; each rank owns a **slice of the math**, then they
**all-reduce** partial results so everyone has the full activation before the next piece.

For this pruned Kimi-K3 GGUF:

| Axis | Full size | Problem for 3 GPUs | TP3 width solution |
|---|---:|---|---|
| Routed experts | **896** | `896 % 3 ≠ 0` — cannot give each rank the same expert count cleanly | **Do not split expert IDs** |
| Expert FFN width | **1536** | `1536 / 3 = 512` exactly, and 512 is 2×256 quant blocks | **Split FFN width** into bands `[0,512)`, `[512,1024)`, `[1024,1536)` |

So **AllExpertsFfnWidth TP3** means:

- Every GPU keeps **all 896 experts**
- Each GPU computes only **1/3 of each expert’s intermediate width**
- After MoE (and after sharded attention), ranks **NCCL all-reduce** the partials
- Resulting hidden state matches (within numeric noise) a single full-width compute

This is **not** the same as:

| Mode | What it splits | Network? |
|---|---|---|
| **Layer split** (SparkInfer `kimi_k3_generate` / llama RPC) | **Layers** across GPUs (pipeline-ish). Decode walks stages **serially** | Same host or RPC |
| **TP / AllExpertsFfnWidth** | **Tensors inside each layer** in parallel | Same host today (`ncclCommInitAll`) |
| **Distributed TP3 on 3 Sparks** | Same tensor idea, but **one process per machine** + `ncclCommInitRank` + RoCE | **Not built yet** |

### Why TP is faster than layer-split on 3× H200

Layer-split: GPU0 does layers 0–31, then GPU1 32–62, then GPU2 63–92 — **one GPU works at a time** per token.

TP3: all three GPUs work **every layer**, then sync. More of the 330 GB is active in parallel.

Measured this session (same prompt, release IQ1_S, WEPS=0, greedy):

| Path | Decode | First token |
|---|---:|---|
| Layer-split 3× H200 | **~24.2 tok/s** | `Paris` (17374) |
| TP3 AllExpertsFfnWidth 3× H200 | **~42.3 tok/s** | `Paris` (17374) |

≈ **1.75×** over layer-split on this box. **Not** a Spark/RDMA number.

## Root cause of the earlier “tile at token 0 failed”

1. Default prefill uses a **tile driver** (batch a few prompt tokens with a fancy phase-major schedule).
2. That tile driver **only** reduces via **owned-buffer** collectives (peer-oneshot / multimem “Mode B”).
3. On this pod, TP falls back to **plain NCCL**, which **does not own** those peer-mapped buffers.
4. Log line was honest: `collective has no owned buffers … running eager`, then the tile still called `allreduce_f32_owned_slot` and failed after loading ~100+ GiB/rank.

**Per-token prefill** already had the correct fallback: `allreduce_f32_group` for NCCL.  
So `SPARKINFER_K3_PREFILL=0` worked immediately.

### Fix shipped in the pod tree

In `kimi_k3_tp_prefill`: if the collective has no owned buffers, **do not arm the tile path**; print once and use per-token prefill. Default TP3+NCCL now works without env hacks (still needs `NCCL_NVLS_ENABLE=0` on this host).

Also required ops lore:

- `NCCL_NVLS_ENABLE=0` or `ncclCommInitAll` dies (same as earlier NCCL microprobe).
- Settle GPUs after failed loads; TP3 residency is ~110–120 GiB/rank on H200 143 GiB — tight.

## Correctness (one-token, same prompt ids)

Prompt: `The capital of France is` → token **17374** / ` Paris` on:

- llama.cpp (prior F16 parity)
- SparkInfer layer-split
- SparkInfer TP3

`compare_logits.py` (1 token):

| Pair | top-1 | mean KLD | RMS Δp |
|---|---:|---:|---:|
| TP3 vs layer-split | **1.0** | 0.0242 | 1.31e-4 |
| TP3 vs llama ref | **1.0** | 0.0166 | 1.28e-4 |

Not bit-identical (expected: TP reduction order + SparkInfer numeric paths). **Argmax agreed.**

Multi-token greedy text (TP3): `Paris. Paris is the capital of France`  
Layer-split chose a different continuation after the same first token (sampling-free but divergent paths after step 0) — compare **first-token** logits for parity, not last-of-N.

## What is still not done

- Fast **tile** prefill on NCCL (needs group-reduce tile implementation or Mode-B backend).
- **Distributed** three-Spark TP (rank bootstrap / `ncclCommInitRank` / RoCE).
- Long-context TP3 quality (PPL/KLD at ≥8K).
- matched-byte-v2 on SparkInfer (still fail-closed).

## How to run (pod)

```bash
export SPARKINFER_K3_MOE_WEPS=0
export SPARKINFER_TP_BACKEND=nccl
export NCCL_NVLS_ENABLE=0
export LD_LIBRARY_PATH=.../nvidia/nccl/lib:build-sm90-tp3width/runtime:$LD_LIBRARY_PATH

./build-sm90-tp3width/runtime/kimi_k3_tp_generate \
  /root/k3/model/Kimi-K3-UD-IQ1_S-00001-of-00009.gguf \
  --ids @prompt.ids --devices 0,1,2 --max-new 8 --ctx 2048 \
  --logits out.spkl
```

Tree: `/root/k3/sparkinfer-tp3-width-run`  
Artifacts: `/root/k3/sparkinfer-tp3-width-results/`
