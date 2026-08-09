# TP4 speed results (4× DGX Spark) — profile + syncfix A/B

**Date:** 2026-08-08 / 2026-08-09  
**Fleet:** spark-78f1 (r0), 9f73 (r1), 366f (r2), b610 (r3) · coord `10.10.10.2:29500`  
**Model:** local NVMe `kimi-k3-neuron-iq1s-local` · max_ctx=8192 · WEPS=0 · GRAPH=0 · NVLS=0  

GDR / peermem: **closed** on Spark GB10 (unsupported). See README fabric section.

---

## Scoreboard

| Config | Median (drop p0) | Mean (all 6) | Peak | vs RPC 2.85 |
|--------|-----------------:|-------------:|-----:|------------:|
| TP3 · 3 Sparks | 6.21 | — | — | 2.18× |
| TP4 eager seal | **7.90** | 7.60 | 8.02 | ~2.77× |
| TP4 GRAPH=1 | 7.84 | 7.85 | 7.89 | ~2.75× (−0.06) |
| **TP4 + stream syncfix** | **7.945** | **7.894** | **8.040** | **~2.79×** |

### Syncfix A/B (n-predict=128 · 6 prompts · multi-prompt)

| Prompt | decode tok/s |
|-------:|-------------:|
| 0 | 7.904 |
| 1 | **8.040** |
| 2 | 7.700 |
| 3 | 8.027 |
| 4 | 7.751 |
| 5 | 7.945 |

- **median_ex0 = 7.945** · **p95_ex0 = 8.040** · mean_all = 7.894  
- delta vs 7.90 seal: **+0.045 t/s** (incremental; keep fix)  
- distance to 10.0: **−2.055 t/s**

### What the syncfix changed (`kimi_k3_dist_forward.cpp`)

1. **Workers:** no production per-token `cudaStreamSynchronize` (profile mode still syncs).  
2. **Position:** `cudaMemcpyAsync(d_pos)` on the compute stream — not `kimi_k3_set_position` default-stream H2D.  
3. **Rank0 logits:** `cudaMemcpyAsync` + single stream sync.

---

## Decode-step profile (steady tokens, n=10 rank0)

`SPARKINFER_K3_DIST_PROFILE=1` · CUDA events per phase · skip 4 cold tokens.  
Profiler forces per-phase sync → wall inflated (~194 ms vs ~126 ms prod); **use % shares**.

| Phase | % |
|--------|--:|
| attn compute | **27.3** |
| ffn_partial (MoE) | **20.3** |
| attn allreduce ×93 | **20.1** |
| moe allreduce ×92 | **19.4** |
| ffn_finish | 6.6 |
| head | 6.3 |
| embed / logits D2H | ~0 |

| Group | % |
|--------|--:|
| Compute (attn + MoE ffn) | **~54** |
| Collectives (both AR) | **~40** (high-side; includes skew wait) |
| Head + other | ~7 |

Standalone NCCL microbench ~**49–50 µs**/AR @ 28–43 KB; in-model AR looks slower mainly as **wait/imbalance**, not broken RoCE.

---

## Next levers (toward 10 t/s)

Need ~**−21%** wall vs 7.9–7.95 band (~100 ms/token).

1. **Attn / MoE kernels** (top compute sinks) inside recipe geometry  
2. True **compute↔NCCL overlap** where deps allow (dual-stream)  
3. TP layout only after measured A/Bs  
4. **Not:** GDR, NCCL flag spam, GRAPH=1 on current dist path  

---

## Hygiene

- Drop prompt0 for median · n-predict 128 · ≥5 prompts same load  
- Local NVMe weights · keep ranks warm with `--prompts-file`  
