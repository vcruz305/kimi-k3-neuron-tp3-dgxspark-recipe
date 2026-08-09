# TP3 speed results (3× DGX Spark)

**Date:** 2026-08-09  
**Fleet:** spark-78f1 (r0), 9f73 (r1), 366f (r2) · coord `10.10.10.2:29500`  
**Geometry:** `AllExpertsFfnWidth` · FFN 512/512/512 · ~113 GiB/rank · `moe_ffn_local=512`  
**Model:** local NVMe `kimi-k3-neuron-iq1s-local` · max_ctx=8192 · WEPS=0 · GRAPH=0  

---

## Scoreboard

| Config | Median (drop p0) | Mean / notes | Peak | vs RPC 2.85 |
|--------|-----------------:|--------------|-----:|------------:|
| Early clean e2e | — | 5.67 @ n=32 | — | 1.99× |
| Longer single window | **6.21** | n=128 single-shot | — | 2.18× |
| **+ multi-prompt hygiene + syncfix** | **6.84** | mean_all **6.85** · p95 **7.08** | **7.08** | **2.40×** |

### Syncfix multi-prompt median (n-predict=128 · 6 prompts)

| Prompt | decode tok/s |
|-------:|-------------:|
| 0 | 6.735 |
| 1 | **7.083** |
| 2 | 6.750 |
| 3 | 6.919 |
| 4 | 6.791 |
| 5 | 6.839 |

- **median_ex0 = 6.8386** · **p95_ex0 = 7.083** · mean_all = 6.8528  
- delta vs prior best 6.21: **+0.63 t/s (~+10%)**  
- vs TP4 best 7.945: **−1.11 t/s**  
- distance to 10.0: **−3.16 t/s**  
- Finish: **OK finished clean**

### Syncfix (same as TP4 path)

1. Workers: no production per-token `cudaStreamSynchronize`  
2. Position: `cudaMemcpyAsync(d_pos)` on compute stream  
3. Rank0 logits: async D2H + one stream sync  

---

## vs TP4

| | TP3 (3 Sparks) | TP4 (4 Sparks) |
|--|---------------:|---------------:|
| Best median | **6.84** | **7.95** |
| Weight/rank | ~113 GiB | ~84 GiB |
| Plan | AllExpertsFfnWidth | ExpertFfn2D |

4th Spark gain is mostly geometry/residency, not a 3-node free lunch.

---

## Next (ROI)

1. ~~Multi-prompt + syncfix median~~ **done (6.84)**  
2. TP3 decode-step profile (sinks on world=3)  
3. Kernels / overlap / engine only if profile still compute-bound  

GDR / NCCL flag spam: **closed** on Spark GB10.

---

## Decode-step profile (2026-08-09)

See [`TP3-DECODE-PROFILE.md`](TP3-DECODE-PROFILE.md).

| Group | % (n=10) |
|--------|---------:|
| attn compute | **36.5** |
| MoE ffn_partial | **23.5** |
| Collectives (attn_ar + moe_ar) | **~26** |
| ffn_finish + head | ~14 |

**Next:** attn / MoE kernels (or vLLM engine path). Not GDR/NCCL spam.
