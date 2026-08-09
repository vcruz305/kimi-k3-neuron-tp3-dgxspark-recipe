# TP3 decode-step profile (3× DGX Spark)

**Date:** 2026-08-09  
**Config:** world=3 · local NVMe GGUF · GRAPH=0 · WEPS=0 · max_ctx=8192 · syncfix binary  
**Profiler:** `SPARKINFER_K3_DIST_PROFILE=1` · skip 4 · **n=10** samples rank0  
**Note:** per-phase `cudaEventSynchronize` inflates wall (~208 ms vs ~146 ms @ 6.84 t/s). Use **% shares**.

## PROFILE_AVG n=10

```
attn=36.5%  ffn_partial=23.5%  attn_ar=13.1%  moe_ar=13.1%
ffn_finish=6.9%  head=6.8%  embed~0%  sync_d2h~0%
mean_n_attn_ar=93  mean_n_moe_ar=92
total_ms(profile)≈208
```

## Breakdown table

| Phase | % |
|--------|--:|
| **attn compute** | **36.5** |
| **ffn_partial (MoE)** | **23.5** |
| attn allreduce ×93 | 13.1 |
| moe allreduce ×92 | 13.1 |
| ffn_finish | 6.9 |
| head | 6.8 |
| embed / logits D2H | ~0 |

## Grouped

| Group | % |
|--------|--:|
| **Compute (attn + MoE ffn_p + ffn_f)** | **~67** |
| **Collectives (both AR)** | **~26** |
| Head + other | ~7 |

## Top sinks
1. **attn compute** — 36.5%  
2. **MoE ffn_partial** — 23.5%  
3. **collectives (tied)** — 13% each AR  

## vs TP4 profile (same profiler)

| | TP3 (3 Sparks) | TP4 (4 Sparks) |
|--|---------------:|---------------:|
| attn | **36.5%** | 27.3% |
| ffn_partial | 23.5% | 20.3% |
| coll (sum) | **~26%** | ~40% |

On TP3, **compute dominates harder**; fabric is secondary.

## Speed context
- Multi-prompt + syncfix median: **6.84** t/s (p95 **7.08**)
- Need ~**−31%** wall for 10 t/s from 6.84

## Next lever
Attack **attn kernels** first, then **MoE ffn_partial**.  
NCCL flag thrash / GDR: closed on Spark.  
Engine alternative: vLLM recipe (H200-qualified graphs; Sparks need multi-node TP).
