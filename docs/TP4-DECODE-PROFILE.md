# TP4 decode-step profile (Step 1)

**Config:** world=4 · local GGUF · GRAPH=0 · WEPS=0 · max_ctx=8192 · n-predict=32  
**Profiler:** `SPARKINFER_K3_DIST_PROFILE=1` CUDA events per phase (steady tokens; skip 4; n=10 samples rank0)  
**Note:** profile inserts `cudaEventSynchronize` per phase → wall **~194 ms/token** inflated vs production **~126.6 ms** @ 7.90 t/s. **Use % shares** for ROI; ms scaled to prod for intuition.

## Average breakdown (rank0, n=10)

| Phase | ms (profile) | % | ms @ 7.90 t/s scale | 
|-------|-------------:|--:|--------------------:|
| **attn compute** | 53.00 | **27.3%** | 34.5 |
| **ffn_partial (MoE)** | 39.45 | **20.3%** | 25.7 |
| **attn allreduce** (n=93) | 39.01 | **20.1%** | 25.4 |
| **moe allreduce** (n=92) | 37.72 | **19.4%** | 24.6 |
| ffn_finish | 12.80 | 6.6% | 8.3 |
| head (norm+lm) | 12.22 | 6.3% | 8.0 |
| embed | 0.018 | 0.0% | 0.01 |
| sync+logits D2H | 0.049 | 0.0% | 0.03 |
| **TOTAL** | 194.26 | 100% | 126.6 |

## Grouped

| Group | % profile | Notes |
|-------|----------:|-------|
| **Collectives (attn_ar + moe_ar)** | **39.5%** | 185 AR/token; ~415 µs/AR under load (>> standalone ~50 µs microbench — larger partials and/or in-model contention) |
| **MoE compute (ffn_partial + ffn_finish)** | **26.9%** | Expert path |
| **Attn compute** | **27.3%** | |
| Head + embed + D2H | 6.3% | D2H negligible |

## Top 3 time sinks
1. **attn compute** — 27.3%
2. **ffn_partial (MoE)** — 20.3%
3. **attn allreduce** — 20.1%  (tie-ish with moe_ar 19.4%)

**Dominates:** split decision — **compute (attn+MoE) ~54%** vs **collectives ~39%**. Neither is a rounding error. Coll share is **much higher in-model** than the 7% microbench story implied.

## Highest-ROI next lever (Step 2)
**Attack collectives under real partial sizes + overlap with compute** (not GDR, not NCCL flag spam):
1. Measure partial buffer element counts (attn vs moe) — explain 50µs → ~415µs gap  
2. Prefer **overlap**: pipeline next-layer independent work with in-flight AR if recipe dependencies allow  
3. Or **fuse/reduce AR frequency** only if geometry-correct  
4. Parallel track: **ffn_partial kernel** (2nd sink) if coll work stalls  

**Do not** start TP layout changes until one coll-or-MoE A/B is measured.

## Distance to 10 t/s
| Rate | ms/token | need cut from 126.6 ms |
|-----:|---------:|------------------------:|
| 7.90 (now) | 126.6 | — |
| 9.0 | 111.1 | **−15.5 ms (−12%)** |
| 10.0 | 100.0 | **−26.6 ms (−21%)** |

Clearing ~half of coll (25 ms scaled) → ~9.84 t/s ballpark if no regression.

## Raw
`k3-thunder/.scratch/tp4-profile-r0.txt`
