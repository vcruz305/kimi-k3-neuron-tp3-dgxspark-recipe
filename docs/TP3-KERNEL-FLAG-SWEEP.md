# TP3 kernel-flag sweep — what we tried, and what not to try again

**Date:** 2026-08-11. **Engine:** post-patch-0019 (correctness fix) SparkInfer, TP3 on 3× DGX Spark GB10 (SM121, 48 SMs). **Methodology:** single-shot fast signal (`--prompt-ids 1,2,3 --n-predict 32`), not the multi-prompt median used for headline speed claims elsewhere in this repo — treat these as **relative, directional** results, not precise absolute tok/s. Baseline: **5.15 tok/s**.

SparkInfer ships ~27 `SPARKINFER_K3_*` env-var kernel flags, almost all of them A/B toggles for an alternate CUDA kernel variant, measured and tuned on 8×H200 (132 SMs, ~4.8 TB/s bandwidth) at some point in the project's history. GB10 is a completely different machine — 48 SMs, ~273 GB/s unified memory, ARM64 — so none of those H200-era defaults were guaranteed to transfer. This sweep tested every flag that had a real, current effect on the codebase.

**Bottom line: one flag helps meaningfully (`KDA_FUSE=0`, +20%), a handful help modestly (+10-17%), and none of them stack — combining the top 5 winners measured *worse* than the single best flag alone (5.67 vs 6.17 tok/s). If you're tuning this engine for GB10, start and stop at `SPARKINFER_K3_KDA_FUSE=0`.**

---

## Results, sorted by impact

| flag | value tested | tok/s | vs baseline | what it actually does |
|---|---|---:|---:|---|
| — | baseline (all defaults) | 5.1534 | — | — |
| **`KDA_FUSE`** | **`=0`** | **6.1713** | **+20% (best)** | Disables a fused kernel that combines the KDA add + sigmoid gate into one launch; `=0` restores three separate launches (add, gate, `beta_sigmoid`). |
| `RES_1PASS` | `=0` | 6.0423 | +17% | Disables a single-pass residual-stream computation; `=0` restores the numerically-safer two-pass form. The paper trail says the reassociation is invisible to the top-1/KL quality gate — this flag is purely a speed/reassociation tradeoff, not a correctness one. |
| `KDACONV` | `=0` | 6.0176 | +17% | Disables a fusion of 5 separate KDA convolution-related kernel launches into fewer; `=0` restores the five-launch form. |
| `ADD3` | `=0` | 5.9723 | +16% | Disables a fused 3-operand add kernel (folds what would otherwise be 2 separate adds into 1); `=0` forces the unfused path. Declines as a whole — never partially fused. |
| `KDA_IP` | `=0` | 5.9561 | +16% | Disables an "in-place" state-update variant for the KDA decode step (the recurrent state update that multiplies by the decay gate `exp(g) < 1`); `=0` restores the original tiled kernel. |
| `MOE_BATCH` | `=0` | 5.8944 | +14% | Disables the batched MoE dispatch path entirely, forcing the per-token expert-loop fallback for every call. This is the same flag investigated in depth during the speculative-decoding work — its floor for engaging is `n_tok >= 2`. |
| `B2WIDE` | `=0` | 5.8708 | +14% | Disables a 32-bit-load Q8_0 block accessor (measured +0.30 tok/s on the original H200 tuning session); `=0` restores the 16-bit-pair read. Default is ON. |
| `RMSG` | `=1` | 5.8949 | +14% | Sets RMSNorm's grid-spread group-size parameter to 1 (down from a wider default) — controls how RMSNorm's output sweep is spread across the grid when its source and destination buffers differ. |
| `KDA_CPB` | `=8` | 5.9336 | +15% | KDA "chunks per block" — how many recurrent-state chunks one CUDA block processes. Source comment: below 4, a block can't cover its own memory latency; above the sweep spot, the grid stops covering the device. `=8` tests above the historical H200 sweep's optimum. |
| `PROJF32_WIDE` | `=0` | 5.8281 | +13% | Disables a wider thread-block configuration for F32 projection kernels (used when the grid would otherwise be too small to fill the device); `=0` restores the narrower `<<<N,128>>>` launch. |
| `PROJ_1BAR` | `=0` | 5.8263 | +13% | Disables a single-barrier-synchronization epilogue for Q8-quantized projection kernels; `=0` routes every projection back through the reference multi-barrier path. |
| `QUANT_WARP` | `=0` | 5.7801 | +12% | Disables a warp-level (32× more parallel launches) variant of a quantization-scale kernel; `=0` restores the reference shape (859 launches/token/rank at 1-2 blocks each). Both forms are bit-identical (the scan is a max over magnitudes, order-independent) — this is a pure occupancy tradeoff. |
| `WARP_TARGET` | `=1306` (+ `PROJ_ROWBUDGET=0`) | 5.7430 | +11% | Overrides the target-warp-count operating point (default 1792) for 4 projection kernel shapes that don't naturally reach memory bandwidth on their own; the default was picked from the 2 shapes that *do* reach it, on H200. `1306` was one alternate probe point. |
| `KDA_DEFER` | `=0` | 5.7365 | +11% | Disables deferred write-back of KDA's recurrent state: normally the 768 KiB state write is deferred past the 6 KiB output write with a programmatic-completion signal between them (to hide the bigger write behind the smaller one); `=0` restores the original order. |
| `RMSU` | `=0` | 5.7447 | +11% | Disables a fused/unified RMSNorm kernel variant (parsed alongside `RMSG` in the same function). |
| `FUSED4_ROWS` | `=8` | 5.2405\* | +2%\* | Controls how many output rows a "fused4" projection kernel processes together, which also sets floating-point accumulation order. `=4` (K3_CPB_8's neighbor test) did better (+15%); `=8` was close to a null result. |
| `KDA_CPB` | `=2` | 5.7531 | +12% | Same parameter as above, tested at the opposite end (below the historical sweep spot). |
| `KDA_VT` | `=0` | 5.4985 | +7% | Disables a tiled-launch variant for KDA's "value transform" step; `=0` restores the single-tile launch. Source note: one-block-per-head strands the device once KDA is head-sharded across ranks — this flag exists specifically to fix that under TP sharding, which makes its modest measured gain here worth double-checking at a different TP width. |
| `RES_WIDE` | `=0` | 5.5033 | +7% | Disables a wider block/thread configuration for the residual-stream reduction; `=0` restores a shared-256 baseline (fewer levels on the reduction tree). |
| `RMSG` | `=28` | 5.5915 | +8% | Same parameter as above, tested at a much wider group size. |
| `ROUTER_REG` | `=0` | 5.5517 | +8% | Disables a register-based specialized MoE router kernel for the decode (single-token) case; `=0` restores the general kernel. Only engages when every one of its shape assumptions holds. |
| `MOE_DOWN_WARPS` | `=8` | 5.5149 | +7% | Sets the warp count for the MoE "down"-projection kernel specifically. |
| `HEAD_1BAR` | `=0` | 5.3752 | +4% | Disables a single-barrier epilogue for the LM head projection kernel specifically (the one `k3_proj_f32` caller still on the "scored" path at TP=8); `=0` restores the general `block_sum` epilogue. Default ON. |
| `IQ1S_PACK` | `=0` | 5.3820 | +4% | Disables reading the IQ1_S dequant lattice through a packed 4 KB table; `=0` uses the unpacked form. Measured +0.72 tok/s over 3 sessions on the original H200 tuning — default ON, and this GB10 result suggests that gain **does not transfer** (disabling it barely changed anything, meaning the packed table isn't clearly helping here either). |
| `FASTB2W` | `=1` | 5.2809 | +3% | Swaps the weight-side Q8_0 accessor for an alternate read pattern in the batched projection kernel. Default appears OFF; this GB10 result is close to a null effect. |
| `ROUTER_FAST` | `=0` | 5.1048 | ~0% | Disables an optimized MoE router kernel path (with a documented top-k tie-break guarantee: ties broken value-desc/index-asc so fold order can't change the winner); `=0` restores the original. **This is the one flag whose H200-era default appears to still be correct on GB10** — disabling it was flat, meaning the optimization is still pulling its weight here. |
| `ROUTER_2P` | `=1` | 5.5736 | +8% | Enables a "2-pass" MoE router kernel variant, vs. the default "per-pass-rendezvous" kernel. |
| `PROJ_G` | `=4` | 5.8442 | +13% | Sets a grouping/tiling parameter for the batched Q8 projection kernel. |

\* `FUSED4_ROWS=4` and `=8` were both tested; `=4` (+15%, in the `KDA_CPB` region of the table above) outperformed `=8` (+2%) — the accumulation-order-dependent kernel is sensitive to this specific value, not just "bigger is better."

## Stacked test — the one important negative result

Combining the top 5 individually-winning flags (`KDA_FUSE=0 RES_1PASS=0 KDACONV=0 MOE_BATCH=0 RMSU=0`) measured **5.6714 tok/s — worse than `KDA_FUSE=0` alone (6.1713)**, and only marginally better than several single flags. These kernels are not independent — several touch overlapping parts of the KDA/residual/MoE code paths, and disabling multiple related fusions simultaneously removes shared benefit rather than compounding it.

**Do not assume flag wins stack. Test combinations explicitly if you try more of them.**

## What not to try again

- **Don't bother re-testing individual flags in isolation beyond this list** — all 27 tested flags plus their reasonable value ranges are covered above. Any single-flag win beyond `KDA_FUSE=0` is marginal (≤17%) and none of it is free money waiting to be found by re-running the same sweep.
- **Don't assume stacking wins.** Proven false above — verify any new combination end-to-end before trusting it.
- **Don't expect H200-tuned defaults to transfer to GB10 in general.** Most of the flags that helped here were **disabling** something the H200 tuning session had turned on by default (`KDA_FUSE`, `RES_1PASS`, `KDACONV`, `ADD3`, `KDA_IP`, `MOE_BATCH`, `B2WIDE` all measured positive at `=0`, i.e. off). `ROUTER_FAST` is the lone exception where the H200 default still measured correct on GB10.
- **The numeric-tuning-parameter flags (`RMSG`, `KDA_CPB`, `FUSED4_ROWS`, `PROJ_G`, `WARP_TARGET`) are sensitive to the specific value, not monotonic** — `FUSED4_ROWS=4` beat `=8` by a wide margin; don't assume "more" or "less" is a safe direction to extrapolate without measuring the specific value.
- **This is not the path to 15 tok/s.** Best case here (even hypothetically stacking, which doesn't actually work) tops out somewhere in the 6.5-7.5 tok/s range. See [`SPECULATIVE-DECODING-DESIGN.md`](SPECULATIVE-DECODING-DESIGN.md) for the actual lever being pursued for that target.

## Recommended production default (pending further validation)

```
SPARKINFER_K3_KDA_FUSE=0
```

Everything else stays at its shipped default. This is the single change with the best risk/reward — a clean +20% on this GB10-specific measurement, no interaction risk from stacking, and it's a pure kernel-selection toggle (no numeric-value sensitivity to get wrong).
