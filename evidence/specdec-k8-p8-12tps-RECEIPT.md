# Recursive-majority K=8/P8: TP3 >12 tok/s structured profile

**Date:** 2026-08-12

**Hardware:** 3× NVIDIA DGX Spark GB10, one process/node, private fabric
`10.10.10.2/4/6`

**Target:** corrected Kimi-K3 Neuron IQ1_S GGUF engine through patch 0026

## Outcome

This is a structured-throughput profile, not a general or prose >12 tok/s claim. The
candidate uses the already-published recursive-majority drafter at K=8 and selects the
measured low-sawtooth eight-row projection dispatch explicitly:

```text
--spec-draft 8 --spec-ngram-min 1 --spec-ngram-max 8
--spec-min-occur 2 --spec-majority 2/3
SPARKINFER_K3_DIST_HEAD_BAND=1
SPARKINFER_K3_PROJ_TOKS=8
```

Matched candidate → baseline → candidate result:

| workload | candidate A | baseline | candidate B | candidate avg | vs baseline |
|---|---:|---:|---:|---:|---:|
| coherent code | 9.3535 | 6.4746 | 8.6543 | 9.0039 | +39.06% |
| structured generation | **12.5468** | 6.4930 | **12.5516** | **12.5492** | **+93.27%** |
| freeform prose control | 6.6040 | 6.4626 | 6.5746 | 6.5893 | +1.96% |
| coherent code repeat | 11.2682 | 6.4705 | 10.7961 | 11.0322 | +70.50% |
| four-request arithmetic mean | 9.9431 | 6.4752 | 9.6442 | 9.7937 | +51.25% |

Both candidates accepted **112/112** proposed tokens. Candidate A and B produced the
same 64 generated token IDs on all four prompts, including the no-draft prose control.
The structured row independently cleared 12 tok/s on both sides of the matched baseline.

## Why P8 matters

The verify-step latency is a sawtooth rather than monotonic in width. The measured K=8
step was 385.63 ms on the generic projection dispatch but 320.44 ms when
`SPARKINFER_K3_PROJ_TOKS=8` selected the exact eight-row projection path, a 65.19 ms
reduction for the same verified batch. This changes the earlier K=8 conclusion: the old
10.8060 structured result and 9.1194 mixed mean are not the right K=8/P8 comparison.

No new model, training, approximation, or acceptance rule is introduced. The target
still verifies every proposal and emits only its own greedy token/prefix.

## Common run shape and gates

- `--max-ctx 4096 --n-predict 64`; prompts ordered code, structured repetition,
  prose, then the identical code prompt again.
- Each candidate/baseline process loaded once and reset KV/KDA state between prompts.
- `SPARKINFER_K3_MOE_WEPS=0`, `SPARKINFER_K3_GRAPH=0`, `NCCL_NVLS_ENABLE=0`,
  `SPARKINFER_K3_PREFILL_CHUNK=16`, `CUDA_VISIBLE_DEVICES=0` on all ranks.
- Generator MD5 all ranks: `03e6c8d76328034de3254b3904eb4246`.
- Runtime MD5 all ranks: `d8a2469ff109ba60c04c10ad36a6ce59`.
- Host speculative suite: **1092/1092 PASS**.
- Raw logs: `/home/victor/work/k3-tp3-0012/run/bracket12-0812/`.

## Non-claims

- Do not call one structured prompt a general >12 tok/s result. Prose remains near
  ordinary target-only speed because the host drafter correctly declines when it has no
  recurring evidence.
- Do not enable `--spec-pad-pow2`: it is a separate rejected experiment and was off here.
- Do not add `RES_1PASS=0`, `KDACONV=0`, `ADD3=0`, or the experimental full KDA fusion;
  their explicit K=8 interactions were rejected.
- K=4 remains the lower-width conservative profile documented in the original >10
  receipt; K=8/P8 is the profile for repeat-heavy structured output.
