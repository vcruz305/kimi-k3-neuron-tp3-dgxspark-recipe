# Recursive-majority speculative decoding: TP3 >10 tok/s receipt

**Date:** 2026-08-12

**Hardware:** 3× NVIDIA DGX Spark GB10, one process/node, private fabric
`10.10.10.2/4/6`

**Target:** corrected Kimi-K3 Neuron IQ1_S GGUF distributed engine through patch 0026

## Outcome

Useful structured generation stayed above 10 tok/s on both sides of a matched baseline
on the real three-node fleet:

| position | workload | candidate A | baseline | candidate B | candidate avg | vs baseline |
|---:|---|---:|---:|---:|---:|---:|
| 0 | coherent code | **10.0983** | 6.3621 | 8.8326 | 9.4655 | +48.8% |
| 1 | structured generation | **10.3002** | 6.4272 | **11.3231** | **10.8117** | **+68.2%** |
| 2 | freeform prose control | 6.6009 | 6.4905 | 6.4693 | 6.5351 | +0.7% |
| 3 | coherent code, same-load repeat | **10.3358** | 6.4162 | 9.9156 | **10.1257** | +57.8% |
| — | four-request arithmetic mean | 9.3338 | 6.4240 | 9.1352 | 9.2345 | +43.8% |

This proves sustained >10 tok/s for the structured-generation workload across a
candidate -> baseline -> candidate bracket. It does **not** prove prose or general
cross-workload mean >10. An exploratory K=5 run later reached the fastest observed point,
**11.9000 tok/s**, but did not improve the overall mean and is not the recommendation.

## Candidate

Patch `0026-tp-specdec-recursive-majority-drafter.diff` adds an opt-in recursive
continuation vote. For each draft position it scans from the longest available n-gram
down, requires at least two historical occurrences and a continuation winning at least
2/3 of votes, appends the proposed token to a temporary history, then predicts the next
draft position. The target model still verifies every proposal; the drafter never decides
which tokens are emitted.

Flags on every rank:

```text
--spec-draft 4
```

Rank-0 draft selection flags:

```text
--spec-ngram-min 1 --spec-ngram-max 8 --spec-min-occur 2 --spec-majority 2/3
```

Environment:

```text
SPARKINFER_K3_DIST_HEAD_BAND=1
SPARKINFER_K3_MOE_WEPS=0
SPARKINFER_K3_GRAPH=0
NCCL_NVLS_ENABLE=0
SPARKINFER_K3_PREFILL_CHUNK=16
CUDA_VISIBLE_DEVICES=0
```

Common run shape: `--max-ctx 4096 --n-predict 64`, four prompts ordered code,
structured repetition, prose, then the identical code prompt again. Each candidate and
baseline process loaded once and reset KV/KDA state between prompts.

## Correctness and build gates

- `kimi_k3_spec_draft_check`: **1091/1091 passed**, including 4000 randomized legacy
  drafter/reference trials, full/partial/zero acceptance arms, protocol round trips,
  greedy-oracle loop equivalence, majority template continuation, and split-vote decline.
- Patch 0026 plain-`git am` applied to the exact clean 0025 audit tip with zero dirty
  files; local verification commit was `5915d4d`.
- Tested generator MD5 on all three ranks:
  `9b259046953b2585374f4c3034e7e087`.
- Known-good unchanged runtime MD5 on all ranks:
  `d8a2469ff109ba60c04c10ad36a6ce59`.
- Both K=4 candidate sides finished clean with **95/95 accepted drafts** and emitted
  identical recorded token sequences (SHA-256 `c53cc6d73c9b06b66dfd3196da093ee9b15ce262683a4ab05bd94bce410b340c`).
  No draft fired for prose; the prose path therefore stayed on ordinary one-token decode.

## Exact first-candidate output

```text
[spec-draft] ON K=4 ngram n=[1,8] min_occur=2 agree=0 majority=2/3 weak_draft=0
OK generated 64 tokens in 6.338s (decode_tok_s=10.0983) prompt=0
OK generated 64 tokens in 6.213s (decode_tok_s=10.3002) prompt=1
OK generated 64 tokens in 9.696s (decode_tok_s=6.6009) prompt=2
OK generated 64 tokens in 6.192s (decode_tok_s=10.3358) prompt=3
[spec-draft] steps=157  with_draft=37  no_match=120
[spec-draft] drafted=95 accepted=95  acceptance=1.0000
[spec-draft] mean_draft_len=0.605  mean_emitted_per_step=1.605
[spec-draft] pos=0 drafted=37 accepted=37 p=1.0000
[spec-draft] pos=1 drafted=29 accepted=29 p=1.0000
[spec-draft] pos=2 drafted=29 accepted=29 p=1.0000
OK multi-prompt summary prompts=4 mean_decode_tok_s=9.3338
OK finished clean
```

## Exact post-baseline output

```text
OK generated 64 tokens in 10.060s (decode_tok_s=6.3621) prompt=0
OK generated 64 tokens in 9.958s (decode_tok_s=6.4272) prompt=1
OK generated 64 tokens in 9.861s (decode_tok_s=6.4905) prompt=2
OK generated 64 tokens in 9.975s (decode_tok_s=6.4162) prompt=3
OK multi-prompt summary prompts=4 mean_decode_tok_s=6.4240
OK finished clean
```

## Exact second-candidate output

```text
[spec-draft] ON K=4 ngram n=[1,8] min_occur=2 agree=0 majority=2/3 weak_draft=0
OK generated 64 tokens in 7.246s (decode_tok_s=8.8326) prompt=0
OK generated 64 tokens in 5.652s (decode_tok_s=11.3231) prompt=1
OK generated 64 tokens in 9.893s (decode_tok_s=6.4693) prompt=2
OK generated 64 tokens in 6.454s (decode_tok_s=9.9156) prompt=3
[spec-draft] steps=157  with_draft=37  no_match=120
[spec-draft] drafted=95 accepted=95  acceptance=1.0000
[spec-draft] mean_draft_len=0.605  mean_emitted_per_step=1.605
[spec-draft] pos=0 drafted=37 accepted=37 p=1.0000
[spec-draft] pos=1 drafted=29 accepted=29 p=1.0000
[spec-draft] pos=2 drafted=29 accepted=29 p=1.0000
OK multi-prompt summary prompts=4 mean_decode_tok_s=9.1352
OK finished clean
```

## Targeted follow-up sweeps

These are negative/diagnostic runs, not replacements for the bracketed K=4 recipe:

| variant | code | structured | prose | code repeat | mean | decision |
|---|---:|---:|---:|---:|---:|---|
| majority K=5 | 8.8557 | **11.9000** | 6.3350 | **10.1170** | 9.3019 | reject overall; keep peak as exploratory |
| majority K=8 | 8.9845 | **10.8060** | 6.1300 | **10.5572** | 9.1194 | reject |
| majority K=4 + `KDA_FUSE=0` | 9.0493 | **11.2400** | 6.5496 | **10.2893** | 9.2821 | reject interaction |

K=5 accepted 108/108 reached drafts and K=8 accepted 112/112. The wider variants lose
on verification cost despite perfect acceptance; widening further is not the next lever.

Rank-0 raw logs on the fleet:

```text
/home/victor/work/k3-tp3-0012/run/majority-0812/majority4warm_r0.stdout
/home/victor/work/k3-tp3-0012/run/majority-0812/majority4warm_r0.stderr
/home/victor/work/k3-tp3-0012/run/majority-0812/baseline_post_r0.stdout
/home/victor/work/k3-tp3-0012/run/majority-0812/baseline_post_r0.stderr
/home/victor/work/k3-tp3-0012/run/majority-0812/majority4post_r0.stdout
/home/victor/work/k3-tp3-0012/run/majority-0812/majority4post_r0.stderr
/home/victor/work/k3-tp3-0012/run/majority-0812/majority5a_r0.stdout
/home/victor/work/k3-tp3-0012/run/majority-0812/majority8a_r0.stdout
/home/victor/work/k3-tp3-0012/run/majority-0812/majority4fusea_r0.stdout
```

## Non-claims / what not to retest

- Do not call the 9.2345 bracketed four-request mean a three-workload median or a general
  >10 TPS result. Bracketed prose averaged 6.5351.
- Do not use a single peak alone as the headline. The supported headline is the structured
  row's 10.3002 / 6.4272 / 11.3231 bracket and 10.8117 candidate average.
- Do not integrate the 7.1 GB `Inferact/Kimi-K3-DSpark` checkpoint merely to cross 10 on
  useful structured output; this host-only drafter already does that. DSpark remains a
  possible future route if the goal changes to prose or cross-workload mean >10.
- Do not widen K on intuition. Majority K=5 and K=8 were explicitly tested; neither beat
  K=4 overall despite perfect acceptance. K=5's 11.9000 is an exploratory peak only.
- Do not add `KDA_FUSE=0` to this recipe by assumption; the measured interaction averaged
  9.2821 and did not improve the first K=4 candidate's 9.3338 mean.
- `--spec-majority` is opt-in. With a zero numerator (the default), patch 0026 preserves
  the 0025 span-copy path byte for byte.
