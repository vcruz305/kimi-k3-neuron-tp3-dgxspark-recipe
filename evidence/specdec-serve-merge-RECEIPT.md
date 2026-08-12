# Serve + speculative-decoding merge receipt

Date: 2026-08-12 UTC

Fleet: 3× DGX Spark GB10, TP3

Pinned upstream base: `7a9b77a043596157d74e4af376cf9f29f68ce368`

## What changed

- Repaired patch 0015: added the two missing protocol-side allowances for the
  distributed `-2` KV-reset sentinel.
- Rebased 0021–0025 onto the real 0014/0016 serve/repetition-guard line.
- Kept the repetition guard on ordinary decode. Speculative verification bypasses
  it because 0019 fixed the original collapse, and guarding verification would
  suppress literal repetition, prompt lookup's measured best case.
- Added speculative execution to the resident `--serve` request loop, including
  streaming, stop ids, deferred rollback, and reset between requests.

## Clean-chain proof

Fresh pinned-base checkout, plain `git am` in documented order:

- 27/27 patches applied.
- Zero dirty files.
- Resulting tree byte-identical to the fleet-built integration tree.
- No `--ignore-whitespace` or manual post-apply edits.

## Build and tests

Built on rank0 with CUDA 13.0.88, `CMAKE_CUDA_ARCHITECTURES=121`, Release:

- `kimi_k3_dist_generate`
- `libsparkinfer_runtime.so`
- `tp_rank_local_loader_cpu_test`: 38/38
- `tp_dist_generate_protocol_cpu_test`: 17/17
- `kimi_k3_spec_draft_check`: 1089/1089
- `kimi_k3_tune_check`: 29/29
- KDA head-shard equivalence: relL2 `9.956e-08`
- MoE width-shard equivalence: relL2 `1.135e-07`
- Batched KDA: bit-identical to sequential for K=2..8

Isolated deployed artifacts, MD5 identical on all three ranks:

- `kimi_k3_dist_generate`: `6a5272a95ccf906c1fa6d5be3a7a5821`
- `libsparkinfer_runtime.so`: `d8a2469ff109ba60c04c10ad36a6ce59`
- `libsparkinfer_moe.so`: `b6a1540f702e0fde6f844efaaf313567`

The pre-existing live `dist/` directory was not overwritten; testing used
`dist-integrated-0812/`.

## Five-run benchmark

64 tokens × 3 workloads. Same binaries, prompts, environment, and launch pattern.
Baselines bracket the candidates to expose wall-clock drift.

| config | code | repetition | prose | mean | vs avg baseline |
|---|---:|---:|---:|---:|---:|
| baseline before | 6.1612 | 5.7251 | 6.1622 | 6.0162 | — |
| confidence-gated K=4 | — | — | — | 8.0591 | +32.61% |
| head-band only | 6.4718 | 6.3173 | 6.3888 | 6.3926 | +5.19% |
| confidence K=4 + head-band | 8.1651 | 10.9018 | 6.5672 | 8.5447 | +40.61% |
| baseline after | 6.1477 | 6.1166 | 6.1496 | 6.1379 | — |

Average baseline: `6.07705` tok/s. Both gated runs accepted 59/64 drafted
positions (`0.9219`). The earlier independent head-band proof compared 129/129
non-empty logit dumps and found every row bitwise identical.

## Serve integration proof

All ranks launched with protocol v2, K=4, confidence gating, and head banding.
Rank0 additionally launched with `--serve 127.0.0.1:38081`.

1. Request 1: streaming enabled, 16 tokens returned.
2. Request 2: distributed KV reset first, then 16 tokens returned.
3. Shutdown request: clean finish on coordinator and workers.

Client result: `SERVE_INTEGRATION_PASS checks=[True, True, True, True]`

Driver result: `SERVE_DRIVER_PASS`

Fleet after test: no `kimi_k3_dist_generate` processes running.

Raw fleet logs remain under:

- `/home/victor/work/k3-tp3-0012/run/integrated-0812/`
- `/home/victor/work/k3-tp3-0012/run/serve-integrated-0812/`
