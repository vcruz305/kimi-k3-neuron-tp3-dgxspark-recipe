# What not to test or claim

This file prevents invalid comparisons and unsafe launch procedures.

## Do not benchmark these as the release result

- A build missing patch **0019**. It skips required model work; its speed is invalid.
- Load time, prefill rate, first-token latency, or a profiler-enabled run as decode TPS.
- A single candidate run without a baseline-before / candidate / baseline-after bracket.
- Output from `--prompt-ids` as an agent-quality test. Use the native chat template and
  tokenizer for user-facing evaluation.
- An arbitrary prose/chat workload as evidence for the structured 12.55 tok/s result.
- A different speculative profile as evidence for the K=8/P8 result. In particular,
  `--spec-require-agree` is not part of the qualified 12.55 tok/s configuration.
- A run with a different quantization, context length, driver/CUDA version, network,
  rank count, or a mixed binary/DSO deployment as directly comparable.
- TP4, same-host H200, llama.cpp RPC, H100, or 5090 receipts as three-Spark TP3 results.

## Do not use these as a public launcher

- `scripts/spec_draft.sh` and `scripts/spec_rollback.sh`: historical lab helpers with
  hard-coded `/home/victor`, private 10.10.10.x addresses, SSH user, process killing,
  and no deployment validation.
- Any `docs/` forecasting or historical investigation document as current operating
  instructions. Start with README, APPLY, and CURRENT_STATE instead.
- `api-server/` directly on the public Internet. It has optional bearer authentication,
  but no TLS, rate limiting, tenant isolation, or request batching.

## Do not make these claims

- “15 tok/s achieved” or “12.55 tok/s on every workload.” Neither is supported.
- Byte-identical speculative and sequential decoding on arbitrary prompts. The
  qualification establishes a valid greedy trajectory; the batched collective can alter
  near-tied choices.
- General production or multi-user readiness. The adapter is a trusted-network,
  one-request bridge.
- Ownership or redistribution rights for SparkInfer or Kimi-K3 weights. Follow their
  upstream/model-card terms; this repository contains patches and documentation only.

## Required release gates

1. `bash scripts/release_check.sh` passes on the clone.
2. The complete patch chain applies to the pinned base and the required targets build.
3. Loader, rank-protocol, drafter, tuning/head-band, KDA batch, and rollback checks pass.
4. Every rank runs the exact same binary and runtime libraries.
5. Rank-0 logs show the patch-0019 capability probe before performance is reported.
6. A real bracketed run uses a rendered chat prompt and preserves raw logs.
