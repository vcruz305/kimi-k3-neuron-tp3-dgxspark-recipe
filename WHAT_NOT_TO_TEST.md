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
  TP participant count, or a mixed binary/DSO deployment as directly comparable.
- TP4 (four-host or one-PC), same-host H200, llama.cpp RPC, H100, or 5090 receipts
  as three-Spark TP3 results.
- An SM120/SM121 mixed-fleet result as a homogeneous-fleet result, or an RTX 6000 Ada
  result as an RTX PRO 6000 Blackwell result.
- Any TP4 run with the TP3 K=8/P8 flags as evidence that TP4 reaches 12.5492 tok/s.
  TP4 is code-supported and experimental; it has no current public speed claim.

## Do not use these as a public launcher

- `scripts/spec_draft.sh` and `scripts/spec_rollback.sh`: historical lab helpers with
  hard-coded `/home/victor`, private 10.10.10.x addresses, SSH user, process killing,
  and no deployment validation.
- Manual copying of a partial `dist/` directory, or manual coordinator/peer starts as
  the public recipe. Use `scripts/k3_cluster.sh`: it creates and verifies the runtime
  manifest, copies no GGUF weights, uses role names, and records only its own PIDs.
- `scripts/k3_cluster.sh stop` against a process not launched by that helper. It
  deliberately refuses a reused/unrecognized PID rather than killing it.
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
- That every Blackwell topology has been qualified. The runtime supports SM120/SM121
  builds and TP4 geometry; driver, NCCL fabric/PCIe topology, available VRAM, and
  performance still require target-hardware validation.
- Ownership or redistribution rights for SparkInfer or Kimi-K3 weights. Follow their
  upstream/model-card terms; this repository contains patches and documentation only.

## Required release gates

1. `bash scripts/release_check.sh` passes on the clone.
2. The complete patch chain applies to the pinned base and the required targets build.
3. Loader, cluster-protocol, drafter, tuning/head-band, KDA batch, and rollback checks pass.
4. Every participant runs the exact same binary and runtime libraries.
5. Coordinator logs show the patch-0019 capability probe before performance is reported.
6. A real bracketed run uses a rendered chat prompt and preserves raw logs.
