#!/usr/bin/env bash
# Apply the audited SparkInfer K3 TP3 patch chain to its pinned upstream base.
set -euo pipefail

recipe_dir=${1:?usage: $0 /path/to/k3-recipe}
base=7a9b77a043596157d74e4af376cf9f29f68ce368

test -d "$recipe_dir/patches/sparkinfer"
test "$(git rev-parse HEAD)" = "$base"
test -z "$(git status --porcelain)"

patches=(
  patches/sparkinfer/0001-tp-plan-K3-all-expert-FFN-width-shards-for-TP3.patch
  patches/sparkinfer/0002-wire-tp3-all-expert-width-init-ffn-prefill.patch
  patches/sparkinfer/next/0003-tp-three-host-rank-bootstrap-protocol.patch
  patches/sparkinfer/next/0004-tp-distributed-rank-transport-and-tp3-tp4-plans.patch
  patches/sparkinfer/next/0005-tp-rank-local-plan-and-nccl-microbench.patch
  patches/sparkinfer/next/0006-tp-rank-local-gguf-load-api-and-moe-budget.patch
  patches/sparkinfer/next/0007-tp-distributed-eager-forward-and-dist-generate.patch
  patches/sparkinfer/next/0008-tp-fix-rank0-loadready-oneshot-and-load-before-ready.patch
  patches/sparkinfer/next/0009-tp-f16-token-embd-output-and-dist-embed.patch
  patches/sparkinfer/next/0010-tp-first-forward-stall-instrumentation.patch
  patches/sparkinfer/next/0011-tp-finish-deadlock-and-decode-tps.patch
  patches/sparkinfer/next/0012-tp-finish-ack-after-wait-token.patch
  patches/sparkinfer/next/0013-tp-pin-per-tensor-host-memory-for-fast-load.patch
  patches/sparkinfer/next/0015-tp-multi-prompt-file-and-kv-reset-sentinel.patch
  patches/sparkinfer/next/0014-tp-serve-mode-for-dynamic-prompt-requests.patch
  patches/sparkinfer/next/0016-tp-repetition-guard-for-greedy-decode.patch
  patches/sparkinfer/next/0017-test-fix-three-bugs-in-kimi-k3-state-check.patch
  patches/sparkinfer/next/0018-tp-attention-shard-kill-switches-for-dist-path.patch
  patches/sparkinfer/next/0018a-tp-attention-shard-equivalence-tests.patch
  patches/sparkinfer/next/0019-tp-probe-gguf-capability-flags-in-dist-rank-load.patch
  patches/sparkinfer/next/0019a-tp-dist-phase-profiler.patch
  patches/sparkinfer/next/0020-tp-per-layer-dump-instrumentation-for-dist-path.patch
  patches/sparkinfer/next/0021-tp-specdec-stages-0-2-batched-verify.diff
  patches/sparkinfer/next/0022-tp-specdec-ar-count-determinism-diagnostic.diff
  patches/sparkinfer/next/0023-tp-specdec-stage3-kda-state-rollback.diff
  patches/sparkinfer/next/0024-tp-specdec-stage4-ngram-drafter-and-ktoken-protocol.diff
  patches/sparkinfer/next/0025-tp-specdec-tuning-confidence-gate-and-dist-head-band.diff
  patches/sparkinfer/next/0026-tp-specdec-recursive-majority-drafter.diff
)

for patch in "${patches[@]}"; do
  # `git am` requires a committer identity even though every patch already carries its
  # author. Use a command-local identity so a fresh appliance does not need global Git
  # configuration and the user's checkout config is not changed.
  git -c user.name='K3 TP3 Patch Builder' \
      -c user.email='k3-tp3-builder@local.invalid' \
      am "$recipe_dir/$patch"
done
