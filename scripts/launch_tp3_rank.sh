#!/usr/bin/env bash
# Portable foreground launcher for one rank of the verified 3-Spark K=8/P8 profile.
# Configure with environment variables; run one copy on each Spark under systemd,
# tmux, or another supervisor. This script never kills existing processes.
set -euo pipefail

die() { printf 'launch_tp3_rank: %s\n' "$*" >&2; exit 2; }

: "${RANK:?set RANK to 0, 1, or 2}"
: "${MODEL:?set MODEL to the first local GGUF shard}"

case "$RANK" in 0|1|2) ;; *) die "RANK must be 0, 1, or 2" ;; esac

DIST=${DIST:-"$HOME/k3-tp3/dist"}
BIN=${BIN:-"$DIST/kimi_k3_dist_generate"}
WORLD=${WORLD:-3}
COORD=${COORD:-}
LISTEN=${LISTEN:-0.0.0.0:29500}
SERVE=${SERVE:-127.0.0.1:29600}
MAX_CTX=${MAX_CTX:-4096}
SPEC_K=${SPEC_K:-8}

[[ "$WORLD" = 3 ]] || die "this public recipe is qualified only for WORLD=3"
[[ -x "$BIN" ]] || die "missing executable: $BIN"
[[ -f "$MODEL" ]] || die "missing first GGUF shard: $MODEL"
[[ -f "$DIST/libsparkinfer_runtime.so" ]] || die "missing $DIST/libsparkinfer_runtime.so"
[[ -f "$DIST/libsparkinfer_moe.so" ]] || die "missing $DIST/libsparkinfer_moe.so"
[[ "$SPEC_K" = 8 ]] || die "the published P8 profile requires SPEC_K=8"

export LD_LIBRARY_PATH="$DIST${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export SPARKINFER_K3_MOE_WEPS=${SPARKINFER_K3_MOE_WEPS:-0}
export SPARKINFER_K3_GRAPH=${SPARKINFER_K3_GRAPH:-0}
export SPARKINFER_K3_PREFILL_CHUNK=${SPARKINFER_K3_PREFILL_CHUNK:-16}
export SPARKINFER_K3_DIST_HEAD_BAND=${SPARKINFER_K3_DIST_HEAD_BAND:-1}
export SPARKINFER_K3_PROJ_TOKS=${SPARKINFER_K3_PROJ_TOKS:-8}

common=(
  --rank "$RANK" --world 3 --model "$MODEL" --max-ctx "$MAX_CTX"
  --spec-draft 8 --spec-ngram-min 1 --spec-ngram-max 8
  --spec-min-occur 2 --spec-majority 2/3
)

if [[ "$RANK" = 0 ]]; then
  printf 'launch_tp3_rank: rank0 listen=%s serve=%s\n' "$LISTEN" "$SERVE" >&2
  exec "$BIN" "${common[@]}" --listen "$LISTEN" --serve "$SERVE"
else
  [[ -n "$COORD" ]] || die "set COORD to rank0's private fabric address, for example 10.10.10.2:29500"
  printf 'launch_tp3_rank: rank%s coord=%s\n' "$RANK" "$COORD" >&2
  exec "$BIN" "${common[@]}" --coord "$COORD"
fi
