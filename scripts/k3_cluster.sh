#!/usr/bin/env bash
# One-command lifecycle helper for the public Kimi-K3 distributed recipe.
#
# Public roles deliberately avoid the transport's numeric terminology:
#   coordinator, compute-a, compute-b, [compute-c]
# The engine still receives its protocol slot internally. This script never copies
# model weights and never force-kills an unrelated process.
set -euo pipefail

usage() {
  cat <<'EOF'
usage: k3_cluster.sh {sync|start|stop|status|dry-run}

Run on the coordinator machine. Required settings:
  K3_HOME=$HOME/k3-neuron
  COORDINATOR_FABRIC=10.10.10.2       # reachable private/fabric IPv4 address
  MODEL=$K3_HOME/model/Kimi-K3-UD-IQ1_S-00001-of-00009.gguf

Remote cluster (default MODE=remote):
  TP_SIZE=3 PEER_A=user@10.10.10.4 PEER_B=user@10.10.10.6
  TP_SIZE=4 PEER_A=user@host-a PEER_B=user@host-b PEER_C=user@host-c

One PC with four Blackwell GPUs:
  MODE=local TP_SIZE=4 COORDINATOR_FABRIC=127.0.0.1 \
    GPU_COORDINATOR=0 GPU_COMPUTE_A=1 GPU_COMPUTE_B=2 GPU_COMPUTE_C=3

Optional: DIST, BIN, LISTEN, SERVE, MAX_CTX, SPEC_K.  `sync` copies only DIST
(the binary, runtime DSOs, manifest, and this helper), never MODEL. `dry-run`
prints the resolved plan without SSH, rsync, or process changes.
EOF
}

die() { printf 'k3_cluster: %s\n' "$*" >&2; exit 2; }
note() { printf 'k3_cluster: %s\n' "$*" >&2; }
quote() { printf '%q' "$1"; }

ACTION=${1:-}
case "$ACTION" in sync|start|stop|status|dry-run|__worker|__stop|__status) ;; *) usage; exit 2 ;; esac

K3_HOME=${K3_HOME:-"$HOME/k3-neuron"}
DIST=${DIST:-"$K3_HOME/dist"}
BIN=${BIN:-"$DIST/kimi_k3_dist_generate"}
MODEL=${MODEL:-"$K3_HOME/model/Kimi-K3-UD-IQ1_S-00001-of-00009.gguf"}
MODE=${MODE:-remote}
TP_SIZE=${TP_SIZE:-3}
COORDINATOR_FABRIC=${COORDINATOR_FABRIC:-}
LISTEN=${LISTEN:-0.0.0.0:29500}
SERVE=${SERVE:-127.0.0.1:29600}
MAX_CTX=${MAX_CTX:-4096}
SPEC_K=${SPEC_K:-8}
RUN_DIR=${RUN_DIR:-"$K3_HOME/run"}

# Calls from the coordinator use this private interface after the helper has been
# copied beside the runtime. Numeric engine slots never appear in this interface.
INTERNAL_ROLE=
INTERNAL_GPU=
if [[ "$ACTION" = __worker || "$ACTION" = __stop || "$ACTION" = __status ]]; then
  shift
  INTERNAL_ROLE=${1:?missing role}; INTERNAL_GPU=${2:-0}
  DIST=${3:?missing dist}; BIN=${4:?missing bin}; MODEL=${5:?missing model}
  TP_SIZE=${6:?missing tp size}; COORDINATOR_FABRIC=${7:?missing coordinator}
  LISTEN=${8:?missing listen}; SERVE=${9:?missing serve}; MAX_CTX=${10:?missing max ctx}
  SPEC_K=${11:?missing spec width}; RUN_DIR=${12:?missing run dir}
fi

case "$MODE" in remote|local) ;; *) die 'MODE must be remote or local' ;; esac
case "$TP_SIZE" in 3|4) ;; *) die 'TP_SIZE must be 3 or 4' ;; esac
if [[ "$ACTION" != dry-run ]]; then
  [[ -n "$COORDINATOR_FABRIC" ]] || die 'set COORDINATOR_FABRIC to the reachable coordinator address'
fi
if [[ "$ACTION" = sync || "$ACTION" = start || "$ACTION" = __worker ]]; then
  [[ -x "$BIN" ]] || die "missing executable: $BIN"
  [[ -f "$MODEL" ]] || die "missing local first GGUF shard: $MODEL"
  [[ -f "$DIST/libsparkinfer_runtime.so" ]] || die "missing $DIST/libsparkinfer_runtime.so"
  [[ -f "$DIST/libsparkinfer_moe.so" ]] || die "missing $DIST/libsparkinfer_moe.so"
fi

roles=(coordinator compute-a compute-b)
[[ "$TP_SIZE" = 4 ]] && roles+=(compute-c)

role_slot() {
  case "$1" in coordinator) echo 0 ;; compute-a) echo 1 ;; compute-b) echo 2 ;; compute-c) echo 3 ;; *) die "unknown role $1" ;; esac
}
role_gpu() {
  case "$1" in
    coordinator) echo "${GPU_COORDINATOR:-0}" ;;
    compute-a) echo "${GPU_COMPUTE_A:-1}" ;;
    compute-b) echo "${GPU_COMPUTE_B:-2}" ;;
    compute-c) echo "${GPU_COMPUTE_C:-3}" ;;
  esac
}
role_peer() {
  case "$1" in compute-a) echo "${PEER_A:-}" ;; compute-b) echo "${PEER_B:-}" ;; compute-c) echo "${PEER_C:-}" ;; *) die "no remote peer for $1" ;; esac
}
role_peer_name() {
  case "$1" in compute-a) echo PEER_A ;; compute-b) echo PEER_B ;; compute-c) echo PEER_C ;; *) die "no remote peer for $1" ;; esac
}
pid_path() { printf '%s/%s.pid' "$RUN_DIR" "$1"; }
log_path() { printf '%s/%s.log' "$RUN_DIR" "$1"; }
is_our_pid() {
  local pid=$1 cmd
  kill -0 "$pid" 2>/dev/null || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline")
  [[ "$cmd" == *"$BIN"* ]]
}

write_manifest() {
  if find "$DIST" -maxdepth 1 -type f -name '*.gguf' -print -quit | grep -q .; then
    die "refusing to sync: GGUF weights must not be stored in $DIST"
  fi
  ( cd "$DIST"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%P\0' | sort -z | xargs -0r sha256sum > SHA256SUMS
  )
}
verify_local_dist() { (cd "$DIST" && sha256sum -c SHA256SUMS >/dev/null); }

worker() {
  local role=$1 gpu=$2 slot pid log
  slot=$(role_slot "$role")
  pid=$(pid_path "$role")
  log=$(log_path "$role")
  mkdir -p "$RUN_DIR"
  if [[ -s "$pid" ]] && is_our_pid "$(<"$pid")"; then
    die "$role is already running (pid $(<"$pid")); use status or stop"
  fi
  rm -f "$pid"
  local common=(--rank "$slot" --world "$TP_SIZE" --model "$MODEL" --max-ctx "$MAX_CTX"
    --spec-draft "$SPEC_K" --spec-ngram-min 1 --spec-ngram-max "$SPEC_K" --spec-min-occur 2 --spec-majority 2/3)
  (
    export LD_LIBRARY_PATH="$DIST${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
    export SPARKINFER_K3_MOE_WEPS="${SPARKINFER_K3_MOE_WEPS:-0}"
    export SPARKINFER_K3_GRAPH="${SPARKINFER_K3_GRAPH:-0}"
    export SPARKINFER_K3_PREFILL_CHUNK="${SPARKINFER_K3_PREFILL_CHUNK:-16}"
    export SPARKINFER_K3_DIST_HEAD_BAND="${SPARKINFER_K3_DIST_HEAD_BAND:-1}"
    export SPARKINFER_K3_PROJ_TOKS="${SPARKINFER_K3_PROJ_TOKS:-8}"
    if [[ "$role" = coordinator ]]; then
      nohup "$BIN" "${common[@]}" --listen "$LISTEN" --serve "$SERVE" >"$log" 2>&1 < /dev/null &
    else
      nohup "$BIN" "${common[@]}" --coord "$COORDINATOR_FABRIC:${LISTEN##*:}" >"$log" 2>&1 < /dev/null &
    fi
    echo $! >"$pid"
  )
  note "started $role on GPU $gpu (pid $(<"$pid"), log $log)"
}

worker_stop() {
  local role=$1 pid
  pid=$(pid_path "$role")
  [[ -s "$pid" ]] || { note "$role is not recorded as running"; return 0; }
  if is_our_pid "$(<"$pid")"; then
    kill -TERM "$(<"$pid")"
    note "sent TERM to $role (pid $(<"$pid"))"
    rm -f "$pid"
  elif kill -0 "$(<"$pid")" 2>/dev/null; then
    die "refusing to signal $role: pid $(<"$pid") was reused by a different process"
  else
    rm -f "$pid"
  fi
}
worker_status() {
  local role=$1 pid
  pid=$(pid_path "$role")
  if [[ -s "$pid" ]] && is_our_pid "$(<"$pid")"; then
    printf '%s: running (pid %s, log %s)\n' "$role" "$(<"$pid")" "$(log_path "$role")"
  else
    printf '%s: stopped\n' "$role"
  fi
}

remote_worker() {
  local peer=$1 action=$2 role=$3 gpu=$4
  ssh "$peer" "bash $(quote "$DIST/k3_cluster.sh") $action $(quote "$role") $(quote "$gpu") $(quote "$DIST") $(quote "$BIN") $(quote "$MODEL") $(quote "$TP_SIZE") $(quote "$COORDINATOR_FABRIC") $(quote "$LISTEN") $(quote "$SERVE") $(quote "$MAX_CTX") $(quote "$SPEC_K") $(quote "$RUN_DIR")"
}

case "$ACTION" in
  __worker) worker "$INTERNAL_ROLE" "$INTERNAL_GPU"; exit 0 ;;
  __stop) worker_stop "$INTERNAL_ROLE"; exit 0 ;;
  __status) worker_status "$INTERNAL_ROLE"; exit 0 ;;
esac

if [[ "$ACTION" = dry-run ]]; then
  printf 'mode=%s tp_size=%s coordinator=%s model=%s\n' "$MODE" "$TP_SIZE" "$COORDINATOR_FABRIC" "$MODEL"
  for role in "${roles[@]}"; do
    if [[ "$role" = coordinator || "$MODE" = local ]]; then endpoint="local GPU $(role_gpu "$role")"; else endpoint="$(role_peer "$role")"; fi
    printf '%s -> %s\n' "$role" "$endpoint"
  done
  exit 0
fi

if [[ "$MODE" = remote ]]; then
  for role in "${roles[@]:1}"; do [[ -n "$(role_peer "$role")" ]] || die "set $(role_peer_name "$role") to that host's SSH destination"; done
fi

case "$ACTION" in
  sync)
    [[ "$MODE" = remote ]] || { note 'local mode has one shared dist; nothing to sync'; exit 0; }
    # Ship this small role-based launcher with the runtime payload. The model is never
    # part of DIST and rsync below has no source path that can include it.
    install -m 0755 "$0" "$DIST/k3_cluster.sh"
    write_manifest; verify_local_dist
    for role in "${roles[@]:1}"; do
      peer=$(role_peer "$role")
      note "syncing runtime distribution to $role ($peer); model files are excluded"
      ssh "$peer" "mkdir -p $(quote "$DIST") $(quote "$RUN_DIR") && test -f $(quote "$MODEL")"
      rsync -a --checksum "$DIST/" "$peer:$DIST/"
      ssh "$peer" "cd $(quote "$DIST") && sha256sum -c SHA256SUMS >/dev/null && test -x $(quote "$BIN")"
    done
    ;;
  start)
    if [[ "$MODE" = remote ]]; then bash "$0" sync; fi
    worker coordinator "$(role_gpu coordinator)"
    sleep 2
    for role in "${roles[@]:1}"; do
      if [[ "$MODE" = local ]]; then worker "$role" "$(role_gpu "$role")"; else remote_worker "$(role_peer "$role")" __worker "$role" 0; fi
    done
    note 'all roles launched; use status and inspect coordinator log before sending traffic'
    ;;
  stop)
    for role in "${roles[@]:1}"; do
      if [[ "$MODE" = local ]]; then worker_stop "$role"; else remote_worker "$(role_peer "$role")" __stop "$role" 0 || true; fi
    done
    worker_stop coordinator
    ;;
  status)
    worker_status coordinator
    for role in "${roles[@]:1}"; do
      if [[ "$MODE" = local ]]; then worker_status "$role"; else remote_worker "$(role_peer "$role")" __status "$role" 0 || true; fi
    done
    ;;
esac
