# Kimi-K3 Neuron with SparkInfer

Run the 330 GB Kimi-K3 Neuron IQ1_S GGUF with the SparkInfer patch series,
distributed tensor parallelism, and optional repeat-aware speculative decoding. The
included API bridge speaks OpenAI `POST /v1/chat/completions`, so Hermes and other
OpenAI-compatible agents work without a custom client.

## Release scope

- **Qualified configuration:** TP3 on three DGX Sparks (GB10 / SM121), with the
  complete patch series through 0026.
- The structured/repeat-heavy K=8/P8 profile measured **12.5492 decode tok/s**
  candidate average (12.5468 and 12.5516) against a 6.4930 matched baseline. It is
  not a general chat, concurrency, long-context, or prose claim: freeform prose was
  6.5893 tok/s and the four-request mix was 9.7937 tok/s.
- **TP4 is experimental.** The patched runtime implements the `world=4`
  ExpertFfn2D plan and the distributed/speculative serve path accepts it, but the
  K=8/P8 profile is qualified only for TP3. Do not quote a TP4 speed from this release.

Read [WHAT_NOT_TO_TEST.md](WHAT_NOT_TO_TEST.md) before benchmarking. The complete
patch and validation chain is in [APPLY.md](APPLY.md).

## Prerequisites

Each participating machine needs CUDA/NCCL, local NVMe (at least 320 GB free), the
same model shard path, and the IQ1_S GGUF **on its own local disk**. Do not use NFS
or SSHFS. Use a private fabric address for cluster coordination, not a management
gateway. The API adapter is a trusted-network, one-request bridge; put TLS and rate
limits in front of it before public exposure.

The `dist/` directory contains only the executable and runtime libraries. It is small
enough to copy; the 330 GB model is deliberately never copied by the launcher.

## 1. Download the model and tokenizer on every participating machine

```bash
python3 -m pip install -U "huggingface_hub[hf_xet]"
hf auth login
export HF_XET_HIGH_PERFORMANCE=1
export K3_HOME=$HOME/k3-neuron
mkdir -p "$K3_HOME/model" "$K3_HOME/tokenizer"

hf download vcruz305/Kimi-K3-GGUF --local-dir "$K3_HOME/model" \
  --include "*.gguf" --include "k3_chat_template.jinja"
hf download moonshotai/Kimi-K3 --local-dir "$K3_HOME/tokenizer" \
  --include "config.json" --include "generation_config.json" \
  --include "tokenization_kimi.py" --include "tiktoken.model" --include "tokenizer_config.json"

test -f "$K3_HOME/model/Kimi-K3-UD-IQ1_S-00001-of-00009.gguf"
test -f "$K3_HOME/tokenizer/tokenization_kimi.py"
```

## 2. Build once and package the runtime distribution

Run this on the machine that will be the **coordinator**. For a mixed SM120/SM121
fleet, build a fat binary as shown. A homogeneous fleet may set only its own
architecture instead. Your CUDA toolchain must recognize every requested architecture.

```bash
export K3_HOME=$HOME/k3-neuron
mkdir -p "$K3_HOME/src"
git clone --depth 1 --branch main https://github.com/vcruz305/kimi-k3-neuron-tp3-dgxspark-recipe \
  "$K3_HOME/src/recipe"
git clone https://github.com/gittensor-ai-lab/sparkinfer-k3.git "$K3_HOME/src/sparkinfer-k3"
cd "$K3_HOME/src/sparkinfer-k3"
bash "$K3_HOME/src/recipe/scripts/apply_sparkinfer_patch_series.sh" "$K3_HOME/src/recipe"

cmake -S runtime -B build -DSPARKINFER_TP=ON -DCMAKE_CUDA_ARCHITECTURES="120;121"
cmake --build build -j"$(nproc)" --target kimi_k3_dist_generate \
  tp_rank_local_loader_cpu_test tp_dist_generate_protocol_cpu_test \
  kimi_k3_tp_kda_check kimi_k3_tp_width_check kimi_k3_kda_batch_check \
  kimi_k3_spec_draft_check kimi_k3_tune_check
ctest --test-dir build --output-on-failure

mkdir -p "$K3_HOME/dist"
cp "$(find build -type f -name kimi_k3_dist_generate -print -quit)" \
   "$(find build -type f -name libsparkinfer_runtime.so -print -quit)" \
   "$(find build -type f -name libsparkinfer_moe.so -print -quit)" "$K3_HOME/dist/"
```

## 3. One-command TP3 launch (qualified 12.5 TPS profile)

Run this on the coordinator. `k3_cluster.sh` copies the runtime distribution to the
two compute peers, verifies its checksum there, confirms that the local model exists
on each peer, starts the coordinator first, then starts the peers. It does **not**
copy weights. The names `coordinator`, `compute-a`, and `compute-b` are operational
roles; numeric protocol slots are kept inside the helper.

```bash
export K3_HOME=$HOME/k3-neuron
export COORDINATOR_FABRIC=10.10.10.2
export PEER_A=youruser@10.10.10.4
export PEER_B=youruser@10.10.10.6
export MODEL="$K3_HOME/model/Kimi-K3-UD-IQ1_S-00001-of-00009.gguf"
cd "$K3_HOME/src/recipe"

# Safe preview first: no SSH, file copy, or process changes.
bash scripts/k3_cluster.sh dry-run

# Copy runtime only, then launch the measured TP3 K=8/P8 configuration.
bash scripts/k3_cluster.sh start
bash scripts/k3_cluster.sh status
```

Logs and PID files are under `$K3_HOME/run/`; stop only processes started by this
helper with `bash scripts/k3_cluster.sh stop`. The helper refuses to signal a reused
PID. Inspect the coordinator log and wait for its local serve listener before sending
traffic. A first load normally takes about six minutes.

## 4. Experimental TP4 recipes

These recipes use the same `kimi_k3_dist_generate` serve and speculative-decode path,
with four one-GPU processes. They are code-supported but **not performance-qualified**
by this release. Start with a short correctness run, retain logs, and benchmark only
after the gates in [APPLY.md](APPLY.md) pass.

### Four Blackwell hosts (SM120 and/or SM121)

Use four hosts with a reachable private fabric, the same driver/CUDA/NCCL generation,
the same absolute model path, and a build that includes the required architectures.
Mixed SM120/SM121 fleets are experimental: validate all four devices and do not mix
binary or DSO hashes. Each host needs its own local GGUF.

```bash
export K3_HOME=$HOME/k3-neuron
export TP_SIZE=4
export COORDINATOR_FABRIC=10.10.10.2
export PEER_A=youruser@10.10.10.4
export PEER_B=youruser@10.10.10.6
export PEER_C=youruser@10.10.10.8
export MODEL="$K3_HOME/model/Kimi-K3-UD-IQ1_S-00001-of-00009.gguf"
cd "$K3_HOME/src/recipe"
bash scripts/k3_cluster.sh dry-run
bash scripts/k3_cluster.sh start
bash scripts/k3_cluster.sh status
```

### One PC with 4× RTX PRO 6000 Blackwell

This is an experimental single-host TP4 layout, intended for **RTX PRO 6000
Blackwell** GPUs (not RTX 6000 Ada). The machine needs four visible Blackwell GPUs,
enough local NVMe for the model, and a working NCCL peer-to-peer topology. The helper
starts four local processes and binds each one to exactly one GPU. It still uses
`world=4` internally, preserving the TP4 loader and speculative serve path.

```bash
export K3_HOME=$HOME/k3-neuron
export MODE=local TP_SIZE=4
export COORDINATOR_FABRIC=127.0.0.1
export GPU_COORDINATOR=0 GPU_COMPUTE_A=1 GPU_COMPUTE_B=2 GPU_COMPUTE_C=3
export MODEL="$K3_HOME/model/Kimi-K3-UD-IQ1_S-00001-of-00009.gguf"
cd "$K3_HOME/src/recipe"
bash scripts/k3_cluster.sh dry-run
bash scripts/k3_cluster.sh start
bash scripts/k3_cluster.sh status
```

For a pure SM120 PC, build with `-DCMAKE_CUDA_ARCHITECTURES=120`. Start at
`MAX_CTX=4096`, close other GPU workloads, and validate peer access/NCCL before a
long load. Use `bash scripts/k3_cluster.sh stop` to stop all four helper-owned roles.

## 5. OpenAI-compatible API for Hermes and other agents

Run this only on the coordinator, after `status` reports the coordinator process
running. It binds the engine control port to localhost and exposes the HTTP bridge on
the chosen trusted-network interface.

```bash
export K3_HOME=$HOME/k3-neuron
cd "$K3_HOME/src/recipe"
python3 -m venv api-server/.venv
api-server/.venv/bin/pip install -r api-server/requirements.txt

export K3_TOKENIZER_DIR="$K3_HOME/tokenizer"
export K3_CHAT_TEMPLATE="$K3_HOME/model/k3_chat_template.jinja"
export K3_SERVE_HOST=127.0.0.1 K3_SERVE_PORT=29600
export K3_API_HOST=0.0.0.0 K3_API_PORT=8000
export K3_MAX_CONTEXT_TOKENS=4096
export K3_API_KEY='replace-with-a-long-random-secret'
bash api-server/run-api.sh
```

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/v1/models -H "Authorization: Bearer $K3_API_KEY"
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $K3_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"kimi-k3-neuron","messages":[{"role":"user","content":"Reply with exactly: SparkInfer is ready."}],"max_tokens":32}'
```

Set an OpenAI-compatible agent’s `base_url` to
`http://<coordinator-fabric-ip>:8000/v1`, its API key to `K3_API_KEY`, and its model to
`kimi-k3-neuron`. The bridge is sequential and greedy: it is suitable for a low-QPS
agent, not a multi-tenant service. Generated tool calls are text, not parsed
`tool_calls` objects.

## Operations and evidence

- [APPLY.md](APPLY.md) — exact patch series and required correctness gates.
- [CURRENT_STATE.md](CURRENT_STATE.md) — present support and release limits.
- [12.5 TPS receipt](evidence/specdec-k8-p8-12tps-RECEIPT.md) — workload and bracket.
- [api-server/README.md](api-server/README.md) — adapter behavior.

Run `bash scripts/release_check.sh` after cloning the recipe. It verifies checksums,
patch syntax, and accidental credential material.

## License

Patches follow the upstream SparkInfer and project license terms. Kimi-K3 Neuron
weights are distributed separately under their Hugging Face terms.
