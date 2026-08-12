# Kimi-K3 Neuron on 3 DGX Sparks

Run the 330 GB Kimi-K3 Neuron IQ1_S GGUF across three NVIDIA DGX Spark nodes with
the SparkInfer TP3 patch and opt-in speculative decoding. The included HTTP bridge
speaks the OpenAI `POST /v1/chat/completions` schema, so Hermes and other
OpenAI-compatible agents can use it without a custom client.

## What this release proves

- The SparkInfer patch series through **0026** clean-applies to the pinned upstream
  base, builds on GB10, and includes the distributed-loader correctness fix. Apply
  the complete series; older throughput measurements from before patch 0019 are not
  valid model results.
- On three real DGX Sparks, the repeat-heavy/structured speculative profile measured
  **12.5468** and **12.5516 tok/s** around a **6.4930 tok/s** matched baseline
  (**12.5492 tok/s** candidate average). Both candidate runs accepted 112/112 drafted
  tokens and emitted identical token IDs.
- That is a deliberately narrow claim: it is not general-purpose or prose throughput.
  The same profile measured **6.5893 tok/s** on freeform prose and **9.7937 tok/s**
  across its four-request mix. Use normal greedy decoding when a workload does not
  benefit from repeat-aware drafting.

Full receipts: [12.5 TPS benchmark](evidence/specdec-k8-p8-12tps-RECEIPT.md) and
[patch/build status](APPLY.md). The service is a single sequential model instance:
one request at a time, greedy decoding, no sampling, and no parsed tool-call output.

## Prerequisites

You need three GB10-based DGX Sparks on a private/fabric network, SSH between them,
CUDA/NCCL installed, and at least 320 GB of **local NVMe** free on every node. Do not
load the GGUF through NFS or SSHFS. You also need access to the gated GGUF and the
official `moonshotai/Kimi-K3` Hugging Face repositories.

The commands below use these example fabric addresses. Set them to your three nodes
once, then keep rank 0's address identical in every command.

```bash
export RANK0=10.10.10.2
export RANK1=10.10.10.4
export RANK2=10.10.10.6
export K3_HOME=$HOME/k3-neuron
```

## 1. Download model and tokenizer on every Spark

Run this block once on **each** Spark. It downloads the GGUF locally and only the
small tokenizer files needed by the API bridge (not the original full-precision model).

```bash
python3 -m pip install -U "huggingface_hub[hf_xet]"
hf auth login
export HF_XET_HIGH_PERFORMANCE=1
export K3_HOME=$HOME/k3-neuron
mkdir -p "$K3_HOME/model" "$K3_HOME/tokenizer"

hf download vcruz305/Kimi-K3-Neuron-IQ1S-GGUF \
  --local-dir "$K3_HOME/model" \
  --include "*.gguf" --include "k3_chat_template.jinja"

hf download moonshotai/Kimi-K3 \
  --local-dir "$K3_HOME/tokenizer" \
  --include "config.json" --include "generation_config.json" \
  --include "tokenization_kimi.py" --include "tiktoken.model" \
  --include "tokenizer_config.json"

test -f "$K3_HOME/model/k3-neuron-iq1s-00001-of-00009.gguf"
test -f "$K3_HOME/model/k3_chat_template.jinja"
test -f "$K3_HOME/tokenizer/tokenization_kimi.py"
test -f "$K3_HOME/tokenizer/tiktoken.model"
```

## 2. Build SparkInfer with the TP3 + speculative-decoding patch

Run on rank 0. The helper applies the exact, ordered 28-patch series. It stops if
the checkout is not the pinned base or has uncommitted changes.

```bash
export K3_HOME=$HOME/k3-neuron
mkdir -p "$K3_HOME/src"
git clone --depth 1 --branch main \
  https://github.com/vcruz305/kimi-k3-neuron-tp3-dgxspark-recipe \
  "$K3_HOME/src/recipe"
git clone https://github.com/gittensor-ai-lab/sparkinfer-k3.git \
  "$K3_HOME/src/sparkinfer-k3"
cd "$K3_HOME/src/sparkinfer-k3"
bash "$K3_HOME/src/recipe/scripts/apply_sparkinfer_patch_series.sh" "$K3_HOME/src/recipe"

cmake -S runtime -B build -DSPARKINFER_TP=ON
cmake --build build -j"$(nproc)" --target kimi_k3_dist_generate \
  tp_rank_local_loader_cpu_test tp_dist_generate_protocol_cpu_test \
  kimi_k3_tp_kda_check kimi_k3_tp_width_check kimi_k3_kda_batch_check \
  kimi_k3_spec_draft_check kimi_k3_tune_check
ctest --test-dir build --output-on-failure
```

Package the executable and runtime libraries, then copy exactly that directory to the
other two ranks. Do not mix a fresh executable with stale `.so` files.

```bash
export K3_HOME=$HOME/k3-neuron
cd "$K3_HOME/src/sparkinfer-k3"
mkdir -p "$K3_HOME/dist"
BIN=$(find build -type f -name kimi_k3_dist_generate -print -quit)
RUNTIME_SO=$(find build -type f -name libsparkinfer_runtime.so -print -quit)
MOE_SO=$(find build -type f -name libsparkinfer_moe.so -print -quit)
test -n "$BIN" && test -n "$RUNTIME_SO" && test -n "$MOE_SO"
cp "$BIN" "$RUNTIME_SO" "$MOE_SO" "$K3_HOME/dist/"
sha256sum "$K3_HOME/dist"/* | tee "$K3_HOME/dist/SHA256SUMS"

rsync -a "$K3_HOME/dist/" "$RANK1:$K3_HOME/dist/"
rsync -a "$K3_HOME/dist/" "$RANK2:$K3_HOME/dist/"
ssh "$RANK1" "cd '$K3_HOME/dist' && sha256sum -c SHA256SUMS"
ssh "$RANK2" "cd '$K3_HOME/dist' && sha256sum -c SHA256SUMS"
```

## 3. Launch the measured 12.5 TPS profile

Open all three terminals first. Start rank 0, then immediately start ranks 1 and 2 while
rank 0 is waiting for workers. The `--spec-draft 8`
flag is required on **all** ranks because it allocates the rollback ring; rank 0 alone
chooses drafts. `SPARKINFER_K3_PROJ_TOKS=8`, head banding, and the majority settings
are load-bearing for the measured structured/repeat-heavy profile.

On rank 0, use `--serve` to keep the model resident and accept API requests:

```bash
export K3_HOME=$HOME/k3-neuron
export LD_LIBRARY_PATH="$K3_HOME/dist:${LD_LIBRARY_PATH:-}"
export SPARKINFER_K3_MOE_WEPS=0 SPARKINFER_K3_GRAPH=0 NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0
export SPARKINFER_K3_DIST_HEAD_BAND=1 SPARKINFER_K3_PROJ_TOKS=8 SPARKINFER_K3_PREFILL_CHUNK=16
"$K3_HOME/dist/kimi_k3_dist_generate" --rank 0 --world 3 --listen 0.0.0.0:29500 \
  --model "$K3_HOME/model/k3-neuron-iq1s-00001-of-00009.gguf" --max-ctx 4096 \
  --serve 127.0.0.1:29600 --spec-draft 8 \
  --spec-ngram-min 1 --spec-ngram-max 8 --spec-min-occur 2 --spec-majority 2/3
```

On rank 1:

```bash
export K3_HOME=$HOME/k3-neuron
export RANK0=10.10.10.2
export LD_LIBRARY_PATH="$K3_HOME/dist:${LD_LIBRARY_PATH:-}"
export SPARKINFER_K3_MOE_WEPS=0 SPARKINFER_K3_GRAPH=0 NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0
export SPARKINFER_K3_DIST_HEAD_BAND=1 SPARKINFER_K3_PROJ_TOKS=8 SPARKINFER_K3_PREFILL_CHUNK=16
"$K3_HOME/dist/kimi_k3_dist_generate" --rank 1 --world 3 --coord "$RANK0:29500" \
  --model "$K3_HOME/model/k3-neuron-iq1s-00001-of-00009.gguf" --max-ctx 4096 --spec-draft 8 \
  --spec-ngram-min 1 --spec-ngram-max 8 --spec-min-occur 2 --spec-majority 2/3
```

On rank 2:

```bash
export K3_HOME=$HOME/k3-neuron
export RANK0=10.10.10.2
export LD_LIBRARY_PATH="$K3_HOME/dist:${LD_LIBRARY_PATH:-}"
export SPARKINFER_K3_MOE_WEPS=0 SPARKINFER_K3_GRAPH=0 NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0
export SPARKINFER_K3_DIST_HEAD_BAND=1 SPARKINFER_K3_PROJ_TOKS=8 SPARKINFER_K3_PREFILL_CHUNK=16
"$K3_HOME/dist/kimi_k3_dist_generate" --rank 2 --world 3 --coord "$RANK0:29500" \
  --model "$K3_HOME/model/k3-neuron-iq1s-00001-of-00009.gguf" --max-ctx 4096 --spec-draft 8 \
  --spec-ngram-min 1 --spec-ngram-max 8 --spec-min-occur 2 --spec-majority 2/3
```

Wait for rank 0 to report that serve mode is listening. A first load normally takes about
six minutes; subsequent requests reuse the loaded model.

## 4. Start the OpenAI-compatible endpoint

On rank 0, in a second terminal:

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

Verify the complete path:

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/v1/models -H "Authorization: Bearer $K3_API_KEY"
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $K3_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"kimi-k3-neuron","messages":[{"role":"user","content":"Reply with exactly: SparkInfer is ready."}],"max_tokens":32}'
```

Point an OpenAI-compatible agent at `http://RANK0:8000/v1`, use the value of
`K3_API_KEY`, and select `kimi-k3-neuron`. For example, a Hermes/OpenAI SDK client uses
`base_url="http://10.10.10.2:8000/v1"`, `api_key=<K3_API_KEY>`, and
`model="kimi-k3-neuron"`.

## Operational notes

- Use the speculative profile for recurrence-heavy, structured work. It is not a
  benchmark promise for open-ended agent prose; removing `--spec-draft` and the
  speculative environment variables returns ordinary greedy decode.
- The wrapper accepts `temperature` and `top_p` for client compatibility but currently
  uses greedy argmax. Requests are serialized; this is a low-QPS agent bridge, not a
  multi-tenant server.
- Tool declarations can be included in the prompt template, but generated tool calls are
  returned as text rather than parsed `tool_calls`. Give the agent a regular textual
  tool protocol if it needs tool execution today.
- Keep the three ranks on the same patch/build hashes. Use local NVMe and the fabric IP,
  not the management gateway, for rank coordination.

## More detail

- [APPLY.md](APPLY.md): exact patch series and validation evidence.
- [api-server/README.md](api-server/README.md): API behavior and response details.
- [12.5 TPS receipt](evidence/specdec-k8-p8-12tps-RECEIPT.md): workload, bracket, and
  non-claims.
- [DETAILS.md](DETAILS.md): architecture and historical investigation material.

## License

Patches follow the upstream SparkInfer and project license terms. The Kimi-K3 Neuron
weights are distributed separately under their Hugging Face terms.
