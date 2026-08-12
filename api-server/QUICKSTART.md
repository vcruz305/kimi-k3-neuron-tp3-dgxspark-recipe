# OpenAI-compatible API quickstart

This adapter keeps the patched three-Spark SparkInfer engine resident and
exposes its single sequential stream at `POST /v1/chat/completions`. It is a
good fit for one trusted agent (for example Hermes) or a small internal tool;
it is not a multi-tenant inference service.

## 1. Start the qualified TP3 engine

Build and copy the exact same `kimi_k3_dist_generate`,
`libsparkinfer_runtime.so`, and `libsparkinfer_moe.so` to all three Sparks
using the top-level recipe. On each Spark, run the foreground launcher with
the same model path and binary directory. Replace the three addresses with
your private Spark network addresses.

```bash
# Spark 0 (rank 0): also hosts the API wrapper.
RANK=0 MODEL=/models/k3/Kimi-K3-UD-IQ1_S-00001-of-00009.gguf \
DIST=/opt/k3-tp3/dist LISTEN=10.10.10.2:29500 SERVE=127.0.0.1:29600 \
bash scripts/launch_tp3_rank.sh

# Spark 1, in a second terminal/host.
RANK=1 MODEL=/models/k3/Kimi-K3-UD-IQ1_S-00001-of-00009.gguf \
DIST=/opt/k3-tp3/dist COORD=10.10.10.2:29500 \
bash scripts/launch_tp3_rank.sh

# Spark 2, in a third terminal/host.
RANK=2 MODEL=/models/k3/Kimi-K3-UD-IQ1_S-00001-of-00009.gguf \
DIST=/opt/k3-tp3/dist COORD=10.10.10.2:29500 \
bash scripts/launch_tp3_rank.sh
```

Wait for rank 0 to print `serve mode listening`. The `SERVE` port must stay on
loopback: it is a small unauthenticated engine-control protocol, not a public
HTTP API.

## 2. Start the HTTP adapter on rank 0

Run this once on rank 0 after the engine is listening. The native K3 tokenizer
and the chat template must match the GGUF/model release.

```bash
cd /opt/k3-neuron-spark-tp3
python3 -m venv api-server/.venv
api-server/.venv/bin/pip install -r api-server/requirements.txt

export K3_TOKENIZER_DIR=/models/k3-tokenizer
export K3_CHAT_TEMPLATE=/models/k3/k3_chat_template.jinja
export K3_MODEL_NAME=kimi-k3-neuron
export K3_SERVE_HOST=127.0.0.1
export K3_SERVE_PORT=29600
export K3_API_HOST=0.0.0.0             # use 127.0.0.1 if the agent is local
export K3_API_PORT=8000
export K3_API_KEY='replace-with-a-long-random-secret'
bash api-server/run-api.sh
```

`K3_MAX_CONTEXT_TOKENS` defaults to 8192 and must not exceed the engine's
`--max-ctx`; `K3_MAX_TOKENS` and `K3_DEFAULT_MAX_TOKENS` default to 512.
Set them explicitly if you launch the engine with a different context window.
The wrapper allows one request at a time and returns a clear `503` while the
engine is unavailable.

## 3. Verify, then connect an agent

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $K3_API_KEY"

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $K3_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"kimi-k3-neuron","messages":[{"role":"user","content":"Reply with one short greeting."}],"max_completion_tokens":64}'
```

For an OpenAI-compatible agent client, set these values (the exact setting
names vary by agent):

```bash
export OPENAI_BASE_URL=http://RANK0_PRIVATE_OR_VPN_ADDRESS:8000/v1
export OPENAI_API_KEY="$K3_API_KEY"
export OPENAI_MODEL=kimi-k3-neuron
```

The API supports `GET /v1/models`, non-streaming and SSE streaming
`POST /v1/chat/completions`, `max_tokens` and `max_completion_tokens`, text
content arrays, and optional `thinking`. It intentionally rejects image
content, `n > 1`, unknown model IDs, and requests exceeding the configured
context/token limits rather than silently producing a different request.

Run the offline adapter contract check at any time; it creates a fake local
engine and does not contact a Spark or load model weights:

```bash
api-server/.venv/bin/python api-server/smoke_test.py
```

## Exposure and operations

Keep port 8000 on a private network or VPN. `K3_API_KEY` is an optional bearer
token, not TLS, rate limiting, or tenant isolation. Put a real authenticated
reverse proxy in front of it before exposing it beyond a trusted network.
Do not expose the loopback `--serve` port. `GET /healthz` is a tokenizer
liveness check; `GET /readyz` also makes a short TCP reachability check to
rank 0's engine-control port.
