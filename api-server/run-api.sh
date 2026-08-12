#!/usr/bin/env bash
# Start the OpenAI-compatible wrapper beside rank0's resident --serve process.
# The rank0 engine itself must already be listening on K3_SERVE_HOST:K3_SERVE_PORT.
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python_bin=${K3_API_PYTHON:-"$script_dir/.venv/bin/python"}

: "${K3_TOKENIZER_DIR:?Set K3_TOKENIZER_DIR to the native K3 tokenizer directory}"
: "${K3_CHAT_TEMPLATE:?Set K3_CHAT_TEMPLATE to k3_chat_template.jinja}"

if [[ ! -x "$python_bin" ]]; then
  echo "API Python not found: $python_bin" >&2
  echo "Create it once with: python3 -m venv $script_dir/.venv && $script_dir/.venv/bin/pip install -r $script_dir/requirements.txt" >&2
  exit 2
fi
if [[ ! -d "$K3_TOKENIZER_DIR" || ! -f "$K3_CHAT_TEMPLATE" ]]; then
  echo "Tokenizer directory or chat-template file does not exist." >&2
  exit 2
fi

exec "$python_bin" "$script_dir/server.py" \
  --host "${K3_API_HOST:-127.0.0.1}" \
  --port "${K3_API_PORT:-8000}" \
  --tokenizer-dir "$K3_TOKENIZER_DIR" \
  --chat-template "$K3_CHAT_TEMPLATE" \
  --engine-host "${K3_SERVE_HOST:-127.0.0.1}" \
  --engine-port "${K3_SERVE_PORT:-29600}" \
  --model "${K3_MODEL_NAME:-kimi-k3-neuron}" \
  --default-max-tokens "${K3_DEFAULT_MAX_TOKENS:-512}" \
  --max-tokens "${K3_MAX_TOKENS:-512}" \
  --max-context-tokens "${K3_MAX_CONTEXT_TOKENS:-8192}" \
  --engine-timeout-s "${K3_ENGINE_TIMEOUT_S:-600}"
