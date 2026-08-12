#!/usr/bin/env python3
"""Offline contract smoke test for server.py; no DGX Spark or model download needed."""

import asyncio
import importlib.util
import json
import socketserver
import sys
import tempfile
import threading
import types
from pathlib import Path


class FakeTokenizer:
    eos_token_id = 1

    def __len__(self):
        return 163840

    def apply_chat_template(self, messages, **kwargs):
        assert messages[-1]["content"] == "hello world"
        assert kwargs["add_generation_prompt"] is True
        return [10, 11]

    def decode(self, ids, skip_special_tokens=False):
        lookup = {100: "Hello", 101: " world<|close|>"}
        return "".join(lookup[token] for token in ids)


class FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(*_args, **_kwargs):
        return FakeTokenizer()


class EngineHandler(socketserver.StreamRequestHandler):
    def handle(self):
        line = self.rfile.readline().decode("utf-8")
        if not line:
            return  # readiness TCP connect: no request to process
        if "stream=1" in line:
            self.wfile.write(b"tok=100\ntok=101\n")
        self.wfile.write(b"ok=1;generated_ids=100,101;stop_reason=stop\n")


class LocalEngine(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def load_server():
    repo_dir = Path(__file__).resolve().parent
    sys.modules["transformers"] = types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)
    spec = importlib.util.spec_from_file_location("k3_api_smoke_server", repo_dir / "server.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


async def run_checks(server, host, port, tokenizer_dir, template_path):
    import httpx

    server.TOKENIZER_DIR = str(tokenizer_dir)
    server.CHAT_TEMPLATE_PATH = str(template_path)
    server.SERVE_HOST = host
    server.SERVE_PORT = port
    server.MODEL_NAME = "kimi-k3-neuron"
    server.DEFAULT_MAX_TOKENS = 8
    server.MAX_TOKENS = 16
    server.MAX_CONTEXT_TOKENS = 32
    server._state.clear()
    server._startup()

    health = await server.healthz()
    ready = await server.readyz()
    models = await server.list_models()
    assert health["ok"] is True and ready["ok"] is True
    assert models["data"][0]["id"] == "kimi-k3-neuron"

    req = server.ChatCompletionRequest(
        model="kimi-k3-neuron",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "hello "}, {"type": "text", "text": "world"}
        ]}],
        max_completion_tokens=2,
    )
    response = await server.chat_completions(req)
    payload = json.loads(response.body)
    assert payload["choices"][0]["message"]["content"] == "Hello world"
    assert payload["usage"] == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}

    streamed = await server.chat_completions(req.model_copy(update={"stream": True}))
    wire = "".join([chunk async for chunk in streamed.body_iterator])
    assert '"role": "assistant"' in wire and '"content": "Hello world"' in wire
    assert wire.endswith("data: [DONE]\n\n")

    try:
        await server.chat_completions(req.model_copy(update={"model": "wrong-model"}))
        raise AssertionError("wrong model should be rejected")
    except server.HTTPException as exc:
        assert exc.status_code == 404 and exc.detail["code"] == "model_not_found"

    server.API_KEY = "smoke-secret"
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/healthz")).status_code == 200
        denied = await client.get("/v1/models")
        assert denied.status_code == 401 and denied.json()["error"]["code"] == "invalid_api_key"
        allowed = await client.get(
            "/v1/models", headers={"Authorization": "Bearer smoke-secret"}
        )
        assert allowed.status_code == 200
    server.API_KEY = ""


def main():
    server = load_server()
    engine = LocalEngine(("127.0.0.1", 0), EngineHandler)
    thread = threading.Thread(target=engine.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            (temp / "generation_config.json").write_text('{"eos_token_id": 2}', encoding="utf-8")
            template = temp / "k3_chat_template.jinja"
            template.write_text("unused by fake tokenizer", encoding="utf-8")
            asyncio.run(run_checks(server, "127.0.0.1", engine.server_address[1], temp, template))
    finally:
        engine.shutdown()
        engine.server_close()
    print("api-server smoke PASS: models, readiness, auth, chat, streaming, and errors")


if __name__ == "__main__":
    main()
