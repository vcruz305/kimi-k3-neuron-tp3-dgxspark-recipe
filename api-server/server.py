#!/usr/bin/env python3
"""OpenAI-compatible HTTP server in front of SparkInfer's kimi_k3_dist_generate.

Bridges POST /v1/chat/completions to the rank0 "--serve" control port added by
patches/sparkinfer/next/0014-tp-serve-mode-for-dynamic-prompt-requests.patch:

  request  -> tokenizer.apply_chat_template(k3_chat_template.jinja) -> prompt_ids
              -> TCP line to rank0 --serve socket
  response <- generated token ids (streamed as "tok=<id>" lines, or batched) <-
              tokenizer.decode() -> strip K3 message-wrapper markers ->
              OpenAI-shaped JSON / SSE

Tokenization uses the model's own native HF tokenizer in-process (see
`K3_TOKENIZER_DIR` below) -- verified on the real fleet against the actual
163840-token vocab this GGUF was quantized from, not a reimplementation. See
api-server/README.md for how that was confirmed and why an earlier
llama.cpp-based design (k3_vocab_helper) was dropped in favor of this.

The engine is a single sequential process (one KV cache, no batching): only one
request is in flight on the SparkInfer socket at a time, serialized here with a
lock. See api-server/README.md for the full design writeup and known limits.
"""

import argparse
import asyncio
import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

# ---- config -----------------------------------------------------------------

TOKENIZER_DIR = os.environ.get("K3_TOKENIZER_DIR", "")
CHAT_TEMPLATE_PATH = os.environ.get("K3_CHAT_TEMPLATE", "")
SERVE_HOST = os.environ.get("K3_SERVE_HOST", "127.0.0.1")
SERVE_PORT = int(os.environ.get("K3_SERVE_PORT", "29600"))
DEFAULT_MAX_TOKENS = int(os.environ.get("K3_DEFAULT_MAX_TOKENS", "512"))
MAX_TOKENS = int(os.environ.get("K3_MAX_TOKENS", "512"))
MAX_CONTEXT_TOKENS = int(os.environ.get("K3_MAX_CONTEXT_TOKENS", "8192"))
MODEL_NAME = os.environ.get("K3_MODEL_NAME", "kimi-k3-neuron")
API_KEY = os.environ.get("K3_API_KEY", "")
ENGINE_TIMEOUT_S = float(os.environ.get("K3_ENGINE_TIMEOUT_S", "600"))

# K3 chat template message-wrapper markers (see k3_chat_template.jinja: otag/ctag).
# add_generation_prompt=True makes the PROMPT end with the opening tag for the
# first channel we ask the model to continue -- <|open|>think<|sep|> when
# thinking=True, else <|open|>response<|sep|> -- so generated token ids never
# contain that opening tag themselves. If thinking=True the model emits the
# channel transition itself once it is done reasoning.
_THINK_TO_RESPONSE = "<|close|>think<|sep|><|open|>response<|sep|>"
_ANY_CLOSE = "<|close|>"


def extract_reply_text(raw: str, thinking: bool) -> tuple[str, Optional[str]]:
    """Split raw generated text into (content, reasoning_content).

    Generated ids never include the channel's own opening tag (that is part
    of the prompt -- see add_generation_prompt in k3_chat_template.jinja), so
    raw text starts mid-channel. With thinking=True the model emits the
    channel transition itself (`<|close|>think<|sep|><|open|>response<|sep|>`)
    once done reasoning; split there. Either way, truncate at the model's own
    closing tag if it emitted one before hitting max_tokens/stop.
    """
    if thinking:
        idx = raw.find(_THINK_TO_RESPONSE)
        if idx >= 0:
            reasoning = raw[:idx].strip()
            rest = raw[idx + len(_THINK_TO_RESPONSE):]
            end = rest.find(_ANY_CLOSE)
            content = (rest[:end] if end >= 0 else rest).strip()
            return content, reasoning
        # Cut off mid-thought: no response channel reached yet.
        end = raw.find(_ANY_CLOSE)
        return "", (raw[:end] if end >= 0 else raw).strip()

    end = raw.find(_ANY_CLOSE)
    return (raw[:end] if end >= 0 else raw).strip(), None


class ServeClientError(RuntimeError):
    pass


def openai_error(status_code: int, message: str, error_type: str = "invalid_request_error",
                 code: Optional[str] = None) -> HTTPException:
    """Create the error envelope expected by OpenAI SDK-compatible clients."""
    detail = {"message": message, "type": error_type}
    if code is not None:
        detail["code"] = code
    return HTTPException(status_code=status_code, detail=detail)


def serve_request_line(prompt_ids: list[int], n_predict: int, stop_ids: set[int], stream: bool) -> str:
    stop_csv = ",".join(str(i) for i in sorted(stop_ids))
    return (
        f"prompt_ids={','.join(str(i) for i in prompt_ids)};"
        f"n_predict={n_predict};stop_ids={stop_csv};stream={'1' if stream else '0'}"
    )


def parse_kv_line(line: str) -> dict[str, str]:
    out = {}
    for field in line.split(";"):
        if not field or "=" not in field:
            continue
        k, v = field.split("=", 1)
        out[k] = v
    return out


class SparkInferClient:
    """One TCP connection per request to rank0's --serve control port.

    The engine has exactly one KV cache and does not batch, so only one
    request may be in flight at a time -- enforced with an asyncio.Lock
    around the (blocking, run in a thread) socket call.
    """

    def __init__(self, host: str, port: int, timeout_s: float):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.lock = asyncio.Lock()

    def _do_request(self, line: str, on_token=None) -> dict[str, str]:
        with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as sock:
            sock.settimeout(self.timeout_s)
            sock.sendall((line + "\n").encode("utf-8"))
            buf = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    raise ServeClientError("SparkInfer serve socket closed unexpectedly")
                buf += chunk
                while b"\n" in buf:
                    raw_line, buf = buf.split(b"\n", 1)
                    text = raw_line.decode("utf-8", "replace")
                    if text.startswith("tok="):
                        try:
                            token_id = int(text[len("tok="):])
                        except ValueError as e:
                            raise ServeClientError("SparkInfer emitted an invalid token line") from e
                        if on_token is not None:
                            on_token(token_id)
                        continue
                    result = parse_kv_line(text)
                    if not result:
                        raise ServeClientError("SparkInfer emitted an empty response line")
                    return result

    async def request(self, prompt_ids: list[int], n_predict: int, stop_ids: set[int],
                       stream: bool, on_token=None) -> dict[str, str]:
        line = serve_request_line(prompt_ids, n_predict, stop_ids, stream)
        async with self.lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._do_request, line, on_token)


# ---- OpenAI-shaped request/response models ------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: object = None
    name: Optional[str] = None
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = Field(default=None, ge=1)
    # The current OpenAI API name. Supporting both keeps older agent clients
    # (which still send max_tokens) and current SDKs interchangeable.
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = None  # accepted, not used -- see README limits
    top_p: Optional[float] = None  # accepted, not used -- greedy engine
    n: int = Field(default=1, ge=1)
    tools: Optional[list] = None
    tool_choice: Optional[object] = None
    thinking: Optional[bool] = None


# ---- app ------------------------------------------------------------------

app = FastAPI(title="SparkInfer OpenAI-compatible API")
_state: dict = {}


@app.exception_handler(HTTPException)
async def _http_exception_handler(_request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {
        "message": str(exc.detail), "type": "invalid_request_error"
    }
    return JSONResponse(status_code=exc.status_code, content={"error": detail})


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(_request: Request, exc: RequestValidationError):
    # OpenAI-compatible APIs use 400, not FastAPI's default 422, for malformed
    # request bodies. Keep the message compact; detailed Pydantic internals are
    # not useful to an agent retrying a request.
    return JSONResponse(status_code=400, content={"error": {
        "message": "invalid request body: " + "; ".join(
            str(item.get("msg", "invalid value")) for item in exc.errors()
        ),
        "type": "invalid_request_error",
        "code": "invalid_request",
    }})


@app.on_event("startup")
def _startup():
    if not TOKENIZER_DIR:
        raise RuntimeError("K3_TOKENIZER_DIR env var is required (the model's native HF tokenizer dir)")
    if not CHAT_TEMPLATE_PATH:
        raise RuntimeError("K3_CHAT_TEMPLATE env var is required (path to k3_chat_template.jinja)")
    if (DEFAULT_MAX_TOKENS < 0 or MAX_TOKENS < 0 or MAX_CONTEXT_TOKENS <= 0
            or ENGINE_TIMEOUT_S <= 0):
        raise RuntimeError("K3 token limits must be non-negative (context must be positive)")
    if DEFAULT_MAX_TOKENS > MAX_TOKENS:
        raise RuntimeError("K3_DEFAULT_MAX_TOKENS cannot exceed K3_MAX_TOKENS")
    if not Path(TOKENIZER_DIR).is_dir():
        raise RuntimeError(f"K3_TOKENIZER_DIR is not a directory: {TOKENIZER_DIR}")
    if not Path(CHAT_TEMPLATE_PATH).is_file():
        raise RuntimeError(f"K3_CHAT_TEMPLATE is not a file: {CHAT_TEMPLATE_PATH}")

    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)
    _state["tok"] = tok
    _state["template"] = Path(CHAT_TEMPLATE_PATH).read_text(encoding="utf-8")
    _state["client"] = SparkInferClient(SERVE_HOST, SERVE_PORT, ENGINE_TIMEOUT_S)
    _state["engine_host"] = SERVE_HOST
    _state["engine_port"] = SERVE_PORT

    # Stop set: generation_config.json's own eos_token_id is the model's real
    # end-of-turn signal (for this tokenizer that's <|end_of_msg|>, distinct
    # from the tokenizer's generic [EOS] special token) -- confirmed on the
    # real fleet, not assumed. tok.eos_token_id is kept too as a safety net.
    stop_ids = {tok.eos_token_id}
    gen_cfg_path = Path(TOKENIZER_DIR) / "generation_config.json"
    if gen_cfg_path.exists():
        gen_cfg = json.loads(gen_cfg_path.read_text(encoding="utf-8"))
        eos = gen_cfg.get("eos_token_id")
        if isinstance(eos, list):
            stop_ids.update(int(x) for x in eos)
        elif eos is not None:
            stop_ids.add(int(eos))
    _state["stop_ids"] = stop_ids

    print(
        f"[api-server] ready model={MODEL_NAME} vocab_size={len(tok)} "
        f"engine={SERVE_HOST}:{SERVE_PORT} stop_ids={sorted(stop_ids)}"
    )


@app.middleware("http")
async def _optional_bearer_auth(request: Request, call_next):
    """Protect OpenAI endpoints when K3_API_KEY is configured.

    Health checks deliberately remain unauthenticated so an orchestrator can
    distinguish a reachable service from a bad application credential.
    """
    if API_KEY and request.url.path.startswith("/v1/"):
        expected = f"Bearer {API_KEY}"
        if request.headers.get("authorization") != expected:
            return JSONResponse(status_code=401, content={"error": {
                "message": "missing or invalid bearer token",
                "type": "authentication_error",
                "code": "invalid_api_key",
            }})
    return await call_next(request)


def prompt_ids_for_request(req: ChatCompletionRequest) -> list[int]:
    messages = []
    for msg in req.messages:
        data = msg.model_dump(exclude_none=True)
        content = data.get("content")
        if isinstance(content, list):
            # OpenAI clients commonly use content-part arrays even for plain
            # text. K3 is text-only; accept text parts and reject multimodal
            # parts explicitly instead of silently producing a wrong prompt.
            text_parts = []
            for part in content:
                if not isinstance(part, dict) or part.get("type", "text") != "text":
                    raise ValueError("only text message content is supported")
                text = part.get("text")
                if not isinstance(text, str):
                    raise ValueError("text content parts require a string 'text' field")
                text_parts.append(text)
            data["content"] = "".join(text_parts)
        elif content is None:
            data["content"] = ""
        elif not isinstance(content, str):
            raise ValueError("message content must be a string or text-part array")
        messages.append(data)
    kwargs = dict(
        chat_template=_state["template"],
        add_generation_prompt=True,
        thinking=req.thinking if req.thinking is not None else False,
        tokenize=True,
    )
    if req.tools:
        kwargs["tools"] = req.tools
    if req.tool_choice:
        kwargs["tool_choice"] = req.tool_choice
    # `conversation` (the messages list) is positional -- apply_chat_template()
    # does not accept it as a `messages=` keyword.
    prompt_ids = _state["tok"].apply_chat_template(messages, **kwargs)
    if not isinstance(prompt_ids, list) or not all(isinstance(token, int) for token in prompt_ids):
        raise ValueError("tokenizer did not return a token-id list")
    return prompt_ids


def decode_ids(ids: list[int]) -> str:
    if not ids:
        return ""
    return _state["tok"].decode(ids, skip_special_tokens=False)


def stop_ids_for_request() -> set[int]:
    return set(_state["stop_ids"])


def requested_token_count(req: ChatCompletionRequest) -> int:
    if req.max_tokens is not None and req.max_completion_tokens is not None:
        if req.max_tokens != req.max_completion_tokens:
            raise openai_error(
                400, "max_tokens and max_completion_tokens disagree", code="invalid_request"
            )
    n_predict = (req.max_completion_tokens if req.max_completion_tokens is not None
                 else req.max_tokens)
    if n_predict is None:
        n_predict = DEFAULT_MAX_TOKENS
    if n_predict > MAX_TOKENS:
        raise openai_error(
            400, f"requested completion exceeds server limit of {MAX_TOKENS} tokens",
            code="max_tokens_exceeded"
        )
    return n_predict


def validate_request(req: ChatCompletionRequest, prompt_ids: list[int]) -> int:
    if req.model != MODEL_NAME:
        raise openai_error(404, f"model '{req.model}' is not served here", "invalid_request_error",
                           "model_not_found")
    if req.n != 1:
        raise openai_error(400, "only n=1 is supported by this sequential engine", code="unsupported_value")
    n_predict = requested_token_count(req)
    if len(prompt_ids) > MAX_CONTEXT_TOKENS:
        raise openai_error(400, f"prompt exceeds server context limit of {MAX_CONTEXT_TOKENS} tokens",
                           code="context_length_exceeded")
    if len(prompt_ids) + n_predict > MAX_CONTEXT_TOKENS:
        raise openai_error(400, "prompt plus requested completion exceeds server context limit "
                           f"of {MAX_CONTEXT_TOKENS} tokens", code="context_length_exceeded")
    return n_predict


async def engine_request(*args, **kwargs) -> dict[str, str]:
    try:
        result = await _state["client"].request(*args, **kwargs)
    except (OSError, TimeoutError, ServeClientError) as e:
        raise openai_error(503, f"SparkInfer engine is unavailable: {e}", "server_error",
                           "engine_unavailable")
    if result.get("ok") != "1":
        raise openai_error(502, f"SparkInfer engine error: {result.get('error', 'malformed response')}",
                           "server_error", "engine_error")
    return result


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    try:
        prompt_ids = prompt_ids_for_request(req)
    except Exception as e:
        raise openai_error(400, f"prompt rendering/tokenize failed: {e}", code="invalid_request")

    n_predict = validate_request(req, prompt_ids)
    stop_ids = stop_ids_for_request()
    thinking = bool(req.thinking)
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    if not req.stream:
        result = await engine_request(prompt_ids, n_predict, stop_ids, stream=False)
        try:
            gen_ids = [int(x) for x in result.get("generated_ids", "").split(",") if x != ""]
        except ValueError as e:
            raise openai_error(502, "SparkInfer engine returned invalid generated token ids", "server_error",
                               "engine_protocol_error") from e
        raw_text = decode_ids(gen_ids)
        content, reasoning = extract_reply_text(raw_text, thinking)
        finish_reason = "stop" if result.get("stop_reason") == "stop" else "length"
        message = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning
        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(gen_ids),
                "total_tokens": len(prompt_ids) + len(gen_ids),
            },
        })

    async def sse():
        token_ids: list[int] = []
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_token(tok: int):
            loop.call_soon_threadsafe(queue.put_nowait, tok)

        async def run_request():
            try:
                return await engine_request(
                    prompt_ids, n_predict, stop_ids, stream=True, on_token=on_token
                )
            finally:
                await queue.put(None)

        task = asyncio.ensure_future(run_request())

        # Emit the OpenAI role-delta chunk first, matching /v1/chat/completions
        # streaming semantics (the first chunk carries role, no content yet).
        first_chunk = {
            "id": completion_id, "object": "chat.completion.chunk", "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(first_chunk)}\n\n"

        # Re-decode+re-extract the whole sequence so far on every flush rather
        # than decoding deltas: decoding a raw id at a time can split a
        # multi-token piece, and extract_reply_text needs to see the
        # think->response channel transition marker, which can itself span a
        # token boundary. A trailing margin of un-flushed ids absorbs both.
        # Cheap enough at conversational token counts; see README limits.
        flushed_content = ""
        flushed_reasoning = ""
        margin = 4

        async def flush(up_to: int):
            nonlocal flushed_content, flushed_reasoning
            raw = decode_ids(token_ids[:up_to])
            content, reasoning = extract_reply_text(raw, thinking)
            reasoning = reasoning or ""
            if reasoning.startswith(flushed_reasoning) and len(reasoning) > len(flushed_reasoning):
                delta_r = reasoning[len(flushed_reasoning):]
                yield_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk", "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"reasoning_content": delta_r}, "finish_reason": None}],
                }
                flushed_reasoning = reasoning
                yield f"data: {json.dumps(yield_chunk)}\n\n"
            if content.startswith(flushed_content) and len(content) > len(flushed_content):
                delta_c = content[len(flushed_content):]
                yield_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk", "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"content": delta_c}, "finish_reason": None}],
                }
                flushed_content = content
                yield f"data: {json.dumps(yield_chunk)}\n\n"

        while True:
            tok = await queue.get()
            if tok is None:
                break
            token_ids.append(tok)
            if len(token_ids) > margin:
                async for out in flush(len(token_ids) - margin):
                    yield out

        try:
            result = await task
        except HTTPException as e:
            # HTTP status cannot change after SSE starts. Emit a structured SSE
            # error and close cleanly so agents can retry rather than hanging.
            yield f"data: {json.dumps({'error': e.detail})}\n\n"
            yield "data: [DONE]\n\n"
            return
        async for out in flush(len(token_ids)):
            yield out

        finish_reason = "stop" if result.get("stop_reason") == "stop" else "length"
        final_chunk = {
            "id": completion_id, "object": "chat.completion.chunk", "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}]}


@app.get("/healthz")
async def healthz():
    return {"ok": "tok" in _state, "model": MODEL_NAME,
            "vocab_size": len(_state["tok"]) if "tok" in _state else None}


@app.get("/readyz")
async def readyz():
    """Readiness includes a bounded TCP connect to rank0's resident engine."""
    if "client" not in _state:
        raise openai_error(503, "tokenizer is not initialized", "server_error", "not_ready")
    try:
        with socket.create_connection((_state["engine_host"], _state["engine_port"]), timeout=1):
            pass
    except OSError as e:
        raise openai_error(503, f"SparkInfer engine is not reachable: {e}", "server_error", "not_ready")
    return {"ok": True, "model": MODEL_NAME,
            "engine": f"{_state['engine_host']}:{_state['engine_port']}"}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("K3_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("K3_API_PORT", "8000")))
    parser.add_argument("--tokenizer-dir", default=TOKENIZER_DIR)
    parser.add_argument("--chat-template", default=CHAT_TEMPLATE_PATH)
    parser.add_argument("--engine-host", default=SERVE_HOST)
    parser.add_argument("--engine-port", type=int, default=SERVE_PORT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--default-max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--max-context-tokens", type=int, default=MAX_CONTEXT_TOKENS)
    parser.add_argument("--api-key", default=API_KEY,
                        help="optional bearer token required for /v1/* endpoints")
    parser.add_argument("--engine-timeout-s", type=float, default=ENGINE_TIMEOUT_S)
    args = parser.parse_args()
    TOKENIZER_DIR = args.tokenizer_dir
    CHAT_TEMPLATE_PATH = args.chat_template
    SERVE_HOST = args.engine_host
    SERVE_PORT = args.engine_port
    MODEL_NAME = args.model
    DEFAULT_MAX_TOKENS = args.default_max_tokens
    MAX_TOKENS = args.max_tokens
    MAX_CONTEXT_TOKENS = args.max_context_tokens
    API_KEY = args.api_key
    ENGINE_TIMEOUT_S = args.engine_timeout_s
    uvicorn.run(app, host=args.host, port=args.port)
