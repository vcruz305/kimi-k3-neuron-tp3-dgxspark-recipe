# api-server -- OpenAI-compatible wrapper for SparkInfer

Serves `POST /v1/chat/completions` (OpenAI schema) in front of the multi-Spark
`kimi_k3_dist_generate` engine from this recipe, without paying the ~5m45s
model load on every request.

## Design decision: rank0 serve mode, not a new engine

`kimi_k3_dist_generate` normally reads a fixed prompt list (`--prompt-ids` /
`--prompts-file`), runs it once, and exits -- fine for benchmarking, useless
for a server that must accept requests arriving at arbitrary times.

Two ways to close that gap were considered:

1. **Add a persistent request-loop mode to rank0** (chosen). After
   bootstrap+load, rank0 opens a second local TCP port and loops accepting one
   request at a time instead of iterating a static file, reusing the exact
   same `one_step()` / KV-reset / prefill / decode code multi-prompt mode
   already has. Smallest possible diff (one file, `runtime/examples/
   kimi_k3_dist_generate.cpp`), zero changes to the inter-rank NCCL/TCP
   protocol (`rank_protocol.*`, `rank_transport.*`) -- rank0 is still the only
   process workers ever talk to.
2. Some external process-supervisor approach that restarts
   `kimi_k3_dist_generate` per request. Rejected outright: the whole point is
   avoiding the multi-minute reload, and restarting would pay it every time.

See `patches/sparkinfer/next/0014-tp-serve-mode-for-dynamic-prompt-requests.patch`.
It adds a rank0-only `--serve HOST:PORT` flag. The existing
`--prompt-ids`/`--prompts-file` static path (used for the TP3/TP4 benchmarks
elsewhere in this recipe) is untouched -- serve mode is a separate branch
taken only when `--serve` is passed.

### Wire protocol (rank0 <-> this wrapper only, not inter-rank)

One line per request, semicolon-separated `key=value` fields:

```
prompt_ids=1,2,3;n_predict=128;stop_ids=17,42;stream=1
```

With `stream=1`: one `tok=<id>` line per sampled token, then one terminating
line. Without it: only the terminating line.

```
ok=1;generated_ids=45,67,89;prefill_s=0.123;decode_s=4.56;stop_reason=eos|length
ok=0;error=...
```

`shutdown=1` cleanly calls the engine's existing `ctr.finish()` and exits all
ranks. This is deliberately not the `rank_protocol::Message` binary framing --
that framing carries cluster-identity fields for inter-rank NCCL bootstrap.
This is a separate, minimal, single-caller protocol; a hand-rolled `key=value`
line matches the style already used elsewhere in this file (`parse_ids_csv`,
`load_prompts_file`) instead of pulling in a JSON library for a flat schema.

One connection at a time: the engine has exactly one KV cache and does not
batch, so there is nothing to gain from concurrency here. The Python wrapper
enforces this with an `asyncio.Lock` around the SparkInfer socket call.

## Tokenization: the model's own native HF tokenizer, in-process

Prompts are raw token ids, but chat messages are text. Turning one into the
other needs the exact vocabulary and pre-tokenizer the GGUF was built with.

Initial design used a llama.cpp-based helper (llama.cpp has a dedicated
`kimi-k2` pretokenizer, `LLAMA_VOCAB_PRE_TYPE_KIMI_K2` in
`src/llama-vocab.cpp`, and the GGUF embeds the matching vocab as metadata).
That was superseded once fleet recon during the live test turned up something
better already sitting on rank0: a native HF tokenizer download at
`/home/victor/models/k3-tokenizer/` (`tokenization_kimi.py` + `tiktoken.model`
-- the actual reference tokenizer, not llama.cpp's independent
reimplementation of it). Verified directly on the fleet before adopting it:

- `AutoTokenizer.from_pretrained(dir, trust_remote_code=True)` loads a
  `TikTokenTokenizer` with `vocab_size == 163840`, matching `spec.vocab` in
  `kimi_k3_dist_generate.cpp` exactly -- confirms it's the right tokenizer for
  this exact quantized model.
- `tok.apply_chat_template(messages, chat_template=<k3_chat_template.jinja
  contents>, add_generation_prompt=True, thinking=..., tokenize=True)` renders
  through transformers' own Jinja engine and produces byte-identical prompt
  text to an independent plain-`jinja2` rendering tried earlier -- and returns
  token ids directly, no separate encode step needed.
- `tok.encode()` does **not** prepend BOS for this tokenizer/template
  (verified: `encode(text)` and `encode(text, add_special_tokens=False)` are
  identical and neither starts with `bos_token_id`) -- confirmed empirically
  rather than assumed.
- `tok.decode(ids, skip_special_tokens=False)` renders special tokens as
  literal text (`<|open|>`, `<|close|>`, `<|sep|>`, `<|end_of_msg|>`), which
  `server.py` needs to find where the assistant's reply starts and ends.
- **The real stop signal is not `tok.eos_token_id`.** `tok.eos_token_id` is
  `[EOS]` (163585), but `generation_config.json` at the tokenizer dir sets
  `eos_token_id: 163586`, which is `<|end_of_msg|>` -- confirmed by encoding
  `<|end_of_msg|>` and getting back the single id 163586. `server.py` reads
  `generation_config.json` at startup and uses *that* as the primary stop id,
  with `tok.eos_token_id` kept as a secondary safety net.

No C++ tokenizer helper, no separate llama.cpp build, no subprocess per
request -- the tokenizer loads once at server startup and runs in-process.

### Why the reply-extraction logic looks the way it does

`add_generation_prompt=True` makes the *rendered prompt* end with the opening
tag for the channel we ask the model to continue --
`<|open|>think<|sep|>` when `thinking=true`, else `<|open|>response<|sep|>`.
That tag is fed into the model during prefill; it is never part of the
*generated* token ids. So `server.py`'s `extract_reply_text()` does not look
for an opening tag -- it looks for the model's own closing tag, and (when
`thinking=true`) for the channel transition the model itself emits
(`<|close|>think<|sep|><|open|>response<|sep|>`) once it is done reasoning.
Verified against the real template with plain messages, `thinking=true`, and
a `tools` declaration, both independently and via `apply_chat_template`.

## Build

### 1. SparkInfer with the verified patch chain

Follow `APPLY.md`'s complete 28-patch chain through 0026. Patch 0014 introduces
`--serve`; patches 0021–0026 are merged with it. The serve/protocol path through 0025
was verified in a real two-request serve test; 0026 uses the shared drafter loop and was
built and TP3-benchmarked, but did not repeat that HTTP test. Build
`kimi_k3_dist_generate` normally and copy the same binary + `.so`s to
every rank (see `APPLY.md` / `DETAILS.md`).

### 2. Python wrapper

```bash
python3 -m venv api-server/venv
api-server/venv/bin/pip install -r api-server/requirements.txt
```

Needs the model's native tokenizer directory (`config.json`,
`tokenization_kimi.py`, `tiktoken.model`, `tokenizer_config.json`,
`generation_config.json` -- see `dl_k3_tokenizer.py`/`.sh` if you need to
re-fetch it) and the `k3_chat_template.jinja` downloaded alongside the GGUF.

## Run

Rank 0 (same env/flags as a normal launch, `--serve` instead of `--prompts-file`):

```bash
./kimi_k3_dist_generate --rank 0 --world 3 --listen 0.0.0.0:29500 \
  --model "$MODEL" --serve 127.0.0.1:29600 --max-ctx 8192
```

Ranks 1-2 use the normal worker launch (`--coord HOST:29500`) at the exact same patch
level. If rank0 uses `--spec-draft K`, pass the same value on every worker so each rank's
rollback ring is armed; see the top-level README for the measured tuning flags.

Once rank0 logs `serve mode listening on 127.0.0.1:29600`, start the wrapper
(same host as rank0, since `--serve` binds loopback by default):

```bash
export K3_TOKENIZER_DIR=$HOME/models/k3-tokenizer
export K3_CHAT_TEMPLATE=$HOME/models/kimi-k3-neuron-iq1s-local/k3_chat_template.jinja
export K3_SERVE_HOST=127.0.0.1
export K3_SERVE_PORT=29600
api-server/venv/bin/python api-server/server.py --host 0.0.0.0 --port 8000
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"kimi-k3-neuron","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":64}'
```

Streaming: add `"stream": true` and read the response as SSE
(`text/event-stream`, standard OpenAI `chat.completion.chunk` shape, plus a
non-standard `reasoning_content` delta field when `"thinking": true` is set).

## Known limitations (be honest about these)

- **Greedy decoding, now with a repetition guard, still no real sampler.**
  `kimi_k3_dist_generate`'s decode loop is fixed argmax -- see "Degenerate
  output" below for why that alone produced unusable completions, and
  `patches/sparkinfer/next/0016-tp-repetition-guard-for-greedy-decode.patch`
  for the deterministic fix (no-immediate-repeat + a rolling occurrence cap,
  applied to rank0's logits before argmax; no RNG, worker/protocol side
  untouched). `temperature`/`top_p` are still accepted in the request schema
  for client compatibility but have no effect -- real sampling would need a
  seeded RNG on rank0 broadcasting its choice the same way argmax's choice
  already is, which is future work if the guard alone isn't enough in
  practice (status of that: see below).
- **No batching, one request at a time.** Single KV cache, single sequence.
  Concurrent requests queue behind an `asyncio.Lock`; this is a low-QPS
  bridge for a 330GB single-stream model, not a multi-tenant server.
- **Streaming re-decodes the whole sequence so far on every flush** (a few
  tokens' worth of margin, not per-token) rather than decoding incremental
  deltas, because decoding a single raw id can split a multi-token UTF-8/BPE
  piece, and the think-to-response channel transition marker can itself span
  a token boundary. Fine at conversational token counts; do not read this as
  a general streaming-detokenization pattern.
- **Tool-call responses are not parsed back out of model output** in this
  version -- tool *declarations* (`tools=[...]` in the request) render
  correctly into the prompt (verified against the real template), but the
  wrapper returns raw `content` rather than populating `choices[].message
  .tool_calls` from the model's own `<|open|>tools<|sep|>...` markers. Add a
  small parser for that if/when a client actually needs function calling.
- Not multi-user production serving -- see the top-level `DETAILS.md`
  non-claims list, which still applies to the underlying engine.

## Status

**Current status: working and verified on the real 3-Spark fleet.** The complete chain
through 0026 applies cleanly from the pinned base. The merged serve + speculative-decoding
path through 0025 was rebuilt on GB10 and exercised with two sequential requests: request 1 streamed
16 tokens, request 2 forced the distributed KV/KDA reset and returned 16 tokens, then all
ranks shut down cleanly. Both client and driver checks passed.

The earlier degenerate-output investigation below was resolved by patch 0019: the
distributed rank loader had never populated the GGUF capability flags, so real
shared-expert, routed-normalization, and MLA-gate computation was silently omitted. That
was an engine bug, not an API-wrapper, tokenizer, template, or quantization limitation.
Ordinary greedy decode retains the repetition guard; speculative verification deliberately
bypasses it so literal repetition remains draftable. See the top-level README and
`evidence/specdec-serve-merge-RECEIPT.md` for current performance and test evidence.

### Historical investigation (resolved by patch 0019)

**The following text records the state before the root cause was known; it is not current.**

What was run for real: patch 0014 applied (plain `git apply`, not `git am` --
see note below), incremental rebuild, all 3 ranks loaded clean in serve mode,
the Python wrapper started against the real native tokenizer, and a real
`curl POST /v1/chat/completions` round-tripped through the whole stack --
prompt rendered via the real `k3_chat_template.jinja`, tokenized by the real
tokenizer, sent over the new `--serve` socket, generated by the real 3-rank
TP3 engine, detokenized, and returned as an OpenAI-shaped JSON response with
correct `usage` counts and timing (`prefill_s`/`decode_s` in line with this
recipe's documented tok/s). One real bug was caught and fixed this way:
`apply_chat_template()`'s messages argument is positional (`conversation`),
not a `messages=` keyword -- the first live request surfaced this immediately
as a clean 400 error, exactly as intended.

**But the actual generated content is degenerate**: greedy decode collapses
to one token repeated for the entire completion (e.g. token 53408 = "atos"),
regardless of prompt. Isolated with a raw single-token prompt id sent
directly to the `--serve` socket, bypassing the chat template and Python
tokenizer entirely -- still degenerate (token 220, which decodes to a plain
space character -- a textbook greedy-decoding collapse attractor). That
rules out this wrapper's tokenization/template/protocol code as the cause.

This is **not believed to be a bug introduced this session**, by this patch
or otherwise. Working with whoever owns the fleet's kernel/collective tuning
work to pin it down: `kimi_k3_dist_forward.cpp`'s uncommitted diff is the
already-validated "stream syncfix" (changes *when* sync happens, not any
computed value; its own token-selection line, `kimi_k3_dist_argmax`, is
byte-identical old vs new). The same repeated-token pattern has also been
observed independently on the standard, long-validated `--prompts-file`
path this same session, and nothing in this repo's documented validation
(`APPLY.md`'s TPS runs, this recipe's benchmarks) ever decodes
`generated_ids` back to text -- only `decode_tok_s` / "OK finished clean"
are checked. Current best explanation: pure greedy argmax (no repetition
penalty, no sampler anywhere in the engine) on an IQ1_S (~1.56 bit/weight)
quantization is simply prone to this failure mode, and nobody had looked at
decoded output content before now. Not fully confirmed -- see the session's
fleet-test report for the live discussion -- but this patch (0014, scoped to
`kimi_k3_dist_generate.cpp` only) is not the leading suspect.

**Update, now live-tested:** `patches/sparkinfer/next/0016-tp-repetition-guard-for-greedy-decode.patch`
adds a minimal, deterministic no-immediate-repeat + occurrence-cap guard
applied to rank0's logits before both decode loops' argmax calls. It does
what it was designed to do: the single-token infinite collapse is gone (no
more 32x "atos" or 8x " "). **But the result is still not usable output.**
With the guard, the same "capital of France" prompt now produces
`"atosetaryatosetaryatosetaryatosetaryitos Textsitos Textsitos Textsitos
Textsettiettoettiettoettiettoettiettoitolsitolsitolsitols"` -- no longer one
token forever, but a sequence of short repeating garbage fragments, none of
them real words. A second test with a maximally trivial prompt ("Say
hello.") produced equally incoherent output
(`"atosetaryatosetaryatosetaryatosetaryettoitosettoitosettoitosettoetti"`)
from the very first generated token.

That second result is the important one: **a repetition guard cannot fix
this.** No plausible amount of anti-repetition tuning explains a
maximally-predictable prompt producing garbage from token 1 -- the problem
is upstream of decoding strategy entirely.

**Three follow-up diagnostics ruled out the cheap explanations (chat
template, instruction-tuning mismatch, thinking-mode artifact, "just needs
more tokens"):**

1. Long generation (250 tokens, `thinking: false` explicit) on the "capital
   of France" prompt: never stabilizes. It cycles through the same handful
   of degenerate fragments ("atosetary", "itos Texts", "UPDinum", "ottootas",
   ...) in a larger repeating meta-cycle for the entire 250 tokens. No
   transition point, no real words at any point.
2. `thinking: false` was already the default in tests 1-2 above and was
   also explicit in test 1 here -- ruled out as the cause on its own.
3. **The most forgiving test possible**: bypass the chat template and
   instruction-following entirely. Tokenize the plain prefix `"The capital
   of France is"` (no system/user/assistant structure) and greedy-generate
   directly. A base-model-style continuation of a simple factual prefix is
   about as easy as generation gets. Result: `"The capital of France is, ,
   , , Je the Je je Je je Je je the je the  the.:.:"` -- still complete
   garbage (with a stray, telling flicker of French "Je"/"je" -- some
   language signal is present, just no coherent sentence formation).

None of the three cheap explanations hold. This is not a chat-template
mismatch, not a thinking-mode issue, and not "too few tokens to reach the
real answer" -- it fails on the easiest possible input. That leaves either
(a) a real numerical bug somewhere in the current forward pass (uncommitted
kernel/collective tuning work, or something else), or (b) this IQ1_S
(~1.56 bit/weight) quantization genuinely cannot produce coherent English
text right now, independent of sampling strategy -- and if so, that is a
significant, previously undocumented finding for this whole recipe, not
just this wrapper. Escalated back to the team with these results rather
than continuing to guess; next step is likely a reference comparison
(e.g. llama.cpp on the same GGUF) to distinguish an engine bug from a
quantization ceiling. See the session's fleet-test report for the live
discussion and whatever comes out of it.

That investigation also exposed an uncaptured multi-prompt/KV-reset dependency between
0013 and 0014. It is now patch 0015, including the protocol-side `-2` reset-sentinel
allowances. The complete series no longer needs a dirty-tree `git apply` workaround;
the documented 28-patch order succeeds with plain `git am`.
