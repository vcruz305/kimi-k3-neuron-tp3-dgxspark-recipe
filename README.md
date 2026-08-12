# Kimi-K3 Neuron — multi-Spark TP3/TP4 (SparkInfer)

Serves the 330 GB Kimi-K3 Neuron IQ1_S GGUF across 3 or 4 NVIDIA DGX Spark (GB10) nodes.

> **✅ Correctness fix applied and verified, 2026-08-11.** The distributed (TP3/TP4)
> generation path had a bug in `kimi_k3_dist_rank.cpp`: it never set the GGUF capability
> flags that gate loading, so the shared experts, the routed-expert normalization, and the
> MLA attention gate were silently skipped on every token — real weights present in the
> file, never loaded, never computed. No crash, no NaN, just wrong output. Fixed by adding
> the same capability probe every other code path already had.
>
> Verified against the actual upstream Kimi-K3 architecture (an in-progress llama.cpp PR
> implementing this model natively, run independently as a reference): first-token
> prediction is now oracle-exact on every prompt tested, and previously-garbled output is
> coherent English. **Honest caveat, not yet fully explained**: beyond the first few tokens
> of a longer context, SparkInfer's output can drift from a bit-perfect reference (both
> stay fluent, but the exact wording can differ) — plausibly near-tied expert routing under
> 1-bit quantization flipping which experts get picked, not confirmed. Read this as
> "mechanism fixed, coherent, closely matches a reference at short range" — not "verified
> byte-identical to any reference implementation."
>
> **The speed numbers below predate this fix** and were measured on an engine skipping
> real computation — they do not apply to the corrected engine. A full kernel-flag sweep
> on the corrected engine found `SPARKINFER_K3_KDA_FUSE=0` as the best single tuning win
> (+20%, see [`docs/TP3-KERNEL-FLAG-SWEEP.md`](docs/TP3-KERNEL-FLAG-SWEEP.md)); a proper
> multi-prompt-median re-measurement of the corrected baseline is still pending — treat
> the table below as stale until that lands.

| | |
|---|---|
| **This recipe** | [`github.com/vcruz305/kimi-k3-neuron-tp3-dgxspark-recipe`](https://github.com/vcruz305/kimi-k3-neuron-tp3-dgxspark-recipe) |
| **Model (GGUF)** | [`huggingface.co/vcruz305/Kimi-K3-Neuron-IQ1S-GGUF`](https://huggingface.co/vcruz305/Kimi-K3-Neuron-IQ1S-GGUF) (~330 GB) |
| Patches | `0001`–`0024`, applied on top of upstream `sparkinfer-k3` — see [`APPLY.md`](APPLY.md) |
| Full data | benchmarks, geometry, run steps, non-claims → [`DETAILS.md`](DETAILS.md) |

## Why SparkInfer

**SparkInfer** is a standalone C++ inference engine (upstream `gittensor-ai-lab/sparkinfer-k3`)
with its own GGUF reader, hand-written CUDA kernels, and NCCL-based tensor-parallel runtime —
it is **not** llama.cpp or vLLM, it just reads the same GGUF format.

We picked it over the alternative vLLM/GGUF-plugin TP3 path because that path was tested
directly on the physical 3-Spark fleet and found to be at or beyond the real memory ceiling
at the current quantization width — not a tuning gap, a hard wall (independent watchdog,
Ray's own OOM monitor, and a real host wedge all confirmed it). SparkInfer's TP3 footprint
(~113 GiB/rank) fits comfortably under GB10's 121 GiB and runs correctly today. Details
in [`DETAILS.md`](DETAILS.md#related-vllm-path).

## Speed

| | Median decode | Load time (per rank) |
|---|---:|---:|
| 3× Spark (TP3) | **6.84 tok/s** | **~5m45s** (was ~30–60 min pre-patch 0013) |
| 4× Spark (TP4) | **7.945 tok/s** | ~5m45s |

Full benchmark tables, decode profile, NCCL fabric receipt, and forecast: [`DETAILS.md`](DETAILS.md#benchmarks-measured-on-real-sparks).

## Quick start

1. 3 or 4 Sparks, local NVMe (≥320 GB free/node), HF access to the gated model.
2. Pull the model + apply patches 0001–0013 — see [`APPLY.md`](APPLY.md).
3. Copy the built `dist/` tree to every rank, launch rank 0 with `--listen`, others with `--coord`.

Full step-by-step (model download, build flags, topology, multi-prompt launch): [`DETAILS.md`](DETAILS.md#full-run-steps).

## OpenAI-compatible API server

[`api-server/`](api-server/README.md) wraps `kimi_k3_dist_generate` with a
`POST /v1/chat/completions` HTTP server (chat-template tokenization, a rank0
`--serve` mode so the model stays resident between requests, streaming SSE
support). Adds `patches/sparkinfer/next/0014-tp-serve-mode-for-dynamic-prompt-requests.patch`
to the chain above. See `api-server/README.md` for build/run steps and known
limitations (greedy decoding only, one request in flight at a time).

## Speculative decoding (experimental, opt-in)

Patches `0021`–`0024` add n-gram/prompt-lookup speculative decoding to the distributed
path: rank0 looks for the current token sequence's most recent earlier occurrence in the
generation history, drafts the tokens that followed it, and verifies the whole draft in
one batched forward pass instead of one token at a time. No second model, no training —
just pattern-matching against what's already been generated.

**Measured on real TP3 hardware, 64-token generations, 3 workload types:**

| workload | baseline | with `--spec-draft 4` | delta |
|---|---:|---:|---:|
| code / structured output | 6.07 | 7.71 | **+27%** |
| literal repetition | 5.73 | 10.51 | **+83%** |
| freeform prose | 6.16 | 5.93 | **−3.7%** |
| **mean across all three** | 5.99 | **8.05** | **+34.5%** |

**Read this honestly, not optimistically.** It's a real, verified speedup on code and
structured/agentic output — the kind of thing this fleet is most likely serving. It is
**not** a general-purpose win: on freeform prose it's a measured, reproducible
*regression*, not just "no benefit." The prose loss comes from occasional false-positive
matches (short, coincidental token overlaps that are common in natural language and rare
in code) that trigger an expensive batched verify which then gets rejected. **Enable it
for code/agentic workloads. Leave it off for open-ended prose generation** until that
failure mode is tuned out (tracked, not yet done).

**Correctness**: verified against the same standard the rest of this engine is held to —
not necessarily byte-identical to non-speculative decoding (a documented, understood
NCCL-reduction-order effect makes that unreachable without a larger collective-comms
change — see [`docs/SPECULATIVE-DECODING-DESIGN.md`](docs/SPECULATIVE-DECODING-DESIGN.md)),
but always a valid greedy decoding trajectory. On a literal-repetition prompt, output is
bit-for-bit identical to non-speculative decoding; on code/prose, output diverges only at
positions where the model was already near-tied between two choices, and stays coherent
throughout. Backed by 1000+ automated correctness checks (see the patch series below).

### Usage

Apply patches `0021`→`0022`→`0023`→`0024` in order on top of `0020` (each depends on the
one before it touching the same files). Rebuild `kimi_k3_dist_generate` and
`libsparkinfer_runtime.so` and redeploy both to every rank (same sha256sum discipline as
the base recipe — **do not skip verifying the `.so`**, a stale runtime library behind a
fresh binary fails silently). Patch `0024` bumps the internal rank-to-rank protocol
version, so **all ranks must run the same patch level** — a mismatched rank fails loudly
at startup ("unsupported protocol version") rather than silently desyncing.

Add `--spec-draft K` to rank 0's launch command (K in `[2,16]`; `4` is the measured-best
value — `8` measured *worse* than `4`, don't assume bigger is better):

```bash
# rank 0
SPARKINFER_K3_PREFILL_CHUNK=64 ./kimi_k3_dist_generate --rank 0 --world 3 \
  --listen 0.0.0.0:29500 --model .../k3-neuron-iq1s-00001-of-00009.gguf \
  --prompt-ids <...> --n-predict <N> --max-ctx 4096 --spec-draft 4

# ranks 1, 2 — unchanged, no new flags needed
./kimi_k3_dist_generate --rank {1|2} --world 3 --coord HOST:29500 \
  --model ... --max-ctx 4096
```

`SPARKINFER_K3_PREFILL_CHUNK` must be `>= K` (the flag validates this and refuses to
start otherwise). Optionally tune the n-gram match window with `--spec-ngram-min N`
(default `2` — `1` cost freeform prose 12.5%, measured; `3` under-fires and drafts on
only ~16% of steps).

With `--spec-draft` omitted (the default), behavior and performance are **unchanged**
from before these patches — every existing benchmark and result elsewhere in this repo
remains valid.

Full technical writeup — design rationale, break-even math, the NCCL-determinism
investigation, and the staged build/verification process — in
[`docs/SPECULATIVE-DECODING-DESIGN.md`](docs/SPECULATIVE-DECODING-DESIGN.md).

## Give this to your agent

Paste this into an AI coding agent (Claude Code, etc.) with SSH access to your Spark fleet:

```
Set up Kimi-K3 Neuron serving across my DGX Spark fleet using this recipe:
https://github.com/vcruz305/kimi-k3-neuron-tp3-dgxspark-recipe

1. Read APPLY.md in that repo and apply patches 0001-0013 on top of the pinned
   gittensor-ai-lab/sparkinfer-k3 base commit, then build kimi_k3_dist_generate.
2. Read DETAILS.md's "Full run steps" section for the model download command,
   required dist/ files, topology, and launch commands. Follow it exactly —
   don't improvise flags.
3. I have [N] DGX Spark nodes reachable at: [list hostnames/IPs]. Use world=N,
   rank 0 as coordinator on [fabric IP], --listen 0.0.0.0:29500.
4. Download the GGUF from huggingface.co/vcruz305/Kimi-K3-Neuron-IQ1S-GGUF to
   local NVMe on every node (not NFS/sshfs — see DETAILS.md for why).
5. Launch a multi-prompt benchmark and report median decode tok/s dropping
   prompt0, plus confirm "OK finished clean" on every rank.
6. If anything fails, check DETAILS.md's Non-claims and the docs/ table first
   before treating it as a new bug — several sharp edges are already documented.
```

### Give this to your agent (API server)

Once the base recipe above is working (rank0 launched, model loaded, "OK
finished clean" confirmed), paste this to add the OpenAI-compatible HTTP
front end:

```
Add the OpenAI-compatible API server on top of my working Kimi-K3 Neuron
Spark fleet, following this recipe:
https://github.com/vcruz305/kimi-k3-neuron-tp3-dgxspark-recipe

1. Read api-server/README.md in that repo end to end first — it explains the
   design (rank0 --serve mode instead of the static --prompts-file path,
   llama.cpp's own vocab for tokenization) and why the wrapper is shaped the
   way it is. Don't improvise around it.
2. Apply patch 0014 (patches/sparkinfer/next/0014-tp-serve-mode-for-dynamic-prompt-requests.patch)
   on top of whatever patch level I already have, rebuild kimi_k3_dist_generate,
   and redeploy dist/ to every rank (same sha256sum discipline as the base recipe).
3. Get the model's native HF tokenizer directory (tokenization_kimi.py +
   tiktoken.model + tokenizer_config.json + generation_config.json — NOT a
   llama.cpp rebuild, the wrapper tokenizes in-process via `transformers`).
   Exact fetch steps and why this beats a llama.cpp-based tokenizer in
   api-server/README.md section "Tokenization".
4. Relaunch rank 0 with --serve 127.0.0.1:29600 instead of --prompts-file
   (ranks 1..N-1 unchanged). Confirm the "serve mode listening on ..." log line.
5. Set up the Python venv (api-server/requirements.txt) and start
   api-server/server.py with K3_TOKENIZER_DIR / K3_CHAT_TEMPLATE /
   K3_SERVE_HOST / K3_SERVE_PORT pointed at what you set up in steps 3-4.
6. Verify with a real curl request against POST /v1/chat/completions (see the
   example in api-server/README.md) and confirm you get back a coherent,
   non-garbled completion — not just an HTTP 200.
7. Report the known limitations honestly (greedy decoding only, one request
   in flight at a time, no tool-call parsing yet) — see api-server/README.md's
   "Known limitations" section. Don't claim more than what was actually tested.
```

## License

Patches follow upstream SparkInfer / project license terms in the base repo.
