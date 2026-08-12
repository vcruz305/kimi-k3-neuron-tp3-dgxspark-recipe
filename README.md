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
> **Historical pre-fix speed tables are invalid** because they were measured on an engine
> skipping real computation. They are retained only in the detailed historical documents
> and must not be cited as current results. The corrected engine now has a bracketed,
> useful structured output past the requested line. The conservative K=4 profile held
> structured generation at **10.3002 and 11.3231 tok/s (10.8117 average)**. The newer
> repeat-heavy K=8/P8 profile independently held **12.5468 and 12.5516 tok/s (12.5492
> average)** around a matched 6.4930 baseline, with 112/112 drafts accepted on both sides
> and identical A/B generated token IDs. This is a structured-output result: its prose
> control averaged 6.5893 tok/s and its four-request mean was 9.7937 tok/s.

| | |
|---|---|
| **This recipe** | [`github.com/vcruz305/kimi-k3-neuron-tp3-dgxspark-recipe`](https://github.com/vcruz305/kimi-k3-neuron-tp3-dgxspark-recipe) |
| **Model (GGUF)** | [`huggingface.co/vcruz305/Kimi-K3-Neuron-IQ1S-GGUF`](https://huggingface.co/vcruz305/Kimi-K3-Neuron-IQ1S-GGUF) (~330 GB) |
| Patches | Complete 28-patch chain through `0026`; 0026 clean-applied, built, correctness-gated, and fleet-benchmarked on top of the audited prior chain — see [`APPLY.md`](APPLY.md#series-status--what-actually-applies) |
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

**Current corrected-engine measurements are available; a repeated-median headline is
still pending.** The old 6.84 / 7.945 tok/s table was measured on an engine that skipped
real computation and must not be cited. What is valid now:

- Load time: **~5m45s/rank** (was ~30-60 min pre-patch 0013) — this number is unaffected
  by the correctness fix and still valid.
- Best single kernel-tuning win on the corrected engine: `SPARKINFER_K3_KDA_FUSE=0`,
  **+20%** over its own (not-yet-headline) baseline — see
  [`docs/TP3-KERNEL-FLAG-SWEEP.md`](docs/TP3-KERNEL-FLAG-SWEEP.md).
- Current bracketed candidate (recursive-majority K=4 speculative decoding + distributed
  head banding): **10.3002 and 11.3231 tok/s on structured generation**, with 95/95
  accepted drafts in each candidate run. The two candidate sides averaged **10.8117
  tok/s**, +68.2% over the matched 6.4272 baseline. See
  [Speculative decoding](#speculative-decoding-experimental-opt-in).
- Repeat-heavy structured profile (recursive-majority K=8, exact P8 projection dispatch,
  distributed head banding): **12.5468 and 12.5516 tok/s**, averaging **12.5492 tok/s**
  around a matched 6.4930 baseline. Both candidates accepted 112/112 drafts and emitted
  identical token IDs. The four-request candidate mean was 9.7937 tok/s; prose was 6.5893.
  See [`evidence/specdec-k8-p8-12tps-RECEIPT.md`](evidence/specdec-k8-p8-12tps-RECEIPT.md).
- Both structured results use same-order candidate -> baseline -> candidate brackets. Do
  not relabel the 9.2345 K=4 or 9.7937 K=8/P8 four-request mean as a three-workload mean,
  or claim prose/general TPS above 10.

Full benchmark tables, decode profile, NCCL fabric receipt, and forecast (**historical,
pre-fix, clearly marked as such**): [`DETAILS.md`](DETAILS.md#benchmarks-measured-on-real-sparks).

## Quick start

1. 3 or 4 Sparks, local NVMe (≥320 GB free/node), HF access to the gated model.
2. Pull the model + apply the verified patch chain through 0026 — see [`APPLY.md`](APPLY.md).
3. Copy the built `dist/` tree to every rank, launch rank 0 with `--listen`, others with `--coord`.

Full step-by-step (model download, build flags, topology, multi-prompt launch): [`DETAILS.md`](DETAILS.md#full-run-steps).

## OpenAI-compatible API server

[`api-server/`](api-server/README.md) wraps `kimi_k3_dist_generate` with a
`POST /v1/chat/completions` HTTP server (chat-template tokenization, a rank0
`--serve` mode so the model stays resident between requests, streaming SSE
support). Serve mode is introduced by patch 0014 and was integration-tested through
0025; 0026 uses the same shared drafter/decode loop and was built and fleet-benchmarked,
but did not repeat the two-request HTTP test. See `api-server/README.md` for build/run steps and known
limitations (greedy decoding only, one request in flight at a time).

## Speculative decoding (experimental, opt-in)

Patches `0021`–`0026` add n-gram/prompt-lookup speculative decoding, match-confidence
gating, recursive-majority continuation, and distributed LM-head banding to the distributed
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

That is the original Stage 4 result. Patch 0025 fixes the false-positive-match failure
mode by requiring recurrence/continuation agreement, and adds bitwise-equivalent
distributed head banding. A five-run sequence on the merged serve+spec tree measured:

| config | code | repetition | prose | mean | vs avg baseline |
|---|---:|---:|---:|---:|---:|
| baseline before | 6.1612 | 5.7251 | 6.1622 | 6.0162 | — |
| head-band only | 6.4718 | 6.3173 | 6.3888 | 6.3926 | +5.19% |
| **confidence K=4 + head-band** | **8.1651** | **10.9018** | **6.5672** | **8.5447** | **+40.61%** |
| baseline after | 6.1477 | 6.1166 | 6.1496 | 6.1379 | — |

The average bracketing baseline is 6.07705 tok/s. Prose is now +8.07% rather than
regressing. Draft acceptance was 59/64 = 0.9219. Head banding was independently proven
bitwise-equivalent on 129/129 dumped logit rows.

Patch 0026 replaces brittle single-span copying, when explicitly enabled, with a
left-to-right recursive majority: at each draft position it selects the highest-order
history whose continuation has at least a 2/3 vote, appends that proposal to a temporary
history, then predicts the next position. On the real TP3 fleet:

| 0026 workload | candidate A | baseline | candidate B | candidate avg |
|---|---:|---:|---:|---:|
| coherent code | **10.0983** | 6.3621 | 8.8326 | 9.4655 |
| structured generation | **10.3002** | 6.4272 | **11.3231** | **10.8117** |
| freeform prose | 6.6009 | 6.4905 | 6.4693 | 6.5351 |
| coherent code, same-load repeat | **10.3358** | 6.4162 | 9.9156 | **10.1257** |
| four-request mean | 9.3338 | 6.4240 | 9.1352 | 9.2345 |

The run used `--spec-draft 4 --spec-ngram-min 1 --spec-ngram-max 8
--spec-min-occur 2 --spec-majority 2/3` and `SPARKINFER_K3_DIST_HEAD_BAND=1`.
Each candidate accepted **95/95 proposed tokens** and finished clean. The sustained 10+
claim applies to the structured-generation row, not prose or the cross-workload mean.
An exploratory K=5 run reached 11.9000 tok/s on that row, but its 9.3019 overall mean
did not displace K=4. Full logs,
hashes, methodology, and non-claims are in
[`evidence/specdec-majority-10tps-RECEIPT.md`](evidence/specdec-majority-10tps-RECEIPT.md).

The exact-width K=8/P8 follow-up changes the structured recommendation for repeat-heavy
output without changing the patch chain:

| K=8/P8 workload | candidate A | baseline | candidate B | candidate avg |
|---|---:|---:|---:|---:|
| coherent code | 9.3535 | 6.4746 | 8.6543 | 9.0039 |
| structured generation | **12.5468** | 6.4930 | **12.5516** | **12.5492** |
| freeform prose | 6.6040 | 6.4626 | 6.5746 | 6.5893 |
| coherent code, same-load repeat | 11.2682 | 6.4705 | 10.7961 | 11.0322 |
| four-request mean | 9.9431 | 6.4752 | 9.6442 | 9.7937 |

It uses `--spec-draft 8 --spec-ngram-min 1 --spec-ngram-max 8 --spec-min-occur 2
--spec-majority 2/3`, `SPARKINFER_K3_DIST_HEAD_BAND=1`, and critically
`SPARKINFER_K3_PROJ_TOKS=8`. Both candidates accepted **112/112** drafts and generated
identical token IDs on all prompts. This is a structured/repeat-heavy profile, not a
general or prose >12 tok/s claim. Full receipt:
[`evidence/specdec-k8-p8-12tps-RECEIPT.md`](evidence/specdec-k8-p8-12tps-RECEIPT.md).

**Correctness**: verified against the same standard the rest of this engine is held to —
not necessarily byte-identical to non-speculative decoding (a documented, understood
NCCL-reduction-order effect makes that unreachable without a larger collective-comms
change — see [`docs/SPECULATIVE-DECODING-DESIGN.md`](docs/SPECULATIVE-DECODING-DESIGN.md)),
but always a valid greedy decoding trajectory. On a literal-repetition prompt, output is
bit-for-bit identical to non-speculative decoding; on code/prose, output diverges only at
positions where the model was already near-tied between two choices, and stays coherent
throughout. Backed by 1000+ automated correctness checks (see the patch series below).

### Usage

Apply patches `0021`→`0022`→`0023`→`0024`→`0025`→`0026` in order on top of `0020`
(each depends on the one before it touching the same files). The chain through 0025 was
fresh-clone audited; 0026 plain-`git am` applied to that clean tip, built on GB10, and
matched the tested generator source. Rebuild `kimi_k3_dist_generate` and
`libsparkinfer_runtime.so` and redeploy both to every rank (same sha256sum discipline as
the base recipe — **do not skip verifying the `.so`**, a stale runtime library behind a
fresh binary fails silently). Patch `0024` bumps the internal rank-to-rank protocol
version, so **all ranks must run the same patch level** — a mismatched rank fails loudly
at startup ("unsupported protocol version") rather than silently desyncing.

Add `--spec-draft K` to rank 0's launch command (K in `[2,16]`; `4` remains the
lower-width conservative profile, while `8` with the exact P8 projection dispatch is the
measured repeat-heavy structured profile). Bigger is not automatically better: the P8
dispatch setting is load-bearing for the K=8 result. The recursive-majority command uses
`--spec-ngram-min 1 --spec-ngram-max 8
--spec-min-occur 2 --spec-majority 2/3` and the environment variable
`SPARKINFER_K3_DIST_HEAD_BAND=1`:

**`--spec-draft K` is required on every rank, not just rank 0.** The n-gram matching
decision is rank0-only, but `--spec-draft K` also arms each rank's local state-rollback
ring — a worker launched without it has no ring to roll back into, and will error out
the first time rank0 sends a rollback instruction (`kimi_k3_dist_forward_rollback`
fails with "was kimi_k3_dist_set_rollback called before the batch?"). All ranks need the
**same** K value.

```bash
# rank 0
SPARKINFER_K3_PREFILL_CHUNK=64 ./kimi_k3_dist_generate --rank 0 --world 3 \
  --listen 0.0.0.0:29500 --model .../k3-neuron-iq1s-00001-of-00009.gguf \
  --prompt-ids <...> --n-predict <N> --max-ctx 4096 --spec-draft 4 \
  --spec-ngram-min 1 --spec-ngram-max 8 --spec-min-occur 2 --spec-majority 2/3

# ranks 1, 2 — SAME --spec-draft K (required to arm rollback), rest unchanged
SPARKINFER_K3_PREFILL_CHUNK=64 ./kimi_k3_dist_generate --rank {1|2} --world 3 \
  --coord HOST:29500 --model ... --max-ctx 4096 --spec-draft 4
```

`SPARKINFER_K3_PREFILL_CHUNK` must be `>= K` **on every rank** (the flag validates this
and refuses to start otherwise). `--spec-ngram-min`/`--spec-ngram-max` are rank0-only.
The old warning that `n_min=1` hurt prose applies to legacy span copying; 0026's 2/3
majority gate declined every prose draft in the measured run. `--spec-majority` is opt-in,
so omitting it preserves 0025 behavior byte for byte.

For the measured K=8/P8 structured profile, use the same launch on every rank with
`--spec-draft 8`, set `SPARKINFER_K3_PROJ_TOKS=8`, and keep
`SPARKINFER_K3_DIST_HEAD_BAND=1`. Do not add `--spec-pad-pow2`; padding was a separate
experiment and was off in the bracketed 12.5492 tok/s result.

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

1. Read APPLY.md in that repo and apply its complete verified 28-patch chain through
   0026 on top of the pinned gittensor-ai-lab/sparkinfer-k3 base commit. Use the exact
   listed order, then build kimi_k3_dist_generate and the documented runtime libraries.
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
2. Follow APPLY.md's complete chain through 0026. Patch 0014 introduces serve mode and
   is already included; do not apply it twice if the full chain is present. Rebuild
   kimi_k3_dist_generate and the runtime libraries, then redeploy the same verified
   dist/ contents to every rank.
3. Get the model's native HF tokenizer directory (tokenization_kimi.py +
   tiktoken.model + tokenizer_config.json + generation_config.json — NOT a
   llama.cpp rebuild, the wrapper tokenizes in-process via `transformers`).
   Exact fetch steps and why this beats a llama.cpp-based tokenizer in
   api-server/README.md section "Tokenization".
4. Relaunch rank 0 with --serve 127.0.0.1:29600 instead of --prompts-file. Keep every
   worker at the identical protocol/patch level; if speculative decoding is enabled,
   pass the same --spec-draft K to every rank. Confirm the "serve mode listening on ..."
   log line.
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
