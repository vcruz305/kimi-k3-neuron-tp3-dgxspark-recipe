# Current release state

Release candidate based on `82abfa9` (`main`), prepared 2026-08-12.

This repository publishes a reproducible **three-DGX-Spark TP3 SparkInfer patch
series** plus an optional, single-request OpenAI-compatible adapter. It does not
ship model weights, credentials, Spark host configuration, or a hosted service.

## Supported recipe

- Start from the exact SparkInfer base commit in [APPLY.md](APPLY.md).
- Apply **every patch 0001 through 0026** in the documented order using `git am`.
- Use all three ranks at the same resulting commit and binary/DSO build.
- Run the mandatory preflight and correctness gates in `APPLY.md` before serving.
- The qualified speculative profile is `--spec-draft 8 --spec-ngram-min 1
  --spec-ngram-max 8 --spec-min-occur 2 --spec-majority 2/3` with
  `SPARKINFER_K3_DIST_HEAD_BAND=1` and `SPARKINFER_K3_PROJ_TOKS=8`.
  It is workload-sensitive and off by default.

## Measured result — scope matters

The latest qualified TP3 structured-generation bracket is **12.55 decode tokens/s**
(candidate average: 12.5492 tok/s) on the specific three-Spark fleet and prompt
shape recorded in the release evidence. Its bracketing baseline was 6.4930 tok/s.
This is a structured-generation result, not a general throughput guarantee.

The same speculative strategy remained near baseline on freeform prose in qualification.
Do not quote 12.55 tok/s for arbitrary chat, concurrent use, long context, sampling,
or a different model/driver/network stack. Re-run the bracket on your own fleet and
publish the raw rank-0 logs with the result.

## API adapter status

`api-server/` is a low-QPS compatibility bridge for agents that speak
`POST /v1/chat/completions` (including Hermes-style OpenAI clients). It is:

- one request at a time, with a single shared KV cache;
- greedy decoding only; accepted `temperature` and `top_p` fields do not alter
  sampling;
- supports optional bearer authentication, but is not multi-tenant, batched,
  TLS-terminating, rate-limited, or Internet-safe by itself;
- dependent on the model's native Hugging Face tokenizer files supplied locally.

Set `K3_API_KEY` even on a trusted network. Put a TLS-terminating, rate-limiting reverse
proxy in front of it before exposing it beyond a trusted network.

## Release integrity

Run `bash scripts/release_check.sh` from the repository root after cloning. It checks
tracked-file checksums, validates patch syntax, rejects common secret material, and
flags historical scripts containing the original private lab paths. The historical
scripts are evidence/operator aids, not the portable public launcher.

For detailed history and measurements see [DETAILS.md](DETAILS.md). For launch steps
use the README and [APPLY.md](APPLY.md), not the older narrative documents.
