# Kimi-K3 Neuron — multi-Spark TP3/TP4 (SparkInfer)

Serves the 330 GB Kimi-K3 Neuron IQ1_S GGUF across 3 or 4 NVIDIA DGX Spark (GB10) nodes.
**Working end-to-end, this is the recommended path.**

| | |
|---|---|
| **This recipe** | [`github.com/vcruz305/kimi-k3-neuron-tp3-dgxspark-recipe`](https://github.com/vcruz305/kimi-k3-neuron-tp3-dgxspark-recipe) |
| **Model (GGUF)** | [`huggingface.co/vcruz305/Kimi-K3-Neuron-IQ1S-GGUF`](https://huggingface.co/vcruz305/Kimi-K3-Neuron-IQ1S-GGUF) (~330 GB) |
| Patches | `0001`–`0013`, applied on top of upstream `sparkinfer-k3` — see [`APPLY.md`](APPLY.md) |
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

## License

Patches follow upstream SparkInfer / project license terms in the base repo.
