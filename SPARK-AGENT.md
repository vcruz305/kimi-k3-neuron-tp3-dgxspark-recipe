# SPARK-AGENT.md — instructions for LLM agents controlling DGX Sparks

> **⚠️ Superseded — historical, kept for reference only.** Written during early
> bring-up, before "rank-local load" and "distributed forward" (listed below as
> milestones still to build) were actually completed, before the correctness fix, before
> speed tuning, and before speculative decoding. **Do not follow this as current
> instructions — use [`README.md`](README.md)'s "Give this to your agent" prompts
> instead**, which are accurate and maintained. This file predates that becoming the
> canonical onboarding path.

**You are an agent operating NVIDIA DGX Spark machines for a developer.**  
This file is your contract for **Kimi-K3 Neuron IQ1_S multi-Spark TP3 bring-up**.

Read **this entire file** before changing fleet state. Prefer evidence over assumption.

---

## 0. Mission and honesty

### You may
- Inventory Sparks, health-check GPUs, measure NCCL, apply frozen patches, run smokes.
- Implement the **next missing milestones** (rank-local load, distributed forward) behind tests.
- Report tok/s **only** after the qualification ladder in §6.

### You must not
- Claim production multi-Spark inference is ready.
- Quote **~42 tok/s** as a three-Spark number (that is **same-host 3×H200**).
- Quote upstream SparkInfer 56–60 tok/s as this 330 GB / WEPS=0 result.
- Load `matched-byte-v2` into SparkInfer.
- Leave credentials, HF tokens, or SSH keys in chat, git commits, or world-readable logs.
- Run `hermes update`, mass restarts, or delete user data without explicit approval.
- Spend cloud GPU budget without a **hard time/cost cap** and health gate first.

### Status one-liner (paste when asked “does it work?”)
> Preview bring-up: TP3 geometry + InitRank collectives + control plane are frozen;  
> full-model 3-Spark generate is **not** shipped. Production path remains llama.cpp RPC  
> layer-split until rank-local load + distributed forward + parity pass.

---

## 1. Fleet facts

| Item | Value |
|---|---|
| Target world | **3** Sparks (TP3 width). Optional later: **4** (ExpertFfn2D) |
| Process model | **1 process / Spark**, `local_device=0`, global rank `0..world-1` |
| TP3 MoE geometry | All **896** experts; FFN bands **[0,512), [512,1024), [1024,1536)** |
| Memory plan | **~112.85 GiB/rank** — tight on 128 GB Sparks |
| Model | HF `vcruz305/Kimi-K3-Neuron-IQ1S-GGUF` (~330 GB, 9 shards, gated) |
| SparkInfer base | `7a9b77a043596157d74e4af376cf9f29f68ce368` |
| Control plane | TCP framed protocol in patches `next/0002` + `0003` |
| Collectives | `ncclCommInitRank` + opaque 128-byte id; **not** `ncclCommInitAll` across hosts |
| Rank 0 role | Coordinator, NCCL id, BeginLoad, Token broadcast, **full LM head + sample** |
| WEPS | `SPARKINFER_K3_MOE_WEPS=0` until long-context parity |

### Runtime split (do not conflate)

| Runtime | Role |
|---|---|
| **llama.cpp** (Unsloth K3 fork) | Production/correctness GGUF serve, PPL, HumanEval |
| **SparkInfer-K3** | K3-native kernels + TP graph; multi-Spark port in progress |
| Same-host TP3 | Proven ~42 tok/s on 3×H200 after width wire-up + tile/NCCL fix |
| Multi-Spark TP3 | Requires this package’s distributed path — incomplete |

---

## 2. Safety and secrets

1. Tokens only in protected env files / secret stores — **never** echo into chat or commits.  
2. Prefer `BatchMode` SSH; do not disable host key checks in shared logs if avoidable.  
3. Before long jobs: set a **removal/cost cap** on cloud pods; on Sparks, document wall-clock intent.  
4. Do not kill unrelated user jobs (`llama-server`, training) without owner approval.  
5. Scan diffs for secrets before `git push`.

---

## 3. Mandatory GPU health gate

**Before any multi-hour download, full load, or NCCL soak**, on **every** Spark:

```bash
python3 scripts/gpu_health_gate.py health
```

Require for each GPU index you will use:

- `matmul_ok: true`
- uncorrectable ECC volatile/aggregate not climbing
- free memory plausible for the plan

If **any** GPU returns `cudaErrorDevicesUnavailable` or matmul fails → **stop**, quarantine node, do not proceed with TP.  
(Prior Lium 3×H200 failure: GPUs 0/2 dead + ECC; only GPU1 worked.)

---

## 4. Network / NCCL rules (Sparks vs cloud)

### On real DGX Sparks
- Expect **RoCE/IB fabric** between nodes.
- After health gate, run 3-rank InitRank microbench (§5).
- Log NCCL interface selection; **reject silent TCP fallback** for qualification runs.
- Common: `NCCL_NVLS_ENABLE=0` if multicast init fails.

### On multi-cloud single-GPU pods (Lium lesson)
- Often **only SSH ports** are peer-reachable; published 8000/8888 may **timeout** between pods.
- Multi-host NCCL **will not work** without tunnels/VPN — do not burn budget fighting this.
- Prefer **one multi-GPU node** for InitRank software proof; use Sparks for real multi-host.

### Never
- `ncclCommInitAll({0,1,2})` intending three machines (each machine only sees its own device 0).
- Put remote ranks into a single-process `ranks[]` vector of device pointers.

---

## 5. First fabric proof (do this before model load)

### Torch multi-process (quick)

On three ranks (adjust hosts):

```bash
# rank 0
MASTER_ADDR=<rank0_ip> MASTER_PORT=29500 WORLD_SIZE=3 RANK=0 \
  python3 scripts/torch_nccl_rank.py

# rank 1 / 2 — same MASTER_*, RANK=1 or 2
```

Payloads inside script: 7168 and 10752 f32 (28,672 / 43,008 bytes).  
Require `correct=True` on all ranks. Record mean_us and `ms185`.

### Native C++ InitRank (stronger)

```bash
# build with nvcc + libnccl (CUDA toolkit template recommended)
# see scripts/k3_dist_nccl_microbench.cpp header
CUDA_VISIBLE_DEVICES=0 ./k3_dist_nccl_microbench --rank 0 --world 3 --host <rank0> --port 29500 &
# ranks 1–2 similarly with their local device 0
```

**Prior anchors (not Spark fabric):**

| Platform | mean allreduce | ×185 |
|---|---:|---:|
| H100×2 native InitRank | ~8.1 µs | ~1.5 ms/token |
| 5090×3 torch InitRank-style | ~41.5 µs | ~7.7 ms/token |
| H200×3 same-host group AR | ~32 µs p50 | ~5.8–6.0 ms/token |

---

## 6. Qualification ladder (no skipping)

1. CPU protocol tests (package patches) — already green upstream of this pack.  
2. Health gate on all three Sparks.  
3. NCCL 3-rank microbench on **Spark fabric** (this pack §5).  
4. Rank-local 8K load — **zero swap/UVM**, ≥8 GiB `MemAvailable` per rank.  
5. One-token greedy: top-1 match vs llama.cpp (token 17374 / ` Paris` is the historical smoke prompt family).  
6. Multi-token + longer prompts: KLD / top-1 / RMS as agreed.  
7. Only then decode tok/s vs llama RPC baseline (~2.85).  
8. Stop if paging/parity fails; abandon single-stream TP3 if ≤3.5 tok/s after graphs+RDMA.

`WEPS≠0` is **approximate** and must not rescue a failed exact gate.

---

## 7. Patch apply (SparkInfer)

```bash
git checkout 7a9b77a043596157d74e4af376cf9f29f68ce368
git am patches/sparkinfer/0001-tp-plan-K3-all-expert-FFN-width-shards-for-TP3.patch
git am patches/sparkinfer/0002-wire-tp3-all-expert-width-init-ffn-prefill.patch
git am patches/sparkinfer/next/0003-tp-three-host-rank-bootstrap-protocol.patch
git am patches/sparkinfer/next/0004-tp-distributed-rank-transport-and-tp3-tp4-plans.patch
git am patches/sparkinfer/next/0005-tp-rank-local-plan-and-nccl-microbench.patch
```

Verify `SHA256SUMS`. Prefer templates/images with **full CUDA toolkit + nvcc** (e.g. CUDA 13 DIND), not torch-wheels-only.

Same-host wire-up extras (if building TP generate):

- honor `MoeShardMode::AllExpertsFfnWidth` in `k3_moe_ffn_local` (return 512);
- auto-disable **tile** prefill when `!coll->owns_buffers()` (NCCL has no owned buffers);
- explicit NCCL include/lib in CMake when using pip NCCL layouts.

---

## 8. Implementation priorities (what to build next)

Work **in this order** unless the developer overrides:

1. **Rank-local GGUF residency** — load only planned bands; never mirror 330 GB × 3.  
2. **Distributed eager forward** — embed → attn → AR → FFN partial → AR → finish; no graphs yet.  
3. **Token lockstep** — rank0 samples; broadcast token id via control plane.  
4. **Abort path** — sticky failure → `ncclCommAbort` on all ranks.  
5. **Parity suite** then graphs / WEPS experiments.

Do **not** start with FlashKDA, speculative decoding, or vLLM ports for this milestone.

---

## 9. Messaging templates

### To developer — healthy progress
> Fabric NCCL 3-rank PASS (XX µs @ 28k/43k B). Next: rank-local 8K load gate.

### To developer — blocked
> Stopped: GPU1 matmul failed / paging detected / parity top-1 miss. No speed claim.

### To external Spark devs
> Sharing a **preview** TP3 bring-up pack: geometry + InitRank + control plane.  
> Not install-and-run production. llama.cpp RPC remains the supported multi-Spark path today.

---

## 10. File map for tools

| Path | Use |
|---|---|
| `THREE-SPARK-TP3-RECIPE.md` | Shared human recipe |
| `docs/TP3-EXPLAINED-AND-FIXED.md` | Teach TP vs pipeline |
| `docs/SPARK-TP3-PERFORMANCE-FORECAST.md` | Numbers + stop rules |
| `docs/ROOT-CAUSE-AND-H100.md` | Ops failure modes |
| `patches/sparkinfer/**` | `git am` chain |
| `scripts/gpu_health_gate.py` | Preflight |
| `scripts/torch_nccl_rank.py` | Quick NCCL |
| `scripts/k3_dist_nccl_microbench.cpp` | Native InitRank |
| `evidence/*` | Citations for claims |

---

## 11. Definition of done (for *you* this session)

A session is successful if you either:

- advance one ladder step with **receipts** (logs + hashes + hostnames), or  
- stop with a **clear blocker** (health, network, memory, parity) without wasting fleet time.

Do not end on “should work” without tool evidence.
