# Stage 3 — KDA state rollback: verification receipt

Date: 2026-08-11. Fleet: TP3 (`wesche-spark-78f1/9f73/366f`), Kimi-K3 Neuron IQ1_S, 93 layers.
Patch under test: `patches/sparkinfer/next/0023-tp-specdec-stage3-kda-state-rollback.diff`
applied on top of 0021 + 0022.

Build/deploy, md5-verified identical on all 3 ranks before every run:

| run | `kimi_k3_dist_generate` | `libsparkinfer_runtime.so` |
|---|---|---|
| `default`, `arpad` | `af5e47494303637dd3b316cc5ca44719` | `8851ec0428828d24db68c9e0a4835c6d` |
| `guard` (final, = the patch) | `af5e47494303637dd3b316cc5ca44719` | `8ff131aa47dcaea75cc82eb7dbd9354f` |

The `guard` build adds two stale-bookkeeping guards found in self-review (`batch_n_tok`
cleared at the top of `forward_batch` and in `forward_token`, so a rollback can never be
served against a batch that failed partway or that has since been continued past). It
produced results **byte-identical** to the `default` run across a full model reload —
both the gates and all 105 Test D figures — confirming the guards are inert and that the
whole result set reproduces across independent fleet runs.

Harness: `kimi_k3_dist_generate --spec-rollback 8 --spec-cont 3 --prompt-ids 1,2,3`,
launched by `scripts/spec_rollback.sh` (rank 0 local, workers over the private fabric).
Env: `SPARKINFER_K3_MOE_WEPS=0 SPARKINFER_K3_GRAPH=0 NCCL_NVLS_ENABLE=0
SPARKINFER_K3_PREFILL_CHUNK=16`. Raw logs on rank 0: `run/roll_default_r0.stdout`,
`run/roll_arpad_r0.stdout`, `run/roll_guard_r0.stdout`.

Coverage: K = 2..8 (both the reliable {2,4,8} branch and the {3,5,6,7} sawtooth branch),
every acceptance level j ∈ [0, K-1], 3 forced continuation steps each.

---

## The acceptance criterion, and why it is not "matches sequential decode"

Greedy speculative decoding does not need the verify pass to be bit-identical to a
sequential run. It needs two things:

- **(C1) suffix independence** — logits row `j` and the state after token `j` must not
  depend on the drafts at positions > j. They may depend on K.
- **(C2) snapshot consistency** — `rollback(j)` must restore the state *that pass*
  computed after token `j`: the same state that produced the row-`j` logits the accept
  decision was made from.

Given C1 and C2 every emitted token is the argmax of logits computed from its true
accepted prefix, so the output is a valid greedy trajectory. Reproducing a *pure
sequential* state is neither achievable (the collective's summation order depends on the
payload shape) nor desirable — it would splice in a state that never produced the logits
the accept decision used. Tests A and C gate C1/C2 bitwise. Test D measures the residual
distance to sequential decode and is reported, never asserted.

---

## Gate results — both runs, all bitwise

| gate | what it proves | default | AR count pinned |
|---|---|---|---|
| **A** suffix independence after rollback | C1 + C2, plus complete 4-buffer restore, plus MLA KV rollback | **35 / 35 YES** | **35 / 35 YES** |
| **C** ring is inert | snapshotting does not perturb the pass it observes | **7 / 7 YES** | **7 / 7 YES** |

Test A runs two batches sharing tokens `0..j` and disagreeing on everything after, rolls
both back to `j`, and runs the same forced continuation. Every one of the 35 (K, j) pairs
came back bitwise identical at `worst cos = 1.000000000`. Because the two arms wrote
*different* values into the KV rows the rejected tokens occupied, this simultaneously
proves the continuation never reads past its own position — the KV rollback holds.

---

## Test D — distance to pure sequential decode (measurement, not a gate)

`forward_batch(K)` → `rollback(j)` → continue, versus consuming the same accepted tokens
one at a time and continuing. 105 comparisons (35 pairs × 3 steps).

| run | top-1 AGREE | bitwise-exact rows | worst cos | mean cos |
|---|---|---|---|---|
| default | 62 / 105 | **0** | 0.820 | 0.9476 |
| `SPARKINFER_K3_AR_PAD=16` | 67 / 105 | **21** | 0.765 | 0.9547 |

Fidelity declines with K in the default run — the count-dependence signature:

| K | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| mean cos | 0.9654 | 0.9586 | 0.9522 | 0.9462 | 0.9429 | 0.9440 | 0.9440 |
| step-0 top-1 agree | 2/2 | 2/3 | 2/4 | 3/5 | 3/6 | 3/7 | 3/8 |

(The decline flattens rather than continuing past K=6; with 2–8 step-0 samples per K these
per-K figures are directional, not precise.)

### The decisive result

Pinning the all-reduce count makes **every j = 0 case bitwise exact and no other case**:

```
[spec-rollback] D K=2 j=0 step=0 cos=1.000000000 top1 spec=2 seq=2 AGREE exact=100.0000% max|d|=0.000e+00
...  (all 7 K values x 3 steps = 21 rows, all cos=1.000000000, all max|d|=0)
[spec-rollback] D K=8 j=0 step=2 cos=1.000000000 top1 spec=3384 seq=3384 AGREE exact=100.0000% max|d|=0.000e+00
```

No `j > 0` row is bitwise exact in either run. 21 exact rows = exactly the 7 `j = 0` pairs
× 3 steps.

**Why this is the whole explanation.** `j = 0` continues from the state after token 0,
and token 0 sits at **offset 0** of the batched payload — the same offset it occupies in
the single-token path. Pin the count and its elements land in the same reduce-scatter
partition, summed in the same order, so the result is bit-identical. For `j > 0` the token
sat at offset `j·width` in the batch but at offset 0 in sequential decode: same count,
different partition, different summation order.

So the residual divergence is **purely the offset term of the collective's
reassociation**. Two consequences:

1. **The rollback mechanism contributes zero numerical error.** At `j = 0`,
   `forward_batch(K) → rollback(0) → forward_token` is bit-identical to pure sequential
   decode, end to end, at every K from 2 to 8. Nothing in the snapshot, the restore, the
   position/KV rollback or the residual-bank handling perturbs the result.
2. **`SPARKINFER_K3_AR_PAD` is not a losslessness fix and cannot be made into one.** It
   delivers count-invariance only. Bit-identical speculative output needs a **count- AND
   offset-invariant** reduction — e.g. allgather-then-local-sum, where each element's
   three partials are summed in fixed rank order regardless of where the element sits.
   That is the design doc's §6 proposal and it remains unbuilt and unmeasured.

Corroborating this reading: patch 0021's single-GPU 16-layer walk-up already proved the
batched KDA projections, MoE expert-major regroup and batched head bitwise correct with no
collective present, which covers rows > 0 independently of the offset effect.

---

## What this means for Stage 4

- Stage 3 is **sound**: the rollback is correct under the criterion that actually governs
  greedy accept/reject, and is provably error-free where a clean comparison exists.
- The design doc's Stage 4 gate — *"end-to-end generation token-identical to
  non-speculative greedy"* — **cannot be met as written** without the offset-invariant
  collective. The honest claim available is "a valid greedy trajectory", which is the same
  standard this engine already meets against the reference oracle and no weaker.
- Smaller K is closer to sequential (mean cos 0.9654 at K=2 vs 0.9440 at K=8; step-0
  agreement 2/2 vs 3/8). This is a fidelity argument for small K that stacks with the
  existing marginal-cost argument for K ∈ {2, 4, 8} — but see the sample-size caveat
  above before treating the per-K numbers as more than directional.
- Test D's sample is small (35 step-0 comparisons on one 3-token prompt) and its
  divergence concentrates where the model is near-tied, which is also where drafts fail
  anyway. Do **not** read 51% step-0 agreement as "acceptance will halve" — the effect on
  acceptance is unmeasured until a real drafter exists.
