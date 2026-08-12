# How to hand this to a Spark-controlling LLM agent

> **This file is superseded.** It predates the working recipe (correctness fix, speed
> tuning, and speculative decoding all landed after this was written) and used to point
> at `SPARK-AGENT.md` / `THREE-SPARK-TP3-RECIPE.md`, which describe an early,
> incomplete "preview bring-up" state that no longer reflects reality — see the
> deprecation notices on those files. **Use [`README.md`](README.md)'s own "Give this to
> your agent" section instead**, which is the maintained, current onboarding path and is
> kept in sync with the actual state of the recipe. This file is kept for history, not
> as instructions.

## What to use instead

Copy the prompt from README.md's [Give this to your agent](README.md#give-this-to-your-agent)
section (base recipe) and, once that's working, the
[speculative decoding follow-up prompt](README.md#speculative-decoding-experimental-opt-in)
if you want that too. Both are kept accurate as the recipe changes; this file is not.

## Still true, kept for reference

- Never put API tokens or SSH keys in chat or commits.
- `scripts/gpu_health_gate.py` is still a reasonable pre-flight check before long work
  on a Spark, if you want it — not required by the current recipe.
- If you also use `vcruz305/sparkfleet` for OS/firmware/driver lifecycle: use
  **sparkfleet** for fleet ops, this repo for K3 Neuron TP3 serving only, and don't mix
  update windows with full-model experiments on the same nodes.
