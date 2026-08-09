# TP3 3-Spark median — syncfix binary
ts=2026-08-09T04:14:14.637196+00:00

## Config
- world=**3** · ranks 78f1/9f73/366f · local NVMe GGUF
- n-predict=128 · max-ctx=8192 · WEPS=0 · GRAPH=0 · multi-prompt 6
- syncfix: no worker per-token sync · async d_pos · async logits D2H

## Prior TP3 single-shot (pre multi-prompt median hygiene)
- best n=128: **6.21** t/s

## This run
- median_ex0=**6.8386**
- p95_ex0=**7.0830**
- mean_all=**6.8528**
- vals_ex0=[7.083, 6.7495, 6.9192, 6.7913, 6.8386]
  - p0: 6.7350 t/s (128 tok / 19.005254s)
  - p1: 7.0830 t/s (128 tok / 18.071358s)
  - p2: 6.7495 t/s (128 tok / 18.964478s)
  - p3: 6.9192 t/s (128 tok / 18.499253s)
  - p4: 6.7913 t/s (128 tok / 18.847701s)
  - p5: 6.8386 t/s (128 tok / 18.717270s)
- delta vs prior 6.21: **+0.629** t/s
- vs RPC 2.85: **2.40×**
- vs TP4 best 7.945: **0.86×** (-1.11 t/s)
- distance to 10: **+3.161** t/s
