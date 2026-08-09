# TP4 Step2 A/B — stream-ordered pos + no worker per-token sync
ts=2026-08-09T03:04:21.574167+00:00

## Change
- Workers: removed production `cudaStreamSynchronize` every token
- Position: `cudaMemcpyAsync(d_pos)` on compute stream (no default-stream H2D)
- Rank0 logits: `cudaMemcpyAsync` + one stream sync
- No GDR / NCCL flag changes

## Baseline (sealed GRAPH=0)
- median_ex0=**7.90** · mean_all=**7.60** · peak=**8.02**

## After
- median_ex0=**7.9450**
- p95_ex0=**8.0395**
- mean_all=**7.8943**
- vals_ex0=[8.0395, 7.6998, 8.0269, 7.7511, 7.945]
- all=[7.9036, 8.0395, 7.6998, 8.0269, 7.7511, 7.945]
  - p0: 7.9036 t/s (128 tok / 16.195185s)
  - p1: 8.0395 t/s (128 tok / 15.921321s)
  - p2: 7.6998 t/s (128 tok / 16.623815s)
  - p3: 8.0269 t/s (128 tok / 15.946316s)
  - p4: 7.7511 t/s (128 tok / 16.513789s)
  - p5: 7.9450 t/s (128 tok / 16.110754s)
- delta median vs 7.90: **+0.045** t/s
- distance to 10.0: **+2.055** t/s
