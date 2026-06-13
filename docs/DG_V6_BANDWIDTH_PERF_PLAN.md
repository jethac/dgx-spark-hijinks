# DG-V6 — DiffusionGemma NVFP4-KV decode-speed (bandwidth) experiment

**Status:** PLAN (not yet run). Owner: Claude / vLLM lane. Platform: Spark (GB10, sm_121).

## Hypothesis
Full-NVFP4 K+V cache makes **DiffusionGemma decode FASTER** (not just higher-capacity),
and the speedup is **larger than for any autoregressive Gemma** and **larger on
bandwidth-constrained hardware (Spark)** than on the PRO 6000.

## Mechanism (why it should be true)
- nvfp4 KV moves **~3.56× fewer bytes per KV read** than bf16 (9/32 of bf16, scale
  factors included).
- DiffusionGemma is **block diffusion**: the decoder re-reads the *entire cached prefix
  on every denoise step* (dozens of reads per generated 256-token block), vs **once per
  token** for an AR model. So KV-read bandwidth is the *dominant* decode cost, not a side
  cost.
- Spark/GB10 is **bandwidth-bound** (~273 GB/s unified). On a bandwidth-bound part, fewer
  bytes moved == directly faster (and the nvfp4 in-kernel dequant compute is cheap relative
  to the bandwidth saved).
- => shrinking the per-step KV read 3.56× should *speed up* decode, biggest where the KV
  re-read dominates the per-step byte budget.

## Roofline prediction (the quantitative model to test against)
If decode is bandwidth-bound, per-step time `~= (W + K) / BW`, where
- `W` = bytes read per denoise step for the **active** weights. DiffusionGemma 26B-A4B is
  MoE (~4B active/token) -> `W ~= 4e9 * 2 (bf16 weights) ~= 8 GB/step` (refine from config).
- `K` = KV bytes re-read per step = `ctx_tokens * bytes_per_kv_token`. bf16 vs nvfp4:
  `K_nvfp4 = K_bf16 / 3.556`.
- **Predicted TPOT speedup** = `(W + K_bf16) / (W + K_nvfp4)`:
  - short ctx (`K << W`): ~1.0× (no win — weight-bound).
  - long ctx (`K >> W`): -> 3.556× (KV-bound limit).
- So the prediction is a **rising curve vs context**, asymptoting toward 3.56×. Overlay
  measured-vs-predicted: if they track, the speedup is bandwidth, *for the reason claimed*.

## Falsifiable predictions
- **P1 (signature):** DiffusionGemma nvfp4 TPOT < bf16 TPOT, and the **gap grows with
  context length**. (Falsified if flat or shrinking.)
- **P2 (block-diffusion attribution):** the AR control (Gemma 4 26B-A4B-it, same sweep) has a
  **much smaller** nvfp4-vs-bf16 TPOT gap than DiffusionGemma at the same context. (Falsified
  if the gaps are equal — then it's generic, not a block-diffusion effect.)
- **P3 (platform attribution):** the speedup is **bigger on Spark than on the PRO 6000**
  (more bandwidth-bound). (Stretch; falsified if equal/inverted.)

## Design
| axis | values |
|---|---|
| model (primary) | `google/diffusiongemma-26B-A4B-it` (block diffusion) |
| model (AR control, P2) | `google/gemma-4-26B-A4B-it` (same base, autoregressive) |
| KV dtype | **nvfp4** vs **bf16** (matched everything else) |
| context sweep (input len) | 512, 1024, 2048, 4096, 8192 |
| output len | 256 (one diffusion block) |
| concurrency | **1** (single-stream) -- CRITICAL: isolates per-step bandwidth from
  capacity/starvation; both dtypes have ample KV for one stream |
| platform | Spark sm_121 (primary). PRO 6000 / Colab sm_120 (P3, stretch) |

**Why concurrency 1:** the YAK throughput gains were *partly* capacity-relief (bf16 31B was
KV-starved). At concurrency 1, bf16 is NOT starved at any tested ctx, so any TPOT delta is
pure per-step bandwidth. (Also run a high-concurrency pass separately to show the *combined*
capacity+bandwidth win, but the single-stream curve is the clean mechanism proof.)

## Metrics
- **TPOT (ms/output token)** — primary (per-step decode cost).
- output throughput (tok/s), TTFT — secondary.
- derived: `speedup(ctx) = TPOT_bf16(ctx) / TPOT_nvfp4(ctx)` vs the roofline curve.

## Procedure
1. Image: `jethac-vllm-aeon-gemma4:e2-dgv-3d6a0d507-sm121a-r11` (warm MoE kernel cache; r12
   bakes it -- mount `~/spark_tmp/fi_cache` if recompile reappears). Spark marker first.
2. For each (model, dtype): serve once, sweep input len via `vllm bench serve`.
   - nvfp4: `--attention-backend FLASHINFER --kv-cache-dtype nvfp4` + env
     `VLLM_NVFP4_KV_LINEAR_V_SF=1` (VO-split knobs baked) `--enforce-eager
     --no-enable-prefix-caching --no-enable-chunked-prefill --max-num-seqs 4
     --gpu-memory-utilization 0.6 --max-model-len 8704`.
   - bf16: same with `--kv-cache-dtype auto`.
   - bench: `vllm bench serve --dataset-name random --random-input-len <ctx>
     --random-output-len 256 --max-concurrency 1 --num-prompts 16 --ignore-eos
     --save-result`. (Random prompts are fine for TIMING -- the denoise loop runs regardless
     of coherence; coherence already proven in DG-V5.)
3. Repeat the sweep for the AR control (gemma-4-26B-A4B-it) -> P2 contrast.
4. (Stretch) repeat DG sweep on PRO 6000 via the Colab benchmark notebook -> P3.
5. Free GPU + release marker after each model.

## Success / kill criteria
- **Confirmed** if P1 holds (rising speedup vs ctx) AND P2 holds (DG gap >> AR gap). Bonus if
  the measured curve tracks the roofline prediction within ~20%.
- **Refuted / re-scope** if the single-stream TPOT delta is flat (~1.0×) across ctx -> then
  the blog's "per-step bandwidth lever" claim is unsupported and must be softened to capacity-
  only. (This is the honest kill switch; do not ship the speed claim without the curve.)

## Artifacts
`results/dg_v6_bandwidth_<date>/`: per-(model,dtype,ctx) bench JSON, a `SUMMARY.md` with the
speedup-vs-ctx table + the roofline overlay, server logs (proof lines: nvfp4 + LINEAR_V_SF +
VO-split engaged). Feeds the blog's DiffusionGemma section (turns the asserted "bandwidth
lever" into a measured curve) and the SGLang lane (Codex can mirror for DG-R6).

## Risks / caveats
- DiffusionGemma serving is block-diffusion; confirm the denoise loop actually runs under the
  random-prompt bench (check generated token counts, not just no-error).
- MoE first-serve recompiles the CUTLASS kernel (~20 min cold) -- use the warm cache image.
- TPOT semantics for diffusion: vLLM reports per-output-token latency; that already folds the
  per-step re-reads across the block, which is exactly what we want to measure.
- Keep util / max-model-len identical across dtypes so the only variable is KV bytes/read.
