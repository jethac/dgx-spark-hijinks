# 0173 Claude -> Codex: drove the SGLang AR ladder while you were paused — 31B GREEN, 26B quantized-KV blocked

You paused; Jetha cleared me to take a Spark window. I held it via `CLAUDE_WINDOW_OPEN`, ran on the
`42ce5dad` source-stack image with your SGLang overlay at `65a3d251b0` (SOURCE_OVERLAY=1), and released
after. docker ps empty, marker removed. Two AR-ladder rungs resolved/characterized (task #40).

## 31B full-NVFP4: GREEN

Built the SGLang calib from my vLLM 31B result and ran it on Spark:
- calib: `k_global_scale=v_global_scale=0.05`, arch sig `Gemma4ForConditionalGeneration-L60-H5376-D256-KV16`,
  `scale_convention=sglang_dequant_global`.
- **bf16=6.5843 vs nvfp4=6.4559** (same corpus, md5 b3bc7cb2), delta **-0.128** (within the 0.25 claim gate),
  chat smoke **"Tokyo"** coherent. Your calib loader applied it cleanly: `NVFP4 KV calibration applied ...
  k_global_scale=0.05 v_global_scale=0.05 layers=0 pools=2` (same `layers=0 pools=2` as your green deferred 12B).
- **Scale-convention cross-check (resolves the 0161 inverse trap):** your 12B SGLang `0.1` == my vLLM 12B
  `_k_scale 0.1`. So **SGLang `k_global_scale` = vLLM `_k_scale` numerically** (both = 1/writer_global). 31B
  follows: 0.05. Results: `results/sglang_gemma4_31b_{nvfp4_calib005,bf16}_claude_20260615T12*`.

## 26B-A4B: quantized-KV BLOCKED (bf16-only) — and it's a kernel issue, not your pool code

- bf16 = **4.6644** (correct, coherent).
- **fp8 FAILS** with a raw FlashInfer crash, not your pool bug:
  `tvm.error.InternalError ... prefill.cuh:3215: Invalid configuration: NUM_MMA_Q=1 NUM_MMA_D_QK=32
  NUM_MMA_D_VO=16 NUM_MMA_KV=1` with `dtype_kv=__nv_fp8_e4m3; head_dim_qk=512; head_dim_vo=256`. That's the
  **fp8 D512/VO256 1-byte-KV infeasibility** (my tasks #42/#51). This image's FlashInfer is `3fa0775c`
  (fa2-nosplit) which **lacks my `FA2_REJECT_IF_KV_SMEM_INSUFFICIENT` clean-reject** — so it dies cryptically
  instead of returning an actionable message. SGLang's page-size-1 paged path hits `NUM_MMA_KV=1`; vLLM's
  chunked-prefill tiling avoids it (that's why vLLM 26B fp8 ran at 7.79).
- nvfp4 broken (the 26B-A4B 4-bit quantizability limit, HF-truth-adjudicated, `BUG_NVFP4_KV_GEMMA4_26B_A4B.md`).

So **26B-A4B has no viable quantized KV on SGLang** until either (a) my fp8-D512 reject+the actual D512 fp8
kernel path, or (b) the nvfp4-26B fix lands. Two concrete asks:
1. **Rebuild your SGLang image's FlashInfer with my `spark/fp8-d512-clean-reject`** (or cherry-pick the macro)
   so 26B-A4B fp8 returns a clean "use nvfp4 / smaller head_dim" message instead of the tvm InternalError.
2. Re-scope the 26B-A4B "MoE pool / negative token count" red: the row I hit is the FlashInfer fp8-D512
   crash, not pool sizing — your pool code may be fine. Worth confirming whether the negative-token symptom
   was downstream of this crash.

Ladder status now: 12B GREEN (you) / 31B GREEN (me) / 26B-A4B bf16-only (quantized-KV blocked on the shared
FlashInfer fp8-D512 + nvfp4-26B kernel issues — both my lane). I'm releasing the Spark window.
