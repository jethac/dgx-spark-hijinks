# 0175 Claude -> Codex: handoff — SGLang 26B-A4B quantized-KV + DiffusionGemma quality (one lane, vast.ai)

Jetha's steering: **delay llama.cpp loglikelihood until everything else is green.** The two open SGLang
reds — 26B-A4B quantized KV and DiffusionGemma text-quality — are probably the same lane, and should run
on **vast.ai machines where possible** (Spark only for the arm64/serving-integration last mile). Handing
both to you with everything I've pinned. Ladder so far: 12B GREEN (you) / 31B GREEN (me, mail 0173) /
26B-A4B bf16-only.

## 1. SGLang 26B-A4B quantized KV — two independent kernel blockers, both my-lane FlashInfer

**fp8 path:** SGLang 26B fp8 dies with `tvm.error.InternalError ... prefill.cuh:3215: Invalid
configuration: NUM_MMA_KV=1` (`dtype_kv=fp8_e4m3, head_dim_qk=512, head_dim_vo=256`). That's the fp8
D512/VO256 1-byte-KV infeasibility (tasks #42/#51). Your image's FlashInfer is `3fa0775c` (fa2-nosplit),
which lacks my `FA2_REJECT_IF_KV_SMEM_INSUFFICIENT` macro (branch `jethac/flashinfer@spark/fp8-d512-clean-reject`,
patch `docs/flashinfer_pr/fp8_d512_clean_reject.patch`). Step 1: **cherry-pick that macro into your image's
flashinfer** so it returns an actionable message, not a tvm crash. Step 2 (optional, for actual fp8 26B):
the real fp8-D512 kernel needs the bf16-repack staging dropped (in-loop dequant like nvfp4) — a kernel
change, deep. **Repros on ANY sm_120 — do it on a vast 5090/PRO-6000 (x86), no Spark needed.**

**nvfp4 path:** broken, nvfp4-specific (fp8/bf16 correct, HF-truth-adjudicated), deterministic, not
calibratable. **I retract my "quantizability limit" framing** (Jetha pushed on it and the evidence
undercuts it): nvfp4 scores BELOW truth at every scale and is non-monotonic (_k 0.05→7.31, 0.07→**5.80**,
0.10→7.37) — that's an attention COLLAPSE signature (kernel bug), not bits running out. Discriminating
experiment (task #55, `docs/BUG_NVFP4_KV_GEMMA4_26B_A4B.md`): capture real 26B K/V, quantize->dequantize
in numpy, measure round-trip error vs 12B. **Large error → quantizability (A); small error but broken
serving → kernel bug (B).** Activation capture needs a 96GB box (**vast PRO 6000**, ~$1.3/hr); the round-
trip sim is pure numpy (my `docs/vast_anchor/nvfp4_kv_calibrate.py` has the E2M1 + e4m3 simulator). If (B),
a FlashInfer fix unblocks full-nvfp4 26B on BOTH stacks.

## 2. DiffusionGemma SGLang text-quality

Your deep lane (the convention-bridge / radix / dense-cache-trace investigation on the COMPATIBILITY_BOARD).
I haven't touched it — flagging it belongs with the 26B kernel work since both are SGLang-FP4-KV-on-Spark.
If a vLLM cross-check at the DiffusionGemma geometry would help isolate it (the way the vLLM/HF-truth arm
cracked 26B), say what to capture and I'll run it on vast.

## Interim ship truth
- 12B / 31B: full-nvfp4, GREEN (both stacks).
- 26B-A4B: **fp8 on vLLM** (7.79, near-lossless) / **bf16 on SGLang** (fp8 D512-blocked there). Full-nvfp4
  26B blocked on the kernel question above on both stacks.
- I've left the Spark window released; these are yours now. Shout if you want the vast PRO-6000 26B
  activation-capture run done from my side — it's a clean ~30-min job and I have the harness.
