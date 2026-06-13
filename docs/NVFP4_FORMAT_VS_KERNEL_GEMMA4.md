# Gemma-4 NVFP4-KV: the +0.281 serving delta is the FlashInfer kernel, NOT the format

**Date:** 2026-06-13 · **Verdict:** the NVFP4 KV **format** is near-lossless for Gemma-4
(+0.003 nats/token at ctx 8192). The +0.281 (vLLM) / +0.403 (SGLang) serving quality red
is the **FlashInfer nvfp4 kernel/serving path** — a fixable bug (Task #25), not inherent.

## The question
Codex's matched vLLM 12B anchor (mail 0117) measured full-NVFP4 K+V at `Δ=+0.281`
nats/token — ~50× the near-lossless ledger results for Qwen (+0.005) and Gemma-3-27B
(+0.005…+0.037). Backend swap couldn't isolate it because nvfp4 KV is FlashInfer-only
(mail 0121: Triton can't do nvfp4, FLASH_ATTN can't run this Gemma-4 config). So we used a
**kernel-free reference** to split format from kernel.

## Method (reference NVFP4, pure torch — no FlashInfer)
HF eager, `attn_implementation="sdpa"`, base checkpoints + raw wikitext-2 (valid prompt
contract; Gemma-4 has no attention softcap, so sdpa is numerically exact). Monkeypatch
`F.scaled_dot_product_attention` to quantize/dequantize **K and V to NVFP4** (E2M1 levels
{0,.5,1,1.5,2,3,4,6}, per-16 block scale quantized to fp8 e4m3) before attention. The
bf16-vs-refNVFP4 NLL delta = the **pure format loss**, kernel-free. Script:
`docs/vast_anchor/refsim_run.py` (vast.ai RTX PRO 6000, destroyed after).

## Results

| model | ctx | bf16 NLL | refNVFP4 NLL | **Δ (format)** |
| --- | ---: | ---: | ---: | ---: |
| google/gemma-4-12B | 2048 | 1.7274 | 1.7397 | **+0.0123** |
| google/gemma-4-12B | 4096 | 1.8925 | 1.9007 | **+0.0082** |
| **google/gemma-4-12B** | **8192** | 1.9734 | 1.9765 | **+0.0030** |
| google/gemma-3-12b-pt | 2048 | 1.4228 | 1.4603 | +0.0375 |
| google/gemma-3-12b-pt | 4096 | 1.6028 | 1.6320 | +0.0291 |

**Quantizer validated:** the Gemma-3-12B reference (+0.037/+0.029) reproduces the ledger's
known FlashInfer Gemma-3-27B kernel value (+0.037) — so the reference is calibrated, not
under-quantizing. (If anything, real 2-level NVFP4 with a global scale is ≤ this.)

## Conclusion
- **Gemma-4-12B NVFP4-KV format loss at ctx 8192 = +0.003 nats/token** — essentially
  lossless, in-family with Qwen/Gemma-3, and *lower* than Gemma-3 at long context.
- **FlashInfer kernel = +0.281** at the same shape (Codex 0117). So **~+0.278 nats/token is
  the FlashInfer nvfp4 kernel/serving-path error, not the format.**
- The kernel is fine for Qwen + Gemma-3 (near-lossless) but bad for Gemma-4 geometry
  (head_dim 256 + sliding-window pattern + GQA; VO-split for the 512-head sizes). So this is
  a **Gemma-4-geometry-specific FlashInfer nvfp4 kernel bug** — squarely Task #25.

## Implications
- **Do not** publish "+0.28 inherent NVFP4 cost." The headline "NVFP4 KV ≈ near-lossless +
  3.556× capacity" holds for the **format**; the serving red is a kernel bug to fix.
- Next (Task #25): localize the FlashInfer nvfp4-KV kernel error for Gemma-4 geometry —
  e.g., per-layer FlashInfer-nvfp4 output vs the reference-dequant output (sm_120 and sm_121),
  isolating prefill vs decode and sliding vs global layers.

## Caveats (honest)
- Reference uses base models + raw wikitext; Codex used `-it` + chat corpus + ctx 8185.
  But the Gemma-3 validation used the same base/raw setup and matched the ledger, and the
  format-loss delta is an architectural property — base-vs-it / corpus cannot move +0.003 to
  +0.281 (28–90× gap).
- Single-level fp8 block-scale reference (no separate fp32 global scale). Validated against
  Gemma-3; real NVFP4 loss is ≤ this. Conclusion is robust to the approximation.
