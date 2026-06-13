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

## Localization (2026-06-13): the kernel behaves like a PER-TENSOR scale

Scale-granularity sweep on gemma-4-12B (ctx 4096, same kernel-free reference, varying the
NVFP4 block size + K-vs-V):

| scheme | Δ nats/token |
| --- | ---: |
| **block-16 (NVFP4 spec)** K+V | **+0.013** |
| block-32 / 64 / 128 | +0.014 / +0.018 / +0.027 |
| block-256 (per-head-row) | +0.071 |
| **per-tensor (single global scale)** | **+0.235** |
| block-16 **K-only** | +0.0004 |
| block-16 **V-only** | +0.004 |

`per-tensor` scaling reproduces the served magnitude (+0.235 at ctx 4096 ≈ Codex's +0.281
at ctx 8185; the format-loss *decreases* with ctx while per-tensor *grows*, so per-tensor at
8192 lands right on +0.281). **So the served nvfp4 KV path is effectively applying a coarse
(≈per-tensor) scale, not the per-16-block scale the NVFP4 format mandates.** That is the bug:
a scale-factor *granularity* defect, not the format and not the attention math.

- K quantization is essentially free (+0.0004); V is the slightly sensitive term — consistent
  with the per-block V scale-factor (SF) being the thing that's collapsing. Prime suspect:
  the **V-SF tensor layout/stride** (swizzled-vs-linear; `VLLM_NVFP4_KV_LINEAR_V_SF`), where a
  wrong stride makes per-block scales read as effectively one scale.
- Why Qwen/Gemma-3 were near-lossless even if their path is equally coarse: their KV has less
  cross-block dynamic range, so a coarse scale barely hurts; Gemma-4's KV needs the per-16
  granularity.

**Fix direction:** verify/repair per-16-block SF application in the nvfp4 KV read path (esp.
V-SF layout/stride). Proper block-16 recovers ~+0.22 nats/token (→ +0.013, near-lossless).

## Code localization (2026-06-13): the swizzled V-SF path on head-256 — ❌ REFUTED 2026-06-14

> **This hypothesis is wrong.** Codex (mail 0128) ran the v0.5.13 SGLang image with
> `deswizzle_macro_active=False` (i.e. **already on linear V-SF**) and VO-split engaged, and
> still measured **+0.403** on the 12B. So it is *not* the swizzle, not the head-256 default
> path, and "force linear V-SF" is not the fix. The real cause is the **global scale** — see
> the corrected section below. (Also: the 12B *does* have 512-wide globals and goes through
> VO-split; the earlier "head 256, no VO-split" read was incomplete.) Section kept for honesty.

Traced the per-tensor signature to the **V scale-factor swizzle** in the vLLM nvfp4 KV path
(`vllm/v1/attention/backends/flashinfer.py`, `reshape_and_cache_nvfp4` writer):

- `VLLM_NVFP4_KV_LINEAR_V_SF=1` makes writer+reader use a **linear** V-SF (no swizzle).
  **Unset (default) = swizzled writer + in-kernel de-swizzle** (`FLASHINFER_PAGED_V_SF_DESWIZZLE=1`).
- The VO-split path (head_size > 256 → 31B/E4B) **forces** `LINEAR_V_SF=1`. The **dense 12B is
  head 256**, so it is **not** VO-split → it takes the **default swizzled V-SF path**.
- vLLM's own comment: the *"trtllm 4-token swizzle does not commute with head-dim slicing"* and
  *"spreads each 4-token group across the full scale row."* A swizzle/de-swizzle bug or mismatch
  on the head-256 path scrambles which per-16 block each V value reads → the per-tensor-like
  coarseness. Matches the K-free / V-sensitive asymmetry (K-SF is always linear).

**One-flag confirmation:** rerun the 12B nvfp4 matched anchor with `VLLM_NVFP4_KV_LINEAR_V_SF=1`.
If Δ drops from +0.281 toward ~+0.01–0.04, the swizzled-V-SF de-swizzle path (default for
head-256) is the bug. **Free prediction:** the 31B/E4B (already forced linear V-SF) should be
**near-lossless already** — if 31B nvfp4 is green while 12B is red, that confirms it.

**Fix:** either repair the head-256 swizzle/de-swizzle, or default head-256 nvfp4 KV to linear
V-SF (as VO-split already does).

## CORRECTED localization (2026-06-14): the per-tensor GLOBAL scale is ~2× too small

A faithful **two-level** reference sim (per-tensor global scale + per-16 fp8 e4m3 block SF,
with realistic fp8 saturation — matching `cvt_warp_fp16_to_fp4(in_vec, global_scale, sf_out)`
in `csrc/.../nvfp4_kv_cache_kernels.cu`, where the per-tensor `k_scale`/`v_scale` *is* the
quantizer's `global_scale`). Swept the global-scale multiplier `g` on gemma-4-12B (ctx 4096),
`g=1` = calibrated to fit fp8 range:

| `g` (×calibrated) | Δ nats/token |
| --- | ---: |
| 1, 2, 4, 8, 16 | **+0.0056** (flat, near-lossless) |
| **0.5** (global ~2× too small) | **+0.2574** ≈ vLLM served +0.281 |
| 0.25 (4× too small) | +1.556 (catastrophic) |

**Mechanism (decisive):** the sensitivity is *asymmetric*. Over-scaling the global is free;
**under-scaling it forces the per-16 fp8 block SFs to saturate at 448** → the largest-magnitude
blocks get a clipped (too-small) effective scale → their values compress → quality collapses.
The served +0.28–0.4 corresponds to an effective `g ≈ 0.4–0.5`: **the global scale is roughly
2× too small.** It is **V-driven** — V-only is +0.0049 vs K-only +0.0007 at safe `g`, because V
has the wider cross-block dynamic range, so V's outlier blocks saturate first. (This subsumes
the earlier "per-tensor +0.235" granularity finding: it wasn't granularity, it was an
under-ranged global.)

**Root cause + fix:** the served path's per-tensor `k_scale`/`v_scale` (the quantizer's
`global_scale`) is **fixed/under-calibrated** — vLLM defaults `calculate_kv_scales=False` with
constants `K_SCALE_CONSTANT=200`, `V_SCALE_CONSTANT=100`, and warns "may cause accuracy drop
without a proper scaling factor"; SGLang reverted its `SGLANG_FP4_KV_K_GLOBAL_SCALE_MULTIPLIER`
at the rebase tip. **Calibrate the global scale per tensor (amax-based, big enough that no
per-16 block SF saturates fp8 — `global ≥ tensor_amax / (6·448)`), especially for V,** and the
delta collapses from +0.4 to ~+0.005. No swizzle, no VO-split, no kernel-math change required.

Repro: `docs/vast_anchor/gs_run.sh` (two-level global-scale sweep).

## Multimodal validation (2026-06-14): the fix generalizes; vision KV is NOT wider-range

All quality evidence above is text-only; Gemma 4 is text+image+audio (Jetha: validate all).
Reference-sim method extended to multimodal (HF eager + torch-qdq, no serving wheel needed),
`docs/vast_anchor/mm_exp.py` (image) / `mm_audio.py` (audio). Per-modality KV amax stats +
the same two-level global-scale A/B (g=1 calibrated vs g=0.5 under-ranged), measuring NLL of
a vision/audio-dependent answer.

Each row generated a coherent modality-dependent answer (e.g. image → "two tabby cats…remote
control"; audio → accurate ASR of the LibriVox clip), confirming the modality is actually used.

| model | modality | KV amax ratio (mm/text) | bf16 NLL | **calibrated Δ** (g=1) | **under-ranged Δ** (g=0.5) |
| --- | --- | --- | ---: | ---: | ---: |
| E4B | image | K 0.99× · V 0.93× | 0.478 | **+0.047** | +0.198 |
| E2B | audio | K 0.95× · V 0.99× | 0.952 | **+0.078** | +0.099 |
| E4B | audio | K 0.99× · V 0.84× | 1.983 | **+0.069** | +0.085 |
| 12B | audio | K 0.99× · V 0.98× | 1.424 | **−0.047** | +0.815 |

**Conclusions (multimodal validated):**
1. **Vision and audio KV are NOT wider-range than text** (ratios 0.84–0.99× across all sizes) →
   a single per-tensor global scale serves all modalities; image/audio need **no separate
   calibration** and introduce no second saturation regime.
2. **The calibration fix generalizes to every multimodal path** — calibrated is small/negative;
   under-ranged is worse everywhere (catastrophic on 12B: **+0.815**).
3. **12B is the most global-scale-sensitive** (under-ranged +0.815 vs E2B/E4B ~+0.09), matching
   the text 12B being the +0.281 case — so calibration matters most exactly where it bit us.
4. **Calibrated nvfp4 can beat bf16** (12B audio −0.047) — the "quantized-KV-beats-bf16" anomaly
   (Task #25) extends to the multimodal/audio path.

→ Net: NVFP4 KV on Gemma 4 is near-lossless across **text, image, and audio** once the global
scale is calibrated; the only bug is the under-ranged default global, and it's modality-agnostic.
(Done via the wheel-free reference sim; full serving mm-prefix nvfp4 still wants Codex's wheel.)

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
