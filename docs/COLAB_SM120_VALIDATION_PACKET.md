# Colab sm_120 (RTX PRO 6000) validation packet — second-platform lane

Date authored: 2026-06-11. Owner: Claude lane. Purpose: convert the campaign's
"RTX PRO 6000 compatible" claim from static (CC 12.x family) to measured, using
Colab Pro G4-class runtimes. No PGX involvement; runs any time.

## Why these runs matter
- Family-wide vs sm_121-specific: every dispatcher finding (trait guard, fp8
  1-byte term, out-width fix) was measured only on GB10. sm_120 rows make the
  upstream filings two-platform.
- vllm#40677 was reported ON this hardware class (FLASHINFER + Gemma 4 NVFP4
  rejected on RTX PRO 6000): reproduce verbatim, then show the fix.
- The eventual public notebook lives under Colab constraints (no Docker,
  x86_64, pip/source installs) — this packet builds that plumbing early.

## Session 0 — runtime bootstrap (one notebook cell each)
1. Verify GPU: `nvidia-smi` → expect RTX PRO 6000 / CC 12.0. Record VBIOS/driver.
2. torch: the Colab default torch must report `torch.cuda.get_device_capability()
   == (12, 0)`; if torch lacks sm_120 kernels, `pip install` the cu130 nightly.
3. FlashInfer from source (NO pip flashinfer — we need the campaign branch):
   `git clone https://github.com/jethac/flashinfer -b spark/hijinks-022-fa2-d512
   && cd flashinfer && git checkout fb7d62ea && pip install -e . --no-build-isolation`
   JIT cache to Drive: `export FLASHINFER_JIT_DIR=/content/drive/MyDrive/fi_jit`
   (mount Drive first; first JIT ~5 min, cached after — note the module-cache
   caveat from results/jit_cache_mode_unsoundness_analysis_20260611.md: NEVER
   reuse a Drive JIT cache across SF-mode changes).
4. hijinks scripts: clone this repo @ branch spark/hijinks-022-gemma4-mixed-kv,
   head >= e7893f5.

## Session 1 — probe matrix (weight-free, ~30 min incl. JIT)
Run and save JSONs to Drive (`results_colab_sm120/`):
- A-series equivalents: `flashinfer_nvfp4_kv_probe.py --vo-split 2 --head-dim 512
  --kv-container tuple --causal --v-scale-layout linear --no-deswizzle-flag
  --layouts NHD HND --cosine-threshold 0.9999 --flashinfer-source-root <clone>`
  at batch 4/qo 16, qo 1, and --signed-values; plus the d128 control.
- Geometry: `vllm_gemma4_mixed_kv_probes.py --probe fa2-vo-split-d512-vo256
  --geometry e4b|31b --flashinfer-source-root <clone>` (both).
- E1 fp8: `--probe fa2-vo-split-d512-vo256-fp8kv` — KEY QUESTION: does the
  1-byte trait rejection reproduce on sm_120? (GB10 verbatim error in
  results/claude_blockE1_20260611/.)
- bf16-d512 control: `--probe fa2-bf16-d512` (expect the trait-guard rejection,
  same as GB10).
Gate: cosines match GB10 rows to ~1e-6; any divergence is a per-arch finding.

## Session 2 — #40677 reproduction + fix demo (needs vLLM source build, ~1-2 h)
1. BEFORE: stock pip vLLM (or upstream main build), Gemma 4 E4B,
   `--attention-backend FLASHINFER --kv-cache-dtype nvfp4` → expect the
   "head_size not supported" rejection verbatim (cite #40677).
2. AFTER: build jethac/vllm@spark/hijinks-022-gemma4-mixed-kv (x86_64 wheel,
   cache to Drive; sm_120a in TORCH_CUDA_ARCH_LIST) + knobs
   VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1 → E4B serves.
   GATED on the r8-verified writer (do NOT run AFTER before the latch diag
   passes on the locally built extension: `nvfp4_linear_latch_diag.py`).
3. If healthy: E4B BEFORE/AFTER tok/s on sm_120 = the second-platform benchmark
   row for the Triton-retirement story (#38887's 4090 is also consumer-class).

## Caveats
- Colab preemption: keep runs <1 h/cell, checkpoint to Drive.
- 96 GB VRAM fits even 31B bf16 — but weights re-download per session (~10 min
  on Colab's pipe); cache to Drive only if quota allows (~60 GB).
- This packet's session 1 has no PGX dependencies; session 2 step 2 waits on
  the r8 latch-diag verdict to avoid re-validating against a defective writer.
