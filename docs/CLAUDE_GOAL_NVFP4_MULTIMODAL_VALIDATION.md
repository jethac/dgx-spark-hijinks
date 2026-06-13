# /goal (Claude, autonomous): prove the nvfp4-KV global-scale fix end-to-end AND close the multimodal gap

**Statement:** Confirm on a real serving stack that calibrating the per-tensor global K/V
scale fixes the Gemma-4 nvfp4-KV quality red, and — because all our evidence is text-only —
**validate it on the multimodal paths too**. Bank everything, mail Codex, leave a clean tree.

## Why (banked context — read these first)
- Root cause + reference proof: `docs/NVFP4_FORMAT_VS_KERNEL_GEMMA4.md` — the per-tensor
  global scale is ~2× too small → per-block fp8 SFs saturate → +0.28–0.4; calibrated = +0.005.
  Sweep repro: `docs/vast_anchor/gs_run.sh`.
- Codex confirmed (mail 0128) the served red is real (+0.403) and refuted the swizzle; my
  redirect/handoff is mail 0130. Codex is doing the SGLang side.
- The gap (Task #48): every quality number is text-only. Gemma 4 is text+image+audio; the
  mm-prefix (PrefixLM bidirectional) path under nvfp4 KV is unvalidated, and vision/audio KV
  have different magnitude stats → possibly a *second* under-ranged-global saturation.

## Steps
1. **Working stack.** The older `g6adc00f70` wheel CANNOT serve the Gemma-4 nvfp4 mm-prefix
   path (selector: `FLASHINFER: partial multimodal token full attention not supported`). Get
   the newer wheel Codex used (`vllm ...ge32459eea.sm120a`) from ghcr / CI artifacts / a
   `jethac/vllm` branch build, and stand it up on vast.ai (runbook: `docs/vast_anchor/SM120_NUMERICS_PLAN.md §5`).
2. **Text A/B (confirm the fix):** gemma-4-12B, ctx 4096, supplied-token PPL. nvfp4 KV with
   `--calculate-kv-scales` (calibrated) vs default constants. Expect Δ vs bf16 to collapse
   from ~+0.28 → ~+0.01–0.04. (bf16 base raw-wikitext NLL ≈ 1.81 is the known-good baseline.)
3. **Multimodal A/B (the gap):** image+text and audio+text inputs, bf16 vs nvfp4-default vs
   nvfp4-calibrated. Metric: coherence at minimum, plus a small VQA/image set and an audio
   probe if tractable. Explicitly check whether vision/audio KV need their own calibration.
4. **Bank + hand off:** every claim an artifact, every red a verbatim error; update
   `NVFP4_FORMAT_VS_KERNEL_GEMMA4.md` and Task #47/#48; mail Codex the results.

## Guardrails
- vast.ai only (off-Spark → no marker contention with Codex's E4B/ladder runs).
- **Print $/hr before every rent; destroy every box the instant its result is banked; run an
  unconditional cleanup sweep; HARD cap total new spend ≈ $15** (balance was $79.79). Log spend.
- Secrets env-only (VAST_API_KEY, HF_TOKEN) — never written to a file. Don't commit to
  `jethac.github.io`. If a step thrashes (>2–3 failed attempts), bank the partial + reds and
  move on rather than burning the budget.
- **Stop point:** results banked + pushed, Codex mailed, no active vast instances, balance reported.
