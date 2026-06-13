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

### PRIMARY — reference-sim multimodal (wheel-free, robust; do this first)
The HF-eager + torch-qdq method (proven: `gs_run.sh`) needs NO serving wheel, so it can't be
blocked by the mm-prefix selector. Extend it to multimodal:
1. HF eager `Gemma4Unified` with an **image+text** input (AutoProcessor + a test image + a
   VQA-style prompt). Intercept K/V (now including vision tokens) via the `sdpa` patch.
2. **Measure KV dynamic range by modality:** per-tensor amax vs per-16 block-amax distribution
   for K and V, split into text-token vs vision-token positions. The hypothesis to test: vision
   (and audio) tokens have *wider* range → more prone to the under-ranged-global saturation.
3. **Global-scale A/B on multimodal KV:** the two-level qdq from `gs_run.sh`, `g=1` (calibrated)
   vs `g=0.5` (under-ranged), measured on the multimodal forward (NLL on the text completion).
   Question: does one per-tensor global suffice for mixed text+vision KV, or do the modalities
   need separate calibration (i.e. is there a *second* saturation bug specific to vision/audio)?
4. **Audio is required, not optional — cover E2B, E4B, AND 12B** (per Jetha). Gemma-4 audio
   processor + a short speech clip; same per-modality KV amax stats + global-scale A/B on each.
   Vision: cover at least one size (E4B); audio: all three sizes (E2B/E4B/12B). The validation
   claim is incomplete until every multimodal size has its image AND audio KV checked.

### SECONDARY — serving e2e (bounded; only if the wheel is cheaply obtainable)
Codex's `ge32459eea` wheel serves the mm-prefix nvfp4 path; mine (`g6adc00f70`) does not.
If that wheel is pullable from a CI artifact / ghcr in a few tries, stand it up and run the
text `--calculate-kv-scales` A/B + a multimodal serving smoke. **If it thrashes (>2–3 tries),
bank the blocker and rely on the PRIMARY reference-sim result** — do not burn the night on it.

### Bank + hand off
Every claim an artifact, every red a verbatim error; update `NVFP4_FORMAT_VS_KERNEL_GEMMA4.md`
and Task #47/#48; mail Codex the multimodal findings (esp. if vision/audio KV need own calibration).

## Guardrails
- vast.ai only (off-Spark → no marker contention with Codex's E4B/ladder runs).
- **Print $/hr before every rent; destroy every box the instant its result is banked; run an
  unconditional cleanup sweep; HARD cap total new spend ≈ $15** (balance was $79.79). Log spend.
- Secrets env-only (VAST_API_KEY, HF_TOKEN) — never written to a file. Don't commit to
  `jethac.github.io`. If a step thrashes (>2–3 failed attempts), bank the partial + reds and
  move on rather than burning the budget.
- **Stop point:** results banked + pushed, Codex mailed, no active vast instances, balance reported.
