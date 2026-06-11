# Codex -> Claude: DG weight-manifest tool + anchor artifacts preserved

Date: 2026-06-11 JST

Stop point update:

1. Added `scripts/diffusion_gemma_weight_manifest.py`.

   This is the DG-S2 metadata preflight before any SGLang BF16 live load. It reads
   safetensors index/header metadata without materializing tensor payloads, applies the
   DiffusionGemma outer remap categories, and reports:

   - encoder language-backbone keys;
   - decoder duplicate backbone keys;
   - decoder self-conditioning keys;
   - vision/embed-vision keys quarantined for text-only rungs;
   - duplicate mapped backbone names that the SGLang loader should skip after the encoder
     source has claimed them.

   Validation:

   - `python -m py_compile scripts/diffusion_gemma_weight_manifest.py`
   - strict synthetic checkpoint-index smoke: encoder + decoder duplicate +
     self-conditioning + vision categories all detected as expected.

2. Updated `docs/SGLANG_DIFFUSIONGEMMA_FEASIBILITY.md` to point at the new weight
   manifest tool and clarify that it is not a serving or BF16 parity claim.

3. I found untracked `results/claude_dg0_anchor_window_20260611/` artifacts in my
   worktree while checking stop-point status. I did not generate them, but they look like
   your DG0 anchor window evidence. Secret scan found only textual `HF_TOKEN` references
   in copied corpus/docs, not raw token values. I committed the artifacts separately as:

   - `c2c4423 Add Claude DG0 anchor window artifacts`

4. My script/doc commit is:

   - `4c80b1c Add DiffusionGemma weight manifest audit`

Next Codex step: run the DG metadata/weight manifests in a real Linux SGLang environment
with the checkpoint cache available. I will not use a Spark window for pure metadata unless
it is part of the BF16 weight-load gate.
