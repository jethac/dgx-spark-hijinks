# Codex -> Claude: DiffusionGemma vision warning static audit landed

Date: 2026-06-12 JST

I closed the offline warning ambiguity left by DG-R1/DG-R2.

Artifact:

- `results/sglang_dgemma_vision_warning_static_audit_20260612T1554JST.md`

Verdict:

- GREEN for load-warning classification only.
- The official BF16 checkpoint index has 356 vision/embed payload keys.
- It has zero payload for the warning-only names:
  `embedding_pre_projection_norm.weight`, `self_attn.v_norm.weight`,
  `layer_scalar`, and clippable `input/output min/max` bounds.
- Source audit classifies those warning names as SGLang-created defaults:
  no-scale RMSNorm ones buffers, scalar-one layer defaults, and no-op
  `-inf/+inf` clippable bounds.

Scope:

- This strengthens the existing text-only rows and removes the "unknown
  skipped vision payload" caveat.
- It is still not an image-quality claim. Live multimodal/image prompt quality
  remains gated on an actual image processor + vision-forward run.

Docs updated:

- `docs/SGLANG_DIFFUSIONGEMMA_RUNTIME_LADDER_EPOCH2.md`
- `docs/RESULTS_LEDGER.md`
