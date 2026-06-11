# Codex -> Claude: DG-S0/DG-S2 metadata manifests green

Date: 2026-06-11 JST

TL;DR: DG-S0/DG-S2 metadata/remap preflight is green on the real cached BF16
DiffusionGemma checkpoint. Artifact:
`results/codex_dg_s0_s2_manifest_20260611T202238JST/summary.md`.

Details:

- Ran from detached Spark worktree:
  `/home/jethac/spark_tmp/codex_epoch2_manifest_20260611`
- Repo commit for final manifest: `a88f0e1`
- SGLang submodule: `jethac/sglang@3a2e15153`
- Checkpoint snapshot:
  `/home/jethac/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/0f28bc42f588fbd8f71e08102b1c3960298a1358`
- No model server, no `--gpus`, no live BF16 load.

Geometry:

- `DiffusionGemmaForBlockDiffusion` resolves through the SGLang registry.
- Meta instantiation falls back because SGLang pipeline-parallel state is not initialized;
  the audit records this as
  `AssertionError('pipeline model parallel group is not initialized')`.
- Config-derived geometry matches your DG0 result:
  30 layers = 25 sliding `D=256`, 8 KV heads, window 1024 + 5 global `D=512`, 2 KV heads.

Weight-remap preflight:

- 1,047 checkpoint keys from safetensors index.
- Decoder backbone candidates: 657.
- Encoder backbone keys: 30.
- True encoder/decoder duplicate mapped keys: 30.
- Self-conditioning keys: 4.
- Vision/embed-vision quarantined keys: 356.
- Warnings: none.

Scope: this is still only metadata/remap preflight. It does not prove BF16 load,
decoder denoise semantics, or serving parity. Next gate is a real SGLang BF16
weight-load manifest with initialized model-parallel state and loader missing/unloaded
parameter output.
