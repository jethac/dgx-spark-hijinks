# DiffusionGemma DG-S0/DG-S2 Metadata Manifests

Date: 2026-06-11 JST

Status: GREEN for metadata/remap preflight. This is not a BF16 weight-load result,
not a serving result, and not parity against the official vLLM image.

## Environment

- Host: `thinkstationpgx-00b4`
- Worktree: `/home/jethac/spark_tmp/codex_epoch2_manifest_20260611`
- Repo commit: `a88f0e1` (`epoch2`)
- SGLang submodule: `jethac/sglang@3a2e15153`
- Checkpoint snapshot:
  `/home/jethac/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/0f28bc42f588fbd8f71e08102b1c3960298a1358`
- Config audit container: `nvcr.io/nvidia/sglang:26.05-py3`, no `--gpus`

## Geometry Manifest

Artifact: `config_geometry_manifest.json`

- Architecture resolved: `DiffusionGemmaForBlockDiffusion`
- Model class resolved: `DiffusionGemmaForBlockDiffusion`
- Geometry source: `hf_config`
- Model instantiation fallback reason:
  `AssertionError('pipeline model parallel group is not initialized')`
- Layers: 30 total
- Sliding layers: 25, `head_dim=256`, `num_key_value_heads=8`, window 1024
- Global layers: 5, `head_dim=512`, `num_key_value_heads=2`
- Canvas length: 256
- Confidence threshold: 0.005

The fallback is expected for this no-server metadata audit: SGLang's Gemma4 model
constructor requires initialized pipeline-parallel state. The script records this and
falls back to config-derived geometry.

## Weight Remap Manifest

Artifact: `weight_manifest.json`

- `ok_for_dg_s2_weight_remap_probe`: `true`
- Keys: 1,047 from safetensors index
- Encoder backbone keys: 30
- Decoder backbone candidate keys: 657
- True encoder/decoder duplicate mapped backbone keys: 30
- Self-conditioning keys: 4
- Vision/embed-vision keys quarantined for text-only rungs: 356
- Warnings: none

`safetensors` was not installed in the bare host Python, so this row uses the
safetensors index rather than shard headers. That is sufficient for name/remap coverage
but not for shape verification. The next BF16 live-load gate must record actual SGLang
loader output and missing/unloaded parameter names.

## Next Gate

Run the BF16 weight-load manifest in a real SGLang runtime with initialized model
parallel state. Green requires the loader manifest to agree with this preflight:
vision quarantined, self-conditioning loaded, expected decoder/encoder duplicates
skipped, and no unexpected unloaded backbone parameters.
