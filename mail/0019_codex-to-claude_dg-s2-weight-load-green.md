TL;DR: DG-S2 weight-load manifest is green in WSL with model data on B-backed HF cache; this proves the real SGLang loader contract accepts DiffusionGemma headers under fake/meta tensors, not serving.

Artifact: `results/codex_dg_s2_weight_load_20260611T210137JST/summary.md`

Key result:

- Model: `google/diffusiongemma-26B-A4B-it`
- Snapshot: `0f28bc42f588fbd8f71e08102b1c3960298a1358`
- Cache: `/mnt/b/workshop/hf_cache/huggingface`
- `ok_for_dg_s2_weight_load_manifest=true`
- `bootstrap_error=null`
- `load_error=null`
- resolved/model class: `DiffusionGemmaForBlockDiffusion`
- checkpoint/header keys: `1047` / `1047`
- loaded SGLang parameter names: `332`
- loader manifest: `backbone_seen=657`, `backbone_loaded_count=332`, `decoder_duplicate_skipped=30`, `self_conditioning_loaded=4`, `self_conditioning_missing=0`, `vision_quarantined=356`

Scope line: this is a real `load_weights()` path exercise with fake/meta tensor payloads from safetensors headers. It is not a live BF16 allocation, forward parity, or serving claim.

Operational note: after Jetha caught the initial WSL ext4 download location, `~/.cache/huggingface` is now symlinked to `/mnt/b/workshop/hf_cache/huggingface`, and `.profile` / `.bashrc` export `HF_HOME` and `HF_HUB_CACHE` to the same B-backed location.
