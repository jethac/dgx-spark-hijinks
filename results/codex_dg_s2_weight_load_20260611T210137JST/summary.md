# DiffusionGemma DG-S2 Weight-Load Manifest

Date: 2026-06-11 JST

Status: GREEN for the DG-S2 metadata-tensor weight-load gate.

Scope: WSL / CPU-side SGLang loader exercise. This instantiates
`DiffusionGemmaForBlockDiffusion` on `meta`, builds fake tensors from the real
BF16 safetensors shard headers, and runs the real SGLang `load_weights`
implementation. Large tensors stay on `meta`; scalar tensors use tiny CPU zeros
because the default scalar loader calls `.item()`.

This is not a live BF16 allocation, forward parity, or serving claim.

## Inputs

- Model: `google/diffusiongemma-26B-A4B-it`
- Snapshot: `0f28bc42f588fbd8f71e08102b1c3960298a1358`
- Resolved path:
  `/mnt/b/workshop/hf_cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/0f28bc42f588fbd8f71e08102b1c3960298a1358`
- Cache policy: WSL `HF_HOME` / `HF_HUB_CACHE` pointed at `/mnt/b/workshop/hf_cache/huggingface`
- SGLang lane source: WSL editable `~/sglang` from Claude's DG-S0 setup (`3a2e15153`, per `mail/0015_claude-to-codex_wsl-env-dg-s0-green-and-upstream-fp4kv.md`)
- Harness: `scripts/diffusion_gemma_weight_load_manifest.py`

## Result

- `ok_for_dg_s2_weight_load_manifest`: `true`
- `bootstrap_error`: `null`
- `load_error`: `null`
- `resolved_architecture`: `DiffusionGemmaForBlockDiffusion`
- `model_class`: `DiffusionGemmaForBlockDiffusion`
- Checkpoint index keys: `1047`
- Safetensors header keys: `1047`
- Loaded SGLang parameter names: `332`

Loader manifest:

- `backbone_seen`: `657`
- `backbone_loaded_count`: `332`
- `decoder_duplicate_skipped`: `30`
- `self_conditioning_loaded`: `4`
- `self_conditioning_missing`: `0`
- `vision_quarantined`: `356`

`weight_load_stderr.log` contains SGLang's expected "Some weights are not
initialized from checkpoints" warning for the per-layer/PLE-style SGLang-owned
parameters. The loader gate is based on the explicit manifest above: the decoder
backbone remap, duplicate skip, self-conditioning load, and vision quarantine all
complete without a load exception.

## Artifacts

- `weight_load_manifest.json`: structured manifest and gate value
- `weight_load_stdout.json`: stdout copy of the same manifest
- `weight_load_stderr.log`: loader warnings

## Next Gate

DG-S2 can advance from name/remap metadata into BF16 runtime work. The next
claim-grade step is a real serving/parity gate against the official vLLM
DiffusionGemma oracle; this row only proves the SGLang loader contract accepts
the checkpoint under metadata tensors.
