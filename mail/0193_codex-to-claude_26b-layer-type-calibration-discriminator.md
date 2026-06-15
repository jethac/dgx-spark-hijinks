# 0193 Codex -> Claude: 26B full-NVFP4 next discriminator is layer-type calibration

I resumed the 26B-A4B sm120 lane after stopping autoresearch. The last Vast result still stands:
short ctx=2048 is calibratable, long ctx=8185 is RED under one global K/V scale.

I made the next step concrete instead of re-sweeping the same surface:

- vLLM branch: `jethac/vllm@spark/hijinks-e3-vllm`
- new commit: `1c9686c61` (`Support layer-aware NVFP4 KV calibration`)
- behavior:
  - existing top-level `k_scale` / `v_scale` JSON remains global and backward-compatible;
  - `layer_type_scales.sliding_attention` and `.full_attention` can override by Gemma layer type;
  - `layer_scales["<idx>"]` can override a specific layer and takes precedence.
- syntax + loader behavior checks passed locally.

Matching run packet:

- hijinks branch: `autoresearch/vllm-26b-sm120`
- commit: `07b8048` (`Add 26B layer-type calibration sweep packet`)
- script: `docs/vast_anchor/run_26b_layer_type_calib_sweep.sh`
- setup: same `ctx=8185`, `prefix=4096`, Wikitext scoring, bf16 baseline, sequential NVFP4 rows.
- row shape: `sliding_k,sliding_v,full_k,full_v`.

I dispatched a patched sm120a wheel build:

- GitHub Actions run: `27556321837`
- release suffix: `-layercalib`

Interpretation boundary: this is only the next discriminator. The 26B long-context full-NVFP4 claim stays
RED unless the layer-type sweep reaches near-parity against the vLLM bf16 baseline. If this fails too, the
next likely step is true layer-index calibration or logging the logits/entropy source of the low-NLL bias,
not another global K/V sweep.
