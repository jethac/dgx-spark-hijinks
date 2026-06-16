# 26B-A4B packet expanded with layer capture; Vast credit still blocks live row

Codex update, 2026-06-16 JST.

I rechecked Vast. No instances are running. There is RTX PRO 6000 WS capacity again; I selected the cheap
single-GPU high-RAM offer `37036746` (257 GB host RAM, 96 GB VRAM) and attempted a guarded create with
`--cancel-unavail` and the Ubuntu 22.04 CUDA 13 devel image. Vast rejected it before allocation:

```text
Your account lacks credit; see the billing page.
```

No instance was allocated or left behind.

Offline progress while blocked: expanded the readout packet from mail 0212 to capture the first sliding block
directly:

- `docs/vast_anchor/vllm_readout_capture_sitecustomize.py` now also supports `VLLM_LAYER_CAPTURE_DIR`.
- `docs/vast_anchor/compare_layer_captures.py` compares layer-capture outputs.
- `docs/vast_anchor/run_26b_readout_capture.sh` enables `LAYER_CAPTURE=1` by default for layers `0,1,2,3,4`.

The layer hook wraps `Gemma4DecoderLayer.forward()` and records selected rows for:

- layer input
- attention output (`self_attn`)
- MLP output
- MoE output
- final layer output

Comparator output is bucketed by local row and reports cosine, rel-L2, RMS delta, mean-abs delta, and max-abs
delta per `(label, layer, phase, bucket)`.

Validation done locally:

- `python -m py_compile` for the readout hook and both comparators: green
- `bash -n` for the readout run/launcher: green
- synthetic `compare_layer_captures.py` smoke: green

Next live order when credit is restored:

1. Run the cheap public top-logprob packet if we still want the public-distribution-only discriminator.
2. Run the expanded readout/layer packet if we want the deeper split in one row.

No model-quality claim changed. 26B-A4B full NVFP4 K+V remains research/red; fp8 KV remains the ship path.
