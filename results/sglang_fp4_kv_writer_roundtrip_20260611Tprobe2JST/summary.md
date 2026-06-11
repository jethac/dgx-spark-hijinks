# SGLang FP4 KV Writer Roundtrip Probe - 2026-06-11

## Verdict

GREEN. SGLang's `MHATokenToKVPoolFP4.set_kv_buffer()` writer plus the FlashInfer FA2 paged reader roundtrip correctly for full NVFP4 K+V at the tested head-256 sliding-window shape with linear scale factors.

## Scope

- Host: `thinkstationpgx-00b4`
- GPU: NVIDIA GB10, compute capability 12.1
- Container: `sglang-source-stack-c3dae30f-e631a13fd:latest`
- Checkout: `/home/jethac/spark_tmp/dgx-spark-hijinks-live-4df2367`
- FlashInfer JIT flags: `-gencode=arch=compute_121a,code=sm_121a`
- Deswizzle macro: not enabled
- KV mode: full NVFP4 K+V, not mixed KV
- Pool: `MHATokenToKVPoolFP4`, page size 1, linear K/V scale factors
- Shape: `head_dim=256`, `num_qo_heads=32`, `num_kv_heads=16`, `kv_len=384`

## Gate Results

| case | window_left | qo_len | output cosine vs dequant reference | output max abs diff | LSE cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| global_qo | -1 | 16 | 0.99999118 | 0.00048828 | 0.99999994 |
| swa_qo | 255 | 16 | 0.99974465 | 0.00317383 | 0.99999994 |
| swa_decode_as_prefill | 255 | 1 | 0.99999142 | 0.00073242 | 0.99999994 |

## Interpretation

This clears SGLang's real writer path and the shared FA2 paged reader for the head-256, SWA-window, linear-SF shape. For Claude's vLLM 31B gibberish hunt, this points away from a shared FlashInfer reader bug for this shape and toward vLLM's writer/runtime-side state or a shape not covered by this probe.

The probe does not prove head-512, mixed head dimensions, or full serving correctness. It is a focused writer-roundtrip gate for the SGLang linear-SF layout.

Artifacts:

- `output.json` - clean probe result JSON.
- `container.stdout` - raw container stdout including NVIDIA/SGLang banner.
- `run.log` - stderr warnings and import/runtime notes.
