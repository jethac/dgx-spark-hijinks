# SGLang Gemma 4 VO-split authoring stop point

Date: 2026-06-11 JST

Branch: `third_party/sglang@spark/hijinks-018-fp4-e2m1-kv-sm121-serving`

## Scope

Started the SGLang-side FlashInfer VO-split wrapper plumbing for Gemma 4 global
attention layers. This is offline authoring only; no GPU serving claim is made here.

The change is gated behind:

```text
SGLANG_FLASHINFER_VOSPLIT=1
```

## Code touched

File:

```text
third_party/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py
```

Added:

- A gated VO-split helper that maps `head_dim=512` to `head_dim_vo=256`.
- A paged-input splitter that keeps K unchanged, slices V on the last dimension, and slices
  the V scale-factor tensor for SGLang's linear V-SF layout.
- A two-pass paged-prefill path in `_run_paged_native()` that runs FlashInfer twice and
  concatenates the two VO halves.
- `head_dim_vo=` propagation into the paged-prefill wrapper plan.

## Verification

Offline syntax gate:

```text
python -m py_compile python/sglang/srt/layers/attention/flashinfer_backend.py
```

Result: pass.

## Known gaps before any serving run

- Decode is not yet rerouted to the decode-as-prefill VO-split path. The current code emits a
  warning when the feature flag is enabled.
- Gemma 4 uses heterogeneous attention geometry: sliding-window layers use the SWA head
  dimensions, while global/full layers use the global `D=512` dimensions. The current scaffold
  still plans wrapper dimensions from `model_config.head_dim`; serving work must make this
  wrapper-id/layer-geometry aware before a claim.
- Head-512 writer-roundtrip is still required. The existing SGLang writer-roundtrip receipt
  is green for head-256/SWA/linear V-SF only:
  `results/sglang_fp4_kv_writer_roundtrip_20260611Tprobe2JST/summary.md`.
- Any SF-layout mode toggle must be part of module identity, not an ambient compile flag. See
  `results/upstream_draft_issue_flashinfer_module_cache_flags_20260611TmanualJST.md`.

## Next authoring step

Build a wrapper-id geometry table for SGLang FlashInfer plans:

- SWA wrapper: `swa_head_dim`, `swa_v_head_dim`.
- Full/global wrapper: `head_dim`, `v_head_dim`, with `head_dim_vo=256` when
  `SGLANG_FLASHINFER_VOSPLIT=1` and QK head dim is 512.

Then route decode through a VO-split-compatible path and add the head-512 writer-roundtrip
gate before attempting Gemma 4 serving.
