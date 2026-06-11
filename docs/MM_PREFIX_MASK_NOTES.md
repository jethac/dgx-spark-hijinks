# mm-prefix span-level bidirectional masking on the vLLM FlashInfer backend

Status: **implemented and probe-validated on P520** (RTX 5060 Ti, CC 12.0,
2026-06-11). This closes the deferred half of the mm-prefix waiver: multimodal
Gemma 3 / Gemma 4 can now serve images on FlashInfer — and therefore with the
NVFP4 KV cache on CC 12.x — instead of dropping multimodality via
`--language-model-only`.

Branches:

| repo | branch | head | change |
| --- | --- | --- | --- |
| jethac/vllm | `spark/hijinks-e2-vllm` | `7df3c67ec8` | mm-prefix custom-mask grouping in the FlashInfer backend + knob-gated gate lift |
| jethac/flashinfer | `spark/hijinks-022-fa2-d512` | `7d5d477b` | `plan()` moves `mask_indptr` to the custom mask's device before `segment_packbits` |

Knob: `VLLM_FLASHINFER_MM_PREFIX=1` (registered in `vllm/envs.py`, same
not-in-("", "0") convention as the other campaign knobs). Without it,
`FlashInferBackend.supports_mm_prefix()` stays False and nothing changes.

## Problem recap

Multimodal Gemma 3/4 prompts are "mm-prefix LMs": image-token spans attend
bidirectionally among themselves while text stays causal. Upstream vLLM only
lets FLASH_ATTN (FA4), FLEX_ATTENTION, TRITON_ATTN and the ROCm backends claim
`supports_mm_prefix()`; FlashInfer was rejected outright by
`validate_configuration` (`vllm/v1/attention/backend.py:302`). Our epoch-2
ladder needs FlashInfer for the NVFP4 KV cache and the FA2 VO split, so the
campaign had waived multimodality (`is_mm_prefix_lm` returns False under
`--language-model-only`, `vllm/config/model.py:1270`).

DG-2 causal grouping (branch `spark/hijinks-e2-dgemma`) handles per-REQUEST
uniform causality with two wrappers + gather/scatter. mm-prefix needs
per-SPAN masks WITHIN a request — wrapper grouping cannot express it;
FlashInfer's `custom_mask` prefill argument can.

## Recon findings (upstream plumbing)

- Span source: `gpu_model_runner.py:2298-2338` builds
  `CommonAttentionMetadata.mm_req_doc_ranges: dict[req_idx ->
  list[(start, end)]]` from `mm_feature.mm_position.extract_embeds_range()`.
  Document positions, END-INCLUSIVE, valid iff `start < end`; ranges longer
  than the text config's `sliding_window` are dropped at the source.
- Mask contract (must match exactly): `(causal AND sliding_window) OR
  (q in span AND kv in span)` — the OR lets spans OVERRIDE the sliding
  window. See `triton_attention_helpers.py:269-353` (`compute_kv_seq_mask`)
  and `flex_attention.py:569-586` (`or_masks` after `and_masks`).
- Gemma4 `use_bidirectional_attention='vision'` applies spans ONLY to
  sliding layers: `gemma4_mm.py:1618-1655` clears
  `mm_prefix_range{,_tensor}` on full-attention layers' metadata each
  forward. Gemma3 (fallback list `model_arch_config_convertor.py:298`) masks
  all layers and has no clearing hook.
- `cuda.py:260` forces `--disable-chunked-mm-input` for mm-prefix models, so
  a span never straddles the query-window boundary.
- FlashInfer mask machinery: `BatchPrefillWithPagedKVCacheWrapper.plan()`
  accepts `custom_mask` (flattened qo_len x kv_len bools per request,
  concatenated); it computes bit counts via `_compute_page_mask_indptr`
  (`prefill.py:1445`) and bit-packs per request with `segment_packbits`
  (byte-aligned segments, little bit-order). `run()` then dispatches
  `MaskMode.CUSTOM` (`prefill.py:2490-2499`). The kernel reads
  `custom_mask_ptr = maybe_custom_mask + maybe_mask_indptr[batch_idx]`, bit
  `qo_idx * kv_len + kv_idx` (`include/flashinfer/attention/variants.cuh:57-86`).
  Both the standard FA2 JIT module and the customize (NVFP4 jit_args) module
  compile all four mask modes (`jit/attention/modules.py:1617`), and the vLLM
  NVFP4 jit_args already declare `maybe_custom_mask`/`maybe_mask_indptr`
  (`flashinfer.py: _fa2_nvfp4_prefill_jit_args`) — so NO kernel work was
  needed for FP4 composition.
- Two FlashInfer sharp edges found:
  1. `plan(packed_custom_mask=...)` (pre-packed) leaves `mask_indptr` in BIT
     units while the kernel does BYTE pointer arithmetic — broken upstream
     for batch_size > 1. We therefore pass the UNPACKED `custom_mask` and
     let `segment_packbits` produce the byte indptr.
  2. `segment_packbits` requires its indptr on the mask's device
     (`CHECK_DEVICE`, `csrc/quantization.cu:42`), but `plan()` derives
     `mask_indptr` from the (CPU) indptr arrays while the mask lives on GPU.
     Fixed on our flashinfer branch (`7d5d477b`, one-liner). **The vLLM
     side depends on this fix.**
- `DefaultAttention<use_custom_mask, use_sliding_window=true, ...>` ANDs the
  in-kernel sliding window with the custom mask — wrong order for spans. So
  the mm wrapper is planned `causal=False, window_left=-1` and the custom
  mask carries causality + SW + spans wholesale. `window_left=-1` under an
  SW-compiled module degrades to a no-op bound (ctor sets
  `window_left = kv_len`), so the NVFP4 jit module (compiled with
  `use_sliding_window=True` for Gemma sliding layers) needs no second
  module variant.

## Design (vLLM `spark/hijinks-e2-vllm`, `7df3c67ec8`)

All in `vllm/v1/attention/backends/flashinfer.py` + the knob in `envs.py`:

- `FlashInferBackend.supports_mm_prefix()` returns
  `envs.VLLM_FLASHINFER_MM_PREFIX`.
- Builder `__init__`: `mm_prefix_enabled = knob AND model.is_mm_prefix_lm`,
  then per layer group: for Gemma4 'vision' it stays enabled only when
  `window_left >= 0` (sliding groups). This is the static, build-time
  equivalent of the gemma4_mm forward-time clearing hook (FlashInfer bakes
  masks into wrapper plans, so forward-time clearing cannot work; the hook
  becomes a no-op because FlashInferMetadata has no `mm_prefix_range`
  attribute). Gemma3 masks every group. DCP is rejected.
- `build()`: `_mm_prefix_prefill_spans` filters each prefill request's
  ranges to `start < end AND end >= context_len AND start < seq_len` (spans
  fully inside the computed context need nothing: K/V projections are
  mask-independent and in-window queries are then pure text). If no request
  survives the filter, the legacy scalar-causal path runs UNCHANGED
  (byte-identical regression guarantee). Otherwise TRTLLM prefill is forced
  off and `_plan_prefill_mm_groups` partitions prefill requests DG-2 style:
  - plain group -> existing causal wrapper, `causal=True`,
    `window_left=self.window_left`;
  - mm group -> second persistent wrapper (`_get_mm_prefill_wrapper`,
    same `_make_paged_prefill_wrapper` factory incl. NVFP4 jit_args),
    `causal=False`, `window_left=-1`,
    `custom_mask=_build_mm_prefix_custom_mask(...)` = per-request
    `(causal AND SW) OR span` bools on GPU, flattened and concatenated.
  - Groups land in `FIPrefill.mm_groups` (`FIPrefillMMGroup`: wrapper,
    token_indices, num_tokens).
- `forward()`: when `mm_groups` is set, gather query rows per group by
  token index, run (VO-split-aware: `_run_vo_split_prefill` per group for
  qk512/vo256), scatter with `index_copy_` — the DG-2 scaffolding. Asserts
  are group-aware (mm wrapper: non-causal, window_left==-1, mask buf set).
- Decode pathway untouched: spans live in the prompt, decode queries are
  strictly causal. Under the VO split (reorder threshold 0) decode-shaped
  qo_len==1 rows flow through prefill but their spans are in-context, so the
  filter routes them to the plain causal group.
- Cascade attention is already hard-disabled upstream
  (`use_cascade_attention` returns False), so no mm/cascade interaction.

## P520 probe results (gate green)

Harness: `wsl_sm120/test_mm_prefix_mask.py` (pattern of
`test_causal_grouping.py`; copy in `results/p520_mm_prefix_mask_20260611/`
with the JSON). Provenance: RTX 5060 Ti (CC 12.0), torch 2.12.0+cu130,
flashinfer source `7d5d477b` clean, fp32 torch reference with the composed
mask, mixed 4-request batches with interleaved pages, context-append
requests, CPU indptr + GPU mask (exercises the device fix).

| case | config | result |
| --- | --- | --- |
| bf16_d256 | dense bf16, qk=vo=256 | PASS (cos >= 0.999998) |
| bf16_d256_sw | + sliding window 32 carried by the mask | PASS |
| bf16_d512_vosplit | qk=512 / vo=256, two VO-split passes | PASS |
| nvfp4_d256 | uint8-packed NVFP4 KV + linear V-SF, vLLM customize jit module (swa=1, plan window_left=-1) | PASS (cos >= 0.999998) |
| bf16_d256_allmm | every request mm (plain group empty, one request with 2 spans) | PASS |
| bf16_d256_nomm | spans in metadata but none in-window (span-in-context, decode-shaped qo=1, degenerate start==end) | PASS, classified all-plain, **byte-identical** to the causal plan |

mm rows demonstrably diverge from a pure-causal plan (`mask_engaged`), plain
rows in mixed batches matched the full causal plan bitwise on every case.

## How to enable (serving)

```
VLLM_FLASHINFER_MM_PREFIX=1 \
VLLM_NVFP4_KV_LINEAR_V_SF=1 VLLM_NVFP4_KV_VOSPLIT=1 \  # for NVFP4 configs
vllm serve <gemma4-mm> --attention-backend FLASHINFER \
    --kv-cache-dtype nvfp4 ...   # NO --language-model-only needed
```

Requires a FlashInfer build at/after `7d5d477b` on
`spark/hijinks-022-fa2-d512`.

## What remains (Spark serving smoke spec)

The P520 validation is plan-level (no compiled vLLM install on the P520
WSL env; same precedent as DG-2). Next agent on the Spark host should run:

1. Gemma 4 E4B multimodal (vision tower loaded, no `--language-model-only`),
   `VLLM_FLASHINFER_MM_PREFIX=1`, bf16 KV first; one image + text chat
   request; verify the mm group plans (log line "FlashInfer mm-prefix:")
   and the response is coherent vs the TRITON_ATTN baseline of the same
   prompt (logprob/token-level compare, not eyeball).
2. Repeat with `--kv-cache-dtype nvfp4` + linear-V-SF/VOSPLIT knobs
   (sliding layers NVFP4 d256 masked; global d512 layers stay causal by
   the 'vision' policy — confirm via the per-group enable log).
3. Mixed batch: image request + concurrent text requests + in-flight
   decodes (exercises group interleaving and decode pathway untouched).
4. Gemma 3 (e.g. 4B-it) spot check: bidi applies to ALL layer groups there,
   including full-attention.
5. Regression: same serving config with a text-only prompt must produce
   byte-identical logits to the knob-off run (no-mm fast path).

Open items / known limits:

- DCP + mm-prefix is rejected (NotImplementedError) — fine for the ladder.
- TRTLLM prefill is bypassed only for batches with in-window spans; mm
  models on SM100 would eat a perf hit on image prefills (irrelevant for
  CC 12.x, which never takes TRTLLM for NVFP4 anyway).
- Upstream FlashInfer `packed_custom_mask` bit/byte indptr mismatch is
  worth an upstream issue eventually; we sidestep it by passing the
  unpacked mask.
- Async scheduling: `_mm_prefix_prefill_spans` touches `seq_lens_cpu`,
  which may force a sync in async-spec-decode mode; acceptable for the
  campaign, note if profiling Spark serving.
