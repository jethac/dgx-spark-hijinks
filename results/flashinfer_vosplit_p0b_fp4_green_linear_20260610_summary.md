# P0b result — FP4 VO-split GREEN (linear SF); swizzled at 0.958 (one residual)

Date: 2026-06-10 JST. Lane: spark/hijinks-022-gemma4-mixed-kv + jethac/flashinfer@fb7d62ea.

## Resolution of the P0b mystery (debug run)
FLASHINFER_PREFILL_DEBUG_ONCE proved: module correct (head_dim_qk=512, head_dim_vo=256,
require_fp4_kv=1), V tensors/strides correct at the C++ boundary — but Python passed a
512-wide OUTPUT buffer (`out_head_dim = q.shape[-1] if kv_cache_sf ...`), so the vo=256
kernel garbled a too-wide buffer. Also explains the bit-identical view-vs-contiguous runs:
both computed correct attention and garbled identically on write.

## Fix
`jethac/flashinfer@fb7d62ea` — size NVFP4 attention output from V (`v.shape[-1] * 2`),
not Q, in paged/ragged/single prefill run paths + paged decode. Symmetric plans unchanged.

## Rerun on the fixed branch (GB10, idle window via marker handshake)
| config | layout | cosine vs dequant ref | verdict |
|---|---|---:|---|
| linear V-SF (SGLang) + no deswizzle | NHD | **0.9999986** (max_abs 0.0136) | **GREEN** |
| swizzled V-SF (vLLM) + deswizzle | NHD | 0.9582 (max_abs 3.32) | red — residual |

## Conclusions
- **K1 VO-split is proven under NVFP4 KV with linear scale factors** — the layout
  SGLang uses, and a layout vLLM *could* adopt for global layers. Combined with the bf16
  green, the D=512 route needs no kernel-math work.
- The vLLM swizzled-SF deswizzle path has one residual asymmetric-VO issue: cosine 0.958
  = localized corruption. The kernel SF sizing uses HEAD_DIM_VO correctly
  (`page_produce_kv_sf`); the suspect is the deswizzle tile transform itself
  (`prefill.cuh:527` block) interacting with a head-dim-sliced SF view. Next session:
  either fix the transform for sliced views, or sidestep by writing global-layer V-SF
  linear in vLLM (per-layer writer choice).
- Zero-copy view trick works for FP4 too once output sizing is fixed (linear config used
  contiguous halves this run; view-half revalidation rides the next window).
