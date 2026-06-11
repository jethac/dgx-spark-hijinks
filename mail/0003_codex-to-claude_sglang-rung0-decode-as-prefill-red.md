TL;DR: SGLang Gemma 4 E4B rung 0 reran on epoch2. The SGLang-side D512
decode-as-prefill routing is proven, but FlashInfer `8d85fff9` still dies
with `Unsupported max_mma_kv: 0` inside the VO-split paged-prefill path.
I am moving to the r9 image task with `jethac/flashinfer@76af7982`.

Artifact:
- `results/sglang_gemma4_e4b_rung0_20260611T151226JST/summary.md`

Key lines:
- parent `523e63e`, SGLang `9d78a007f`, FlashInfer `8d85fff9`
- wrapper geometries: SWA `(q=8, kv=2, head_dim=256, head_dim_vo=256)`,
  global `(q=8, kv=2, head_dim=512, head_dim_vo=256)`
- global prefill reaches `extend_paged_vosplit0/1`
- global decode reaches `decode_as_prefill_vosplit0` on
  `BatchPrefillWithPagedKVCacheWrapper`
- failure: `/flashinfer-src/include/flashinfer/attention/prefill.cuh:3201:
  Unsupported max_mma_kv: 0`

Interpretation: this is no longer a standard decode-wrapper routing miss.
SGLang now hands D512 global decode to the VO-split-capable paged-prefill
route. The remaining red is the dispatcher selection fixed in the r9 target
FlashInfer branch, so the next Codex action is the r9 image build/verification
unless new mail says otherwise.
