# vLLM Gemma 3 NVFP4-KV Next Root-Cause Target

Date: 2026-06-09 JST

Scope: code-read sidecar, no file edits.

Current evidence says the vLLM Gemma 3 NVFP4-KV failure is below the Python-visible vLLM
page/scale plumbing:

- sampled write/read packed K/V and FP8 scale bytes match;
- short failing prompts do not require SWA eviction;
- fresh wrapper replay matches reused wrapper output;
- Python-visible plan/run signatures match offline replay;
- live `BatchPrefillWithPagedKVCacheWrapper.run(...)` returns byte-like BF16 output whose
  first values match active packed V bytes, while CPU dequant replay is sane.

Next most plausible root cause: the live FlashInfer generated/C++ path is either not the
same object/source as the offline replay path, or the FA2 paged-prefill FP4 V fragment path
is treating packed carrier bytes as numeric BF16 before output writeback.

Recommended next patch points:

1. FlashInfer C++ custom-op argument identity:
   - `third_party/flashinfer/csrc/batch_prefill.cu`
   - `BatchPrefillWithPagedKVCacheRun`
   - `PagedParams` construction
   - `ADDITIONAL_PARAMS_SETTER`

   Add env-gated one-shot host logging for data pointer, dtype, ndim, shape, and stride of
   `q`, `paged_k_cache`, `paged_v_cache`, `o`, `maybe_k_cache_sf`, and `maybe_v_cache_sf`,
   plus compiled dtype constants. Compare live server against offline replay.

2. Generated module/source stamping:
   - `third_party/flashinfer/flashinfer/jit/attention/modules.py`
   - `gen_customize_batch_prefill_module`
   - generated source/copy path
   - `third_party/flashinfer/csrc/batch_prefill_customize_config.jinja`

   Stamp generated source path, URI, compile flags, `DTypeKV`, FP4 enable state, and module
   identity from inside the compiled module. This is distinct from the already-falsified
   stale JIT URI hypothesis.

3. FP4 V fragment conversion path:
   - `third_party/flashinfer/include/flashinfer/attention/prefill.cuh`
   - `page_produce_kv`
   - `compute_sfm_v`
   - FP4 `ldmatrix` branch
   - `frag_layout_swizzle_16b_to_4b_trans`
   - `third_party/flashinfer/include/flashinfer/vec_dtypes.cuh`
   - `vec_cast<nv_bfloat16, __nv_fp4x2_e2m1>`

   Add a narrow kernel diagnostic for one block/lane: V shared-memory bytes after
   `page_produce_kv<true>`, quantized V fragment after swizzle, converted BF16 fragment, V
   scale bytes, and output fragment before writeback.

4. Output writeback sanity:
   - `third_party/flashinfer/include/flashinfer/attention/prefill.cuh`
   - `write_o_reg_gmem`

   Determine whether `o_frag` is already byte-like before writeback. If converted BF16 V is
   byte-like, patch the FP4 swizzle/cast path. If V is sane but `o_frag` or output is
   byte-like, patch MMA/writeback.

Useful isolation patch: a debug-only paged-prefill mode that forces FP4 V through a simple
dequant-to-BF16 staging path before `compute_sfm_v`. If live output becomes sane, the bug
is in the direct FP4 V fragment path rather than vLLM metadata or wrapper planning.

Follow-up patch staged in `jethac/flashinfer` branch `spark/hijinks-021-prefill-debug`
commit `1230341d`: `FLASHINFER_PREFILL_DEBUG_ONCE=1` now adds one-shot C++ host-side
identity logging in `batch_prefill.cu` plus generated JIT metadata stamps in
`batch_prefill_customize_config.jinja`. Run packet:
`tasks/vllm_gemma3_flashinfer_prefill_debug_packet_20260609.md`.
