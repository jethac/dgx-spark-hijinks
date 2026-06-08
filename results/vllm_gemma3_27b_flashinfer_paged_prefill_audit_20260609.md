# vLLM Gemma 3 27B FlashInfer Paged-Prefill Audit, 2026-06-09

## Purpose

Record the code-map audit after the active-page replay localized the Gemma 3
NVFP4-KV failure below vLLM page/scale pairing.

This is diagnostic-only. It does not claim a fix or performance result.

## Evidence Before The Code Read

The current failing boundary is the real FlashInfer FA2 paged-prefill wrapper:

- `results/vllm_gemma3_27b_wrapper_trace_20260609T0148JST_summary.md` shows sane signed
  BF16 `query` / `out_before` entering the wrapper and byte-like BF16 `out_after` leaving
  it.
- `results/vllm_gemma3_27b_active_page_dump_20260609T0216JST_summary.md` dumps the exact
  active pages for layer 5 and shows the first 16 `out_after` BF16 values exactly match the
  first 16 active packed V bytes.
- `results/vllm_gemma3_27b_active_page_replay_20260609T0216JST_summary.md` dequantizes
  those exact active pages and computes a CPU causal GQA reference. The reference is sane
  signed output (mean near `-0.03`, RMS around `1.9..2.0`), while real `out_after` remains
  byte-range (mean `128..129`, RMS `147..148`) with near-zero cosine versus the reference.

This weakens a simple "stale FP8 scale page" explanation for the vLLM Gemma 3 row. The
cross-lane FP4-KV reuse framing remains useful, but vLLM now points at the FlashInfer
paged-prefill NVFP4 specialization itself: it appears to return the packed V carrier as if
it were the output, or to bind the V element conversion path incorrectly.

## Code Map

Python/JIT wrapper:

- `third_party/flashinfer/flashinfer/prefill.py`
  - `BatchPrefillWithPagedKVCacheWrapper.run()` unpacks tuple `paged_kv_cache` into
    `k_cache, v_cache` and tuple `kv_cache_sf` into `maybe_k_cache_sf, maybe_v_cache_sf`.
  - For NVFP4 KV, `kv_cache_sf` is required and `out_head_dim` comes from `q.shape[-1]`.
- `third_party/flashinfer/flashinfer/jit/utils.py`
  - `dtype_map_kv[torch.uint8] = "__nv_fp4x2_e2m1"`, so vLLM's packed `uint8` carrier is
    supposed to enter the generated kernel as the FP4 container type.
- `third_party/flashinfer/flashinfer/jit/attention/modules.py`
  - `get_batch_prefill_uri()` names the generated module using
    `filename_safe_dtype_map[dtype_kv]`. For `torch.uint8`, the URI still contains
    `dtype_kv_u8` even when the generated C++ body renders `DTypeKV=__nv_fp4x2_e2m1`.
    This is a plausible stale AOT/JIT cache collision surface.
  - The generated FA2 prefill module includes additional tensor arguments
    `maybe_k_cache_sf` and `maybe_v_cache_sf`.
- `third_party/flashinfer/flashinfer/jit/core.py`
  - `JitSpec.build_and_load()` can load from an existing built library path for a given
    spec name. If the name only says `u8`, an older raw-`uint8_t` module can plausibly be
    reused for a newer FP4-container build.
- `third_party/flashinfer/flashinfer/jit/attention/utils.py`
  - Scale strides are explicit for `maybe_k_cache_sf` and `maybe_v_cache_sf`; they are not
    derived from packed data strides.

C++/CUDA wrapper:

- `third_party/flashinfer/csrc/batch_prefill.cu`
  - `BatchPrefillWithPagedKVCacheRun()` builds `paged_kv_t<DTypeKV, IdType>` from separate
    K and V data pointers and separate K and V strides, then calls
    `BatchPrefillWithPagedKVCacheDispatched`.
  - For this Gemma 3 layer, `HEAD_DIM_QK == HEAD_DIM_VO == 128`, so a QK/VO dimension
    mismatch is not the current leading theory.
- `third_party/flashinfer/include/flashinfer/page.cuh`
  - `paged_kv_t` carries separate K and V stride fields, and V loads should use
    `protective_get_v_offset()`.

Kernel sites most relevant to the byte-like output:

- `third_party/flashinfer/include/flashinfer/attention/prefill.cuh`
  - `is_fp4_type<__nv_fp4x2_e2m1>` marks packed NVFP4 as a special one-byte type.
  - The paged path calls `page_produce_kv<true>()` for V data and
    `page_produce_kv_sf<true>()` for V scales.
  - `page_produce_kv_sf<true>()` has the `FLASHINFER_PAGED_V_SF_DESWIZZLE` path used by
    vLLM's interleaved V scale-factor layout.
  - `compute_sfm_v()` is supposed to:
    - load packed V fragments from shared memory;
    - run `frag_layout_swizzle_16b_to_4b_trans()`;
    - call `vec_cast<DTypeQ, DTypeKV>::cast<8>()` to convert FP4 to BF16/FP16;
    - multiply by V FP8 scale factors;
    - feed the converted V fragment into `mma_sync_m16n16k16_row_col_f16f16f32`.
- `third_party/flashinfer/include/flashinfer/vec_dtypes.cuh`
  - `vec_cast<nv_bfloat16, __nv_fp4x2_e2m1>` converts FP4 through `cvt.rn.f16x2.e2m1x2`
    on CUDA 13.0 because direct `cvt.rn.bf16x2.e2m1x2` is gated to CUDA 13.2+.
  - This conversion site is a prime suspect because the live vLLM Gemma row uses BF16 Q/O
    and CUDA 13.0.

## Ranked Hypotheses

1. **JIT/AOT cache-key collision for `torch.uint8` KV.** The generated kernel type map can
   render `torch.uint8` as `__nv_fp4x2_e2m1`, but the batch-prefill URI still advertises
   `dtype_kv_u8`. A stale module built with raw `uint8_t` semantics would skip
   `is_fp4_type_v<DTypeKV>` branches and could return packed V bytes as byte-like BF16
   output. This exactly matches the live symptom and is cheaper to test than kernel surgery.
2. **Paged-prefill V conversion bug for BF16 + FP4.** The raw output matching packed active
   V bytes suggests `compute_sfm_v()` may not be converting/scaling the packed
   `__nv_fp4x2_e2m1` V fragment before the output path consumes it, at least in the real
   paged-prefill variant used by vLLM.
3. **Paged-prefill split/variant mismatch not covered by the standalone probe.** The
   standalone Gemma-shaped FlashInfer probe passed decode and prefill, but the real wrapper
   call may select a different JIT URI, split-KV plan, CTA shape, or paged specialization.
4. **Scale deswizzle is not the primary reproducer.** CPU replay with and without V-scale
   deswizzle keeps the reference signed and small while the wrapper output remains byte-like.
   The V-scale path still needs auditing, but it does not explain the exact carrier-byte
   output by itself.
5. **vLLM page/scale pairing is no longer the leading theory.** vLLM traces show sampled
   read/write data and scale bytes match, active pages were dumped from the failing wrapper,
   and replaying those pages gives a sane reference.

## Next Smallest Experiments

1. Force a fresh JIT namespace for NVFP4 paged prefill by making the URI encode the logical
   C++ KV type (`fp4x2_e2m1`) or by adding an NVFP4 paged-prefill version suffix. Clear the
   FlashInfer JIT cache and rerun the exact Gemma 3 first-token wrapper-boundary probe.
2. Add a generated guard for NVFP4 paged prefill: when `maybe_k_cache_sf` /
   `maybe_v_cache_sf` are present and `dtype_kv` is `torch.uint8`, emit a compile-time or
   runtime assertion that the C++ `DTypeKV` is `__nv_fp4x2_e2m1` / `is_fp4_type_v<DTypeKV>`.
3. Add an env-gated FlashInfer paged-prefill debug print or dump that records the selected
   JIT URI plus compile-time values for `DTypeQ`, `DTypeKV`, `DTypeO`, `HEAD_DIM_QK`,
   `HEAD_DIM_VO`, `USE_KV_REPACK`, `is_fp4_type_v<DTypeKV>`, and
   `FLASHINFER_PAGED_V_SF_DESWIZZLE`.
4. Build a minimal real-paged-prefill reproducer from the dumped payloads on GB10:
   - run `BatchPrefillWithPagedKVCacheWrapper` with the original packed FP4 K/V and scales;
   - run the same metadata with materialized BF16 K/V pages from the replay script;
   - compare both against the CPU reference.
5. If materialized BF16 K/V passes and packed FP4 fails, instrument or patch
   `compute_sfm_v()` / `vec_cast<nv_bfloat16, __nv_fp4x2_e2m1>` first.
6. Only after the real paged-prefill wrapper produces signed output should the Gemma ladder
   climb to Gemma 4 31B text-only.
