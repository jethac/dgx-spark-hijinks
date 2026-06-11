TL;DR: eager mixed-KV is a true split-dtype FlashInfer paged-plan from Python:
K is stored/read as fp8, V is stored/read as packed NVFP4. There is no two-pass merge
and no pool-side conversion on read.

Code pointers:

- Pool write/storage:
  `third_party/sglang/python/sglang/srt/mem_cache/memory_pool.py`
  `MHATokenToKVPoolFP4._create_buffers()` sets `mixed_fp8_k_nvfp4_v` from
  `SGLANG_FP4_KV_MIXED_KV=1`, allocates `k_buffer` as `torch.float8_e4m3fn`, allocates
  `v_buffer` as packed `uint8`, omits `k_scale_buffer`, and keeps `v_scale_buffer`.
  `set_kv_buffer()` writes `cache_k.to(torch.float8_e4m3fn)` directly and quantizes only V
  through `NVFP4KVQuantizeUtil.quantize(...)`.
- Scale convention:
  `get_kv_scale_buffer()` returns `(None, v_sf)` in mixed mode; `get_kv_global_scale()`
  returns `(1.0, v_global)`.
- Wrapper detection:
  `third_party/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py`
  `_is_fp8_k_nvfp4_v_pool()` detects the pool mode and `FlashInferAttnBackend.__init__()`
  sets `self.is_fp8_k_nvfp4_v`.
- Plan call:
  `FlashInferIndicesUpdaterDecode.__init__()` and `FlashInferIndicesUpdaterPrefill.__init__()`
  set `kv_data_type=torch.uint8`, `k_data_type=torch.float8_e4m3fn`, `v_data_type=torch.uint8`
  when mixed mode is active. The paged begin-forward call then passes both
  `k_data_type=` and `v_data_type=` if they differ. The same pattern is used by the
  D=512 decode-as-prefill VO-split path.

Scope consequence: graph capture should describe the same split-dtype plan. If the CUDA-graph
capture wrapper can pass `k_data_type/v_data_type`, it is a Python capture-site fix. If that
wrapper path can only key a single `DTypeKV`, the blocker is real FlashInfer split-dtype
module keying / template support. Collapsing to a single dtype would be wrong for the
working eager mixed-KV path unless we intentionally abandon fp8-K and make it full NVFP4 or
full fp8.
