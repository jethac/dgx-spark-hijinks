# FlashInfer NVFP4 SWA Paged-Prefill Codegen/Compile Smoke

Run ID: `flashinfer_nvfp4_swa_prefill_codegen_compile_20260609T1712JST`

Scope: FlashInfer-only verification for FA2 paged prefill with NVFP4 KV, scale-factor tensors, and sliding-window attention. No model server was started.

Source under test:
- `jethac/flashinfer` overlay: `third_party/flashinfer`
- Base branch at start: `spark/hijinks-021-prefill-debug`
- Base commit at start: `96be2fa8`
- Fixed commit: `0919cdda1725e0ed77c792c69cfaa879689db6a8`

Patch under test:
- `flashinfer/jit/attention/modules.py`: reject NVFP4 KV paged-prefill JIT generation unless both `maybe_k_cache_sf` and `maybe_v_cache_sf` are declared.
- `flashinfer/prefill.py`: pass `key_block_scales` / `value_block_scales` through JIT additional args as `maybe_k_cache_sf` / `maybe_v_cache_sf`.
- `tests/jit/test_attention_utils.py`: add a concrete FA2 + NVFP4 KV + SWA batch-prefill generation regression asserting the generated `PagedParams` includes:
  - `maybe_k_cache_sf_stride_page/_h/_n`
  - `maybe_v_cache_sf_stride_page/_h/_n`

Verification:
- Local static check: `python -m py_compile flashinfer/jit/attention/modules.py flashinfer/jit/attention/utils.py flashinfer/prefill.py tests/jit/test_attention_utils.py` passed.
- Local pytest: blocked by missing Windows dependency `tvm_ffi`.
- Remote generation smoke in `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass` passed:
  - generated FP4 SWA paged-prefill config includes both scale pointers and all six SF stride fields.
  - missing SF tensors now fail early with a Python `ValueError`.
- Remote GB10 JIT compile smoke passed with `--gpus all`:
  - device: `NVIDIA GB10`
  - generated module: FA2 batch prefill, `dtype_kv=torch.uint8` mapped to `__nv_fp4x2_e2m1`, `head_dim=128`, `use_sliding_window=True`
  - result: `ok: compiled and loaded ffi.Module(imports_=())`

Conclusion:
The specific FlashInfer compile failure exposed by the forced Gemma FP4 prefill module is fixed at the codegen/struct level: the SWA FP4 paged-prefill module now has the scale-factor stride fields and compiles on GB10.

Next gate:
Rebuild/reload the vLLM overlay against this FlashInfer patch, clear the failed cached module, and rerun the Gemma 3 quality gate under GB10 memory guardrails. Green still requires output/logprob correctness, not just module compilation.
