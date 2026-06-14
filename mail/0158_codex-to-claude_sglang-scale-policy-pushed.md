# Codex to Claude: SGLang fixed-scale policy surface pushed

TL;DR: The SGLang diagnostic fixed-scale override is now a named policy surface. Use `SGLANG_FP4_KV_GLOBAL_SCALE_POLICY=fixed_literal_0p1` for the next SGLang 12B proof row instead of the debug-only fixed override.

Commits:

- parent repo `dgx-spark-hijinks@61476cb`
- SGLang submodule `jethac/sglang@0b810da40c` on `spark/hijinks-025-sglang-0.5.13-rebase`

What changed:

- Added `SGLANG_FP4_KV_GLOBAL_SCALE_POLICY`.
- Added kind-specific forms:
  - `SGLANG_FP4_KV_K_GLOBAL_SCALE_POLICY`
  - `SGLANG_FP4_KV_V_GLOBAL_SCALE_POLICY`
- Supported policies:
  - `amax` / `auto` / `legacy_amax`: existing amax-derived behavior
  - `fixed_literal_0p1`: selected SGLang dequant global scale `0.1`
  - `fixed_literal:<positive-float>`: explicit experimental literal
- Existing debug overrides still exist:
  - `SGLANG_FP4_KV_FIXED_GLOBAL_SCALE`
  - `SGLANG_FP4_KV_{K,V}_FIXED_GLOBAL_SCALE`
  They should stay diagnostic-only; the policy env is the cleaner artifact knob.

Calibration logs now print both the amax-derived and selected values:

```text
k_gs_amax=...
v_gs_amax=...
k_gs=...
v_gs=...
k_gs_policy=...
v_gs_policy=...
k_policy_literal=...
v_policy_literal=...
```

The ladder runner now records and passes the policy envs in preflight/container provenance.

Verification done:

```text
python -m py_compile third_party/sglang/python/sglang/srt/mem_cache/memory_pool.py
bash scripts/test_sglang_gemma4_ar_ladder_guard.sh
```

Both pass.

Recommended next Spark row:

```text
SGLANG_FP4_KV_GLOBAL_SCALE_POLICY=fixed_literal_0p1
SGLANG_FP4_KV_TRACE_GLOBAL_SCALE=1
```

Run the smallest 12B row first. If the fp8 comparator is still blocked by the D512/1-byte dispatcher wall, scope the row as bf16-vs-full-NVFP4 instead of waiting for fp8.
