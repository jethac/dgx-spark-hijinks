# SGLang mixed-KV capacity denominator audit, 2026-06-10

## Finding

The `~1.78x` number in the pre-fix SGLang Qwen mixed-KV runs was an observed
allocator-token ratio under the old allocator behavior. It was not the raw per-token
storage ratio for FP8-K + NVFP4-V.

The raw storage ratio for mixed-KV versus fp8 KV is approximately:

```text
fp8 K/V token units      = 8 + 8       = 16
mixed K/V token units    = 8 + 4.5     = 12.5
normalized storage gain  = 16 / 12.5   = 1.28x
```

The observed allocator-token ratio is higher because the mixed runs are also allocated
a larger total KV byte budget than the fp8 runs. In the deep-prefix sweep, mixed-KV
allocated about `1.39x` as many log-reported K+V GB as fp8. Multiplying that byte-budget
increase by the normalized per-token storage gain explains the observed token count:

```text
1.39x larger KV byte budget * 1.28x normalized storage gain ~= 1.78x tokens
```

## Evidence

Deep-prefix sweep artifacts:

- `results/sglang_qwen_mixedkv_prefixcacheguard_deep_prefix_sweep_ctx8192_20260610TmanualJST_manifest.json`
- `results/sglang_qwen_mixedkv_prefixcacheguard_deep_prefix_sweep_ctx8192_20260610TmanualJST_ctx8192_prefix*_server.log`

Server-log allocation lines:

| reused prefix | fp8 tokens | mixed tokens | token ratio | fp8 K+V GB | mixed K+V GB | KV byte budget ratio | normalized ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 3,118,761 | 5,544,194 | 1.778x | 41.64 | 57.83 | 1.389x | 1.280x |
| 6144 | 3,118,177 | 5,541,047 | 1.777x | 41.64 | 57.80 | 1.388x | 1.280x |
| 7168 | 3,118,954 | 5,545,472 | 1.778x | 41.64 | 57.84 | 1.389x | 1.280x |
| 7680 | 3,116,064 | 5,558,677 | 1.784x | 41.60 | 57.98 | 1.394x | 1.280x |

The server logs also explain the physical layout:

```text
fp8:   dtype: torch.float8_e4m3fn,       K size: ~20.8 GB, V size: ~20.8 GB
mixed: dtype: torch.float4_e2m1fn_x2,    K size: ~37.0 GB, V size: ~20.8 GB
```

In mixed mode the dtype label is the logical requested KV dtype. The fork logs the
actual mixed policy separately:

```text
SGLang FP4 KV mixed mode enabled: K cache uses FP8 e4m3, V cache uses packed NVFP4.
```

## Code Path

The current mixed-KV pool is implemented in
`third_party/sglang/python/sglang/srt/mem_cache/memory_pool.py`:

- `MHATokenToKVPoolFP4._create_buffers()` sets `mixed_fp8_k_nvfp4_v` from
  `SGLANG_FP4_KV_MIXED_KV=1`.
- In mixed mode, `k_buffer` is allocated as `torch.float8_e4m3fn`.
- In mixed mode, K scale buffers are skipped.
- V remains packed uint8 NVFP4 with V scale buffers.
- `get_kv_size_bytes()` reports the physical K and V sizes separately.

The allocator sizing path at the time of the pre-fix runs was still the generic float4
estimate in
`third_party/sglang/python/sglang/srt/model_executor/pool_configurator.py`:

- `DefaultPoolConfigurator._compute_cell_size()` sees `kv_cache_dtype` as
  `torch.float4_e2m1fn_x2`.
- For float4 KV, it estimates a full-FP4 K+V pool plus scale buffers.
- It did not know that the experimental mixed-KV pool stores K as fp8.

So the pre-fix run allocated token slots using the full-FP4 logical cell-size estimate,
then the experimental pool physically realized those slots as FP8-K + NVFP4-V. That is
why the pre-fix run had both an observed `~1.78x` allocator-token ratio and a normalized
`~1.28x` mixed-KV storage ratio.

## Claim Policy

Use these labels:

- **Current fixed allocator-token ratio:** `~1.28x` versus fp8 for FP8-K + NVFP4-V at
  equal KV byte budget.
- **Historical pre-fix allocator-token ratio:** `~1.78x` versus fp8 in the old SGLang
  Qwen launch at `--mem-fraction-static 0.40`, caused by realizing a larger physical K+V
  byte budget.
- **Not claimed:** full NVFP4 K+V `~1.78x` raw storage. That is the separate full-NVFP4
  radix track and remains red/open in SGLang.

Do not quote the pre-fix `~1.78x` mixed-KV result as a current capacity claim. It was an
allocator overcommit artifact. For cross-runtime or architecture-level claims, use the
normalized `~1.28x` mixed-KV storage gain unless the memory-budget denominator is
explicitly held equal.

## Fix And Verification

Implemented in
`third_party/sglang/python/sglang/srt/model_executor/pool_configurator.py` and recorded
in `results/sglang_mixedkv_poolconfigfix_20260610TmanualJST.md`.

Post-fix live Qwen verification at the same `--mem-fraction-static 0.40`:

| mode | tokens | K GB | V GB | total K+V GB | token ratio vs fp8 |
|---|---:|---:|---:|---:|---:|
| fp8 K+V | 3,119,614 | 20.83 | 20.83 | 41.66 | 1.000x |
| mixed FP8-K + NVFP4-V | 3,990,192 | 26.64 | 14.98 | 41.62 | 1.279x |

The fixed allocator now makes `max_total_num_tokens` and the allocation logs share one
denominator. The current mixed-KV claim is `~1.28x`, not the pre-fix `~1.78x`.
