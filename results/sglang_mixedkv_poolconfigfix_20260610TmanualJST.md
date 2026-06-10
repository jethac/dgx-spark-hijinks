# SGLang mixed-KV pool-configurator cell-size fix, 2026-06-10

## Summary

The SGLang mixed FP8-K + NVFP4-V pool now uses a matching allocator cell-size
estimate. Before this fix, `DefaultPoolConfigurator._compute_cell_size()` sized the pool
as if the logical `fp4_e2m1` dtype meant full FP4 K+V, while the experimental pool
physically allocated FP8 K plus NVFP4 V. That made `--mem-fraction-static 0.40` realize
about `57.8 GB` of K+V cache instead of the fp8 row's `~41.6 GB`.

After the fix, mixed-KV allocates the expected normalized `~1.28x` token capacity at
approximately the same physical K+V byte budget as fp8.

## Code Change

File:

```text
third_party/sglang/python/sglang/srt/model_executor/pool_configurator.py
```

Added a mixed-KV cell-size helper:

```text
K data:      head_num * head_dim                  * layers * 1 byte  (fp8)
V data:      head_num * (v_head_dim / 2)          * layers * 1 byte  (packed NVFP4)
V SF:        head_num * (v_head_dim / 16)         * layers * 1 byte  (FP8 scale)
```

The helper is used when:

```text
kv_cache_dtype is fp4_e2m1fn_x2
SGLANG_FP4_KV_MIXED_KV=1
```

Coverage:

- `DefaultPoolConfigurator`: ordinary MHA models such as Qwen.
- `HybridSWAPoolConfigurator`: hybrid SWA models such as the upcoming SGLang Gemma rung.

The physical pool implementation in
`third_party/sglang/python/sglang/srt/mem_cache/memory_pool.py` already used this layout:
`k_buffer` as `torch.float8_e4m3fn`, no K scale buffer, and packed NVFP4 V plus V scale
buffer. This patch only makes sizing agree with allocation.

## Verification

Syntax check:

```text
python -m py_compile third_party/sglang/python/sglang/srt/model_executor/pool_configurator.py
```

Live GB10 verification, same image/model/page/memory fraction, sequential containers,
100g cgroup cap:

```text
RUNTIME_IMAGE=sglang-source-stack-c3dae30f-e631a13fd
MODEL=Qwen/Qwen2.5-1.5B-Instruct
PAGE_SIZE=1
MEM_FRACTION_STATIC=0.40
BENCHMARK_CASES=short_decode
DISABLE_GRAPHS=1
GB10_DOCKER_MEMORY=100g
GB10_DOCKER_MEMORY_SWAP=100g
```

Artifacts:

- `results/sglang_qwen_fp8_poolconfigfix_verify_20260610TmanualJST_row_manifest.json`
- `results/sglang_qwen_mixedkv_poolconfigfix_verify_20260610TmanualJST_row_manifest.json`
- `results/sglang_qwen_fp8_poolconfigfix_verify_20260610TmanualJST_server.log`
- `results/sglang_qwen_mixedkv_poolconfigfix_verify_20260610TmanualJST_server.log`
- `results/sglang_qwen_fp8_poolconfigfix_verify_20260610TmanualJST_openai_benchmark.json`
- `results/sglang_qwen_mixedkv_poolconfigfix_verify_20260610TmanualJST_openai_benchmark.json`

| mode | tokens | K GB | V GB | total K+V GB | token ratio vs fp8 | short decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| fp8 K+V | 3,119,614 | 20.83 | 20.83 | 41.66 | 1.000x | 57.438 |
| mixed FP8-K + NVFP4-V | 3,990,192 | 26.64 | 14.98 | 41.62 | 1.279x | 58.308 |

The live result matches the byte math:

```text
fp8 units   = 8 + 8     = 16
mixed units = 8 + 4.5   = 12.5
16 / 12.5   = 1.28x
```

## Impact

This retires the unsafe pre-fix mixed-KV allocator behavior. The mixed-KV claim should
now use:

- `~1.28x` allocator-token ratio at equal physical K+V byte budget;
- decode parity on the short Qwen row;
- no `~1.78x` mixed-KV claim unless explicitly referring to the old pre-fix overshoot
  artifacts.

The fix also protects the next SGLang Gemma 3 hybrid-SWA rung from silently overcommitting
the GB10 unified-memory pool.
