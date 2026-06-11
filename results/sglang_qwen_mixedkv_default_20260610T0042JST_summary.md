# SGLang Qwen mixed-KV default radix row, 2026-06-10 00:42 JST

## Scope

This row tests the SGLang Qwen default radix-cache failure after switching the FP4 KV
pool to mixed storage:

- K cache: FP8 e4m3
- V cache: packed NVFP4 with FP8 scale factors
- model: `Qwen/Qwen2.5-1.5B-Instruct`
- runtime image: `sglang-source-stack-c3dae30f-e631a13fd`
- source overlay: mounted `third_party/sglang` and `third_party/flashinfer`
- launch: `--attention-backend flashinfer --kv-cache-dtype fp4_e2m1 --page-size 1 --mem-fraction-static 0.40`
- graph mode: CUDA graph and piecewise graph disabled
- env: `SGLANG_FP4_KV_MIXED_KV=1`

This is a default radix first-token quality gate. It is not a long-form quality run,
throughput run, graph-safety run, or Gemma row.

## Artifacts

- request JSON: `results/sglang_qwen_mixedkv_default_20260610T0042JST_default.json`
- server log: `results/sglang_qwen_mixedkv_default_20260610T0042JST_default_server.log`
- dense/cache compare: `results/sglang_qwen_mixedkv_default_20260610T0042JST_default_dense_cache_compare.json`
- summary JSON: `results/sglang_qwen_mixedkv_default_20260610T0042JST_summary.json`
- summary audit: `results/sglang_qwen_mixedkv_default_20260610T0042JST_dense_cache_trace_summary_audit.json`

## Result

Green for the targeted default-radix first-token gate.

The old failure was that the second request reused a 55-token FP4 cached prefix and
changed the first token from `**` to `ark`. With mixed KV, both cached-prefix second
requests now emit `**`:

| row | OpenAI cached | OpenAI token | OpenAI logprob | native cached | native token | native logprob |
|---|---:|---|---:|---:|---|---:|
| baseline_openai_then_native | 0 | `**` | -0.7235294580 | 55 | `**` | -0.7601577044 |
| reverse_native_then_openai | 55 | `**` | -0.7601577044 | 0 | `**` | -0.7235294580 |
| flush_between_openai_native | 0 | `**` | -0.7235294580 | 0 | `**` | -0.7235294580 |
| namespace_isolation_extra_key | 0 | `**` | -0.7235294580 | 0 | `**` | -0.7235294580 |

The dense/cache summary audit passes:

```json
{
  "ok": true,
  "case_count": 1,
  "findings": []
}
```

## Capacity evidence

The server log records:

```text
KV Cache is allocated. dtype: torch.float4_e2m1fn_x2, #tokens: 5573469, K size: 37.21 GB, V size: 20.93 GB
SGLang FP4 KV mixed mode enabled: K cache uses FP8 e4m3, V cache uses packed NVFP4.
max_total_num_tokens=5573469
```

This is close to the earlier full-FP4 row in the same SGLang lane (`5,517,572` to
`5,577,596` tokens depending on artifact) and above the older matched fp8 row
(`3,105,240` tokens). Treat this as strong capacity evidence for the mixed implementation,
but rerun a fresh sequential fp8 comparator before publishing a final ratio.

## Remaining caveats

The run still records a tensor-level difference between dense full-prefill and cached-prefix
merge:

```json
{
  "kind": "attention",
  "label": "attention_output",
  "layer": 0,
  "field": "attention_output_rows",
  "vector": {
    "cosine": 0.4661444810372346,
    "max_abs": 0.2578125,
    "rms": 0.11784679304779001
  }
}
```

Unlike the full-FP4 K+V path, this residual difference no longer flips the first token on
the default radix request-order probe. Do not overstate this as exact tensor equality.

Next gates:

1. Fresh sequential fp8 comparator for a current mixed-vs-fp8 capacity ratio.
2. Longer coherent generation / benchmark quality gate with radix cache ON.
3. Throughput row.
4. Graph-safety row, if SGLang FP4 KV graph capture is still desired.
