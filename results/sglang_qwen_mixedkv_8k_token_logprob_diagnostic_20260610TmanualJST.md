# SGLang Qwen mixed-KV 8k token-logprob diagnostic, 2026-06-10

## Scope

This diagnostic localizes the material 8k supplied-token PPL gap observed in
`results/sglang_qwen_mixedkv_reuse_ppl_20260610TmanualJST.md`.

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`
- Attention backend: FlashInfer
- KV modes compared: fp8 K/V vs mixed FP8-K + NVFP4-V
- Context: `8192`
- Reused prefix: `4096`
- Scored continuation tokens: `4095`
- Graph policy: CUDA graphs enabled; mixed cached-prefix prefill guarded to eager
- Detail mode: `INCLUDE_TOKEN_LOGPROBS=1`

## Artifacts

- fp8 report: `results/sglang_qwen_mixedkv_reuse_ppl_ctx8192_detail_20260610TmanualJST_fp8_ppl.json`
- mixed report: `results/sglang_qwen_mixedkv_reuse_ppl_ctx8192_detail_20260610TmanualJST_mixed_ppl.json`
- comparison: `results/sglang_qwen_mixedkv_reuse_ppl_ctx8192_detail_20260610TmanualJST_compare.json`
- token comparison: `results/sglang_qwen_mixedkv_reuse_ppl_ctx8192_detail_20260610TmanualJST_token_compare.json`
- fp8 server log: `results/sglang_qwen_mixedkv_reuse_ppl_ctx8192_detail_20260610TmanualJST_fp8_server.log`
- mixed server log: `results/sglang_qwen_mixedkv_reuse_ppl_ctx8192_detail_20260610TmanualJST_mixed_server.log`
- comparator script: `scripts/compare_sglang_token_logprobs.py`
- no-reuse control manifest: `results/sglang_qwen_mixedkv_noreuse_ppl_ctx8192_20260610TmanualJST_manifest.json`
- no-reuse control compare: `results/sglang_qwen_mixedkv_noreuse_ppl_ctx8192_20260610TmanualJST_compare.json`
- fixed-8k reuse-prefix sweep: `results/sglang_qwen_mixedkv_reuse_prefix_sweep_ctx8192_20260610TmanualJST.md`
- fixed-8k sweep manifest: `results/sglang_qwen_mixedkv_reuse_prefix_sweep_ctx8192_20260610TmanualJST_manifest.json`

## Result

The detailed rerun reproduces the aggregate 8k loss:

| ctx | reused prefix | scored tokens | PPL fp8 | PPL mixed | delta PPL | delta nats/token |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 4096 | 4095 | 7.238053 | 8.052973 | 0.814921 | 0.106689 |

Both fp8 and mixed-KV report `cached_tokens=4096`, `num_missing_tokens=0`,
`num_mismatched_tokens=0`, and one skipped logprob-span boundary placeholder.

Runtime evidence:

```text
fp8:   Prefill batch, #new-token: 4096, #cached-token: 4096, cuda graph: False
mixed: Prefill batch, #new-token: 4096, #cached-token: 4096, cuda graph: False
```

The mixed row used the intended practical path:

```text
SGLang FP4 KV mixed mode enabled: K cache uses FP8 e4m3, V cache uses packed NVFP4.
```

## No-Reuse Control

The same 8192-token supplied-token PPL comparison was rerun with `reuse_prefix_len=0`.
This keeps the same model, runtime image, mixed-KV allocation, graph policy, page size,
and full 8192-token scoring window, but it forces both fp8 and mixed-KV to score the prompt
without a cached-prefix hit.

| ctx | reused prefix | scored tokens | PPL fp8 | PPL mixed | delta PPL | delta nats/token |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 0 | 8191 | 7.195014 | 7.195014 | 0.000000 | 0.000000 |

Runtime evidence:

```text
fp8:   Prefill batch, #new-token: 8192, #cached-token: 0, cuda graph: False
mixed: Prefill batch, #new-token: 8192, #cached-token: 0, cuda graph: False
```

Capacity in the no-reuse launch remained on the intended mixed-KV path:

| KV mode | allocatable tokens | K size | V size |
|---|---:|---:|---:|
| fp8 K + fp8 V | 3,116,223 | 20.80 GB | 20.80 GB |
| fp8 K + NVFP4 V | 5,542,470 | 37.00 GB | 20.81 GB |

Observed allocator-token ratio: `1.779x`.

Capacity in this launch:

| KV mode | allocatable tokens | K size | V size |
|---|---:|---:|---:|
| fp8 K + fp8 V | 3,115,982 | 20.80 GB | 20.80 GB |
| fp8 K + NVFP4 V | 5,546,772 | 37.03 GB | 20.83 GB |

Observed allocator-token ratio: `1.780x`.

## Token-Level Shape

Delta convention: `delta_nll = fp8_logprob - mixed_logprob`, so positive means mixed-KV
made the supplied token less likely than fp8.

| metric | value |
|---|---:|
| tokens compared | 4095 |
| mean delta nats/token | 0.106689 |
| total delta nats | 436.892 |
| positive-delta tokens | 2272 |
| negative-delta tokens | 1823 |
| positive delta sum | 861.844 |
| negative delta sum | -424.951 |
| median delta | 0.002143 |
| p75 delta | 0.189310 |
| p95 delta | 1.108504 |
| p99 delta | 2.302306 |
| max delta | 8.759079 |
| min delta | -4.996746 |

Interpretation: the loss is not one catastrophic token. Mixed-KV improves many tokens, but
the positive tail is larger than the negative tail. The median is near zero; the net PPL
cost comes from the upper tail and from a strong early-continuation concentration.

Worst 256-token windows:

| positions | mean delta nats/token | sum delta nats |
|---|---:|---:|
| 4097-4352 | 0.385912 | 98.793478 |
| 4865-5120 | 0.244509 | 62.594413 |
| 4609-4864 | 0.165781 | 42.439937 |
| 4353-4608 | 0.157824 | 40.402956 |
| 5889-6144 | 0.126424 | 32.364622 |

The first 1024 scored tokens after the 4096-token cached prefix account for about
`244.231` of `436.892` total delta nats, roughly `55.9%` of the total loss. That points at
cached-prefix transition / early continuation behavior as the first thing to inspect, not
a uniform per-token penalty.

Best 256-token window:

| positions | mean delta nats/token | sum delta nats |
|---|---:|---:|
| 6913-7168 | -0.000964 | -0.246682 |

## Conclusion

The practical mixed-KV path remains capacity-positive and mechanically correct at 8k cache
reuse, but its cached-prefix quality claim is not clean. The 8k no-reuse control is exactly
equal between fp8 and mixed-KV, so this is not a broad long-prefill mixed-KV penalty. The
loss is introduced by cached-prefix reuse and is concentrated in the early continuation
span after the reused prefix.

The fixed-8k reuse-prefix sweep initially sharpened that conclusion: with the same total
context, the graph-enabled mixed-KV penalty rose with reused-prefix length: `0.000000`,
`0.003079`, `0.026555`, `0.106689`, and `0.114980` nats/token for prefixes `0`, `1024`,
`2048`, `4096`, and `6144`. That interpretation is now superseded by
`results/sglang_qwen_mixedkv_prefix4096_graph_vs_eager_20260610TmanualJST.md`: rerunning
the 4096-prefix row with all CUDA graphs disabled on the same corpus path changes the
result to fp8 PPL `5.176347` versus mixed PPL `5.169650`
(`delta_nats_per_token=-0.001295`). The scored cached-prefix request was eager in both
runs; the graph-enabled sweep differed because the cache-populating warmup prefill used
`cuda graph: True`.

Updated conclusion: the 8k graph-enabled penalty is not a clean NVFP4-V quality result.
The current SGLang repair target is graph-written prefix cache state. A 32k row should wait
until cache-populating graph capture is fixed or disabled in the benchmark policy.
