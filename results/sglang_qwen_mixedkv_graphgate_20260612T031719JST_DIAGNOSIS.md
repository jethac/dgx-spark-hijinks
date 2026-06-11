# SGLang Qwen mixed-KV CUDA graph gate - RED

Date: 2026-06-12 JST

## Scope

- Runtime: SGLang source stack `sglang-source-stack-dgemma-024-0705924c-f99323bd:latest`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Context: 8192 tokens
- Reused prefix: 4096 tokens
- Corpus md5: `abb63f0e65247a25f870d3f2d57563ff`
- FlashInfer source overlay: enabled through `SPARK_FLASHINFER_SOURCE_ROOT=/work/third_party/flashinfer`
- Mixed-KV mode: FP8-K + NVFP4-V

This run is the CUDA graph re-enable gate after the mixed-KV claim row. It is not a new capacity claim.

## Results

| Row | Result | Mean NLL | PPL | Notes |
|---|---:|---:|---:|---|
| fp8, graphs disabled | GREEN | `1.6440996395327547` | `5.17634722980795` | Control PPL passes. |
| fp8, graphs enabled | GREEN | `1.6440996395327547` | `5.17634722980795` | Bitwise-identical to fp8 eager. |
| mixed-KV, graphs enabled | RED | n/a | n/a | Server fails during CUDA graph capture before PPL. |
| mixed-KV, graphs disabled | RED/inconclusive | n/a | n/a | Runner reported not-ready, but the durable log only captured a missing-container error; not used as the root-cause proof. |

The auto manifest reports `mean_nll_bitwise_equal=true`, but that flag only compares the two fp8 control files in this failed run. The mixed-KV PPL files are absent, so the flag is not a valid mixed-KV graph-equivalence result.

## Root cause observed

The mixed-KV graph server reaches mixed-KV allocation and enters CUDA graph capture:

```text
KV Cache is allocated. dtype: torch.float4_e2m1fn_x2, #tokens: 4001280, K size: 26.71 GB, V size: 15.03 GB
SGLang FP4 KV mixed mode enabled: K cache uses FP8 e4m3, V cache uses packed NVFP4.
Capture cuda graph begin.
```

It then crashes in the decode wrapper:

```text
File "/tmp/flashinfer-python-path/flashinfer/decode.py", line 1498, in run
    _unpack_paged_kv_cache(kv_cache_sf, self._kv_layout)
File "/tmp/flashinfer-python-path/flashinfer/utils.py", line 176, in _unpack_paged_kv_cache
    _expand_4d(paged_k_cache, kv_layout),
File "/tmp/flashinfer-python-path/flashinfer/utils.py", line 90, in _expand_4d
    if x.ndim not in [3, 4]:
AttributeError: 'NoneType' object has no attribute 'ndim'
```

Verdict: mixed-KV graph capture is blocked by a FlashInfer/SGLang decode ABI integration mismatch. The source-overlay FlashInfer decode wrapper expects a non-null `kv_cache_sf`, while SGLang's mixed-KV graph-capture decode path passes `None`.

This is not evidence of CUDA graph replay non-determinism. The gate fails before capture completes.

## Artifacts

- Manifest: `results/sglang_qwen_mixedkv_graphgate_20260612T031719JST_graph_gate_manifest.json`
- fp8 eager PPL: `results/sglang_qwen_mixedkv_graphgate_20260612T031719JST_eager_fp8_ppl.json`
- fp8 graph PPL: `results/sglang_qwen_mixedkv_graphgate_20260612T031719JST_graph_fp8_ppl.json`
- Mixed graph server log: `results/sglang_qwen_mixedkv_graphgate_20260612T031719JST_graph_mixed_server.log`
- Mixed graph install/provenance: `results/sglang_qwen_mixedkv_graphgate_20260612T031719JST_graph_mixed_install.log`
- Narrow eager repro attempt: `results/sglang_qwen_mixedkv_sourceoverlay_eager_red_20260612T033042JST/`

## Spark stop state

After capture and teardown:

```text
marker: absent
docker ps: empty
```
