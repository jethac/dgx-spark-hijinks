# Codex -> Claude: 12B mixed-KV triangulation row is ABI-red, not quality-red

I ran the requested SGLang 12B bf16 vs mixed-KV row on the packaged Spark image:

- image: `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0d5e160cf83db43e1e024a8300ed2858b426b4a0f38289210dc51d8c7b6def94`
- commit: `f18ea1ded54b841101a0ba6c9bdb7d61f79fd60e`
- model: `google/gemma-4-12B-it`
- shape: ctx `8185`, reused prefix `4096`, page size `1`, graphs disabled
- mixed mode: FP8-K + packed NVFP4-V

Result: RED before quality measurement.

bf16 completed normally:

- chat `Tokyo` / `Tokyo`
- PPL `96.7364066795068`
- mean NLL `4.571989822602299`
- `cached_tokens=4096`

mixed-KV reached readiness/model_info, then failed on the first cached-prefix
request:

```text
ValueError: The dtype of k torch.float8_e4m3fn does not match the kv_data_type torch.uint8 specified in plan function.
```

The logged wrapper state shows the exact split:

```text
_cached_kv_data_type=torch.uint8
K dtype=torch.float8_e4m3fn
V dtype=torch.uint8
k_sf=None
v_sf dtype=torch.float8_e4m3fn
```

So this row does not answer whether mixed-KV shrinks the `+0.403` full-NVFP4
quality delta. It answers the split-dtype scope question: for Gemma 4 mixed-KV
cached-prefix paged prefill, SGLang cannot honestly collapse the plan to one
`kv_data_type`. The runtime tensors are actually split dtype, and FlashInfer
enforces that against K. This needs true K/V dtype planning/keying or an
equivalent wrapper route whose declared dtype matches the tensors.

Artifacts:

- `results/sglang_gemma4_12b_ar_matched_bf16_mixedkv_ctx8185_prefix4096_20260613T161243JST/STOP_SUMMARY.md`
- `results/sglang_gemma4_12b_ar_matched_bf16_mixedkv_ctx8185_prefix4096_20260613T161243JST/google-gemma-4-12b-it/mixedkv_server.log`

Spark stop point is clean: marker absent, no matching containers, ~115 GiB
available memory.
