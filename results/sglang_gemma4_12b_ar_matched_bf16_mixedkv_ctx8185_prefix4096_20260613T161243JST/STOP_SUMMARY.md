# SGLang Gemma 4 12B Matched bf16 vs Mixed-KV Attempt

Status: RED before quality measurement. The mixed-KV arm crashes at the first
cached-prefix paged-prefill request because FlashInfer is planned with one packed
FP4 `kv_data_type=torch.uint8` while SGLang passes mixed tensors:
K=`torch.float8_e4m3fn`, V=`torch.uint8`.

## Scope

- Runtime: SGLang on DGX Spark, packaged Ubuntu 22.04 / arm64 / torch 2.11 image
- Image:
  `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0d5e160cf83db43e1e024a8300ed2858b426b4a0f38289210dc51d8c7b6def94`
- Commit: `f18ea1ded54b841101a0ba6c9bdb7d61f79fd60e`
- Model: `google/gemma-4-12B-it`
- Row labels: `bf16 mixedkv`
- Shape: ctx `8185`, reused prefix `4096`, page size `1`, graphs disabled
- Mixed-KV mode: FP8-K + packed NVFP4-V (`SGLANG_FP4_KV_MIXED_KV=1`)
- Memory guardrail: one server at a time, Docker `--memory 100g`

This row was requested to triangulate the matched full-NVFP4 quality red
(`+0.4029692160381897` nats/token). It does not produce a mixed-KV quality
delta because the mixed-KV arm never completes a request.

## Result

The bf16 arm is green:

- bf16 chat: `Tokyo` / `Tokyo`
- bf16 supplied-token PPL: pass, `cached_tokens=4096`
- bf16 PPL: `96.7364066795068`
- bf16 mean NLL: `4.571989822602299`

The mixed-KV arm reaches server startup/readiness but fails on the first real
cached-prefix request:

- `mixedkv_chat_status.txt`: `chat_1_request_failed`, `chat_2_request_failed`
- `mixedkv_status.txt`: `ppl_request_failed`
- no `mixedkv_ppl.json`
- no `compare_bf16_vs_mixedkv.json`

The decisive server proof line is:

```text
ValueError: The dtype of k torch.float8_e4m3fn does not match the kv_data_type torch.uint8 specified in plan function.
```

The logged wrapper state immediately before the failure shows why this is a real
split-dtype planning issue:

```text
_cached_kv_data_type: torch.uint8
kv_cache=(
  K: shape (449368, 8, 256), dtype torch.float8_e4m3fn,
  V: shape (449368, 8, 128), dtype torch.uint8
)
k_sf=None
v_sf: shape (449368, 1, 8, 16), dtype torch.float8_e4m3fn
```

## Interpretation

This is not a quality result for mixed-KV. It is an ABI/planning red for SGLang
Gemma 4 mixed-KV cached-prefix paged prefill on the packaged image.

The capture site cannot legitimately collapse to one `kv_data_type`: the actual
cache tensors are split dtype by construction (`K=fp8`, `V=packed nvfp4`), and
the paged-prefill wrapper enforces the planned dtype against the K tensor.
Claim-grade Gemma 4 mixed-KV therefore needs FlashInfer/SGLang planning that
passes and keys separate K/V data types for this path, or an equivalent wrapper
route whose runtime tensors match the declared module dtype.

## Host State

At stop point:

- marker: absent
- Docker: no matching containers
- memory: ~115 GiB available

## Key Artifacts

- `google-gemma-4-12b-it/bf16_ppl.json`
- `google-gemma-4-12b-it/bf16_server.log`
- `google-gemma-4-12b-it/mixedkv_server.log`
- `google-gemma-4-12b-it/mixedkv_chat_status.txt`
- `google-gemma-4-12b-it/mixedkv_status.txt`
- `manifest.json`
- `../sglang_gemma4_12b_ar_matched_bf16_mixedkv_ctx8185_prefix4096_20260613T161243JST_runner.log`
