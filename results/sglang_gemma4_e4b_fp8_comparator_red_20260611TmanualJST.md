# SGLang Gemma 4 E4B fp8 Comparator Red

Date: 2026-06-11 JST

Status: RED, formally documented; now root-caused by
`results/sglang_gemma4_e4b_fp8_diag2_20260612T030647JST/DIAGNOSIS.md`.

Scope: this is the fp8 KV comparator path for the SGLang Gemma 4 E4B rung. It is
separate from the full-NVFP4 K+V short-green row and from the allocator capacity
ratio. Do not cite this as a quality baseline until it produces a valid response
and a decode-side D=512 proof.

## Inputs

- `results/sglang_gemma4_e4b_rung1_fp8_20260611T182428JST/summary.md`
- `results/sglang_gemma4_e4b_rung1_fp8_retry1_20260611T183110JST/summary.md`

Both runs used:

- Model: `google/gemma-4-E4B-it`
- SGLang commit: `9d78a007f`
- FlashInfer commit: `76af798243d11c4910eaceaf1d62ba4227656d4a`
- VO split requested: `True`
- KV mode: fp8 comparator

## Evidence

Run 1:

- Request curl status: `28`
- `Unsupported max_mma_kv`: `False`
- coherent Tokyo answer: `False`
- response parse failed with an empty raw body
- D=512 decode proof: missing

Run 2:

- Request curl status: `0`
- `Unsupported max_mma_kv`: `False`
- coherent Tokyo answer: `False`
- response body: `Internal Server Error`
- D=512 decode proof: missing

Both summaries show useful prefill routing:

- SWA/local prefill plans at `head_dim=256, head_dim_vo=256`
- global prefill enters `extend_paged_vosplit0` with `head_dim=512, head_dim_vo=256`
- cached KV dtype in the sampled wrapper state is `torch.float8_e4m3fn`

The archived artifacts do not include raw server logs or traceback text, so the current
evidence is insufficient to distinguish model-quality failure, request/warmup timeout,
fp8 scale provenance, or a later decode-side wrapper problem.

## Current Interpretation

Updated 2026-06-12: the current epoch2 diagnostic reproduces the E4B fp8
failure with full logs. The server reaches readiness, allocates the fp8 hybrid
SWA KV pool, and enters the intended global D=512 VO-split paged-prefill route.
It then crashes inside FlashInfer:

```text
BatchPrefillWithPagedKVCacheDispatched ... prefill.cuh:3215:
Invalid configuration : NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=16 NUM_MMA_KV=1 NUM_WARPS_Q=4 NUM_WARPS_KV=1
```

The failing module is fp8 KV with `head_dim_qk=512`, `head_dim_vo=256`,
`num_qo_heads=8`, `num_kv_heads=2`, `page_size=1`, `split_kv=1`, and
`window_left=-1`.

The row is still not a valid fp8 quality comparator: it never returns a parseable
model response, and it never records D=512 decode-side routing because it dies in
prefill. The full-NVFP4 rung can quote allocator/capacity observations separately,
but quality comparison should remain against the working bf16/auto row until this
fp8 FlashInfer/SGLang selector path is repaired.

## Next Repro Requirements

The next fp8 comparator attempt should be treated as a diagnostic run, not a benchmark row.
It should capture:

- full server log and Python traceback;
- request timeout and SGLang internal timeout settings;
- fp8 K/V scale source and whether default scale `1.0` is being used;
- SWA and global prefill plan parameters;
- D=512 decode or decode-as-prefill plan parameters;
- binary/source proof lines for SGLang, FlashInfer, and the loaded extension artifacts.

Green criteria:

- OpenAI chat request returns a parseable coherent Tokyo answer;
- `Unsupported max_mma_kv` remains absent;
- D=512 global decode routes through the intended VO-split/decode-as-prefill path;
- PPL/logprob comparator can run against the same prompt as the full-NVFP4 row.
