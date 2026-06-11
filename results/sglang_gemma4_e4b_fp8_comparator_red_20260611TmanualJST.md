# SGLang Gemma 4 E4B fp8 Comparator Red

Date: 2026-06-11 JST

Status: RED, formally documented; not root-caused.

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

The row is not failing at the old FlashInfer D=512 dispatcher wall: both summaries record
`Unsupported max_mma_kv: False`, and global D=512 prefill reaches the VO-split route.

The row is also not a valid fp8 quality comparator: it never returns a parseable model
response, and it never records D=512 decode-side routing. The full-NVFP4 rung can quote
its allocator ratio versus the fp8 allocator row, but its quality comparison should remain
against the working bf16/auto row until this fp8 path is repaired.

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
