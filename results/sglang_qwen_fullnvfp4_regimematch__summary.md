# SGLang Qwen Full NVFP4 K+V Regime-Matched Radix Probe

Run ID: `sglang_qwen_fullnvfp4_regimematch_`

Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`

Model: `Qwen/Qwen2.5-1.5B-Instruct`

KV mode: full `fp4_e2m1` K+V, radix cache enabled

Patch under test: `FlashInferAttnBackend.extend_merge_paged` quantizes/dequantizes the new
suffix K/V through the same NVFP4 path before computing the ragged suffix partial. The goal
was to feed `_safe_merge_state` FP4-regime `o1/s1` and FP4-regime `o2/s2`, avoiding the
previous FP4-prefix + raw-suffix LSE mismatch.

## Verdict

Red for serving quality, but the regime-matching hypothesis was tested cleanly.

The patch successfully makes the merged cached-prefix output match a direct all-FP4 recompute,
but that all-FP4 attention state still does not match the fresh BF16/raw dense request closely
enough to preserve the first token.

Token result:

- dense/no-prefix rows: `**`, logprob `-0.7235294580459595`
- cached-prefix rows: `odel`, logprob `-2.0344042778015137`

Layer-0 dense-cache comparator:

- attention output cosine vs dense fresh row: `0.0138981`
- first divergence remains layer-0 attention output

## What Changed

Compared to the prior full-NVFP4 run, the suffix partial is now in the FP4 regime:

- suffix BF16/raw reference vs `o1`: cosine `0.9956782`, LSE max abs `439.41796875`
- suffix FP4 reference vs `o1`: cosine `0.9999998`, LSE max abs `9.716796875`
- independent torch merge vs SGLang merge: cosine `1.0`, max abs `0.0`
- direct full recompute over FP4 prefix + FP4 suffix vs merged output: cosine `0.9991559`
- direct full recompute over FP4 prefix + raw/BF16 suffix vs merged output: cosine `0.8879871`

So the patch fixed the precision-regime mismatch at the merge boundary.

## Interpretation

This falsifies the stronger claim that full NVFP4 K+V radix should pass simply by making
prefix and suffix both FP4-regime. The current evidence says:

- `_safe_merge_state` arithmetic is correct.
- The old cached path was indeed mixing FP4-prefix LSE with raw-suffix LSE.
- The regime-matched full-NVFP4 path now behaves like an all-FP4 attention recompute.
- That all-FP4 attention recompute is still not quality-equivalent to the BF16/raw fresh path
  for this Qwen first-token gate.

The mixed-KV path remains the only observed SGLang radix configuration that preserves this
first-token gate, because keeping K at FP8 avoids the QK/LSE sensitivity that full NVFP4 K
still triggers. It should remain documented as a fallback/design candidate, not as the final
full-1.78x claim.

## Artifacts

- Server log: `results/sglang_qwen_fullnvfp4_regimematch__default_server.log`
- Request/results JSON: `results/sglang_qwen_fullnvfp4_regimematch__default.json`
- Dense-cache compare: `results/sglang_qwen_fullnvfp4_regimematch__default_dense_cache_compare.json`
- Run summary: `results/sglang_qwen_fullnvfp4_regimematch__summary.json`
- Trace audit: `results/sglang_qwen_fullnvfp4_regimematch__dense_cache_trace_summary_audit.json`
