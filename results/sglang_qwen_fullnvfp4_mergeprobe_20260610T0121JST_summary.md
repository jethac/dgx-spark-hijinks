# SGLang Qwen Full NVFP4 K+V Radix Merge Probe

Run ID: `sglang_qwen_fullnvfp4_mergeprobe_20260610T0121JST`

Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`

Model: `Qwen/Qwen2.5-1.5B-Instruct`

KV mode: full `fp4_e2m1` K+V, radix cache enabled

## Verdict

The run reproduces the full-NVFP4 radix-cache quality failure, but the decisive trace does
not support a broken online-softmax merge.

The cached-prefix request still returns `ark` where dense/fresh requests return `**`.
However, the traced layer-0 merge matches a direct full recompute over the actual cached-path
inputs:

- cached prefix partial `o2` vs recomputed FP4-prefix reference: cosine `0.9999972`
- cached prefix LSE `s2` vs recomputed FP4-prefix reference, base-2: cosine `1.0000001`, max abs `0.001953125`
- suffix raw/BF16 partial `o1` vs recomputed suffix reference: cosine `0.9999999`
- suffix raw/BF16 LSE `s1` vs recomputed suffix reference, base-2: cosine `1.0000001`, max abs `0.001953125`
- `_safe_merge_state` output vs direct full recompute over FP4 prefix + raw/BF16 suffix: cosine `0.9999968`, max abs `0.0078125`

This means `_safe_merge_state` is reconstructing the current cached-path attention state
correctly. The failure is not an obvious merge arithmetic bug.

## Important Correction

The earlier framing "dense full-attention FP4 KV is clean, therefore full-NVFP4 radix should
match it" was too strong for SGLang.

In the fresh dense/full-prefill path, SGLang attends over the current request's raw K/V while
writing quantized KV into the cache. In the radix cached-prefix path, the prefix is read back
from the FP4 KV cache while the new suffix still uses raw K/V. Those are different numerical
regimes.

That distinction matters for interpreting the "merge bug" hypothesis. The trace already
contains an independent torch reference merge over the exact tensors passed to
`_safe_merge_state`:

```python
m = torch.maximum(s1, s2)
w1 = torch.exp2(s1 - m)
w2 = torch.exp2(s2 - m)
manual_merged = ((o1 * w1) + (o2 * w2)) / (w1 + w2)
```

That independent merge matches SGLang's merged output with cosine `0.9999999` and max abs
`0.0`. So the current evidence does not show `_safe_merge_state` arithmetic disagreeing with
an exact reference over its supplied `o1/s1/o2/s2` inputs.

The trace shows the difference directly:

- dense fresh attention output vs dense full reference: cosine `0.9999831`
- cached merge vs dense fresh attention output: cosine `0.7218509` in the direct full-reference trace, and `0.0064679` on the sampled trace row used by the dense-cache comparator
- BF16/full prefix partial vs FP4 cached-prefix partial: cosine `0.8517225`
- BF16/full prefix LSE vs FP4 cached-prefix LSE, base-2: max abs `484.75`
- recomputed FP4 suffix partial vs raw/BF16 suffix partial used as `o1`: cosine `0.9956781`
- recomputed FP4 suffix LSE vs raw/BF16 suffix LSE used as `s1`, base-2: max abs `429.703125`
- direct full recompute over FP4 prefix + raw/BF16 suffix vs merged output: cosine `0.9999968`
- direct full recompute over FP4 prefix + FP4 suffix vs merged output: cosine `0.9024510`

That LSE gap is enough to flip the first token under radix reuse.

The actionable interpretation is therefore not "change the merge formula." It is: if SGLang
wants full NVFP4 K+V radix reuse to equal an all-FP4 dense reference, the merge path must feed
`_safe_merge_state` suffix partials computed under the same FP4-cache regime as the prefix, or
the dense comparator must be changed to measure the actual raw-suffix cached regime. Today it
mixes an FP4 cached prefix with a raw/BF16 suffix.

## Mixed-KV Implication

The mixed-KV row (`FP8-K + NVFP4-V`) should remain a fallback/design candidate, not be
dismissed as merely hiding a confirmed merge bug. The current evidence says preserving K
precision is the correctness lever for radix reuse, while V can still be compressed.

Capacity claims must stay separate:

- full NVFP4 K+V: target capacity win, still red under SGLang radix cache
- mixed FP8-K + NVFP4-V: correctness-preserving fallback to validate further, with its own
capacity denominator

## Artifacts

- Server log: `results/sglang_qwen_fullnvfp4_mergeprobe_20260610T0121JST_default_server.log`
- Request/results JSON: `results/sglang_qwen_fullnvfp4_mergeprobe_20260610T0121JST_default.json`
- Dense-cache compare: `results/sglang_qwen_fullnvfp4_mergeprobe_20260610T0121JST_default_dense_cache_compare.json`
- Run summary: `results/sglang_qwen_fullnvfp4_mergeprobe_20260610T0121JST_summary.json`
- Trace audit: `results/sglang_qwen_fullnvfp4_mergeprobe_20260610T0121JST_dense_cache_trace_summary_audit.json`
