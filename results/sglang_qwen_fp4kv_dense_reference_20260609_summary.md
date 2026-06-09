# SGLang Qwen FP4-KV Dense-Reference Partial-State Probe, 2026-06-09

Status: red, but the root-cause boundary is now sharper.

Run:

- Runner: `scripts/run_sglang_fp4_dense_cache_trace.sh`
- Run id: `sglang_qwen_fp4kv_dense_ref_c3dae30f_5b71bef3c_20260609T1455Z`
- Runtime image: `sglang-source-stack-c3dae30f-a8ad6a3ac`
- Source overlays: `jethac/flashinfer@c3dae30f`, `jethac/sglang@5b71bef3c`
- Case: `default`, radix cache on, page size 1, FP4 KV
- Memory guardrail: single server, Docker `--memory=100g --memory-swap=100g`

Serving behavior:

- Dense/no-prefix rows emit `**` with logprob `-0.7235294580459595`.
- Cached-prefix rows reuse 55 tokens and emit `ark` / token id `838` with logprob
  `-0.5874708890914917`.
- Flush and namespace-isolated controls keep `cached_tokens=0` and emit `**`.

Decisive partial-state comparison:

| Comparison | Result |
| --- | ---: |
| Cached Q vs dense no-prefix Q | `0.9999930262565613` |
| Suffix partial `o1` vs dense suffix recompute | `0.9999936819076538` |
| Cached-prefix `o2` vs FP4 cached-page reference | `0.9999971985816956` |
| Cached-prefix `s2` max abs vs FP4 cached-page reference | `0.001953125` |
| Cached-prefix `o2` vs BF16 dense-prefix recompute | `0.8517225384712219` |
| Cached-prefix `s2` max abs vs BF16 dense-prefix LSE | `484.75` |
| Manual merge vs `_safe_merge_state` output | `0.9999998807907104` |
| BF16 dense full recompute vs cached merged output | `0.7218508720397949` |

Interpretation:

- The cached-prefix read path is internally consistent with the FP4 cache.
- The suffix partial and merge math are not the culprit.
- The no-prefix request serves over BF16 K/V and only writes FP4 KV after attention, so
  it is not proof that FP4 K quality is acceptable.
- The failing cached request is the first time the prompt has to attend through the
  FP4-compressed prefix. That prefix state is close to its own FP4 reference but too far
  from the BF16 dense-prefix state.

Next:

- Stop chasing radix page pairing, stale scales, suffix attention, and `_safe_merge_state`
  for this row.
- Either make the FP4 prefix state closer to BF16 with a stronger quality-oriented policy,
  or test a mixed cache policy such as FP8/BF16 K with NVFP4 V and report the capacity
  cost explicitly.
