# SGLang deep prefix sweep stop point, 2026-06-10

## Status

The active sweep row was allowed to finish cleanly before yielding the GB10 GPU.
No comparator was interrupted mid-row.

- Run id: `sglang_qwen_mixedkv_prefixcacheguard_deep_prefix_sweep_ctx8192_20260610TmanualJST`
- Context: `8192`
- Completed prefixes: `4096, 6144, 7168, 7680`
- Runtime image: `sglang-source-stack-c3dae30f-e631a13fd`
- Docker cap: `--memory=100g --memory-swap=100g`
- Graph policy: CUDA graph capture enabled globally; prefix-cache-writing and cached-prefix
  prefills guarded to eager.
- Corpus:
  `results/sglang_qwen_mixedkv_prefixcacheguard_reuse_prefix_sweep_ctx8192_20260610TmanualJST_corpus.md`

## Completed Rows

| reused prefix | scored tokens | delta nats/token | delta PPL | allocator token ratio |
|---:|---:|---:|---:|---:|
| 4096 | 4095 | -0.001295 | -0.006697 | 1.778x |
| 6144 | 2047 | -0.066797 | -0.599492 | 1.777x |
| 7168 | 1023 | 0.008188 | 0.085177 | 1.778x |
| 7680 | 511 | 0.010784 | 0.159553 | 1.784x |

All rows report `ok=true` for fp8 and mixed-KV. The deepest prefixes have short scored
continuations, so their deltas are more sensitive to corpus slice/noise and should not be
over-interpreted without a broader corpus.

## Yield State

After the `7680` fp8+mixed comparator pair completed, the host was returned to idle:

- `docker ps`: empty
- `free -h`: about `114 GiB` available
- Claude marker created:
  `/home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN`

No GPU-serving work should resume until the marker is removed, or the 45-minute yield
timeout expires.

## Next SGLang Step

After Claude's idle window closes, inspect whether any probe containers are still present,
then continue with the capacity-denominator audit before promoting the mixed-KV row to
claim-ready status.
