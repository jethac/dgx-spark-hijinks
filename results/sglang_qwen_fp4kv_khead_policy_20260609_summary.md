# SGLang Qwen FP4-KV Per-Head K Scale Policy Probe, 2026-06-09

Status: red. The current KV layout can represent a per-head effective K scale by folding
head/global ratios into the FP8 block-scale buffer, but the tested amax policy is not a
strong enough next serving fix.

Run:

- Runner: `scripts/run_sglang_fp4_dense_cache_trace.sh`
- Run id: `sglang_qwen_fp4kv_khead_policy_c3dae30f_d6fa9d104_20260609T1430Z`
- Runtime image: `sglang-source-stack-c3dae30f-a8ad6a3ac`
- Source overlays: `jethac/flashinfer@c3dae30f`, `jethac/sglang@d6fa9d104`
- Case: `default`, radix cache on, page size 1, FP4 KV
- Trace env: `K_HEAD_SCALE_POLICY=1`,
  `K_SCALE_MULTIPLIERS=0.125,0.25,0.5,1,2,4,8`
- Memory guardrail: single server, Docker `--memory=100g --memory-swap=100g`

Serving behavior is unchanged:

- Dense/no-prefix rows emit `**` with logprob `-0.7235294580459595`.
- Cached-prefix rows reuse 55 tokens and emit `ark` / token id `838` with logprob
  `-0.5874708890914917`.
- Flush and namespace-isolated controls keep `cached_tokens=0` and emit `**`.
- First localized divergence remains layer-0 attention output, dense `o_rows` vs cached
  `merged_rows`: cosine `0.006467887232207366`.

Trace method:

For each KV head, the trace computes an amax-derived K global, quantizes that head under
the head-specific global, folds `head_global / base_global` into the stored FP8 block
scales, and dequantizes under the single scalar K global that FlashInfer receives. This
is a storage-compatible reference for a finer effective K policy without changing the
FlashInfer wrapper ABI.

Main row (`row=55`, layer 0):

| Policy | K+V attention cosine vs BF16 | K-only attention cosine vs BF16 | K reconstruction cosine |
| --- | ---: | ---: | ---: |
| Baseline scalar K global | `0.7876883745193481` | `0.7893718481063843` | `~0.9967` |
| Per-head folded-SF, multiplier `0.125` | `0.954595148563385` | `0.9574685096740723` | `0.8836419582366943` |
| Per-head folded-SF, multiplier `0.25` | `0.8113424181938171` | `0.815320611000061` | `0.9156984090805054` |
| Per-head folded-SF, multiplier `0.5` | `0.740337073802948` | `0.7418009042739868` | `0.9615521430969238` |
| Per-head folded-SF, multiplier `1.0` | `0.5636622905731201` | `0.5680369138717651` | `0.9962771534919739` |

Interpretation:

- Per-head amax scaling is a real offline lever, but it is not materially better than the
  scalar `0.125` regime that already failed actual radix-on serving.
- Higher multipliers saturate the folded FP8 scale buffer (`sf_after_max=448`) and make
  attention quality worse despite high direct K reconstruction.
- Do not implement this per-head amax policy as a serving fix yet. The next useful
  experiment is a stronger quality-oriented K policy or a mixed cache policy such as
  FP8/BF16 K with NVFP4 V.

Capacity note:

- NVFP4 K+V is roughly `4.5 + 4.5 = 9` bits per KV element pair, or about `1.78x` the fp8
  KV pool.
- Naive FP8 K + NVFP4 V is roughly `8 + 4.5 = 12.5` bits per pair, or about `1.28x` the
  fp8 KV pool. That fallback may still be useful, but it gives up a large part of the
  NVFP4 capacity headline and must be reported as such.
