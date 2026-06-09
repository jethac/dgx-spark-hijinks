# SGLang Qwen FP4-KV K-Scale Policy Probe, 2026-06-09

Status: red. K global-scale policy is a real lever, but the scalar multiplier tested here
is not a sufficient serving fix.

Runs:

- Trace sweep run id: `sglang_qwen_fp4kv_kscale_trace_c3dae30f_dfd426442_20260609T`
- Actual serving run id:
  `sglang_qwen_fp4kv_kscale_actual_c3dae30f_e4f24bbd3_20260609Tactual0125`
- Runtime image: `sglang-source-stack-c3dae30f-a8ad6a3ac`
- Source overlays: `jethac/flashinfer@c3dae30f`, `jethac/sglang@dfd426442` for the trace
  sweep, `jethac/sglang@e4f24bbd3` for the runtime multiplier
- Case: `default`, radix cache on, page size 1, FP4 KV
- Memory guardrail: single server, Docker `--memory=100g --memory-swap=100g`

Trace-sweep result:

| Main row metric | Baseline K scale | K scale `0.125` |
| --- | ---: | ---: |
| K global scale | `0.1197916716337204` | `0.01497395895421505` |
| FP4 K-only attention cosine vs BF16 | `0.7893718481063843` | `0.9584803581237793` |
| FP4 K+V attention cosine vs BF16 | `0.7876883745193481` | `0.9561840295791626` |
| Direct K reconstruction cosine | `~0.9967` | `0.8765701055526733` |

The same `0.125` multiplier also improved the 4096-token warmup row's K-only attention
reference from `0.7645635604858398` to `0.996171772480011`, and K+V from
`0.7661007642745972` to `0.9926666617393494`.

Actual serving result with `SGLANG_FP4_KV_K_GLOBAL_SCALE_MULTIPLIER=0.125`:

- Fresh rows still emit `**` with logprob `-0.7235294580459595`.
- The 55-token radix-hit row emits `To` / token id `1249` with logprob
  `-1.7186779975891113`, not the expected `**`.
- Flush-between and namespace-isolated controls keep `cached_tokens=0` and emit `**`.
- First localized layer-0 attention divergence improves from `0.006467887232207366` to
  `0.1657561728524288`, but remains red.

Interpretation:

- The current autocalibrated K global scale optimizes direct K reconstruction, not
  attention-logit stability.
- Reducing the K global scale can improve offline attention references substantially, but a
  single scalar multiplier does not recover correct radix-on serving output.
- Do not bless the multiplier. The next work should test finer K scaling policy
  (per-head/per-group), mixed K/V policy such as FP8 or BF16 K with FP4 V, or a
  model-specific gate. Any K-not-FP4 fallback must report the capacity cost.
