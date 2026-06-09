# SGLang Qwen FP4-KV K/V Split Trace, 2026-06-09

Status: red, but root cause is narrower.

Run:

- Runner: `scripts/run_sglang_fp4_dense_cache_trace.sh`
- Run id: `sglang_qwen_fp4kv_kvsplit_trace_c3dae30f_8b95253af_20260609T1335Z`
- Runtime image: `sglang-source-stack-c3dae30f-a8ad6a3ac`
- Source overlay: `jethac/sglang@8b95253af`, `jethac/flashinfer@c3dae30f`
- Case: `default`, radix cache on, page size 1, FP4 KV
- Memory guardrail: single server, Docker `--memory=100g --memory-swap=100g`

Observed behavior is unchanged:

- Dense/no-prefix rows emit `**` with logprob `-0.7235294580459595`.
- Cached-prefix rows reuse 55 tokens and emit `ark` / token id `838` with logprob `-0.5874708890914917`.
- Flush and namespace-isolated controls keep `cached_tokens=0` and emit `**`.
- First localized divergence remains layer-0 attention output, dense `o_rows` vs cached `merged_rows`: cosine `0.006467887232207366`, max abs `0.318359375`, RMS `0.13599727805129772`.

New finding:

- The cached-prefix paged contribution and merge math are not the current culprit.
- Previous prefix-reference trace showed paged prefix `o2/s2` matches a manual FP4 dequant attention reference: `o2_compare.cosine=0.9999971985816956`, base-2 `s2_compare.cosine=1.0000001192092896`, and `merge_compare.cosine=0.9999998807907104`.
- The dense FP4 reference already diverges from BF16 on the same failing row, especially through K quantization.

K/V split on the main row (`row=55`, layer 0):

| Reference | Cosine vs BF16 | Max abs |
| --- | ---: | ---: |
| Actual FlashInfer dense vs BF16 reference | `0.9999995231628418` | `0.00390625` |
| FP4 K+V reference vs BF16 | `0.7876883745193481` | `1.796875` |
| FP4 K-only reference vs BF16 | `0.7893718481063843` | `1.8515625` |
| FP4 V-only reference vs BF16 | `0.996927797794342` | `0.10546875` |

Interpretation:

- V quantization is comparatively benign for this row.
- K quantization perturbs the attention logits enough to move the attention output and downstream first token.
- Direct write-time K reconstruction still has high raw cosine (`~0.9967`) but large absolute error (`max_abs ~41-42`, RMS `~6`) on layer-0 K, which is enough to flip softmax behavior in this Qwen row.
- The simple FlashInfer/SGLang global-scale convention suspicion was falsified by `scripts/sglang_fp4_quant_scale_probe.py`: SGLang's current convention and FlashInfer's helper convention both reconstruct a synthetic KV tensor at about `cosine ~0.9955`.
- A direct dense-vs-cached scale diff on the same failing artifact also falsifies a stale or inverted cached-prefix global-scale explanation. Layer-0 dense write/dequant, dense-quant attention, and the failing `extend_merge_paged` cached-prefix call all use `k=0.1197916716337204`, `v=0.0016276042442768812`; FlashInfer's wrapper applies the same handedness as the local FP4-dequant reference.

Next target:

- Stop chasing radix page pairing and `_safe_merge_state` for this bug unless new evidence appears.
- Investigate K-side policy: calibrated K scale quality, per-head/per-group K scaling, FP8/BF16 K with FP4 V, or model-specific gating for Qwen. A K-not-FP4 fallback would sacrifice part of the capacity win, so quantify the memory delta before blessing it.
