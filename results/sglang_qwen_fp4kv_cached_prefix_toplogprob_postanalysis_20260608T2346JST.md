# SGLang Qwen FP4-KV Cached-Prefix Top-Logprob Post-Analysis, 2026-06-08 23:46 JST

Status: red serving path; no-reuse remains diagnostic only.

Source artifact:

- `results/sglang_qwen_fp4kv_request_order_20260608T2340JST.json`
- source overlay: `jethac/sglang@d4fe78078633e70fde968c58032a675c72f13bc1`
- model: `Qwen/Qwen2.5-1.5B-Instruct`
- KV dtype: `fp4_e2m1`
- attention backend: `flashinfer`
- probe: `medium_decode`, first generated token, `top_logprobs_num=20`

Purpose: quantify whether the FP4 cached-prefix failure is a small rank wobble or a
large distribution change.

Rows:

| row | first request cached tokens | first token | second request cached tokens | second token | top-20 token overlap |
|---|---:|---|---:|---|---:|
| baseline OpenAI then native | 0 | `**` | 55 | `ark` | 0 / 20 |
| reverse native then OpenAI | 0 | `**` | 55 | `ark` | 0 / 20 |
| flush between requests | 0 | `**` | 0 | `**` | 20 / 20 |
| namespace isolation | 0 | `**` | 0 | `**` | 20 / 20 |

Interpretation:

- The failure follows the cached FP4 prefix, not endpoint identity.
- The failed cached-prefix rows do not merely reorder close candidates: the first-token
  top-20 set is disjoint from the full-prefill top-20 set.
- The no-reuse rows match the full-prefill distribution exactly for the top-20 tokens.
- Therefore, disabling radix/prefix reuse can only be treated as a diagnostic or
  emergency correctness workaround. It is not an acceptable final fix for the FP4-KV
  capacity path, because prefix reuse is part of the serving behavior the capacity win
  must survive.

Next focus:

1. Compare full dense prefill against a cached-prefix FP4 attention path at the logits
   level, not just sampled layer-0 attention output.
2. Test scale/calibration variants only if they preserve prefix reuse and recover the
   top-logprob distribution.
3. Keep any selective no-reuse knob opt-in and label rows produced with it as
   workaround rows, not blessed FP4-KV serving rows.
