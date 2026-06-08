# SGLang Qwen FP4 KV Logprob Quality Probe

Date: 2026-06-08 JST

Purpose: localize why the matched `jethac/sglang@d7d931f` FP4-KV row passes raw/chat
smoke but fails the standardized OpenAI benchmark prompts.

## Setup

- Runtime: `nvcr.io/nvidia/sglang:26.05-py3` with `jethac/sglang@d7d931f`
  source overlay.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`.
- Comparator: `--kv-cache-dtype fp8_e4m3`.
- Candidate: `--kv-cache-dtype fp4_e2m1`.
- Shared settings: FlashInfer attention, page size 1, memory fraction 0.40,
  CUDA graph disabled, piecewise CUDA graph disabled.
- Probe: `scripts/openai_quality_probe.py` with generated-token logprobs enabled on
  `short_decode` and `medium_decode`.

## Artifacts

- `results/sglang_qwen_fp4kv_d7d931f_logprob_quality_20260608T1609JST_fp8_quality_probe.json`
- `results/sglang_qwen_fp4kv_d7d931f_logprob_quality_20260608T1609JST_fp4_quality_probe.json`
- `results/sglang_qwen_fp4kv_d7d931f_logprob_quality_20260608T1609JST_compare.json`
- `results/sglang_qwen_fp4kv_d7d931f_logprob_quality_20260608T1609JST_fp8_server.log`
- `results/sglang_qwen_fp4kv_d7d931f_logprob_quality_20260608T1609JST_fp4_server.log`
- `results/sglang_qwen_fp4kv_d7d931f_logprob_quality_20260608T1609JST_fp4_trace_excerpt.txt`

## Result

- fp8 passed both quality probes.
- FP4 `short_decode` began with the same high-confidence English prefix as fp8
  (`A local AI workstation`) but then drifted into mixed Chinese text and repeated
  phrases. The heuristic did not mark this case failed, but similarity to fp8 was only
  `0.0883`.
- FP4 `medium_decode` diverged immediately: first generated tokens were
  `the following code:` rather than fp8's `**Engineering Note:`. It then collapsed into
  repeated `import` text and was flagged for low unique-word ratio, dominant repeated
  word, repeated bigrams, and repeated trigrams. Similarity to fp8 was `0.0084`.
- The combined comparison has `ok=false`.

## Interpretation

This is a quality-localization result, not a fix and not a speed row. It shows the FP4
path can be wrong from the first generated token for a normal benchmark prompt, while
another prompt starts plausibly and corrupts later. That points away from "the raw
backend never works" and toward prompt/path-sensitive serving state: calibration
application, backend wrapper metadata, scale convention/layout coupling, or another
model-serving path not covered by the synthetic pool bridge.

Next useful probe: add a prefill/decode boundary or per-token divergence window around
the failing `medium_decode` prompt, while keeping the same fp8 comparator and backend
trace.
