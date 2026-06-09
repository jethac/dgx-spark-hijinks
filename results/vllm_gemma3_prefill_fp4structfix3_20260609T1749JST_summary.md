# vLLM Gemma 3 NVFP4-KV Prefill Struct Fix, 2026-06-09 17:49 JST

Status: green for the short first-token/logprob gate; not yet a long-context PPL or throughput row.

Purpose: verify the FlashInfer FP4 paged-prefill fix after `jethac/flashinfer@0919cdda`
made the SWA FP4 module compile and `jethac/flashinfer@c3dae30f` completed the Python
wrapper argument plumbing.

Stack:

- model: `google/gemma-3-27b-it`
- served model: `gemma3-27b-it`
- vLLM overlay: local `third_party/vllm` branch with FA2 NVFP4 prefill JIT args
- FlashInfer overlay: `jethac/flashinfer@c3dae30f` plus the same source mounted into the
  vLLM container
- run id: `vllm_gemma3_prefill_fp4structfix3_20260609T1749JST`
- memory guardrails: single server, `--gpu-memory-utilization 0.72`, `--memory 100g`,
  `--memory-swap 100g`

Artifacts:

- `results/vllm_gemma3_prefill_fp4structfix3_20260609T1749JST_server.log`
- `results/vllm_gemma3_prefill_fp4structfix3_20260609T1749JST_first_token.json`
- `results/vllm_gemma3_prefill_fp4structfix3_20260609T1749JST_first_token_compare.json`
- `results/vllm_gemma3_prefill_fp4structfix3_20260609T1749JST_quality_gate.json`
- `results/vllm_gemma3_prefill_fp4structfix3_20260609T1749JST_nvfp4_kv_first_token_row_manifest.json`
- `results/vllm_gemma3_prefill_fp4structfix3_20260609T1749JST_flashinfer_prefill_debug_audit.json`

Result:

- The server reached readiness.
- The previous `Expected 29 but got 24/26 arguments` warmup failures are fixed.
- Runtime debug lines show FP4 paged-prefill modules generated for both Gemma attention
  modes:
  - SWA: `dtype_kv=__nv_fp4x2_e2m1`, `fp4_kv=1`, `window_left=1023`
  - global: `dtype_kv=__nv_fp4x2_e2m1`, `fp4_kv=1`, `window_left=-1`
- Warmup also emitted raw-byte variants for larger warmup batch shapes. The serving probe
  nevertheless passed the first-token comparator; keep this distinction in later audits.

First-token comparison against the prior fp8 baseline:

| case | fp8 first token | NVFP4 first token | overlap ratio |
|---|---|---|---:|
| `exact_spark_ok` | `spark` | `spark` | `0.9047619048` |
| `simple_math` | `4` | `4` | `0.7727272727` |
| `short_decode` | `A` | `A` | `0.9047619048` |

`scripts/gemma_nvfp4_kv_quality_gate.py` passed with no findings. This clears the known
bad first-token signatures (`Reigns`, Gujarati/CJK token, `ioane`) from the earlier Gemma 3
NVFP4-KV row.

Caveats:

- This is a short-prompt first-token/logprob gate only. It does not prove long-context SWA
  behavior, supplied-token PPL, or throughput.
- `scripts/flashinfer_prefill_debug_log_audit.py` is stale for the current debug-line format:
  it fails to parse tensor fields that are present in the raw log. Treat the failed audit JSON
  as an audit-tool issue, not a runtime failure.

Next:

- Run the queued PPL comparator sequentially, not concurrently, under the GB10 memory rules.
- Then climb Gemma rung 2 only after the long-context/SWA confidence row is green.
