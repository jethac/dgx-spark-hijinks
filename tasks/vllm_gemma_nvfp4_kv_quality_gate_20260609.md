# vLLM Gemma NVFP4-KV Quality Gate

Status: queued after the current Gemma 3 FlashInfer paged-prefill debug/fix.

Purpose: prevent a Gemma NVFP4-KV capacity/routing row from being blessed unless it also
has Gemma-specific quality evidence. The existing heuristic quality comparator missed the
known Gemma 3 corruption, so this gate requires a stricter artifact.

## Why This Exists

Gemma attention is outlier-sensitive. The Gemma 4 NVFP4 weight releases keep attention
weights in BF16, and the checked-in Gemma 3 NVFP4-KV candidate proves why smoke tests are
not enough: capacity and FA2 routing pass, but first-token logits are unrelated to the fp8
baseline.

The known bad row is now machine-rejected by:

```bash
python3 scripts/gemma_nvfp4_kv_quality_gate.py \
  --baseline-manifest results/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_row_manifest.json \
  --candidate-manifest results/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_row_manifest.json \
  --first-token-compare results/vllm_gemma3_27b_rung1_20260608T205432JST_first_token_compare.json \
  --output results/gemma3_nvfp4_kv_quality_gate_current_red_20260609.json
```

Expected current result: `ok=false`.

## Green Criteria For A Future Gemma Row

Run a matched fp8-or-bf16 KV baseline and NVFP4-KV candidate on the same Gemma rung, then
pass:

```bash
python3 scripts/gemma_nvfp4_kv_quality_gate.py \
  --baseline-manifest results/${RUN}_fp8_or_bf16_row_manifest.json \
  --candidate-manifest results/${RUN}_nvfp4_kv_row_manifest.json \
  --first-token-compare results/${RUN}_first_token_compare.json \
  --ppl-compare results/${RUN}_ppl_compare.json \
  --output results/${RUN}_gemma_nvfp4_kv_quality_gate.json
```

At least one strict quality proof must pass:

- `openai-first-token-compare/v1` with `ok=true` and top-logprob overlap above the gate
  threshold; or
- `vllm-prompt-ppl-comparison/v1` with `ok=true`, finite supplied-token logprobs, and
  Gemma fp8/bf16-vs-NVFP4 delta within the configured nats/token threshold.

For a blessed rung, prefer both. PPL is the public quality number; first-token/logprob is
the fast regression check that catches the current failure mode.

## Acceptance

- The candidate manifest is a Gemma `openai-serving-row-manifest/v1` with
  `kv_cache_dtype=nvfp4` and `ok=true`.
- The baseline manifest is the same Gemma rung with fp8, bf16, auto, or otherwise
  non-NVFP4 KV.
- The gate artifact has schema `gemma-nvfp4-kv-quality-gate/v1` and `ok=true`.
- The summary names the model rung, KV dtype selected per layer where relevant, capacity
  delta, and quality delta.
- Do not climb from Gemma 3 to Gemma 4 31B until this gate is green for Rung 1.

Expected queue artifacts:

- `results/${RUN_ID}_gemma_nvfp4_kv_quality_gate.json`
- `results/${RUN_ID}_first_token_compare.json`
- `results/${RUN_ID}_ppl_compare.json`
