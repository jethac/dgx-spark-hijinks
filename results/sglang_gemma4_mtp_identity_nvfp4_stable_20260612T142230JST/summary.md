# SGLang Gemma 4 MTP Identity Gate

Status: GREEN

## Scope

Greedy spec-off vs spec-on identity for SGLang Gemma 4 Frozen-KV MTP with `fp4_e2m1` KV, graphs disabled. Servers are run sequentially; this is not a speedup claim.

## Provenance

- Run: `sglang_gemma4_mtp_identity_nvfp4_stable_20260612T142230JST`
- Target model: `google/gemma-4-E2B-it`
- Draft model: `google/gemma-4-E2B-it-assistant`
- Image: `sglang-source-stack-dgemma-024-0705924c-f99323bd:latest`
- SGLang: `98bf8f129d701d2829f2d1a82c4ce6a8b2f5a968`
- FlashInfer: `f99323bd7d1c`
- Spec flags: `--speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-num-draft-tokens 1 --speculative-eagle-topk 1 --speculative-draft-model-quantization unquant`
- KV cache dtype: `fp4_e2m1`
- Graphs disabled: `1`

## Gates

- Chat/native text identity: PASS
- Token IDs exposed: PASS
- Token identity: PASS
- Native `/generate` text is treated as not comparable when both sides return empty text; matching exposed token IDs remain the native endpoint gate.

## Runtime Notes

- This rerun replaced the prior open-ended DGX Spark prompt with a literal
  low-entropy prompt after target-only NVFP4 `spec_off` captures showed the old
  prompt varied across sequential server boots.
- The spec-on server reached readiness under full NVFP4 K+V target cache.
- The Frozen-KV MTP draft-worker calibration guard fired:
  `Skipping NVFP4 KV cache calibration for Frozen-KV MTP draft worker`.
- Target NVFP4 calibration warmup still reported partial coverage
  (`only calibrated 15/35 NVFP4 KV layers`) and fell back to first real prefill
  for remaining auto-calibration. This row is an identity gate only, not a
  speedup, graph-capture, or long-quality claim.

## Prompt Checks

- `capital_japan`: chat_text_match=True native_text_match=None chat_token_ids_match=None native_token_ids_match=True off='The capital of Japan is Tokyo.' on='The capital of Japan is Tokyo.'
- `arithmetic`: chat_text_match=True native_text_match=None chat_token_ids_match=None native_token_ids_match=True off='2 + 2 equals 4.' on='2 + 2 equals 4.'
- `literal_spark_ok`: chat_text_match=True native_text_match=True chat_token_ids_match=None native_token_ids_match=True off='spark-ok' on='spark-ok'

## Artifacts

- `spec_off.capture.json`
- `spec_on.capture.json`
- `identity_comparison.json`
- `spec_off.server.log`
- `spec_on.server.log`
- `hf_access_target.json`
- `hf_access_draft.json`
