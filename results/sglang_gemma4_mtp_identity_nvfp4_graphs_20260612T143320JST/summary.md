# SGLang Gemma 4 MTP Identity Gate

Status: GREEN

## Scope

Greedy spec-off vs spec-on identity for SGLang Gemma 4 Frozen-KV MTP with `fp4_e2m1` KV, graphs enabled. Servers are run sequentially; this is not a speedup claim.

## Provenance

- Run: `sglang_gemma4_mtp_identity_nvfp4_graphs_20260612T143320JST`
- Target model: `google/gemma-4-E2B-it`
- Draft model: `google/gemma-4-E2B-it-assistant`
- Image: `sglang-source-stack-dgemma-024-0705924c-f99323bd:latest`
- SGLang: `98bf8f129d701d2829f2d1a82c4ce6a8b2f5a968`
- FlashInfer: `f99323bd7d1c`
- Spec flags: `--speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-num-draft-tokens 1 --speculative-eagle-topk 1 --speculative-draft-model-quantization unquant`
- KV cache dtype: `fp4_e2m1`
- Graphs disabled: `0`

## Gates

- Chat/native text identity: PASS
- Token IDs exposed: PASS
- Token identity: PASS
- Native `/generate` text is treated as not comparable when both sides return empty text; matching exposed token IDs remain the native endpoint gate.

## Runtime Notes

- The launch requested graphs (`Graphs disabled: 0`), but native FP4 KV graph
  capture was intentionally disabled by SGLang's safety guard:
  `Disabling CUDA graph capture for native FP4 KV cache. Current FlashInfer FA2
  NVFP4 KV graph capture can produce corrupt decode output; set
  SGLANG_FP4_KV_ENABLE_CUDA_GRAPH=1 only for graph-safety experiments.`
- Therefore this row proves the graph-request guard path preserves MTP identity;
  it does not prove native-FP4 CUDA graph replay is safe.
- The Frozen-KV MTP draft-worker calibration guard also fired:
  `Skipping NVFP4 KV cache calibration for Frozen-KV MTP draft worker`.

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
