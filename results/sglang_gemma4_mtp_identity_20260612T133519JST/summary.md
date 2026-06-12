# SGLang Gemma 4 MTP Identity Gate

Status: GREEN

## Scope

BF16 greedy spec-off vs spec-on identity for SGLang Gemma 4 Frozen-KV MTP. Servers are run sequentially; this is not a speedup claim and not an NVFP4 row.

## Provenance

- Run: `sglang_gemma4_mtp_identity_20260612T133519JST`
- Target model: `google/gemma-4-E2B-it`
- Draft model: `google/gemma-4-E2B-it-assistant`
- Image: `sglang-source-stack-dgemma-024-0705924c-f99323bd:latest`
- SGLang: `0211f87b234b0f05eb10005b3791d81a962883ca`
- FlashInfer: `f99323bd7d1c`
- Spec flags: `--speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-num-draft-tokens 1 --speculative-eagle-topk 1 --speculative-draft-model-quantization unquant`
- Graphs disabled for the first identity gate.

## Gates

- Chat/native text identity: PASS
- Token IDs exposed: PASS
- Token identity: PASS
- Native `/generate` text is treated as not comparable when both sides return empty text; matching exposed token IDs remain the native endpoint gate.

## Prompt Checks

- `capital_japan`: chat_text_match=True native_text_match=None chat_token_ids_match=None native_token_ids_match=True off='The capital of Japan is Tokyo.' on='The capital of Japan is Tokyo.'
- `arithmetic`: chat_text_match=True native_text_match=None chat_token_ids_match=None native_token_ids_match=True off='2 + 2 equals 4.' on='2 + 2 equals 4.'
- `spark_use`: chat_text_match=True native_text_match=None chat_token_ids_match=None native_token_ids_match=True off='A DGX Spark is useful for accelerating large-scale data processing and machine learning workloads.' on='A DGX Spark is useful for accelerating large-scale data processing and machine learning workloads.'

## Artifacts

- `spec_off.capture.json`
- `spec_on.capture.json`
- `identity_comparison.json`
- `spec_off.server.log`
- `spec_on.server.log`
- `hf_access_target.json`
- `hf_access_draft.json`
