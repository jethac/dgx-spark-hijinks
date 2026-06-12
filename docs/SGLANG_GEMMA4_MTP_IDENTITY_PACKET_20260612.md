# SGLang Gemma 4 MTP Identity Packet

Date: 2026-06-12 JST

Goal: run the first live SGLang Gemma 4 Frozen-KV MTP identity gate on Spark.
This is a correctness row only, not a speedup row.

## Scope

- Target: `google/gemma-4-E2B-it`
- Draft: `google/gemma-4-E2B-it-assistant`
- Runtime: SGLang on `epoch2`, source stack image
  `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0d5e160cf83db43e1e024a8300ed2858b426b4a0f38289210dc51d8c7b6def94`
- SGLang pin: `98bf8f129d701d2829f2d1a82c4ce6a8b2f5a968`, which adds
  `gemma4_assistant` / `gemma4_unified_assistant` config aliases for the
  native assistant checkpoints and passes prefix lengths for Frozen-KV MTP
  verify prefill planning, plus narrows MRoPE positions for the one-token draft
  seed, narrows seed cache slots to the last target token per request, and
  reports eager buffer shape mismatches by slot. It also skips NVFP4 KV-cache
  calibration for the Frozen-KV draft worker, because the native assistant has
  no own KV cache and reads the already-calibrated target cache.
- First row: BF16 target, unquantized draft, `topk=1`, `num_steps=1`,
  `num_draft_tokens=1`, graphs disabled
- Second row: full NVFP4 K+V target cache (`KV_CACHE_DTYPE=fp4_e2m1`),
  unquantized draft weights, same speculative settings, graphs disabled.
- Memory rule: spec-off and spec-on servers are run sequentially; no concurrent
  comparator servers on GB10.

## Command

```bash
cd /home/jethac/spark_tmp/dgx-spark-hijinks-sglang-live
bash scripts/run_sglang_gemma4_mtp_identity_gate.sh
```

Full NVFP4 K+V row:

```bash
KV_CACHE_DTYPE=fp4_e2m1 SGLANG_FP4_KV_MIXED_KV=0 DISABLE_GRAPHS=1 \
  bash scripts/run_sglang_gemma4_mtp_identity_gate.sh
```

Optional cache/online controls:

```bash
HF_LOCAL_FILES_ONLY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  bash scripts/run_sglang_gemma4_mtp_identity_gate.sh
```

## Gates

- Spec-off and spec-on both reach readiness.
- Spec-on server log proves `NEXTN` was promoted to `FROZEN_KV_MTP`.
- Greedy spec-on output is identical to spec-off at temperature 0.
- Token IDs must be exposed and identical. If text matches but token IDs are
  absent, the row is RED under the zero-bug bar.
- Transcripts and raw responses are banked verbatim.

## Stop Conditions

- `CLAUDE_WINDOW_OPEN` present, or `docker ps` non-empty: yield with no partial
  run.
- HF access probe fails for target or draft: record the failure; do not treat as
  a model/runtime RED.
- Any token mismatch or incoherent transcript: RED artifact, no speed claim.
