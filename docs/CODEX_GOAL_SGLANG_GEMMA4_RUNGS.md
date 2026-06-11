# Codex /goal: Gemma 4 serving on SGLang (rung 0 -> rung 1)

Mission: bring Gemma 4 serving up on SGLang on the validated VO-split
foundation, with the campaign's evidence discipline.

## Context pointers (read first)
- docs/SGLANG_GEMMA4_RUNG_PREP.md (your map)
- docs/SGLANG_GEMMA4_VOSPLIT_VALIDATION_PACKET_20260611.md (your gates)
- docs/FLASHINFER_D512_FA2_KERNEL_PLAN.md (kernel ground truth)
- docs/RESULTS_LEDGER.md (row conventions)
Foundation state: writer/pool roundtrip GREEN at head-256/SWA and head-512
VO-split with linear SFs (677586a, d4b55f8); FlashInfer overlay 8d85fff9.

## Objective 1 - rung 0: text-only bring-up
Gemma 4 E4B (cached on box), bf16 weights, NO KV quantization,
SGLANG_FLASHINFER_VOSPLIT=1 handling the D=512 global layers.
Gates:
- coherent temp-0 output;
- measured per-layer geometry logged from SERVING DISPATCH (heads /
  kv-heads / page-size per attention group) - not config-file values;
  vLLM's crash proved they differ (config said 16 kv heads; dispatch
  ran group 8 at page 32);
- binary-md5 + resolved-path proof lines per your packet.
CAUTION: vLLM's equivalent config crashes at warmup with
"Unsupported max_mma_kv: 0" (page_size=32, group-8 dispatch). If SGLang
hits the same wall: set FLASHINFER_PREFILL_DEBUG_ONCE=1, capture the
printer dump, and STOP - that feeds the shared dispatcher fix (Claude,
task 17). A captured crash is a valuable result, not a failure.

## Objective 2 - rung 1: NVFP4 KV
Same model, --kv-cache-dtype fp4_e2m1, FULL K+V (linear SFs are your
native layout). Mixed-KV fallback only if full-NVFP4 is blocked - and
document why. Gates: coherence; capacity row vs fp8 comparator
(sequential servers); short PPL pair - exactly your Gemma 3 rung 1
evidence pattern.

## PARKED - do not work on
- mixed-KV CUDA-graph gate: waits on FlashInfer split-dtype module
  keying (Claude, task 22; your d4b55f8 stop note is the spec);
- the max_mma_kv dispatcher fix (Claude, task 17, in progress);
- anything touching prefill.cuh kernel math.

## Protocols
- Marker handshake unchanged: /home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN
  present = yield; Claude takes occasional ~15-min probe windows.
- Memory guardrails: one server at a time; --mem-fraction-static per
  your Gemma 3 greens; --memory 100g cgroups; sequential comparators.
- Evidence: every claim has an artifact; every red a verbatim error;
  every binary a provenance line (md5 vs blessed build); every stop
  point a clean tree and a pushed summary.
- If blocked >2 attempts on the same error: stop and report. Clean stop
  notes get unblocked within one rotation; improvised workarounds have
  cost this campaign days.
