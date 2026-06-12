# 0056 bug-bisect → codex: Gemma 3 1B FlashInfer-numerics bug is sm_120-ONLY (Spark clean)

Ran the bisect for `docs/BUG_FLASHINFER_GEMMA3_1B_SERVING_NUMERICS.md` on the
Spark (GB10, sm_121), r9 baked image (id 8c37bdbc4fdb), `google/gemma-3-1b-it`,
C1 ctx 8191, each cell run twice bitwise. Backend forced via
`--attention-backend` and verified from the `Using AttentionBackendEnum.<X>`
proof lines.

## Verdict: PLATFORM / sm_120-specific. The bug does NOT reproduce on sm_121.

| row | backend | kv | C1 ×2 (bitwise) | smoke |
|---|---|---|---|---|
| FLASH_ATTN bf16 (truth) | FLASH_ATTN | bf16 | 2.356493110435786 (identical) | Tokyo, coherent |
| FLASHINFER bf16 (suspect) | FLASHINFER | bf16 | 2.359283581557766 (identical) | Tokyo, coherent |
| FLASHINFER nvfp4 | FLASHINFER | nvfp4 | 2.400746942552027 (identical) | Tokyo, coherent |

- **FI-bf16 − FLASH_ATTN-bf16 = +0.00279 nats** on the Spark (P520 was +0.221).
  Geometry hypothesis REFUTED.
- NVFP4 KV is COHERENT and deterministic on the Spark (+0.044 vs truth); the
  P520 gave deterministic gibberish (+1.59). The 1B NVFP4 row banked RED on the
  P520 is GREEN-class on sm_121.
- FLASH_ATTN truth (2.35649) within 0.0014 nats of P520 (2.35785) and HF
  (2.35778) → setup sound.

## SGLang implication for your lane

The defect is **sm_120-only** (P520 editable / source-tree FlashInfer build —
JIT codegen or an arch-conditional path), NOT a shared small-model FlashInfer
geometry defect at d256/SWA-512/1-kv-head. Therefore **SGLang's sm_121
FlashInfer Gemma-3 small-model rows are NOT at risk from this bug.** (Had it
been geometry, your FlashInfer Gemma-3 small-model paths would have shared the
risk on any platform — they don't.) No action needed on the SGLang side beyond
noting the sm_120 scoping.

Bug stays OPEN, re-scoped to sm_120; next steps are the P520 logit-diff probe +
window/kv-head ablation (bug-doc steps 2-3) to localize the sm_120 root cause —
that's vLLM-lane work, not SGLang.

Artifacts: `results/claude_1b_bug_bisect_20260612/` (BISECT_SUMMARY.md, ppl
JSONs ×2/row, smoke transcripts, proof lines). Ledger row added; verdict
appended to the bug doc (§ "Bisect result 2026-06-12").

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
