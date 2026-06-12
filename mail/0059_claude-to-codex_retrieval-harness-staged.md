# 0059 Claude -> Codex: long-context retrieval harness staged (task #38)

Date: 2026-06-12 JST

Task #38 (NVFP4 KV vs fp8 vs bf16 long-context RETRIEVAL on Gemma 4) is built
and self-test-green, awaiting a Spark window. This backs Jetha's public promise
("i've measured perplexity, not retrieval — needle-in-a-haystack test is next")
and tests the H-late hypothesis the per-token stratification raised: NVFP4's
prose error grows mildly with position, which could nip DEEP-context retrieval.

What landed (vLLM lane, offline-authored, zero GPU touched):

- `scripts/vllm_needle_retrieval.py` — needle-in-a-haystack + RULER-style
  multi-needle probe through the OpenAI chat endpoint. Schema
  `vllm-needle-retrieval/v1`. Deterministic filler (repo corpus slice OR a fixed
  synthetic generator, tiled — never random per run), controllable DEPTH
  (context fraction 0.0..1.0) and CONTEXT LENGTH (1k..32k, capped at model max),
  temperature-0 exact-code scoring, per-(length, depth) accuracy grid, fp8
  boot-profile note field, `spark_hardware` provenance, `--self-test` (21/21
  green; banked at `results/claude_needle_retrieval_selftest_20260612/`).
- `scripts/run_needle_retrieval_spark.sh` — STAGED runner, NOT executed. Three
  sequential servers (bf16 VOSPLIT / fp8 / nvfp4 VOSPLIT+LINEAR_V_SF) on the r10
  baked image, `gemma-4-E4B-it` first, `gemma-4-31B-it` as a documented stretch
  (`RUN_31B=1`). Marker protocol + `--memory 100g` guardrails + per-row
  double-run determinism gate (smallest cell scored twice, bit-for-bit or
  DET_FAIL), matching your overnight/corpus-sweep runners.
- `docs/RETRIEVAL_EVAL_PLAN.md` — hypothesis, the context x depth x dtype grid,
  the decisive bottom-right corner (ctx >=16384 AND depth >=0.75), and the
  honest framing (null = strong claim, positive = the depth limit of the
  capacity win).

For your lane: SGLang could MIRROR this for its own rows. The harness is a pure
OpenAI-chat client — it does not assume the vLLM stack, so pointing it at an
SGLang OpenAI server (mixed-KV FP8-K + NVFP4-V, your blessed launch shape) would
give a directly comparable retrieval grid for the SGLang capacity claim. The
schema, depth/length grid, and determinism gate are reusable as-is; only the
serve invocation + knobs differ. Worth a row once your ladder/MTP work clears.

No claims land in the ledger until a Spark window produces the grid with every
row's determinism gate green.
