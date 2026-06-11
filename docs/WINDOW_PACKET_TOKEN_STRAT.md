# Window packet: token-level stratification capture (anomaly arm 3)

Status: READY — waiting on a Spark gap (Codex holds the window as of 2026-06-11 ~22:25 JST).
Owner: Claude. Task #25 remaining arm.

## Question

The corpus sweep established WHAT (fp8 general, nvfp4 corpus-dependent) and the
llama.cpp arm established the two-component model (small general effect +
~10x-larger stack-specific component). This window answers WHERE: do the
per-token NLL deltas concentrate in particular surprisal bands, positions, or a
catastrophic tail — especially for the nvfp4/C2 prose inversion (+0.253 nats)?

Hypotheses to discriminate:
- H-broad: deltas are diffuse (small shifts on most tokens) → numerical
  regularization story.
- H-tail: a small set of tokens with huge deltas dominates → quantization
  occasionally destroys specific attention reads (bug-shaped).
- H-late: deltas grow with position → KV accumulation effect.
- H-hard: deltas concentrate in high-surprisal tokens → quantization mostly
  perturbs already-uncertain predictions (benign).

## Plan (single window, ~3 server cycles, same budget as the corpus sweep)

1. Stage `R=/home/jethac/spark_tmp/claude_token_strat_20260612`: copy `docs/`
   corpora from the corpus-sweep window dir (md5s must match status.txt),
   copy current repo `scripts/` (sweep script now has `--dump-token-logprobs`).
2. Run `scripts/run_token_strat_capture.sh` — identical 3x3 matrix (r9 image,
   31B, util 0.72, ctx 8191, sequential servers, memory guardrails) with
   per-token dumps to `results/token_dumps/` (9 files, ~250KB each).
3. Determinism gate: every cell's mean_nll must equal the banked sweep value
   EXACTLY (table embedded in the runner; REPRO_FAIL poisons the window).
4. Pull `results/` back to the repo under `results/claude_token_strat_20260612/`.

## Offline analysis (no Spark needed)

`scripts/anomaly_token_strata_analyze.py` (self-test green) on the six
baseline/comparator pairs (fp8 vs bf16, nvfp4 vs bf16; C1/C2/C3):
surprisal bands, 8 position buckets, sign decomposition, top-40 |delta| tokens
with text context. Cross-check: recomputed mean delta must match the corpus
sweep deltas. Priority read-out: nvfp4/C2 (the inversion).

Follow-on (stretch, P520-compatible): if H-tail wins, replay the worst tokens'
contexts through the FlashInfer probe harness to localize the read path.
