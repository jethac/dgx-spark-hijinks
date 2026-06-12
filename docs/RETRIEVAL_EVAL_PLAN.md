# Retrieval eval plan: NVFP4 KV vs fp8 vs bf16 on Gemma 4 (task #38)

Date: 2026-06-12 JST
Status: harness READY (self-test green), awaiting a Spark window.
Lane: dgx-spark-hijinks / epoch2.

## Why now (the public commitment)

Jetha's tweet thread says: *"i've measured perplexity, not retrieval --
needle-in-a-haystack test is next."* The PPL sweeps measure the **average**
next-token quality across a context; they do not test whether a single fact
buried deep in a long context can be **retrieved exactly**. This window closes
that gap and makes the promise true.

It is not a generic "let's add retrieval" task. The per-token stratification
arm found that **NVFP4's prose NLL error grows mildly with position** -- the
exact signature (call it **H-late**) that could selectively nip *deep-context*
retrieval while leaving average PPL almost untouched. So the retrieval grid is
the discriminating experiment for a hypothesis the PPL numbers already raised.

## Hypothesis: H-late

If NVFP4 KV degrades late-position information more than bf16/fp8, then NVFP4
retrieval accuracy should fall off specifically in the **high context_len AND
deep-position** corner of the grid, while bf16/fp8 stay flat. The PPL deltas are
small and mixed-sign, so this is genuinely open.

- **Null result** (NVFP4 grid matches bf16/fp8 everywhere, including the deep
  corner): "retrieval holds" -- this is the *strong* claim. The capacity win
  (NVFP4 KV ~3.5x vs bf16) costs nothing measurable for needle retrieval.
- **Positive result** (NVFP4 drops in the deep/late corner only): we have found
  **the real depth limit of the capacity win** -- an honest, specific caveat
  ("NVFP4 KV is safe up to depth/length X; beyond that, retrieval degrades").
  That is a better story than a vague "it's fine," because it is falsifiable and
  it is exactly what a careful reader of the stratification would predict.

Either way the verdict is a clean, defensible line for the blog and the tweet
follow-up. We do not pre-commit to a direction.

## The grid (context lengths x depths x dtypes)

Harness: `scripts/vllm_needle_retrieval.py` (schema `vllm-needle-retrieval/v1`).
Runner: `scripts/run_needle_retrieval_spark.sh` (STAGED; not executed here).

- **Context lengths** (tokens): 1024, 2048, 4096, 8192, 16384, 32768.
  Capped at the served `--max-model-len` via `--max-context-len`. The campaign's
  proven E4B PPL window is 8192; the runner serves at 32768 as a documented
  STRETCH and falls back to 8192 (set `SERVE_MAX_LEN=8192 MAX_CTX=8192`) if the
  server will not boot.
- **Depths** (needle position as context fraction): 0.0, 0.1, 0.25, 0.5, 0.75,
  0.9, 1.0. 0.0 = front, 1.0 = end. The H-late corner is depth >= 0.75 at the
  larger context lengths.
- **Dtypes / rows** (one server each, sequential, campaign knobs):
  - bf16  : no `--kv-cache-dtype`, `VLLM_FLASHINFER_VOSPLIT=1`.
  - fp8   : `--kv-cache-dtype fp8_e4m3`, no knob envs (forced TRITON_ATTN route).
            fp8 is **per-boot bistable** (order-control row) -- the JSON's
            `boot_profile_note` field records which profile this boot landed on.
  - nvfp4 : `--kv-cache-dtype nvfp4`, `VLLM_NVFP4_KV_VOSPLIT=1`
            `VLLM_NVFP4_KV_LINEAR_V_SF=1`.
- **Models**: `google/gemma-4-E4B-it` FIRST (cheap). `google/gemma-4-31B-it`
  is the D=512-global flagship the tweet asker asked about -- staged as a
  documented STRETCH (`RUN_31B=1`).

Single-needle grid = 6 x 7 = **42 cells per row** (the accuracy grid). A second
**RULER-style multi-needle mode** (`--mode multi`: K=7 facts, one per depth,
retrieve ALL with strict recall) runs once per context length per row as a
harder cross-check.

## Decisive cells

The verdict lives in the **bottom-right corner**: context_len in {16384, 32768}
crossed with depth in {0.75, 0.9, 1.0}. Read-out per row:

1. nvfp4 deep-corner accuracy ~= bf16/fp8 deep-corner accuracy -> **H-late
   refuted for retrieval**, null/strong claim.
2. nvfp4 deep-corner accuracy < bf16/fp8 while shallow cells match -> **H-late
   confirmed**, report the depth/length threshold as the capacity-win caveat.
3. All three drop together in the deep corner -> it is a *model/context-length*
   limit, not a KV-dtype effect; do not attribute it to NVFP4.

## How the verdict feeds the deliverables

- **Blog**: the quality story currently rests on PPL deltas. This adds a
  task-level retrieval grid -- the thing readers actually care about for KV
  quantization. A null result lets the blog say "capacity 3.5x, retrieval
  intact"; a positive result gives the blog its most honest and most memorable
  caveat.
- **Tweet promise**: directly fulfills "needle-in-a-haystack test is next" with
  a per-(length, depth, dtype) grid, not a single anecdote.

## Determinism + provenance

- Temperature 0 throughout; needle/code is a deterministic function of
  (seed, context_len, depth, k) -- same cell -> same verdict every run.
- Per-row **double-run determinism gate**: the smallest single-needle cell
  (ctx 1024, depth 0.5) is scored twice and the answer must match bit-for-bit
  or the row is `DET_FAIL` (mirrors the PPL ladder's C1 x2 gate).
- Filler is deterministic (a repo corpus slice, or a fixed synthetic
  Paul-Graham-style generator) tiled to length -- never random per run.
- `spark_hardware.collect_cuda_hardware()` provenance, container image, and the
  fp8 boot-profile note are recorded in every JSON.

## Self-test (offline, no GPU)

`scripts/vllm_needle_retrieval.py --self-test` runs the insertion + scoring +
grid logic against a MOCK server response and asserts: needle inserted at the
right token offset (front/mid/end), multi-needle offsets ascending and distinct,
scoring catches right/wrong/missing, RULER strict recall (all vs partial), grid
assembles correctly, codes deterministic. 21/21 checks green; banked at
`results/claude_needle_retrieval_selftest_20260612/self_test.json`.

## Exact Spark run command (future window)

```bash
# E4B only (default), 32768 stretch:
bash scripts/run_needle_retrieval_spark.sh
# If the server will not boot at 32768, fall back to the proven window:
SERVE_MAX_LEN=8192 MAX_CTX=8192 bash scripts/run_needle_retrieval_spark.sh
# Add the 31B flagship stretch when the window has budget:
RUN_31B=1 bash scripts/run_needle_retrieval_spark.sh
```

No claims land in the ledger until the grid exists and the determinism gate is
green for every row.
