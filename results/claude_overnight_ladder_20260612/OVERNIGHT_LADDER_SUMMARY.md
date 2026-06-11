# Overnight vLLM ladder — Spark block, 2026-06-12 (finisher wrap-up)

Runner completed (`LADDER_DONE TOTAL_WALL=2884` + e4bafter bench); the driving
agent died before wrap-up, so this summary was banked by the finisher.
Marker `~/CLAUDE_WINDOW_OPEN` cleared and verified at 2026-06-12 ~01:30 JST
(2026-06-11T16:30Z); `docker ps` empty; box handed to Codex.

Window: `R=/home/jethac/spark_tmp/claude_overnight_ladder_20260612`, baked r9
image `jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9`
(id `8c37bdbc4fdb`), util 0.72, ctx 8191, `--language-model-only`, corpora
C1/C2/C3 md5-verified (abb63f0e / 1686a33b / 28dfeba9). Zero-bug gates per
`docs/OVERNIGHT_LADDER_PLAN_20260612.md` Amendment 1: C1 double-run bitwise,
|delta| > 0.5 nats vs bf16 = RED, verbatim smoke transcripts, proof lines.

## Row table (all PPL = mean nats/token, ctx 8191)

| row | attn route (proof) | KV dtype | KV capacity (tok) | C1 mean x2 | C1 determinism | C2 | C3 | delta vs bf16 (C1/C2/C3) | smoke | verdict |
|---|---|---|---:|---:|---|---:|---:|---|---|---|
| g312b_bf16 | FLASH_ATTN (homog. heads, default) | auto | 335,944 | 1.991684 | IDENTICAL (bitwise) | 2.215285 | 1.155823 | baseline | coherent (Tokyo) | GREEN |
| g312b_nvfp4 | FLASHINFER, linear V-SF | nvfp4 | 1,072,364 | 2.065823 | IDENTICAL | 2.166118 | 0.992964 | +0.0741 / -0.0492 / -0.1629 | coherent | GREEN |
| g312b_fp8 | FLASHINFER | fp8_e4m3 | 618,254 | 2.032127 | IDENTICAL | 2.213406 | 1.094120 | +0.0404 / -0.0019 / -0.0617 | coherent | GREEN |
| g412b_bf16 | n/a — never reached backend selection | auto | n/a | n/a | n/a | n/a | n/a | n/a | n/a | **RED (failed to serve)** |
| g412b_nvfp4 | n/a | nvfp4 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | **RED (failed to serve)** |
| g412b_fp8 | n/a | fp8 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | **RED (failed to serve)** |
| g426b_bf16 | FLASHINFER FA2 VO-split (auto KV), knob `VLLM_FLASHINFER_VOSPLIT` | auto | 370,652 | 3.255458 | IDENTICAL | 6.632085 | 2.766721 | baseline | coherent | GREEN |
| g426b_nvfp4 | FLASHINFER NVFP4 VO-split, linear V-SF | nvfp4 | 1,321,656 | 3.384581 | IDENTICAL | 6.804527 | 2.888919 | +0.1291 / +0.1724 / +0.1222 | coherent | GREEN |
| g426b_fp8 | **TRITON_ATTN** (forced: heterogeneous heads, no vosplit knob) | fp8_e4m3 | 756,092 | 3.257858 | IDENTICAL | 6.785276 | 2.869311 | +0.0024 / +0.1532 / +0.1026 | coherent | GREEN (Triton-route row, see note) |
| e4bafter | FLASHINFER NVFP4 VO-split | nvfp4 | (speed row) | n/a | n/a | n/a | n/a | n/a | coherent | GREEN (bench banked) |

All quantized deltas are far inside the 0.5-nat band. The smoke-json
`ok:false` on the Tokyo cells is the harness's strict-string-match flag; the
banked transcripts are verbatim coherent ("The capital of Japan is
**Tokyo**.") and the driver verdict is `SMOKE_TOKYO=COHERENT` in status.txt.

### Capacity ratios

| model | nvfp4/bf16 | nvfp4/fp8 | fp8/bf16 |
|---|---:|---:|---:|
| Gemma 3 12B-it | 3.19x | 1.73x | 1.84x |
| Gemma 4 26B-A4B-it | 3.57x | 1.75x | 2.04x |

### RED rows: gemma-4-12B-it (all three dtypes), verbatim cause

`SERVER_DID_NOT_BECOME_READY` in ~16 s, identical for bf16/nvfp4/fp8:

> pydantic_core._pydantic_core.ValidationError: 1 validation error for
> ModelConfig — Value error, The checkpoint you are trying to load has model
> type `gemma4_unified` but Transformers does not recognize this
> architecture. This could be because of an issue with the checkpoint, or
> because your version of Transformers is out of date.

The r9 image's Transformers does not know the G4 12B dense `gemma4_unified`
arch (the 26B-A4B MoE arch IS known — it served green). The crash occurs in
`create_model_config()`, BEFORE engine init and before any attention-backend
selection; it is a packaging gap, not an attention-route failure. The
"zero-code-ready per task 18" assumption for G4 12B was wrong at the image
level. Crash excerpts banked per dtype
(`results/claude_g412b_*_crash_excerpt.txt`).

## REVERT REQUIRED for e2-vllm@20196b5946 (revert rule, Amendment 3)

The rule is mechanical: "If ANY overnight bf16 row goes RED, the default
reverts before morning and the red ships as the reason." **g412b_bf16 is
RED** (failed to serve), so the text-flip default on e2-vllm@20196b5946 must
revert until the Gemma 4 12B dense rows run green; this red ships as the
reason. NOT performed by the finisher — revert is the owner's call.

Adjudication context for the owner: the red is a Transformers
arch-resolution crash that precedes backend selection, is identical across
all three KV dtypes, and occurred on the r9 image (vLLM `9759e3b06`, which
does not contain the flip commit). It is therefore not evidence of a
FlashInfer-route regression; it is a missing validation cell — the G4 12B
dense size was never exercised on the retirement path tonight. Both bf16
rows that did run (G3 12B, G4 26B-A4B) are green under the full zero-bug
gates.

Triton-retirement reads from tonight (scorecard `docs/TRITON_RETIREMENT_SCORECARD.md`):
- R5 provenance: both FlashInfer bf16/nvfp4 rows show zero Triton ATTENTION
  dispatch ("...instead of TRITON_ATTN"). The `TRITON Unquantized MoE`
  lines on 26B rows are the MoE GEMM backend, not attention — expected.
- I2 refinement: g426b_fp8 was force-routed to TRITON_ATTN and served fp8
  KV with green quality (C1 +0.0024) and 2.04x capacity. So "Triton cannot
  read quantized KV at all" overstates: in this build the Triton route DOES
  handle fp8 KV. The FlashInfer-only capability claim holds for NVFP4
  (3.19x/3.57x) and for VO-split head-512 configs; scorecard I2 wording
  should be narrowed accordingly.
- G3 12B bf16 selected FLASH_ATTN (homogeneous heads), so that row
  validates the default selector, not the FlashInfer retirement path
  specifically.

## E4B speed AFTER (nvfp4 + VO-split) vs Triton bf16 baseline

Baseline: `results/claude_blockE23_20260611/BLOCKE23_SUMMARY.md` (E3a,
Triton force, bf16 KV, medians of 3, same harness/params, temp 0, nonces).
AFTER: `claude_e4bafter_benchmark.json`, run_id `claude_e4bafter_nvfp4_vosplit`,
schema blockE23-benchmark/v2, warmup ok.

| metric | Triton bf16 BEFORE | FlashInfer nvfp4+vosplit AFTER | ratio AFTER/BEFORE |
|---|---:|---:|---:|
| decode tok/s (256 tok, batch 1, median) | 19.03 | 18.40 | 0.967x |
| TTFT (2083-tok prompt, median) | 0.317 s | 0.346 s | 1.091x (slower) |
| prefill tok/s (2083-tok prompt) | 6570 | 6019 | 0.916x |
| concurrent x4 aggregate decode tok/s | 92.04 | 89.61 | 0.974x |

Read: this is NOT the I1 same-precision speed pair — it pairs bf16-Triton
against NVFP4-KV-FlashInfer (different KV dtype). The ~3% decode / ~9% TTFT
cost buys the NVFP4 capability Triton does not have at all for this config
(3.57x KV capacity on the 26B measurement; VO-split head-512 serving). The
clean bf16-FI vs bf16-Triton speed pair remains the morning scorecard cell
(Amendment 3 open box / scorecard I1).

## Token-strata analysis (step-6 check)

`results/claude_token_strat_20260612/STRATA_SUMMARY.md` already exists
(banked by the prior agent) — determinism gate CLEAN (REPRO=EXACT x9),
six-pair hypothesis verdicts: fp8-vs-bf16 C1/C2/C3 all **H-hard** (benign
direction); nvfp4-vs-bf16 C1 and C3 **H-hard**; the priority pair
nvfp4-vs-bf16 **C2 (+0.2526 prose inversion) = H-broad with secondary
H-late, explicitly NOT H-tail** (diffuse 56.4% worsening, positional growth
after ~3k tokens, hardest band still improves). No rerun needed.

## Dropped / not reached tonight

- G4 12B-it rows (all 3): RED as above; rerun needs an image/install whose
  Transformers knows `gemma4_unified`.
- Stretch items: mm-prefix overlay smoke, G4 12B base-checkpoint PPL pair —
  not attempted (runner ended at e4bafter per plan order).
- Token PPL dumps banked in `results/token_dumps/` (7.3 MB total, 20 files
  — small, kept in repo).

## Artifacts

- `status.txt` — full row/cell log incl. double-run means and walls.
- `results/claude_*_{ppl,stdout,stderr}*` — PPL cells (a/b doubles on C1).
- `results/claude_*_proof_lines.txt`, `*_server.log` — route/dtype proofs.
- `results/claude_*_smoke_{tokyo,sparkok}.json` — verbatim transcripts.
- `results/claude_g412b_*_crash_excerpt.txt` — verbatim RED evidence.
- `results/claude_e4bafter_benchmark.json` (+stdout) — AFTER speed row.
- `results/preflight_access_*.json`, `prefetch.log` — pre-flight gates.
