# Triton retirement scorecard — Spark morning block (2026-06-12)

Window: claimed 03:40 JST (write-first, two consecutive free checks, after
Codex's SGLang block), released 06:02 JST (markers ls-verified absent). Runner:
`run_retirement_scorecard.sh` + `run_scorecard_phase2.sh` (this dir), r9 baked
image `jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9`
(id 8c37bdbc4fdb), util 0.72, `--memory 100g`, one server at a time, ctx 8191,
`--language-model-only`, corpora md5-gated (c1 abb63f0e / c2 1686a33b /
c3 28dfeba9). Per docs/TRITON_RETIREMENT_SCORECARD.md (R1-R5 / I1-I4) and
docs/TRITON_RETIREMENT_NOTES.md §6: Triton comparator rows = NO campaign knobs
(upstream heterogeneous-head force); FlashInfer rows = `VLLM_FLASHINFER_VOSPLIT=1`
(Gemma 4) / `--attention-backend FLASHINFER` (Gemma 3). Every C1 cell run
TWICE (bitwise gate). Speed = bench_e3.py, params identical to the banked
E4B baseline (`results/claude_blockE23_20260611/`).

## THE TABLE (C1 mean nats/token, ctx 8191; speed = median of 3)

| size | Triton C1 (x2 bitwise) | FI C1 (x2 bitwise) | delta FI-Tri (nats) | R1 (band +0.05) | Tri decode tok/s | FI decode tok/s | decode ratio | Tri TTFT s | FI TTFT s | Tri x4 tok/s | FI x4 tok/s | coherence (both routes) |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| G4 31B | 4.65317496471429 | 4.613162683323541 | **-0.040012** | **PASS (FI better)** | 3.72 | 3.74 | 1.005x | 2.199 | 1.958 | 14.56 | 14.63 | both COHERENT, banked |
| G4 E4B | 2.9470637031470144 | 2.9510406338369775 | +0.003977 | PASS | 18.87 | 18.66 | 0.989x | 0.354 | 0.334 | 91.89 | 90.18 | both COHERENT, banked |
| G4 12B† | 3.4373001938921166 | 3.464887691589146 | +0.027587 | PASS† | 7.53 | 7.43 | 0.987x | 0.998 | 0.914 | 36.86 | 36.70 | both COHERENT, banked |
| G4 26B-A4B | 3.2462895786054022 | 3.255457864166645 | +0.009168 | PASS | 24.08 | 23.63 | 0.981x | 0.599 | 0.555 | 56.32 | 55.81 | both COHERENT, banked |
| G3 12B‡ | 1.991683653495193 (FLASH_ATTN) | 2.010738572891384 | +0.019055 | PASS | 7.59 | 7.37 | 0.972x | 0.763 | 0.763 | 37.48 | 34.16 | both COHERENT, banked |

† G4 12B rows are LABELED dep-overlay cells: r9's Transformers does not know
`gemma4_unified` (tonight's ladder RED, adjudicated NO-REVERT, named open
box), so both rows ran with an in-container `pip install --upgrade
transformers` (5.11.0, recorded in server logs). Identical stack on both
routes, so the PAIRED comparison is internally valid; the size's retirement
CLAIM still waits for the r10 image per the adjudication log. Bonus: vLLM
9759e3b06 + transformers 5.11.0 serves G4-12B green — de-risks the r10 spec.

‡ Gemma 3 12B has uniform-256 heads; its default route on r9 is FLASH_ATTN
(proof: "Using FLASH_ATTN attention backend out of potential backends
['FLASH_ATTN','FLASHINFER','TRITON_ATTN','FLEX_ATTENTION']"), NOT Triton. The
pair pins "what retirement routes away from" for Gemma 3 = upstream priority
order. Stretch cell, both routes green.

E4B cross-check vs the banked Triton baseline (r8-era source install,
19.03 / 0.317 s / 92.04): today's same-image Triton row reproduces it within
~1% (18.87 / 0.354 / 91.89) — baseline confirmed, and the FI pair is
same-image so the ratio above is confound-free.

## Verdicts per criterion

- **R1 quality: PASS, all five paired sizes.** Every FI cell within
  +0.028 nats of its comparator (band +0.05); at 31B FI is BETTER by 0.040.
  Every C1 cell double-run bitwise-identical.
- **R2 coherence: PASS.** Temp-0 chat transcripts banked verbatim for all 11
  servers (spark-ok + "The capital of Japan is **Tokyo**."), both routes, no
  degradation.
- **R3 multimodal: NOT RUN HERE (pending, assigned elsewhere).** No
  post-merge e2-vllm image existed on the box (checked `docker images` +
  mail/ before and after claim; newest vLLM image is r9), and the scorecard
  assigns mm cells to the P520 protocol + post-merge-image Spark rows. The
  one stretch mm Spark cell is therefore SKIPPED, not failed.
- **R4 scope: route identity PASS + a LOUD side-finding.** Fresh 31B
  fp8-KV server, no knobs: proof lines match the banked fp8 rows exactly
  ("Gemma4 model has heterogeneous head dimensions ... Forcing TRITON_ATTN
  backend", 2x "Using AttentionBackendEnum.TRITON_ATTN", zero FlashInfer
  attention dispatch) — the non-flipped config still routes to Triton,
  identical pre/post flip. HOWEVER the row's C1 value does NOT reproduce the
  banked number: see "fp8 drift" below.
- **R5 provenance: PASS.** All FI rows: TRITON_ATTN selection count = 0,
  FLASHINFER selection lines present (+ "FA2 VO split" proof on D512 rows).
  All Triton rows: FLASHINFER selection count = 0. Proof-line files banked
  per row. Noted exception (pre-existing, also present in the banked-green
  FI rows): the jit-monitor logs a Triton *utility* kernel compilation
  (`_compute_slot_mapping_kernel`) on FI rows — slot-mapping helper, not
  attention dispatch. Gemma 3 rows additionally log FLASH_ATTN "for vit
  attention" (mm-encoder side) on BOTH routes.

## Improvement quantification

- **I1 speed: parity, NOT a speed win — stated plainly.** Decode and x4
  ratios FI/Triton are 0.97-1.005x across sizes (within a few percent,
  FI marginally slower below 31B); TTFT is consistently BETTER on FI
  (-6% to -11%) and prefill +6-12%. Per the scorecard's own rule this is a
  finding, not a fail: on GB10 the upstream Triton fallback is NOT the ~9
  tok/s disaster of vllm#38887 (that row is an RTX 4090 number). The
  retirement case rests on I2 + I3 + determinism, not raw speed.
- **I2 capability unlock (narrowed wording per 2026-06-12 amendment):**
  Triton CAN serve fp8 KV (today's R4 row + ladder 26B fp8 row) but CANNOT
  read NVFP4/packed-FP4 nor run the D512 VO-split path. All NVFP4 rows
  (3.19x-3.57x KV capacity at bounded quality deltas) are conditional on the
  FlashInfer route. Stated as precondition, not double-counted as speed.
- **I3 Triton-tax adjudication: SUSPECT CONFIRMED, exhibit stands.** Fresh
  fresh-server double-run on the r9 BAKED image reproduces the r8+overlay
  "suspect" value BITWISE: 4.65317496471429. The 31B bf16 Triton-route
  quality deficit vs FlashInfer (+0.0400 nats) is real, reproducible across
  images, and is now a primary exhibit for the retirement filing. (Note the
  smaller sizes show the opposite sign at much smaller magnitude — the tax
  is size/geometry-dependent, only 31B crosses 0.03 nats.)
- **I4 mm speed pair:** not run here (same reason as R3).

## OVERALL VERDICT: NO REVERT REQUIRED from this block

All regression criteria testable in this window (R1, R2, R4-route, R5) are
GREEN across 31B / E4B / 26B (+ labeled 12B pair, + G3 stretch pair). The
known-risk row — E4B bf16-FlashInfer serving, never green on vLLM before —
is now GREEN end-to-end (served, bitwise PPL, coherent, benched). R3/I4
(multimodal) remain open boxes for the P520 protocol + r10 image rows, and
G4-12B's claim remains gated on r10 per the adjudication log. Within text
bf16 scope on CC 12.1, the Amendment-3/4 retirement stands on this evidence.

## Loud flags (deliverables, not embarrassments)

1. **fp8-Triton C1 cross-window drift (zero-bug investigation item).** The
   banked fp8 C1 4.473945385741097 was bitwise-stable across THREE windows
   and TWO images on 2026-06-11 (ppl_pair r8+overlay, anchor_matched
   r8+overlay, corpus-sweep r9-baked). Today the same r9 image, same
   template, returns 4.591455999476844 — internally bitwise across its own
   double-run, but 0.1175 nats off the banked value. bf16 rows do NOT show
   this (4.613162683323541 and 4.65317496471429 both repinned bitwise across
   windows today; 3.255457864166645 and 1.991683653495193 repinned vs
   tonight's ladder). Phase-2 probe (fresh server, corpus-sweep cell order:
   C1 before any smoke): **4.473945385741097 x2 bitwise — the banked value
   exactly.** So fp8-Triton C1 is REQUEST-ORDER-DEPENDENT, not random:
   score-first reproduces the banked number bitwise; two chat smokes before
   scoring shift it +0.1175 nats, then everything is bitwise-stable again.
   Suspected mechanism: first-request fp8 KV-scale calibration latching
   scales from whatever runs first (verify against the fork's fp8 path /
   `calculate_kv_scales`-class logic) — bf16 rows (no scales) repin bitwise
   across windows regardless of order. CONSEQUENCES: (a) every banked fp8
   PPL row must carry its request-order provenance — tonight's overnight
   ladder ran smokes BEFORE C1, so its fp8 cells (g312b 2.032127, g426b
   3.257858) are smoke-calibrated numbers, internally consistent but NOT
   comparable to score-first fp8 rows like the 4.4739 family; (b) the
   nvfp4 rows should get the same order-dependence probe before any
   cross-window fp4 quality claim is sharpened. Affects the fp8-KV row
   family's determinism story, NOT the bf16 retirement comparison (both
   bf16 routes are order-insensitive on today's evidence).
2. **Transient 31B serving crash, first row after window transition.** The
   first s31b_tri attempt died in torch.compile autotuning:
   `torch._inductor.exc.InductorError: RuntimeError: Failed to run autotuning
   code block: CUDA driver error: operation not permitted` (route selection
   itself was already correct: TRITON_SEL=2). Identical retry 100 min later
   served green. Crash excerpt banked
   (`results/claude_s31b_tri_crash_excerpt.txt`); watch for recurrence —
   if it correlates with window transitions, runners should auto-retry the
   first row.
3. **FI decode parity-minus at small sizes** (0.97-0.99x decode, 0.91x x4 on
   G3-12B): if Jetha wants the flip to be Pareto-dominant on speed too,
   that residual is the number to beat; TTFT/prefill already favor FI.

## Artifacts

Per-row in `results/`: server logs, proof-line files, smoke transcripts
(verbatim JSON), C1 a/b PPL JSONs + stdout/stderr, bench JSONs, crash
excerpts (s31b_tri r1). Status ledger: `status.txt` (phase 1 + phase 2).
Window discipline: `poll_log.txt` (claim at 03:40:18+09:00 after two free
checks; write-first noclobber + docker re-verify), `monitor_log.txt`.
