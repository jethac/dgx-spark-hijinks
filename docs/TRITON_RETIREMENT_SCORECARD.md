# Triton retirement scorecard - regression proof + improvement quantification

Jetha (2026-06-12 ~02:00): "for such a major change we need to prove that
there are no regressions caused by retiring triton, and we also have to
prove how much of an improvement there is."

Design: PAIRED cells. Every claim cell pairs a Triton route run and a
FlashInfer route run, same model, same image where possible, same params,
zero-bug gates (double-run bitwise PPL, verbatim transcripts).
The r9 image serves BOTH routes for bf16 (knobs select); mm + flip-default
cells use the post-merge e2-vllm head image/install.

## Regression criteria (ALL must hold per size+modality, else REVERT default)

- R1 quality: bf16-FlashInfer PPL <= bf16-Triton + 0.05 nats on C1 (C2/C3
  where run). FI better is fine and expected (31B precedent: 4.6132 vs
  4.6532). Each cell double-run bitwise.
- R2 coherence: temp-0 chat transcripts on both routes, banked; no
  degradation.
- R3 multimodal: image-grounded equivalence (P520 protocol) on 4B/E4B;
  Spark 12B/26B/31B mm cells from the post-merge image.
- R4 scope: one serving spot-check that a non-flipped config (fp8 KV
  explicit) routes identically pre/post flip (selection tests cover the
  matrix statically; this is the live sanity cell).
- R5 provenance: FI rows show zero Triton mentions; Triton comparator rows
  show zero FlashInfer attention dispatch. Proof lines banked.

## Improvement quantification

- I1 speed pairs per size (decode tok/s, TTFT, x4 concurrent), Triton vs
  FI identical params: Spark for E4B (vs banked 19.03/0.317s/92.04),
  12B G4, 26B, 31B; P520-local pairs for 1B/4B/E2B (paired on-box, not
  cross-box).
- I2 capability unlock (argued + measured, NARROWED 2026-06-12 after the
  overnight ladder): Triton CAN serve fp8 KV (26B fp8 row green at 2.04x on
  the forced-Triton route) but CANNOT read NVFP4/packed-FP4 nor serve the
  D512 VO-split path. The NVFP4 rows (3.19x-3.57x capacity) are conditional
  on the FlashInfer route; fp8 rows are not. State it exactly that way.
- I3 the 31B Triton-tax adjudication: fresh bf16-Triton C1 cell to confirm
  or retire the suspect 4.6532 (vs FI 4.6132 bitwise reproduction).
- I4 mm speed pair on at least one size per family (4B G3, E4B G4 on P520;
  one Spark size in the morning block).

## Cell assignment

- P520 small-size chain (1B/4B/E2B/E4B): adds bf16-Triton comparator cell +
  on-box speed pair per size; mm pairs per the Amendment-4 smoke protocol.
- Spark morning scorecard block (after Codex's SGLang block; marker
  protocol, write-first): E4B/12B/26B/31B Triton comparator cells (PPL C1
  double-run + speed), I3, one mm Spark cell, R4 spot-check. Runs on r9 for
  bf16 pairs + post-merge-head image for flip/mm cells (image build spec in
  TRITON_RETIREMENT_NOTES 6/8/9).

Verdict rule: scorecard PASS = R1-R5 all green across sizes -> retirement
stands, improvement table published. Any red -> revert the relevant flip
commit (text: 20196b5946 revert rule banked; mm: pre-merge branch simply
not merged), red ships with verbatim evidence.

## Adjudication log

- 2026-06-12 ~03:00 (Claude): Amendment-3 revert rule TRIPPED by g412b_bf16
  RED, adjudicated NO REVERT. Evidence: pydantic ModelConfig crash from
  Transformers not knowing `gemma4_unified`, BEFORE backend selection,
  byte-identical across bf16/nvfp4/fp8 (incl. the Triton-routed fp8 cell) on
  an image (r9) that predates the flip commit. Not route evidence. Both bf16
  rows that served are fully green. CONSEQUENCE: G4-12B is a NAMED OPEN BOX -
  retirement is NOT claimed for that size until a Transformers-bumped image
  (r10 spec: r9 recipe + transformers >= the version knowing gemma4_unified;
  same provenance gates) serves its paired cells green. Jetha may overrule
  the no-revert call in the morning.
- 2026-06-12 06:02 (Claude, Spark morning scorecard block): paired cells run,
  full table in `results/claude_retirement_scorecard_20260612/SCORECARD_SUMMARY.md`.
  R1 PASS all five paired sizes (31B/E4B/26B + labeled 12B overlay pair + G3
  12B stretch; max delta +0.028 nats, 31B FI BETTER by 0.040); R2 PASS; R4
  route PASS; R5 PASS. I3 ADJUDICATED: suspect 4.6532 CONFIRMED bitwise on
  r9-baked (4.65317496471429 x2) — the 31B Triton tax is real. I1: speed
  parity (decode 0.97-1.005x), TTFT/prefill consistently better on FI; NOT a
  speed win, stated plainly. E4B bf16-FI: FIRST GREEN serving row on vLLM.
  R3/I4 mm: skipped (no post-merge image existed; P520 + r10 scope).
  VERDICT: NO REVERT REQUIRED from this block. New loud finding: fp8-Triton
  C1 is request-order-dependent (score-first 4.4739 bitwise == banked;
  smokes-first 4.5915) — fp8/nvfp4 rows need order provenance; bf16
  unaffected.
- 2026-06-12 08:1x (Claude, order-control + r10 window): G4-12B OPEN BOX
  CLOSED. r10 exists per spec (`aed0da3f96b2` = id-pinned r9 + transformers
  5.11.0 baked, builder `scripts/build_vllm_gemma4_rebuiltc_r10_image.sh`,
  all r9-equivalent provenance gates green) and serves the paired cells
  green end-to-end: Tri 3.4373001938921166 x2 bitwise vs FI
  3.464887691589146 x2 bitwise (+0.0276, R1 PASS), both routes coherent,
  R5 proof clean, speed parity (0.99x decode, TTFT -9% FI) — every value
  bitwise-reproduces the labeled dep-overlay pair, so the overlay is
  validated and G4-12B joins the retirement claim set. Bonus: first 12B
  quantized cell (nvfp4 VO-split, 3.53x capacity, C1 3.6834130552987028 x2
  bitwise, score-first cold provenance, +0.246 vs bf16 — in band, largest
  nvfp4 delta in the family). SAME WINDOW, order-control matrix
  (`results/claude_order_control_20260612/ORDER_CONTROL_SUMMARY.md`): the
  06:02 entry's "request-order-dependent" interpretation is SUPERSEDED —
  fp8 C1 is per-boot BISTABLE (today's cold/warmed A/B inverted yesterday's
  exactly); fp8 rows need boot-profile provenance (A=4.4739/B=4.5915 at
  31B), not just order; bf16 (6 bitwise repins) and nvfp4 (3 boots, both
  orders, one bitwise profile) are unaffected, so no retirement criterion
  moves.
