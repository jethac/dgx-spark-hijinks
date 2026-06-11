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
- I2 capability unlock (argued + measured): Triton CANNOT read quantized
  KV at all; every fp8/nvfp4 row in the campaign (1.70x/2.8x/3.57x
  capacity, fp8-beats-bf16 quality) is conditional on the FlashInfer route.
  Retirement is the precondition, stated as such - not double-counted as a
  speed number.
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
