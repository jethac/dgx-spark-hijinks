# Overnight ladder plan - 2026-06-12 (Jetha directive, ~midnight JST)

Goal by morning: full Gemma 3 + Gemma 4 support and verification, ALL sizes,
both engines. Full use of P520 and Spark granted to both agents.

## Checkpoint policy

All rows use -it checkpoints (deployment target; consistent with every banked
row to date - we have ALWAYS tested -it). Checkpoint-independence spot-check:
ONE base-model PPL pair (gemma-4-12B base, nvfp4 vs bf16) IF time permits.
Multimodal: overnight rows are text-only (--language-model-only / r9 image);
the mm-prefix knob exists only on e2-vllm @ 7df3c67ec8 (Python-only diff), so
the mm serving smoke runs as a verified-symlink overlay window at the END of
the Claude block, time permitting. Full multimodal rows are day-after scope.

## Platform assignment

P520 (Claude, parallel to Spark, no contention):
- G3 1B vLLM 3 rows (bf16/fp8/nvfp4) - RUNNING (build phase)
- then G3 4B, G4 E2B, G4 E4B vLLM rows (reuse the WSL vLLM install)
- first serving-level CC 12.0 evidence, all sizes that fit 16GB

Spark Claude block (vLLM lane, ONE server at a time, util 0.72, r9 image):
- [in flight] token-strat capture (ends ~00:30-00:45)
- G3 12B-it: bf16 / nvfp4 / fp8 (capacity + smoke + C1/C2/C3 PPL, ctx 8191)
- G4 12B-it: same 3 rows (zero-code-ready per task 18)
- G4 26B-A4B-it: same 3 rows (KV rows on the MoE)
- G4 E4B speed AFTER row: nvfp4+vosplit serving benchmark, same params as the
  19.03 tok/s Triton baseline (results/claude_blockE23_20260611/)
- stretch: mm-prefix overlay smoke; G4 12B base-checkpoint PPL pair
- then CLEAR MARKER + mail handoff

Spark Codex block (SGLang lane, after Claude marker clears, est. ~04:00):
- E2B + 12B + 26B-A4B + 31B SGLang rows (E4B path generalizes; 31B = head-512
  vosplit serving, first SGLang flagship row)
- E4B fp8 comparator red root-cause (its known queue item)
- CUDA-graph integration gate
- DG-R2 only AFTER the Gemma ladder or in genuine gaps

Pre-flight for every block: hf_model_access_probe.py on exact model names
BEFORE claiming a window (a typo must not eat an overnight window); corpora
md5 check (c1 abb63f0e / c2 1686a33b / c3 28dfeba9); EXT_PATH + latch
provenance gates as always; row order bf16 -> nvfp4 -> fp8 per model so a
blown window still leaves the essential pair.

Parallel engineering (no GPU contention):
- Triton-retirement selector work for bf16 Gemma 4 configs on e2-vllm
  (agent-authored; resolves the banked selector-vs-kernel head-512 bug;
  Spark validation deferred to morning windows)

## Marker protocol amendment (after tonight's 23:45 race)

WRITE-THEN-VERIFY: claim by writing the marker FIRST, then check docker ps.
If you find the marker present but docker empty for >15 min, the holder
stalled - mail and self-clear. Mail numbering: ls mail/ immediately before
writing, use max+1 (two collisions already).

## Amendment 1 (Jetha, 00:0x): zero-bug bar

"We cannot afford even the most minor bugs." Operational meaning, every row:
- Determinism gate: C1 PPL cell run TWICE per server; bitwise-identical or the
  row is RED (no averaging, no shrugging).
- Sanity bands: bf16 absolute C1 PPL must be family-plausible; any quantized
  row with |delta| > 0.5 nats vs bf16 is RED pending investigation, never
  banked as a support claim.
- Smoke transcripts banked verbatim; incoherent output = RED row even if the
  server is green. RED rows ship with verbatim errors; they are deliverables,
  not embarrassments - what we cannot afford is a WRONG green.
- Engineering authored tonight (Triton-retirement selector, MTP) is NOT
  "support" until it passes serving validation; overnight it lands as
  validated-on-P520 code + a morning Spark validation spec, clearly labeled.
- No overlay shortcuts on claim rows: claim rows run on baked images or
  fully-verified installs; overlays only for explicitly-labeled smokes.

## Amendment 2 (Jetha, 00:0x): MTP drafters

Full support for MTP (multi-token-prediction) drafters in scope. Lane:
1. Recon: what MTP/drafter checkpoints exist for Gemma 3/4 (native MTP heads
   in checkpoints? released drafter models? EAGLE-style heads?), what vLLM and
   SGLang spec-decode infra expects, and where quantized KV touches it: the
   verify step is multi-token attention against the NVFP4 cache (our
   decode-as-prefill path generalizes - qo_len=k+1), drafter KV pool dtype,
   rejection-sampler interaction.
2. Enablement on the vLLM lane + P520 validation where it fits.
3. THE verification gate (fits the zero-bug bar): greedy spec decode must be
   OUTPUT-IDENTICAL to non-spec greedy at temp 0, per size, plus acceptance
   rate + speedup recorded. Identity failure = RED, full stop.
4. Spark serving rows interleave into morning windows.

## Amendment 3 (Jetha, 00:2x): retire Triton TONIGHT

Decision: yes. Structure that keeps the zero-bug bar honest:
- The overnight bf16 ladder rows ARE the retirement validation: every bf16
  row tonight (G3 12B / G4 12B / G4 26B on Spark; 1B/4B/E2B/E4B on P520; 31B
  already banked, bf16-FlashInfer 4.6132 BEATS bf16-Triton 4.6532) exercises
  exactly the path the retirement selector routes to, under the determinism
  and coherence gates.
- When the selector branch lands tonight (spark/hijinks-e2-triton-retire),
  Claude flips the default to ON for TEXT-ONLY Gemma configs on CC 12.x and
  re-runs the selection truth-table tests. If ANY overnight bf16 row goes
  RED, the default reverts before morning and the red ships as the reason.
- CARVE-OUT: multimodal configs keep upstream (Triton) routing unless
  VLLM_FLASHINFER_MM_PREFIX is set - Triton is the current mm-prefix-capable
  backend and our custom-mask replacement has not passed a serving smoke yet.
  mm retirement follows the mm smoke, not tonight.
- Open box for morning: the bf16 speed pair (E4B bf16-FlashInfer bench vs the
  19.03 tok/s Triton baseline, same params) - quality+coherence validate
  tonight, the speed datapoint completes the scorecard in a morning window.
