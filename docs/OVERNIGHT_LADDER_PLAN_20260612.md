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
