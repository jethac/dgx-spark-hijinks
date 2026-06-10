# Window packet — anchor trio: bf16 reference, fp8 rerun, E2 provenance test

Authored 2026-06-11. ~75 min. Completes the 31B flagship row AND tests the
E2 bf16-crash provenance hypothesis. Protocol/guardrails unchanged.

Common: r8 image (`...-r8`), overlay clone @ 9759e3b06 (Python-only + the two
fa2/fa3 symlinks; PRE-FLIGHT: `find clone/vllm -name '*.so' -not -type l` must
be empty, symlink realpaths must point into /opt/jethac-vllm, EXT_PATH proof
line mandatory), `-w /work`, `--language-model-only`, util 0.72, max-model-len
8192, ctx-8191 PPL (same corpus md5 abb63f0e), first-token gate + 3 decode reps.

## Row 0 — bf16 KV anchor (~20 min, weights cached)
`google/gemma-4-31B-it`, NO kv-cache-dtype flag (bf16), NO knobs. Gemma4Config
will force TRITON_ATTN (fine - this is the lossless QUALITY reference; backend
does not change exact-math quality). Record PPL/nats-per-token + KV tokens.

## Row 1' — fp8 comparator RERUN on clean overlay (~20 min)
`--kv-cache-dtype fp8_e4m3`, no knobs (Triton force expected). Identical to
the tainted row 1 except clean overlay. Compare to tainted 4.473945: if it
moves materially, the taint was real; if it reproduces, fp8's gap vs bf16 is
genuine (Gemma outlier-clipping hypothesis gains weight - note Gemma 4 QK-norm
status in the writeup either way).

## Row 2 stands: nvfp4 4.281335 nats/tok (03720c7). Ledger deltas to publish:
nvfp4-vs-bf16 and fp8-vs-bf16, plus capacity ratios all three ways.

## Bonus — E2 provenance test (~20 min, answers task #17's biggest question)
Gemma 4 **E4B** bf16, `VLLM_FLASHINFER_VOSPLIT=1`, NO kv-cache-dtype flag, on
r8 (our FlashInfer fb7d62ea baked in). The original E2 crash
("max_mma_kv: 0", prefill.cuh:2964) happened on the STOCK vllm-openai image's
FlashInfer; every probe that failed to reproduce it ran OUR branch. If this
serves: the mystery was FlashInfer version provenance, the Triton-retirement
lane unblocks, and the E3b AFTER benchmark (vs the banked 19.03 tok/s BEFORE)
can run in the same session if time permits (bench_e3.py pattern, 3 reps).
If it crashes identically on OUR FlashInfer: real dispatcher work remains -
capture verbatim, that narrows task #17 honestly.
