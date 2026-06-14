# Goal — Claude (vLLM + FlashInfer lane)

**Mission (phase 2 — localize & fix):** the +0.40 is reproduced and the lane is decided
(mail 0138). Now LOCALIZE the long-context NVFP4 accumulation mechanism and SHIP a FlashInfer-side
fix, plus the remaining FlashInfer pieces the SGLang lane needs. Coordinate with Codex via mail.

## Context (what's already settled — don't re-derive)
- NVFP4 KV **format/default is near-lossless** across text + image + audio, confirmed e2e on vLLM
  (base 12B default +0.0073, -it +0.032 @ ctx 4096) and reference sim. Capacity 3.5556× exact.
- **The +0.40 is REPRODUCED on vLLM and lane-decided** (`docs/NVFP4_LONGCTX_REPRO_VLLM.md`,
  mail 0138): single-prefill long-ctx +0.4215, chunked/reuse merge +0.1906, ctx-4096 +0.0359.
  → **general nvfp4 long-context accumulation, NOT SGLang-radix.** Radix/partial-state-merge is
  EXONERATED (it is the *better* path). VO-split is not a fixed tax. Global-scale retracted (0132);
  `--calculate-kv-scales` *breaks* nvfp4 — never enable it.
- mm-prefix nvfp4 serves on vLLM via `VLLM_FLASHINFER_MM_PREFIX=1`. Codex landed the SGLang-side
  FlashInfer image-prefix mask (mail 0137, source-overlay green).
- Repro env gotchas: full-vocab prompt_logprobs spikes ~8 GiB → `--gpu-memory-utilization 0.5`;
  12B nvfp4 needs BOTH `VLLM_NVFP4_KV_VOSPLIT=1` and `VLLM_NVFP4_KV_LINEAR_V_SF=1` (else Triton
  fallback rejects nvfp4). Full state: `docs/NVFP4_FORMAT_VS_KERNEL_GEMMA4.md`.

## Tasks (priority order)
1. **Localize & fix the long-context NVFP4 accumulation.** The lever: single-pass online-softmax
   over all KV (+0.4215) is worse than chunked partial-state renormalization (+0.1906). Find why
   the single-prefill path accumulates NVFP4 dequant noise worse (LSE / unnormalized sum / running
   max), and ship a FlashInfer-side change that brings long-ctx nvfp4 back toward near-lossless.
   Confirm on the real serving path at ctx 8185 (the 0130 lesson). **Also re-check whether the
   campaign's "12B intrinsically +0.39" headline (Task #25) is just this long-ctx effect** before
   it ships in the blog — 12B is near-lossless at ctx 4096.
2. **E4B fp8 D512/VO256 dispatcher (mail 0136).** My current patch *rejects* this shape; the real
   fix is to bias `CTA_TILE_Q→16` (→ `NUM_WARPS_Q=2`, making `NUM_MMA_KV=1` valid, q-tile smem
   65536→16384) for 1-byte-KV D512 on tight-smem archs, instead of rejecting. Build + test, then
   tell Codex so it can rerun the fp8 comparator.
2. **FlashInfer mm-prefix bidirectional masking (Task #31).** SGLang's image path falls back to
   causal (`Bidirectional attention for image tokens requires TritonAttnBackend`). Make the
   FlashInfer bidirectional mm-prefix support usable on SGLang's FlashInfer path; deliver the
   FlashInfer-side change + a serving smoke so Codex can wire it.
3. **FlashInfer dispatcher PR (Task #42).** Finalize `docs/flashinfer_pr/` for filing (rebase-check
   vs upstream main, polish). Filing itself is Jetha's call (outward-facing).

## When/how to communicate with Codex
Channel: committed `mail/NNNN_claude-to-codex_subject.md`. **Before writing, run
`python3 scripts/mail_next_number.py --sender claude --recipient codex --slug <subject>` or
`git ls-tree -r --name-only origin/epoch2 -- mail/` and use the next FREE number** (we collided on
0132 — never reuse). Poll incoming mail **at the start of every work session AND after banking any
cross-lane result.**

**Mail Codex the moment you:**
- **Determine whose lane the +0.40 bug is in** (general vs SGLang-radix) — this unblocks Codex's
  decision to fix its radix path vs wait for a general fix. This is the key handoff.
- Localize the mechanism or have a fix.
- Have the FlashInfer mm-prefix support ready for SGLang to wire.
- **Refute/retract any prior claim** — promptly and visibly (the 0130 lesson).
- Need a cross-check from the SGLang side.

**Do NOT mail:** routine same-lane progress, or speculative direction into Codex's active GPU
window (Jetha's rule — wait for a confirmed/banked result before sending direction).

## Discipline (campaign rules)
Every claim an artifact, every red a verbatim error, every binary its provenance. Never conclude a
root cause from a simulation without testing the real path. vast.ai: print $/hr before renting,
destroy boxes on bank, secrets env-only, log spend. Stop point = clean tree + pushed summary.
