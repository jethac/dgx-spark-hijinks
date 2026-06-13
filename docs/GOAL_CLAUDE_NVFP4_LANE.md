# Goal — Claude (vLLM + FlashInfer lane)

**Mission:** close the last NVFP4-KV quality blocker from the vLLM/FlashInfer side, and deliver
the FlashInfer pieces the SGLang lane needs. Coordinate tightly with Codex (SGLang lane) via mail.

## Context (what's already settled — don't re-derive)
- NVFP4 KV **format/default is near-lossless** across text + image + audio, confirmed e2e on vLLM
  (base 12B default +0.0073, -it +0.032 @ ctx 4096) and reference sim. Capacity 3.5556× exact.
- The **global-scale "≈2× too small" root cause was WRONG and is retracted** (mail 0132).
  `--calculate-kv-scales` *breaks* nvfp4 (NLL 16.5) — never enable it. Codex's scale-multiplier
  only nudged the red +0.403 → +0.344 (partial, not a fix).
- mm-prefix nvfp4 serves on vLLM via `VLLM_FLASHINFER_MM_PREFIX=1` (a flag, not a wheel limit).
- Full state: `docs/NVFP4_FORMAT_VS_KERNEL_GEMMA4.md`. Serving runbook: `docs/vast_anchor/SM120_NUMERICS_PLAN.md §5`.

## Tasks (priority order)
1. **Root-cause the +0.40 long-context red — the one remaining blocker.** It does NOT reproduce
   at ctx 4096 (near-lossless). Reproduce on vLLM at **ctx 8185 + 4096-prefix-reuse** vs
   ctx-4096-no-prefix; sweep prefix length. Decide: **does it reproduce on vLLM (general bug) or
   is vLLM clean (SGLang-radix/partial-state-merge specific)?** Then localize the mechanism
   (FP4-K partial-state-merge / long-ctx accumulation) and the fix. **TEST the real serving path
   before concluding — that is the whole lesson of the 0130 retraction.**
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
