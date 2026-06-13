# Goal — Codex (SGLang + infra lane)

**Mission:** clear the SGLang NVFP4-KV quality reds, complete the Gemma-4 AR ladder, and wire the
multimodal bidirectional path — coordinating with Claude (vLLM/FlashInfer lane) on the +0.40 red.

## Context (what's already settled — don't re-derive)
- NVFP4 KV **format/default is near-lossless** (Claude e2e + reference sim, text+image+audio).
  Capacity 3.5556×. SGLang multimodal serving is green-scoped (E4B + 12B, mail 0133/0134/0135).
- The **global-scale "≈2× too small" root cause is RETRACTED** (Claude mail 0132). Do not pursue
  `calculate_kv_scales`/global-scale calibration as *the* fix — it isn't (your multiplier only got
  +0.403 → +0.344, and on vLLM `calculate_kv_scales` *breaks* nvfp4).
- The +0.40 12B text red is **general, not SGLang-radix-specific** (Claude mail 0138).
  vLLM reproduces it at the same ctx-8185 shape (`+0.4215` single prefill), while chunked/reuse
  paged+ragged merge is smaller (`+0.1906`). Do **not** rewrite SGLang radix / partial-state
  merge for this red; wait for Claude's FlashInfer/numerics fix, then rerun the matched SGLang
  12B row.

## Tasks (priority order)
1. **The +0.40 12B text red — FlashInfer/numerics lane blocker.** Claude owns the mechanism/fix.
   Once it lands, apply the FlashInfer update and verify with a matched bf16-vs-NVFP4 SGLang
   12B row at the failing shape (ctx 8185, prefix 4096). Keep the current SGLang row scoped red
   until then.
2. **Complete the SGLang Gemma-4 AR ladder (Task #40, the ship gate):** matched bf16-vs-NVFP4
   12B / 26B-A4B / 31B once the quality red clears. Matched-claim rule: one image/corpus/shape per
   pair, flip only the KV dtype.
3. **Clear the E4B fp8 comparator red.**
4. **Multimodal bidirectional path:** E4B full-NVFP4 is green on a baked SGLang package image
   carrying `jethac/sglang@f920e2d88a` (mail 0139). Extend/reuse this path for remaining Gemma 4
   rows as needed, preserving scope labels until matched quality/capacity gates are complete.

## When/how to communicate with Claude
Channel: committed `mail/NNNN_codex-to-claude_subject.md`. **Before writing, run
`git ls-tree -r --name-only origin/epoch2 -- mail/` and use the next FREE number** (we collided on
0132 — never reuse). Poll incoming mail **at the start of every work session AND after banking any
cross-lane result.**

**Mail Claude the moment you:**
- Apply a candidate fix and have a result for Claude's vLLM cross-check.
- Hit a red whose root cause points at **FlashInfer or vLLM** (Claude's lane).
- Land a ladder rung or a quality claim becomes ready (Claude folds it into the ship-gate/blog).
- Need the FlashInfer mm-prefix support, or anything Claude owns.
- **Refute/retract any prior claim** — promptly and visibly.

**Do NOT mail:** routine same-lane serving smokes/progress, or speculative direction into Claude's
active GPU window — wait for a confirmed/banked result before sending direction (Jetha's rule).

## Discipline (campaign rules)
Every claim an artifact, every red a verbatim error, every binary its provenance. Don't conclude a
root cause without serving the real path. Stop point = clean tree + pushed summary, marker absent.
