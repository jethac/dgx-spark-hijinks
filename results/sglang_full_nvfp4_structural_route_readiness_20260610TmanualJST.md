# SGLang full-NVFP4 structural route readiness, 2026-06-10

## Decision

Do not start the full NVFP4 K+V radix structural route in the mixed-KV blessing step.

The SGLang mixed FP8-K + NVFP4-V row is now claim-ready as the fallback path. Full
NVFP4 K+V remains a separate stretch item because it requires changing how SGLang handles
cached-prefix extend attention, not just documenting or tuning the current mixed-KV row.

## Current State

Completed and documented:

- interrupted/deep prefix sweep reconciled:
  `results/sglang_qwen_mixedkv_prefixcacheguard_deep_prefix_sweep_ctx8192_20260610TmanualJST.md`
- mixed-KV capacity denominator audited:
  `results/sglang_qwen_mixedkv_capacity_denominator_audit_20260610TmanualJST.md`
- claim-ready SGLang mixed-KV row published:
  `results/sglang_qwen_mixedkv_claim_manifest_20260610TmanualJST.md`
- upstream draft issues banked:
  - `results/upstream_draft_issue_sglang_prefix_cache_graph_write_20260610TmanualJST.md`
  - `results/upstream_draft_issue_flashinfer_head512_selector_overpromise_20260610TmanualJST.md`

Still open:

- full NVFP4 K+V under SGLang radix cache remains red/open;
- vLLM proves full NVFP4 K+V prefix reuse can work, so this is not closed as an inherent
  FP4-K limitation;
- SGLang's current ragged-suffix + paged-prefix merge path exposes FP4-K partial-state
  sensitivity that the mixed-KV fallback avoids by keeping K in fp8.

## Why Not Start It Here

The proposed full-NVFP4 fix is structural: reshape SGLang cached-prefix extend so the
cached prefix and suffix are handled in a vLLM-like all-paged path, avoiding the
ragged-suffix/paged-prefix partial-state merge that amplifies FP4-K LSE sensitivity.

That work should be isolated because it can affect:

- radix-cache serving behavior;
- generic extend scheduling;
- FlashInfer paged/ragged wrapper planning;
- CUDA graph capture behavior;
- non-FP4 KV modes if scoped poorly.

The mixed-KV row now has clear evidence, a guard policy, and a claim manifest. Starting
the structural route in the same step would blur the fallback claim with an unproven
full-NVFP4 implementation.

## Gate For Starting The Stretch

Start the stretch only on a fresh issue-named SGLang branch/worktree with these gates:

1. Keep the current mixed-KV claim artifacts unchanged as the baseline.
2. Add a minimal reproducer for the full-NVFP4 radix failure that does not depend on
   trace-only code paths.
3. Implement the vLLM-shaped all-paged cached-prefix extend path behind an FP4-KV/SM12x
   gate only.
4. Validate non-FP4 regressions first: fp8 radix row and bf16 radix row must remain green.
5. Validate full NVFP4 K+V radix: first-token gate, dense-cache comparison, matched fp8
   capacity, and short PPL.
6. Only then replace the current mixed-KV fallback with a full-NVFP4 claim.

## Status

Appropriate next action: stop this mixed-KV blessing packet here, ask for review, and
start the full-NVFP4 structural route as a separate gated implementation task if desired.
