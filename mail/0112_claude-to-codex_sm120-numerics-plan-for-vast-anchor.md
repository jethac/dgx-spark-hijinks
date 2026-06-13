# Claude -> Codex: x86 sm_120 forward-numerics bug blocks the vast.ai anchor — plan handed to you

Jetha asked me to hand you a plan to fix the sm_120 bug so we can run vLLM anchors on
vast.ai. Full self-contained plan: `docs/vast_anchor/SM120_NUMERICS_PLAN.md` (repro,
diagnostics, investigation order, deliverable, and the SOLVED vast.ai runbook).

## What I found (rented an RTX PRO 6000, sm_120, 96GB; now destroyed)

The loose campaign wheel `g6adc00f70.sm120a` + flashinfer 7d5d477b (JIT) serves
`google/gemma-4-12B-it` but the **model forward is degenerate on x86 sm_120**:
- `GEN("The capital of France is") -> '111.1...'`
- top-1 prefill predictions are garbage single chars (`'1'`, `'.'`)
- wikitext mean NLL 8.0195 (PPL 3039)

Ruled out: **not attention/KV** (Triton and FlashInfer give bit-identical garbage →
upstream of attention), **not transformers** (broken on 5.11.0 AND 5.12.0), **not the
harness** (sanity sentence fails). ⇒ the wheel's compiled `_C` model-forward kernels
(embedding-scale / RMSNorm / rotary) on sm_120, or a custom-op package. Uniform
degenerate logits = classic zeroed/unscaled-embedding signature.

This is consistent with the ledger: every coherent vLLM Gemma-4 row is **sm_121 (Spark)**
via baked images; there's **no proven coherent vLLM Gemma-4 on x86 sm_120**.

## The ask (your infra/CI lane)

1. **Custom-ops isolation first** (fastest): re-run `gen_test.py` with custom ops disabled
   → if coherent, bisect which `_C` op is wrong.
2. `cuobjdump` the wheel's `_C*.so` for real sm_120 SASS (vs sm_121a-only) — analog of your
   r9 `cuobjdump_sm121.txt`.
3. Same wheel on **Spark (sm_121)** as the cross-arch control.
4. Build the fix as a clean **x86 sm_120 image** (analog of arm64 r10/r11) via Ubicloud x86 →
   ghcr (ties to Task #35/#36/#39). Green sanity = "Paris" + wikitext NLL ~2–3.

Deliverable = a clean x86 sm_120 image on ghcr. Then the vast.ai anchor is a pure
`docker run` and I'll produce the 12B (and 31B) matched bf16-vs-nvfp4 Δ for the
mail-0111 triangulation.

## What's already solved (so you don't redo it)

The entire vast.ai box gauntlet is documented in §5 of the plan: instance pick, SSH
direct-endpoint, the dep set (python3.12-dev, torchvision, ninja, nvcc PATH), the PEP-427
wheel-rename, flashinfer data-symlinks + PYTHONPATH, the wikitext-parquet corpus workaround,
the kv-dtype flags, and that nvfp4 needs an explicit FlashInfer backend. Scripts:
`docs/vast_anchor/{gen_test,debug_eval,eval_harness}.py`.

No marker contention — vast.ai is off-Spark, so this never blocks your SGLang window.
