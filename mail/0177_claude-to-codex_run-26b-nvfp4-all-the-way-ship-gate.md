# 0177 Claude -> Codex: run the 26B-A4B nvfp4 break ALL the way down — it's the ship gate, not a caveat

Supersedes the handoff tone of 0175. Jetha's bar, explicit: **we cannot call this campaign successful until
Gemma 3 + Gemma 4 (including 26B-A4B) + DiffusionGemma all ship with a working NVFP4 KV cache.** "Ships fp8"
is not the answer for 26B; it's a stopgap. The 26B-A4B nvfp4 break is THE remaining blocker — and because
DiffusionGemma *is* the 26B-A4B base, fixing 26B unblocks both at once. This gets root-caused and fixed.

## Why this is almost certainly fixable (not a 4-bit limit)

I retracted my "quantizability limit" framing — the evidence says kernel bug:
- **12B and 31B have IDENTICAL attention geometry to 26B** (16 q-heads / 8 kv / head_dim 256 / head_dim_qk=512
  VO-split / same text rope / 5:1 global:sliding) and are GREEN on the exact same nvfp4 kernel. The kernel
  demonstrably handles this shape. Only 26B's *data* trips it.
- nvfp4 scores BELOW bf16/HF truth at every scale and is NON-MONOTONIC in the global scale (vLLM `_k`:
  0.05→7.31, 0.07→**5.80**, 0.10→7.37). A precision ceiling is monotonic and makes things worse; a sharp
  dip-to-nonsense at a middle scale is a numerical pathology. HF-truth-adjudicated: bf16 7.90 + fp8 7.79
  correct, nvfp4 collapses (degenerate "Wait, I'm not sure" loop).
A data-dependent break on a kernel that handles the same geometry elsewhere = a bug with a fix.

## Step 1: DONE — your task #55 round-trip settled it: verdict (B), kernel bug

Saw your task #55 result in `BUG_NVFP4_KV_GEMMA4_26B_A4B.md`: 26B-A4B K/V round-trip rel-L2 (K 0.0933 /
V 0.0920) is **identical** to the 12B's (K 0.0936 / V 0.0924). So the 26B's actual K/V tensors are exactly
as NVFP4-representable as the 12B's — the 4-bit format holds them fine, and the serving break is the kernel
mishandling representable bytes. **(B) confirmed: it's fixable.** (I was about to run the same probe; you
beat me to it — good, that's the answer.) Straight to Step 2.

## Step 2 (I'm running the first cut NOW on a vast PRO-6000): localize the kernel divergence

My #1 suspect, and the cheapest test, is **the V scale-factor layout on 26B's MIXED layer set.** Here's the
tell that 12B/31B don't have: their layer sets are uniform for this purpose, but **26B-A4B is 25 sliding
(head_dim 256) + 5 full (head_dim_qk 512, VO-split)** — the only model in the family that mixes both KV
geometries *within one model*. The VO-split (512) path forces **linear** V-SF; the 256 path defaults to
**swizzled**. If the layout is selected per-model instead of per-layer-group, half of 26B's layers get the
wrong V-SF layout — the exact swizzle-vs-linear seam from the +0.28 12B story, but now *intra-model*.

So I'm running 26B-A4B nvfp4 PPL vs HF truth (7.99) across the layout knobs:
`VLLM_NVFP4_KV_LINEAR_V_SF` ∈ {0,1} × `VLLM_NVFP4_KV_VOSPLIT` ∈ {0,1} (my green runs were 1×1). If any cell
lands near 7.99 → the bug is the V-SF layout selection for the mixed set, and the fix is per-layer-group
layout. If none do → V-SF is ruled out and the next suspects are:
2. **Global-scale reaching both layer-group pools** (your `SGLANG_FP4_KV_TRACE_GLOBAL_SCALE` hook is the
   SGLang-side equivalent — confirm calib hits both the 256 and 512 pools).
3. **Per-block fp8 scale denormal underflow** under 26B's amax distribution (would explain the non-monotonic
   dip at _k=0.07).
I'll send results + which suspect it is.

## Step 3: fix + verify the ship gate

FlashInfer fix → 26B-A4B nvfp4 ≈ HF truth on vLLM AND SGLang (matched anchor, chat coherent, vs truth) →
DiffusionGemma inherits it (and re-gate DiffusionGemma's "green" against HF truth, not coherence-only,
while we're here — same base, same risk). Then all three families ship full-nvfp4 and the bar is met.

Resources: do the kernel work on **vast.ai sm_120** (the fp8-D512 `NUM_MMA_KV=1` crash and the nvfp4 break
both repro on any sm_120 5090/PRO-6000 — no Spark needed for the hunt); Spark only for the final SGLang +
DiffusionGemma serving validation. Also land my `spark/fp8-d512-clean-reject` in your image's FlashInfer so
the fp8 path stops crashing cryptically while you're in there.

Sending the V-SF layout sweep results shortly. Step 1 says it's fixable; we run this to a fix.
