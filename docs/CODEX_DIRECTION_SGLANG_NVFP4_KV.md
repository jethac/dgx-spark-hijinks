# Direction: SGLang → NVFP4 KV on Spark (convert proven capacity into blessed quality)

> Standing direction for the SGLang lane. SGLang is the lane where NVFP4-KV **capacity**
> is already proven on GB10 (1.779× fp8 pool) — the closest result to the campaign's
> founding goal (memory / context / concurrency). It is **one correctness bug away** from
> being the headline capacity win, and that bug is the highest-leverage thing in the
> whole NVFP4-KV effort.

## Why this, why now
The SGLang FP4 KV row already expands the KV pool ~1.78× over fp8 on GB10. But it
produces **corrupt output even in eager/no-graph mode** (raw `2+2` fails; benchmark text
degenerates). The underlying FlashInfer FA2 NVFP4-KV kernel is proven correct standalone
(`jethac/flashinfer@e152cf4d`, cosine ≥ 0.99999946), and the layout probe ruled out
scale-rank as the cause (`results/sglang_nvfp4_kv_layout_probe_20260608.json`, dequant
cosine 0.9999957). **So the bug is in SGLang's integration of a correct kernel, not in
the kernel math.** The job: find and fix that bug, then land a blessed matched
fp8-vs-FP4 serving row. This is plausibly higher leverage than the vLLM KV lane, which
hasn't even served NVFP4 KV yet.

## Prime suspect — read this first
Turn 1 of this whole investigation was: "SM120/121 should use the `fp4_quantize`
fallback with **inverted global scale**, not `nvfp4_kv_quantize` — why?" The answer:
quantize and consumer are a matched pair on global-scale convention. `nvfp4_kv_quantize`
applies the encode scale by **multiply** (`s_enc = 6·448/amax`); the `fp4_quantize`
fallback applies the global scale by **divide** (`s_dec = 1/s_enc`). Crossing kernels
without reciprocating = off-by-`s_enc²` → NaN/garbage logits.

The current SGLang patch (`jethac/sglang@67c7967` → `eefe8aded`) **routes SM12x through
`nvfp4_kv_quantize`** (the multiply convention). The eager-mode corruption is exactly
the symptom you'd expect if its GB10-runnable FlashInfer FA2 consumer expects the
**divide** convention. **First hypothesis to test: is SGLang's quantize convention
matched to what the FlashInfer FA2 NVFP4-KV consumer actually consumes?** A direct test
is swapping SM12x to `fp4_quantize` + inverted global scale and checking whether raw
`2+2` returns `4`. If it does, the rest is mostly bookkeeping.

## Methodology / sequencing constraint
Do NOT build a container image in the dev loop. SGLang already iterates the right cheap
way and should keep doing so:
  1. **Editable source overlay on stock `nvcr.io/nvidia/sglang:26.05-py3`** with the
     `jethac/sglang` source + local FlashInfer JIT source (exactly the autosafe-row
     setup). Note overlay loses `.git`, so it is overlay evidence, not a pinned/wheel
     proof — fine for dev, label it honestly.
  2. **The proven standalone FlashInfer reference is ground truth.** Use
     `scripts/flashinfer_nvfp4_kv_probe.py` / `jethac/flashinfer@e152cf4d` and extend
     `scripts/sglang_nvfp4_kv_layout_probe.py` into a per-op numerical bridge to localize
     divergence. Most root-causing happens here, with zero serving.
A blessed serving row + clean container is the final deliverable, gated on quality
passing — not a dev step.

## What is already proven — do NOT redo these
- KV4Compatibility server-arg gates: `jethac/sglang@eefe8aded`, `3 passed / 56
  deselected` (`results/sglang_fp4_kv_sm121_pytest_20260608T0320JST.md`). Python-level
  arg compatibility only.
- `KVFP4QuantizeUtil` alias of `BlockFP4KVQuantizeUtil`: `jethac/sglang@98ad46961`.
- **Capacity**: matched autosafe row,
  `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_summary.md` — FP4 KV `5,519,481`
  tokens vs fp8 `3,101,822` = **1.779×**; calibration runs (`NVFP4 KV cache calibrated
  28 layers from 4096 eager prefill tokens`). This is capacity-proven; treat it as done.
- **Negative results that narrow the search**: scale-rank (4D vs 3D) is NOT the bug
  (layout probe dequant cosine 0.9999957); the FlashInfer FA2 NVFP4-KV kernel is correct
  standalone (e152cf4d). Do not re-investigate either — they're cleared.

Read `docs/NVFP4_KV_PORTING_MAP.md` (SGLang Reference Map) and the autosafe summary
before starting.

## The SGLang problem, precisely
1. **Quality corruption in eager mode** (keystone). Output is wrong even with CUDA graph
   and piecewise capture disabled — so it is not merely a graph bug. A correct kernel is
   being fed wrong-convention or wrong-scale data by the SGLang path.
2. **Worse corruption with CUDA graphs.** Graph-enabled decode corrupts output, so the
   fork currently auto-disables capture (`SGLANG_FP4_KV_ENABLE_CUDA_GRAPH=1` opt-in).
   This forces the slow no-graph path (an early variant measured 0.276 tok/s). Separable
   from #1 and likely a calibration-vs-capture state problem.
3. **FlashInfer FP4 decode kernel was force-compiled past errors** (`vec_dtypes.cuh`,
   group6 dtype mismatch, packed head-dim — `fi_fp4_decode_*` probes, commit `fb7f0a1`).
   Confirm the decode path is numerically correct, not merely compiling.

## Objectives, in order
**A. Root-cause and fix the eager-mode quality corruption (keystone).**
Build the numerical bridge: feed identical KV through (i) SGLang's
quantize→memory-pool→FlashInfer-wrapper path and (ii) the proven standalone FlashInfer
reference, and diff per layer / per op to localize divergence. Test suspects in order:
   1. **Global-scale convention** — encode-multiply (`nvfp4_kv_quantize`) vs decode-divide
      (`fp4_quantize` + inverted scale). Confirm the convention matches the consumer; try
      the fallback path and check raw `2+2`.
   2. **V-scale layout** — SGLang default is **symmetric-linear** V scale-factors;
      vLLM uses **B2 swizzle + in-kernel deswizzle**. SGLang must consume the FlashInfer
      kernel built **without** `FLASHINFER_PAGED_V_SF_DESWIZZLE`. A layout/macro mismatch
      here corrupts V exactly like a convention mismatch.
   3. **Per-layer calibration application** — the 28-layer / 4096-token calibration must
      apply the same global scales at quantize and at in-kernel dequant.
   4. Only then the decode kernel itself (Objective C overlap).

**B. Confirm the FlashInfer FP4 decode kernel is numerically correct.**
The decode compile fixes (`vec_dtypes.cuh`, group6 dtype, packed head-dim) must be
validated against the standalone reference at the SGLang shapes, not just "it builds."
Decode is the daily-driver path; a subtly wrong decode kernel reproduces #1's symptom.

**C. Fix the CUDA-graph-capture corruption.**
Once eager is correct, graph corruption means calibration/global-scale state isn't
captured. Likely calibration-before-capture ordering or graphs capturing stale/
uncalibrated scales. Goal: serve FP4 KV with graphs on, so the capacity win isn't stuck
behind a slow no-graph path. Keep `SGLANG_FP4_KV_ENABLE_CUDA_GRAPH` as the gate until
proven.

**D. Land the blessed matched fp8-vs-FP4 serving row.**
Same model / prompts / `--mem-fraction-static` / `--page-size` / graph mode. Quality
must pass: raw `2+2` = `4`, coherent benchmark content, plus a real quality comparator
(PPL or retrieval sanity vs fp8/bf16). Record KV pool tokens, max concurrency, memory
telemetry, TTFT, warmed decode tok/s. Server log must prove native FlashInfer FP4 KV
selection — not fp8/bf16 fallback. Use Qwen2.5 1.5B (the established comparator) first.

**E. SWA / Gemma is explicitly later.** SGLang FP4 KV with hybrid-SWA subpool delegation
(Gemma's alternating local/global) is a separate, harder workstream — and SGLang Gemma
already has open blockers (issue #14). Prove FP4 KV on Qwen/Step-like non-SWA models
first; keep Gemma SGLang blockers out of the NVFP4-KV quality fix. Coordinate Gemma
attention geometry with the vLLM lane (`docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md`).

## Evidence gates (a row isn't a claim without these)
- Source-overlay/build evidence with a valid `sm_121a`/`compute_121a` FlashInfer target.
- `cuobjdump`/JIT-cache proof the running FP4 KV decode kernel matches the claimed path.
- Server log proving native FlashInfer FP4 KV selection (not fallback).
- Deterministic sanity (raw `2+2` = `4`) AND a quality comparator vs fp8/bf16/dequant.
- Capacity/concurrency vs an fp8 comparator at matched settings (already have 1.779×).
- CUDA-graph-replay coverage once graphs are re-enabled.
- Explicit scope labels: SWA/Gemma, page-size variants, TP>1, MTP/spec-decode untested.

## Guardrails
- **Keep capacity and quality claims separate.** Capacity (1.779×) is proven; quality is
  not. Never let the proven capacity number imply a usable serving row.
- **Convention discipline (the turn-1 lesson).** Document the matched quantize↔consumer
  pair explicitly: which global-scale convention (multiply vs divide) and which V-scale
  layout (SGLang symmetric-linear, NOT vLLM B2 swizzle). Most of this lane's risk is a
  convention/layout mismatch between a correct quantizer and a correct consumer.
- **Lane ownership.** FlashInfer owns kernel/page/stride and the symmetric-linear V-scale
  behavior (even when a reference patch ships under the SGLang overlay). SGLang owns
  memory pool, KV dtype, calibration, FlashInfer wrapper plumbing, and server args. Do
  not put kernel math fixes in SGLang.
- **Validate on `sm_121a` only; SM120 ride-along.** Same policy as the vLLM doc: build on
  `hikarioyama/sglang-nvfp4-kv-sm120@9b2160f0` as prior art, keep patches SM12x-family-
  shaped, re-derive (do not vendor the overlay tree), emit `120a`+`121a` not `120f`,
  label SM120 compiled-but-unclaimed (hikari-validated, not us). The 99 KB/block SMEM
  ceiling is confirmed family-wide. See the "SM120 ride-along" section of
  `docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md`.
- **Maintain RTX PRO 6000 (SM120) compatibility in the quality fix, not just the build.**
  The convention/V-scale/calibration fix must land on the SM12x-family gate
  (`is_sm120_supported()` covers SM120 and GB10/SM121), derived from hikari's SM120
  reference — never an sm_121-only special case. Turn-1's `fp4_quantize`-fallback +
  inverted-scale guidance was always framed for the whole SM120/121 family; keep it that
  way so a correct GB10 result is also correct on the RTX PRO 6000 we can't test.
- **Hikari's working SM120 path is a debugging oracle for Objective A.** Their SGLang
  SM120 reference serves correctly on SM120, so it has already resolved the global-scale
  convention and V-scale layout for the family. If our GB10 output is corrupt and theirs
  is clean, diff our quantize/scale/calibration handling against hikari's — the
  divergence likely *is* the bug. This makes RTX PRO 6000 compatibility and the quality
  fix the same problem, not competing ones.
- Use issue-named worktrees (issue #18 for SGLang NVFP4 KV); reference #2555-style
  in-flight upstream work to avoid collisions on the backend selector.

## First concrete step (no image builds)
Build the SGLang-vs-standalone-FlashInfer **numerical bridge** on Qwen2.5 1.5B by
extending `scripts/sglang_nvfp4_kv_layout_probe.py`, and run the single decisive test:
swap SM12x quantization to `fp4_quantize` + inverted global scale and check whether raw
`2+2` returns `4` in eager mode. That one result either confirms the convention
mismatch (and points straight at the fix) or clears it and sends the search to V-scale
layout / calibration. Everything downstream depends on it. No serving image required.
