# PLAN: Hadamard-rotated NVFP4 KV cache — true 4-bit KV for Gemma-4-26B-A4B (and all Gemma 4)

**North star (Jetha, non-negotiable):** Gemma-4-26B-A4B must serve with a working **full 4-bit (NVFP4)**
KV cache at claim-grade long-context quality. Calibration alone does not get there (proven: below-truth +
knife-edge + context-drift; Codex's 2D sweep + block attribution; Claude's read-capture). The fix is a
**transform**, not a scale.

## Root-cause thesis (to be confirmed in Phase 0)

The 26B nvfp4 collapse is localized to the **first sliding block (layers 0-4)** with knife-edge K-scale
sensitivity. The FlashInfer paged reader is FAITHFUL (out == dequant+SDPA) and per-tensor round-trip rel-L2
== 12B at short context. The remaining suspect is **per-channel outliers in those layers' post-RoPE K**
(and possibly V): NVFP4 stores a per-16-element-block fp8 scale, so one channel far larger than its
block-mates forces the whole block to clip-or-underflow. Scaling only trades clipping for underflow; it
cannot add levels. The standard remedy is an **orthogonal rotation** (Hadamard) applied to K (and V) along
head_dim that spreads each channel's energy across all channels, flattening the per-block amax so 4 bits
suffice — then undo it so attention math is unchanged.

## The math (why this is lossless to attention, only changes what gets quantized)

H is a normalized Walsh-Hadamard matrix (H Hᵀ = I), dim = head_dim (256 for sliding qk & all vo; 512 for
global qk — both powers of 2, exact FWHT).

- **K/Q (scores preserved):** score = qᵀk = qᵀ(HᵀH)k = (Hq)ᵀ(Hk). Store **Hk** in the cache (flat → 4-bit
  clean); rotate **Hq** before the dot. Apply to BOTH after RoPE (RoPE is position-dependent and must be
  applied first, identically, on Q and K; H is a fixed matrix so the identity holds for every position).
- **V (output rotates, then we undo it):** out = softmax·V. Store **V R** (R orthogonal on vo dim); then
  out_rot = softmax·(V R) = (softmax·V) R = out·R; recover **out = out_rot·Rᵀ** before o_proj.
- Net: attention is mathematically identical; only the *stored* K/V are rotated to be outlier-free, so the
  EXISTING nvfp4 writer (per-block amax → fp8 scale) and the EXISTING faithful reader handle them at 4-bit.

## Phasing (correctness-first; the reader we already trust does the heavy lifting)

### Phase 0 — CONFIRM THE MECHANISM (cheap; GATES the whole effort)
1. Per-channel post-RoPE **K amax histogram**, 26B layers 0-4 vs 12B same layers, at ctx 8185. Expect 26B to
   show channels whose amax >> their 16-block neighbours; 12B not. (Reuse the `sitecustomize.py` capture +
   an offline numpy reducer; Codex also has bf16/base/e0 activations from the block-attribution run.)
2. Offline numpy: take captured 26B layer-0-4 K (and V), apply normalized FWHT, re-measure per-16-block amax
   spread and the **nvfp4 round-trip rel-L2 rotated vs unrotated**. Expect rotation to flatten the spread and
   cut round-trip error materially.
3. **GATE:** rotation demonstrably flattens outliers AND lowers round-trip error → build it. If not, outliers
   aren't the mechanism → stop and re-diagnose (don't build on a wrong thesis).

### Phase 1 — OFFLINE CORRECTNESS PROTOTYPE (no serving)
- Implement normalized FWHT for d∈{256,512} (torch, fp32 accumulate). Unit-test H Hᵀ = I and FWHT == dense H·x.
- Harness on captured 26B layer-0-4 tensors: rotate Q/K/V → nvfp4 quant K/V → dequant → masked SDPA →
  un-rotate output; compare to bf16 attention. TARGET: rotated-nvfp4 attention error ≈ 12B's (~bf16 noise),
  unrotated stays broken. This proves the fix end-to-end before any serving code.

### Phase 2 — SERVING INTEGRATION (vLLM model-path; reuses the proven nvfp4 reader, NO kernel change yet)
- In the Gemma4 unified attention forward: after RoPE compute `q←Hq`, `k←Hk`; before cache write `v←vR`;
  after attention `out←out·Rᵀ` (fold into / before o_proj). Gate by layer via config (default: first sliding
  block; allow "all layers" — rotation is lossless so applying everywhere is safe and simplest to start).
- The nvfp4 writer stores rotated K/V (unchanged); the FlashInfer reader is unchanged. Build sm120a wheel.
- **Validate:** 26B matched anchor ctx 8185, nvfp4+Hadamard vs HF-eager bf16 truth. TARGET: claim-grade
  near-parity (≈ +0.02..+0.10 vs truth, like 12B), robust across global scale (the knife-edge should vanish).
  Re-confirm 12B/31B unaffected (rotation is identity-safe on already-working models).

### Phase 3 — KERNEL FUSION (perf; only if Phase 2 latency is unacceptable)
- Move the FWHT into FlashInfer: apply H to Q on load, fold the K rotation into the dequant epilogue, apply
  Rᵀ in the output epilogue — FWHT in shared memory (O(d log d)). Lands in the SHARED
  `spark/hijinks-e3-flashinfer` ref → benefits vLLM AND SGLang. Re-validate parity + measure tokens/s vs the
  model-path version and vs fp8.

### Phase 4 — GENERALIZE + SHIP
- Per-arch config: which layers carry the rotation (likely all sliding, or all). DiffusionGemma (26B base)
  inherits the fix — truth-gate it. SGLang parity (Codex integrates the same FlashInfer + the mirror
  model-path change on his stack). Bank green ladder rows; add the rotation chapter to the blog.

## Risks / open decisions
- **RoPE ordering:** rotation MUST be post-RoPE, identical on Q and K. Verify in Phase 1 against a bf16 ref.
- **VO-split global layers** (qk=512, vo=256): H₅₁₂ for Q/K, H₂₅₆ for V and the output un-rotation. Both
  exact-power-of-2.
- **Numerics:** normalized Hadamard (÷√d); accumulate FWHT in fp32 to avoid bf16 error reintroducing spread.
- **Does V even need rotation?** Test K-only first (V may be benign); V-rotation adds the output un-rotate
  cost. Phase 0/1 decides per-tensor.
- **Masking/SWA:** independent of head_dim rotation — no interaction.
- **Random vs Hadamard:** start parameter-free Hadamard (no stored matrix, deterministic, exact for pow2).

## Lane split
- **Claude OWNS** this rotation/kernel effort (vLLM model-path → FlashInfer fusion → shared e3 ref).
- **Codex** continues SGLang e3 + his 26B localization; his block-attribution activations feed Phase 0; he
  mirrors the model-path rotation on SGLang once Phase 2 lands and integrates the fused kernel from the shared
  FlashInfer ref. fp8 stays the interim 26B ship until rotated-nvfp4 is claim-grade.

## Done =
26B-A4B serves **full NVFP4 KV** (Hadamard-rotated) at claim-grade long-context quality (matched anchor
near-parity vs HF truth, robust, coherent) on the e3 stack, both runtimes; 12B/31B unaffected; DiffusionGemma
truth-gated; rotation fused in the shared FlashInfer ref; ladder + blog updated.
