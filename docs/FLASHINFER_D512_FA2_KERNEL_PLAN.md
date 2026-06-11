# FlashInfer FA2 D=512 Plan — Full NVFP4 K+V for Gemma 4 Global Layers

Status: **P0 + P0b GREEN, P3 AUTHORED (2026-06-10) — K1 confirmed end to end on GB10.**
- P0 (bf16): (512,256) two-pass VO split matches torch fp32 at **cosine 0.9999978**
  (`results/flashinfer_fa2_vo_split_d512_vo256_probe_20260610T0520JST.json`).
- P0b (NVFP4 KV): green at **cosine 0.9999986** with linear V-SF
  (`results/flashinfer_vosplit_p0b_fp4_green_linear_20260610_summary.md`). Root cause of
  the first red: FlashInfer's Python sized O from Q width under FP4 — fixed on
  `jethac/flashinfer@spark/hijinks-022-fa2-d512` (out width from V, 4 sites; head
  `fb7d62ea`). Swizzled V-SF does NOT slice along the head dim (token-quads spread across
  the full SF row) — sidestepped via `VLLM_NVFP4_KV_LINEAR_V_SF=1`
  (`results/d512_vosplit_sf_layout_decision_20260610.md`).
- P3 (vLLM orchestration): authored and pushed,
  `jethac/vllm@spark/hijinks-022-gemma4-mixed-kv` head `f9159f41b`. Opt-in via
  `VLLM_NVFP4_KV_VOSPLIT=1` + `VLLM_NVFP4_KV_LINEAR_V_SF=1`: builder plans
  (qk=512, vo=256), all requests route through the prefill wrapper
  (`reorder_batch_threshold=0`; the decode wrapper has no `head_dim_vo`), impl runs two
  zero-copy V-half passes and concatenates. Gemma4Config keeps per-layer backend
  resolution under the knob instead of forcing TRITON_ATTN. GPU validation (P1 gates +
  serving smoke) rides the next idle window.

**Scope expansion (2026-06-11, Jetha: "do all of it"):** the VO split is now also the
**wholesale TRITON_ATTN retirement** for Gemma 4 — `VLLM_FLASHINFER_VOSPLIT=1` extends
the two-pass split to **all KV dtypes** (bf16/fp16/fp8, not just NVFP4) and makes
`Gemma4Config` force FLASHINFER for the whole model instead of TRITON_ATTN (single
backend, so upstream's mixed-backend divergence concern does not apply). Authored and
pushed: `jethac/vllm@ad2337814`. Receipts this answers: vllm#38887 (E4B ~9 tok/s on
RTX 4090 under the Triton force; upstream PR #38891 incomplete), vllm#40677 (RTX PRO
6000 + Gemma 4 NVFP4, FLASHINFER rejected), vllm#42068, #39133, #39965 — see
`docs/ISSUE_TRACKER.md` upstream table. Gates before any claim: fp8 (512,256)
trait-pair probe (bf16 is already P0-green; fp8 is NOT yet probed), bf16 serving
smoke, and the **E4B tok/s benchmark vs TRITON_ATTN on GB10** — the speed pitch must
be measured, not asserted. Python-only: validates via source overlay, no extension
rebuild.

**Coherence-hunt verdict (2026-06-11, heads a612230/e7893f5):** the 31B full-NVFP4
gibberish was NOT attention math. Chain: VO-split serving crash fixed (`e08a6f3ae`,
ctor jit_args pinned symmetric vo); rerun served but emitted deterministic garbage;
config bisects + the two-runtime writer-roundtrip pincer (SGLang green at
head-256/SWA/linear; vLLM stage-A red at head-128 calibration) convicted the **r7
image's compiled writer: it swizzles V-SF unconditionally, ignoring
VLLM_NVFP4_KV_LINEAR_V_SF** (source correct; stale csrc in the image build). The
Block C paradox (knob-on green on the same image) is explained by **FlashInfer module
-cache unsoundness**: AOT modules shadow JIT rebuilds and EXTRA_CUDAFLAGS is not in
the module key, so deswizzle-built head-128 readers accidentally re-paired with the
swizzling writer (`results/jit_cache_mode_unsoundness_analysis_20260611.md` —
upstream candidate; campaign rule: clear module caches when toggling SF mode;
long-term fix: SF mode as a jit-arg in the module name). Gate sequence on r8:
latch diag -> AOT-dir proof -> calibration -> head-256/SWA matrix -> Block C re-run
-> 31B smoke. Kernel-side K1 remains fully validated; everything outstanding is
binary/build hygiene.

Two findings beyond the plan (from P0):
- The wrapper's `paged_k_cache.stride(i) == paged_v_cache.stride(i)` check is the only
  asymmetric-pair blocker hit so far, and it is dodged **zero-copy**: pass V halves as
  strided VIEWS of the full V (identical strides to K; the half is selected by base
  pointer offset). Zero FlashInfer changes for bf16. P2 may still relax the check
  properly for robustness.
- The feared q-register bust at D_QK=512 did not materialize (bf16, NUM_WARPS defaults).
Next gates: **P1 correctness sweep** (HND layout, batch>1, qo_len=1-as-decode) + linear
V-SF writer regression on the existing Qwen/Gemma 3 rows, then **P4** (Gemma 4 full-NVFP4
serving rung). Codex: the probes + results live on the 022 branch; the same VO-split
pattern + the out-width fix on `spark/hijinks-022-fa2-d512` apply to SGLang's Gemma
rungs when they arrive (SGLang's linear SFs slice trivially). Owner: Claude lane
(`spark/hijinks-022-*` branches; worktrees — Codex's checkouts untouched). Goal: upgrade
the Gemma 4 mixed-KV **global D=512 layers** from bf16/Triton fallback to **NVFP4 on
FlashInfer**, completing full NVFP4 K+V on Gemma 4 (the 1.78×-class capacity claim instead
of the mixed ~1.25–1.57×).

## 1. The wall, precisely (code-anchored, hardware-confirmed)

`include/flashinfer/attention/prefill.cuh` (`c3dae30f`), `KernelTraits::IsInvalid()`:

```cpp
return ((NUM_MMA_D_VO < 4) || (NUM_MMA_D_VO == 4 && NUM_MMA_KV % 2 == 1) ||
        (POS_ENCODING_MODE == kRoPELlama && NUM_MMA_D_VO > 4 &&
         NUM_MMA_D_VO % (2 * NUM_WARPS_Q) != 0) ||
        (NUM_MMA_Q * (8 * NUM_MMA_D_VO + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >= 256) ||
        ((sizeof(DTypeK) == 1 || sizeof(DTypeV) == 1) && NUM_MMA_KV * 2 % NUM_WARPS_Q != 0) ||
        (sizeof(DTypeK) == 1 && POS_ENCODING_MODE == kRoPELlama));
```

For `HEAD_DIM_VO=512`: `NUM_MMA_D_VO=32` → `8*32=256` saturates the register budget with
`NUM_MMA_Q=1` before the KV term — the per-thread output-accumulator fragments for a
512-wide VO row do not fit. **Dtype-independent** (probed on GB10 with bf16:
`results/flashinfer_fa2_bf16_d512_probe_20260610T0438JST.json`, identical failure to FP4).

## 2. The key insight — a kernel-math-free route exists

Two facts, both verified in the source:

1. **The guard counts only `NUM_MMA_D_VO`** (output accumulator + softmax state), not
   `NUM_MMA_D_QK`. A `(HEAD_DIM_QK=512, HEAD_DIM_VO=256)` trait scores
   `1*(8*16 + 2*4*1) = 136 < 256` → passes.
2. **Asymmetric (QK, VO) pairs are first-class** in the JIT: modules are generated per
   `head_dim_qk_*_head_dim_vo_*` (DeepSeek 192/128 is shipping precedent); the C++ takes
   both as template params.

And the math: attention decomposes **exactly** along the VO dimension. With
`V = [V_left | V_right]` (256+256):

```
S = Q·K^T            (full 512-dim QK — identical in both passes)
P = softmax(S)       (identical)
O = [P·V_left | P·V_right]   (exact concatenation, no merge, no LSE games)
```

Unlike KV-length splits (online-softmax merge — see the SGLang LSE saga) or QK-dim splits
(changes the logits), a VO split needs **no recombination math at all**. Two passes with
the same Q/K and half the V each, concatenate outputs.

## 3. Strategy ladder (cheapest first; each rung gated)

### K1 — VO-split multi-pass via existing asymmetric templating (target)
No kernel-math changes. JIT a `(512, 256)` FA2 paged module; run the global layers'
attention **twice per call** (V halves), concat outputs.
- **NVFP4 slicing is clean by construction:** FP4 block scales are per-16-elements along
  the head dim; a 256-boundary split is 16 whole SF blocks per half — data and SF views
  slice exactly. (vLLM's *swizzled* V-SF layout + in-kernel deswizzle is the one layout
  risk: the slice must respect the swizzle tile; SGLang's linear SFs slice trivially.
  Probe before trusting.)
- **Cost (honest):** QK logits + softmax computed twice; K read twice. Bounded: global
  layers are ~1/6 of Gemma 4 layers; prefill-dominant. Decode adds ~2× K-traffic on those
  layers (≈ +8% total decode KV traffic). This is a **capacity** play, not speed — same as
  the whole campaign.
- **Fallback within K1:** if (512,256) busts registers in practice, (512,128)×4 passes
  (`NUM_MMA_D_VO=8` → trivial budget).
- **Known unknown:** the guard does not count **q-fragment registers at D_QK=512**
  (largest shipping QK is 256/192). ptxas may still spill/explode. This is exactly what
  the P0 probe measures empirically — no guessing.

### K2 — single-pass warp-partitioned VO (real kernel surgery; only if K1 fails)
Restructure `prefill.cuh` so warps partition `D_VO` (new `NUM_WARPS_VO` axis): per-thread
o_frag covers `D_VO/NUM_WARPS_VO`, epilogue and V smem staging reworked, softmax state
shared. Weeks-class CUDA work. Trigger: K1 register-busts at both (512,256) and (512,128),
or K1 perf is unacceptable on the rung gates.

### K3 — guard relaxation + smem-staged O accumulation (last resort)
Most invasive; not planned unless K1 and K2 both fail.

## 4. Phases and gates

- **P0 — trait probe (GB10, weight-free):** JIT + run `(512,256)` and `(512,128)` paged
  prefill, bf16 first then FP4-KV. Gate: compiles, runs, finite output, sane ptxas
  register counts. Extends `scripts/vllm_gemma4_mixed_kv_probes.py`. *This single probe
  decides K1 vs K2.*
- **P1 — correctness:** extend the standalone NVFP4-KV probe to D=512 via two-pass
  orchestration in the probe itself; cosine ≥ 0.9999 vs bf16 reference, NHD+HND,
  decode+prefill shapes, on GB10.
- **P2 — FlashInfer enablement** (branch `jethac/flashinfer@spark/hijinks-022-fa2-d512`,
  base `c3dae30f`): whatever Python-level head-dim whitelists/plan() validations need the
  (512,256) pair admitted (cf. the wrapper whitelist at `prefill.py:3992`). K1 should not
  touch `prefill.cuh` math at all.
- **P3 — vLLM orchestration** (existing branch `jethac/vllm@spark/hijinks-022-gemma4-mixed-kv`):
  for Gemma 4 global layers under mixed-KV, an opt-in flag switches the global layers from
  the Triton/bf16 fallback to FlashInfer NVFP4 two-pass VO-split (plan two wrappers or one
  wrapper with V-half views; concat in the impl). Mixed-KV stays the default until gates
  pass.
- **P4 — the rung row:** Gemma 4 (31B first) **full NVFP4 K+V**: capacity comparator
  (expect the full ~1.78×-class pool, minus page-padding now removable), quality gate
  (`gemma_nvfp4_kv_quality_gate.py` + PPL), memory guardrails as always.

## 5. Coordination + scope

- Worktree lanes only; Codex's checkouts untouched. FlashInfer base `c3dae30f` (current
  campaign head) so the NVFP4-KV SF-stride work is inherited.
- K1 deliberately avoids `prefill.cuh` math — no collision with any SGLang-lane kernel
  work. K2 (if ever) coordinates first.
- PGX runs follow the idle-window protocol (check `docker ps`/GPU/memory; weight-free
  probes in capped `--rm` containers; never alongside a serving run).
- Upstream: (512,·) pair enablement + the two-pass orchestration pattern is a
  FlashInfer/vLLM contribution candidate; the **selector-vs-kernel head-512 disagreement**
  (validate_configuration over-promise, probe-proven) should be filed regardless.

## 6. Success criteria

Full NVFP4 K+V serving on a Gemma 4 server variant: server log proves NVFP4 KV on **all**
layers (no bf16/fp8 global fallback), capacity row at the full ratio, first-token +
supplied-token PPL gates green vs the fp8 comparator, prefix-reuse row included. That
retires the mixed-KV capacity compromise on vLLM and makes "full NVFP4 K+V on both
runtimes" a Gemma-4-inclusive claim.
