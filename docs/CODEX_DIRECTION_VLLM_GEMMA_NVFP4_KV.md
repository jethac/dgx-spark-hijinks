# Direction: vLLM → Gemma on Spark with NVFP4 KV

> Standing direction for the vLLM lane. The product target is **Gemma 4 (and earlier
> Gemmas) serving well on the Spark with NVFP4 KV**. Qwen was the clean first lane that
> proved the packaging/native-FA2 machinery; Gemma is the destination and is where the
> remaining hard problems live.

## Why this, why now
NVFP4 KV cache on GB10 is the founding goal of this whole campaign — it targets Spark's
actual bottleneck (memory / context length / concurrency), not decode FLOPs. Reorient
the vLLM NVFP4-KV work so that **Gemma is the target model**, not Qwen.

## Methodology / sequencing constraint
Do NOT build a fresh vLLM image as part of the dev loop. An image build is the final
packaging *deliverable*, gated on Gemma 4 actually serving correctly with NVFP4 KV
proven by logs — not a step you repeat while iterating. Iterate using the two cheap
loops this repo already established:
  1. **Standalone FlashInfer kernel harness on GB10** for the D=512 diagnosis and any
     kernel fix — this is exactly how `e152cf4d` correctness was proven, no image.
  2. **Source overlay on the existing proven image**
     (`jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`): mount the forked
     vLLM editable + local FlashInfer JIT source, exactly as the SGLang FP4-KV work
     overlaid the stock sglang container. Let FlashInfer JIT-compile the kernel from
     mounted source.
Only bake a new image once Gemma 4 serves correctly with the FA2 NVFP4-KV path proven.

## What is already proven — do NOT redo these
- Clean vLLM + native FA2 `sm_121a` packaging: `jethac/vllm@a919d635d` +
  `jethac/flash-attention@7d53245`, image
  `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`. Qwen3.6 NVFP4-*weights*
  + DFlash served at ~61 tok/s with cuobjdump-proven `sm_121a` FA2 cubins. (This is
  NVFP4 weights + normal KV — NOT NVFP4 KV.)
- vLLM NVFP4-*KV* routing: `jethac/vllm@8916796` (`spark/hijinks-007-nvfp4-kv-sm121`)
  routes SM12x `--kv-cache-dtype nvfp4` → FlashInfer FA2 (not trtllm-gen), with V-SF
  deswizzle. Routing is probe-proven on GB10
  (`results/vllm_nvfp4_sm12x_routing_probe_20260607T171227Z.json`).
- FlashInfer FA2 NVFP4-KV kernel correctness: `jethac/flashinfer@e152cf4d`
  (`spark/hijinks-007-fa2-nvfp4-kv-sm121`), cosine ≥ 0.99999946 standalone, including
  Gemma sliding/local `D=256`.
- **vLLM NVFP4-KV now SERVES on Qwen (2026-06-08) — founding goal reached for the
  standard-attention path.** Image
  `jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv` (vLLM fork +
  FlashInfer `e152cf4d`) served Qwen3.6 with a matched fp8-vs-NVFP4 row
  (`results/vllm_qwen_nvfp4_kv_capacity_20260608T1455JST_summary.md`). Server log proves
  the path: `Using FlashInfer FA2 backend for NVFP4 KV cache on SM12x with vLLM
  V-scale-factor deswizzle enabled`. **1.751× fp8 KV pool/concurrency** (11,146,226 vs
  6,364,935 tokens at 262k ctx, 0.85 mem), decode parity (~43 tok/s), **normal content**.
  Done — do not redo. Caveats: derived proof image (not the overlay loop), still MARLIN
  weight-only FP4 (not native FP4 MoE). **The only remaining vLLM NVFP4-KV target is
  Gemma.**

Read `docs/NVFP4_KV_PORTING_MAP.md` (esp. "Smallest GB10 Proof Sequence") and
`docs/GEMMA4_ON_DGX_SPARK.md` before starting.

## The Gemma problem, precisely (three intertwined blockers)
Gemma's blocker and the NVFP4-KV blocker are the same blocker: heterogeneous/dual head
dimensions (local layers `D=256`, global layers `D=512`) plus alternating SWA.

1. **Model-arch support.** Gemma 4 needs `Gemma4UnifiedForConditionalGeneration`;
   released vLLM 0.22.1 can't load it. So far it only ran via source `da1daf40` +
   Transformers-main surgery (7.7 tok/s, forced Triton).
2. **Native attention.** vLLM forces `TRITON_ATTN` for Gemma 4's heterogeneous head
   dims + alternating SWA. That means the native FA2 `sm_121a` path does not engage for
   Gemma at all — and NVFP4 KV rides on FA2.
3. **NVFP4 KV global-attention failure.** The FA2 NVFP4-KV path passes Gemma local
   `D=256` but fails global `D=512` with FlashInfer "invalid configuration" at
   `prefill.cuh:3215`
   (`results/flashinfer_nvfp4_kv_probe_gemma4_26b_global_20260608T0335JST.json`).
   A focused NHD rerun on `jethac/flashinfer@e152cf4d` in the Qwen-proven NVFP4-KV
   image reproduced the same blocker: decode fails with `NUM_MMA_Q=1
   NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 NUM_MMA_KV=1 NUM_WARPS_Q=1 NUM_WARPS_KV=4`, and
   prefill fails with `NUM_MMA_KV=2 NUM_WARPS_Q=4 NUM_WARPS_KV=1`
   (`results/flashinfer_nvfp4_kv_probe_gemma4_26b_global_nhd_debug_20260608.json`).

## Objectives, in order

**A. Get Gemma onto the clean jethac stack with the backend recorded (overlay, no image).**
Load Gemma 4 12B and 26B via source overlay on the `cleanfa2-patchedfa2-cutlass` image.
Record the *actually selected* attention + MoE backend, quantization path, and whether
KV is bf16/fp8/nvfp4. Goal: replace the `da1daf40` + Transformers surgery with a
reproducible clean-stack Gemma baseline. This is compatibility, not a speed/KV claim.

**B. Break the `prefill.cuh:3215` D=512 failure — this is the keystone.**
The standalone harness reproduction is done, and the first trait audit has a concrete
answer: the `D=512` launch reaches `DISPATCH_HEAD_DIM`'s `case 512`, then trips
`KernelTraits::IsInvalid()` in `include/flashinfer/attention/prefill.cuh`. The rejecting
clause is `NUM_MMA_Q * (8 * NUM_MMA_D_VO + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >=
256`. For `D=512`, `NUM_MMA_D_VO=32`, so `8 * NUM_MMA_D_VO` is already `256` before the
positive KV term is added (`264` for the decode trait, `272` for prefill). This is a
compile-time fragment/register-shape guard, not a missing head-dim table and not
primarily a 99 KiB SMEM overflow. The practical next fixes are:
   - **per-layer mixed KV** — NVFP4 KV on local (`D=256`) layers, fp8/bf16 on the global
     (`D=512`) layers. Gemma's ~5:1 local:global ratio captures most of the capacity win
     immediately while a true `D=512` kernel matures; or
   - an alternate FlashInfer FA2 `D=512` kernel/trait that changes the fragment/register
     shape enough to satisfy the guard. A simple head-dim support-table change or smaller
     CTA/warp tweak is not expected to work under the current trait math.

Explorer read-only audit (2026-06-08): prototype **NVFP4 local + bf16 global** first,
not fp8 global. vLLM already has a manual `kv_cache_dtype_skip_layers` escape path that
falls skipped layers back to `auto`/model dtype, which is enough for bf16 global. fp8
global needs a real per-layer fallback dtype string, so it is a second step.

Likely insertion points and risks:

- `vllm/model_executor/models/gemma4.py`: `Gemma4DecoderLayer.__init__` already
  classifies global/full layers and selects `global_head_dim` vs `head_dim`;
  `Gemma4Attention.__init__` constructs the per-layer `Attention`.
- `vllm/model_executor/layers/attention/attention.py`: `Attention.__init__` resolves
  `kv_cache_dtype`, applies `kv_cache_dtype_skip_layers`, selects the backend, and
  instantiates the impl. Extending skip matching to `full_attention` is the smallest
  Gemma-local policy hook for bf16 global fallback.
- `vllm/v1/kv_cache_interface.py`: `AttentionSpec` already carries per-layer `dtype` and
  `kv_quant_mode`. fp8 fallback likely needs a `cache_dtype_str`-style field to preserve
  `fp8`/`fp8_e4m3`/`fp8_e5m2`, similar to MLA specs.
- `vllm/v1/core/kv_cache_utils.py`: allocator grouping is the main risk. Local NVFP4
  `D=256` and global bf16/fp8 `D=512` may have incompatible page sizes; generalize the
  existing multi-page-size/bucketing path instead of adding a Gemma-only manager.
- `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/worker/gpu/attn_utils.py`, and
  `vllm/v1/worker/utils.py`: several reshape/zeroing paths still use global
  `cache_config.cache_dtype`; these must derive layout from each `AttentionSpec` or a
  global `--kv-cache-dtype nvfp4` run can reshape bf16 global layers as packed NVFP4.
- `vllm/v1/attention/backends/flashinfer.py`: `FlashInferMetadataBuilder` uses global
  `cache_config.cache_dtype` when `kv_cache_spec.kv_quant_mode != NONE`; mixed
  fp8/NVFP4 must dispatch from the layer/group spec.
- `vllm/model_executor/models/config.py`: Gemma 4 currently forces `TRITON_ATTN` for
  heterogeneous `head_dim/global_head_dim`; relax this only after per-layer routing is
  explicit, or global `D=512` can accidentally hit the known FlashInfer NVFP4 guard.

**C. Decouple SWA handling from the dual-head-dim fix using an earlier Gemma.**
Earlier Gemmas (2/3) use SWA but uniform head dims, so they exercise the
hybrid-local/global KV plumbing *without* the `D=512` blocker. Use one as the first
end-to-end NVFP4-KV-on-Gemma proof; it de-risks the SWA path independently.

**D. Prove NVFP4-KV-on-Gemma in a running server (overlay, no image build).**
The Qwen equivalent is DONE (see proven list) and is your template: the
`...-flashinfer-e152cf4d-nvfp4kv` row hit exactly this gate on standard attention. For
Gemma, marry the re-derived `jethac/vllm@8916796` routing with
`jethac/flashinfer@e152cf4d` and start Gemma with `--kv-cache-dtype nvfp4`. The **server
log must prove FA2 NVFP4 KV was selected** — not fp8/bf16 fallback, not trtllm-gen. Then
record a matched fp8-vs-NVFP4 row (same model/prompts/mem-fraction/graph-mode): KV pool
tokens, max concurrency, memory telemetry, deterministic output, quality (PPL or
retrieval sanity), TTFT, warmed decode tok/s. The only delta from the working Qwen row
is Gemma's heterogeneous attention + the `D=512` global fix (Objectives A–C) — once those
clear, D should fall out the same way Qwen did.

**E. Only now: bake the deliverable image.** Once Objective D's row passes its gates,
build the clean Gemma NVFP4-KV image as the reproducible artifact + CI handoff. The
build matrix must emit **both `sm_120a` and `sm_121a`** cubins (see "SM120 ride-along"
below) — not `120f`, which cannot emit native block-scaled FP4 MMA (#3170). Before
Objective D passes, building an image is wasted cycles around a red gate.

## SM120 ride-along (build on hikarioyama, claim nothing)
We have no RTX PRO 6000 (SM120) to validate on, but SM120 must ride along as a
first-class *compiled* target, because RTX PRO 6000 is far more common than DGX Spark
and an upstream maintainer will care about SM120 more than sm_121a. Shaping the work
for the SM12x family — not as sm_121a-only hacks — is what makes the eventual PR
mergeable beyond Spark.

- **Foundation is hikarioyama's SM120 prior art.** Our `hijinks-007` forks are already
  derived from `hikarioyama/vllm-nvfp4-kv-sm120@f6156ee3` and
  `hikarioyama/sglang-nvfp4-kv-sm120@9b2160f0` (porting map "Patch ownership" tables).
  Re-derive minimal upstream-shaped patches; do not vendor their overlay trees.
- **Keep patches family-shaped.** The FA2 NVFP4-KV route gates on
  `is_device_capability_family(120)`, which covers SM120 and SM121. Do not regress that
  into sm_121a-only as you fix Gemma. The family gate is correct for FA2 NVFP4-KV; it is
  NOT a license to assume weight/MoE FP4 MMA behaves identically across `120a`/`121a`.
- **Emit `120a` AND `121a` — and understand why we can't merge them.** Both are
  arch-specific (`a`) targets; `120f` is a fake FP4 target (#3170: family targets can't
  emit native block-scaled FP4 MMA). Critically, an `a` cubin is **not portable across
  compute capabilities**: a `sm_120a` cubin will NOT execute on GB10 (sm_121), and
  `sm_121a` won't run on a RTX PRO 6000. The only thing spanning the 12.x family is
  `120f`, which by definition lacks native FP4. So there is no single native-FP4 cubin
  for both — ship per-arch or lose native FP4 on one. This is also why we *physically
  cannot* validate SM120's native path on our Spark: its native cubin can't run here.
- **Provenance label (use this wording):** "derived from hikarioyama's SM120 reference,
  validated by them on SM120; compiled here; validated by us only on sm_121a." This is
  stronger than "compiled, untested" — but still never a Spark-team SM120 claim. The
  `sm_120a` non-portability above is why hikari, not us, is the SM120 validation source.
- **The SMEM ceiling is family-wide (confirmed) — use it.** Per NVIDIA's Blackwell
  Tuning Guide, every CC 12.x part (RTX PRO 6000 sm_120 and GB10 sm_121 alike) has
  128 KB shared memory/SM but a **99 KB max per thread block** — vs datacenter CC 10.0
  (B200) at 228 KB/SM, 227 KB/block. So #11368 (SM120 CUTLASS FP4 GEMM tiles needing
  >99 KiB overflow GB10) is a **12.x-family constraint, not a GB10 quirk**. A tile fix
  that fits 99 KB is correct for both sm_120 and sm_121 — our constraint and the RTX PRO
  6000's converge. Prefer GeForce/consumer-class small-SMEM tiles (cf. CUTLASS ex.79
  `blackwell_geforce_gemm`, ~41.6 TFLOPS FP4 on GB10-appropriate tiles).
- **Still: don't assume SM120 == SM121 for arch-specific code.** The SMEM ceiling
  converges, but they remain different SMs with non-interchangeable `a` cubins. The
  family gate (`is_device_capability_family(120)`) is correct for FA2 NVFP4-KV routing;
  it is not a license to assume weight/MoE FP4 MMA behaves identically across the two.
- **Upstream demand is concrete:** vllm #31085 asks for SM120 (RTX 6000/5000 Blackwell)
  native NVFP4 MoE kernels. Family-shaped, hikari-derived work targets that audience.

Verified facts (NVIDIA Blackwell Tuning Guide, 2026-06): RTX PRO 6000 = sm_120 / CC 12.0
(GB202 family); GB10 = sm_121 / CC 12.1; both CC 12.x = 128 KB SMEM/SM, 99 KB/block;
B200 = CC 10.0 = 228 KB/SM, 227 KB/block.

## Evidence gates (a row isn't a claim without these)
- Spark build evidence: build/JIT logs showing a valid `sm_121a` target.
- `cuobjdump`/JIT-cache proof the running kernels match the claimed FA2 NVFP4-KV path.
- Server log proving native FA2 NVFP4 KV selection (not fallback).
- Correctness vs dequant/fp8/bf16 reference; quality comparison.
- Capacity/concurrency numbers vs an fp8 comparator at matched settings.
- Explicit scope labels for what's untested (MLA, attention sinks, TP>1, page sizes).

## Guardrails
- **Validate on `sm_121a` only.** This campaign's validation hardware is GB10. SM120 is a compiled-but-unclaimed build
  target; never put a correctness/capacity claim on hardware we can't run. See the
  "SM120 ride-along" section for how to build for it without claiming it.
- **Re-derive against current vLLM main; do not blind patch-apply.** The reference was
  pinned to `vllm 0.1.dev16944` / `flashinfer 0.6.11.post2`; main has NVFP4-KV
  scaffolding but still forces trtllm-gen. Re-derive the routing logic.
- The FA2 NVFP4-KV route correctly gates on `is_device_capability_family(120)` (it's
  the family-generic `mma.sync` path). Do **not** reuse that gate for native FP4/MXFP4
  weight/MoE MMA — that still needs true `sm_121a`.
- **Never merge NVFP4-weights serving rows into NVFP4-KV claims.** They are different
  capabilities. The Qwen ~61 tok/s row is weights+normal-KV; keep it out of KV stats.
- Keep the lanes split per the porting map: FlashInfer owns kernel/page/stride; vLLM
  owns routing/tensor plumbing. Use issue-named worktrees; reference the relevant
  issues so we don't collide on `flashinfer.py` / the SM12x backend selector.

## First concrete step (no image builds)
Two things, both on the existing image / standalone harness:
  1. Prototype the Gemma mixed-KV fallback as **NVFP4 local + bf16 global** first: keep
     FA2 NVFP4 KV on local `D=256` layers and route global `D=512` layers to model dtype,
     with logs proving the per-layer split. Do not start with fp8 global until the
     per-layer dtype-string/plumbing risk is solved. In parallel, treat a true FlashInfer
     `D=512` FA2 NVFP4 kernel as a separate kernel-design task, not a dispatch-table
     tweak.
  2. Bring up Gemma 4 12B via **source overlay** on the clean FA2 image and capture the
     server log + backend audit (which attention/MoE backend actually got selected).
Defer every image build until Objective E.
