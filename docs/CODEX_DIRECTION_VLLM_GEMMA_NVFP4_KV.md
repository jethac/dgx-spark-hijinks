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

## FP4 KV quality breaks after the linear Qwen path (cross-lane finding, 2026-06-09) — NEXT FOCUS
Plain linear / full-attention FP4 KV is **clean** (Qwen, 1.751×). But FP4 KV breaks
outside that simple Qwen path, and **both runtimes now point above raw kernel bring-up**:
- **vLLM Gemma 3 27B Rung 1** (`results/vllm_gemma3_27b_rung1_nvfp4_20260608T1924JST.md`):
  capacity ✓ (1.777×), geometry ✓ (D=128, 52 local SWA + 10 global), FA2 NVFP4 path
  selected ✓ — but **output quality FAILS**.
- **SGLang Qwen** (see its doc): the FP4 first-token divergence localized to the
  **radix/prefix cache** — `--disable-radix-cache` makes FP4 output match fp8 again.

Earlier wording over-weighted SWA/window reuse. The updated vLLM trace
(`results/vllm_gemma3_27b_rung1_trace_20260609T0015JST_summary.md`) proves the failing
short prompts do **not** require sliding-window eviction (`max_request_num_skipped_tokens=0`)
and sampled read-side packed data/scale bytes match write-side bytes (`195 / 195`,
`0` mismatches). SGLang similarly cleared simple stale/wrong-page scale-buffer hypotheses
for sampled cached-prefix pages. So the live shared root-cause class is narrower:
**linear full-attention NVFP4 KV is proven, but non-Qwen/reused/windowed model paths need
tensor-level attention/logit comparison because byte-level page pairing is no longer enough.**
Compare root causes with the SGLang radix finding, but do not assume the vLLM Gemma 3
failure is SWA eviction until a long-prompt trace proves it.

First-token update (`results/vllm_gemma3_27b_rung1_first_token_20260608T205432JST.md`):
the refreshed packet keeps the no-downgrade source-overlay policy, reruns fp8 and NVFP4,
and proves NVFP4 corruption is present on the first generated token. The fp8 row chooses
`spark`, `4`, and `A`; the NVFP4 row chooses unrelated Cyrillic/CJK tokens with `0.0`
top-logprob overlap for every probe case. So this is not late decode compounding or
sampling noise; attention/KV state or logits are already wrong before sampling.

Audit caveat: those first-token prompts are only `18`, `23`, and `24` tokens, far below
Gemma 3's `sliding_window=1024`. They therefore do not require SWA eviction or window
rotation to fail. SWA/hybrid metadata is still the only new model-family variable versus
Qwen, but the next trace must first prove the base Gemma NVFP4 write/read/page-pairing
path before attributing the defect to eviction.

vLLM code-map audit: Gemma 3 SWA uses common `SlidingWindowSpec` and common block-table
machinery. NVFP4 data and scales are not separate pools; they are slices of the same
physical KV page (`[K_data | K_scale | V_data | V_scale]`). Therefore the next audit is:
`SlidingWindowManager` / skipped-block lifecycle, `BlockTable` slot mapping,
`FlashInferImpl.do_kv_cache_update`, `reshape_and_cache_nvfp4_dispatch`, and FlashInfer
read-side `nvfp4_kv_cache_split_views` / paged metadata. The invariant to prove is that
write-side slot mapping and read-side page IDs pair packed data and FP8 scale views from
the same physical block.

Minimal vLLM trace plan (read-only audit, 2026-06-08): keep this Python-side so it works
in the source-overlay loop and does not require a new image or C++ rebuild. Add env-gated
JSONL tracing with `VLLM_SPARK_KV_TRACE=1`,
`VLLM_SPARK_KV_TRACE_FILE=/tmp/vllm_gemma3_kv_trace.jsonl`,
`VLLM_SPARK_KV_TRACE_LAYERS=...`, `VLLM_SPARK_KV_TRACE_LIMIT=512` or higher, and
`VLLM_SPARK_KV_TRACE_VALUES=16`. Patch points:

- `vllm/v1/attention/backends/flashinfer.py`:
  `_compute_flashinfer_kv_metadata`, `FlashInferMetadataBuilder.build`,
  `FlashInferImpl.do_kv_cache_update`, and `FlashInferImpl.forward`.
- `vllm/utils/torch_utils.py`: `nvfp4_kv_cache_split_views`.
- `vllm/v1/core/single_type_kv_cache_manager.py`:
  `remove_skipped_blocks` and `SlidingWindowManager.get_num_skipped_tokens`.

Trace events should cover `fi_metadata`, `kv_write_pre`, `kv_write_post_nvfp4`,
`kv_read_views_nvfp4`, and `swa_skip`. Required short-prompt gate: Gemma 3 first-token
probes should show `swa_skip.num_skipped_tokens == 0`; if they still corrupt output with
clean slot/page metadata, the next suspect is NVFP4 data/scale contents or V-scale
swizzle/deswizzle, not SWA eviction.

Implementation update (2026-06-08): `jethac/vllm@e2a8197a9` implements the Python-side
trace hooks above in `vllm/v1/attention/backends/flashinfer.py` and
`vllm/v1/core/single_type_kv_cache_manager.py`. Events are inactive unless
`VLLM_SPARK_KV_TRACE=1` is set, and include `fi_metadata`, `kv_write_pre`,
`kv_write_post_nvfp4`, `kv_read_views_nvfp4`, and `swa_skip`. Next vLLM action is to rerun
the Gemma 3 27B fp8/NVFP4 first-token packet with `VLLM_SPARK_KV_TRACE_FILE` enabled for
layers 0/1, then compare slot mapping, block-table pages, NVFP4 split-view offsets, sampled
data/scale bytes, and `swa_skip` against the first-token quality split.

Trace run result (2026-06-09):
`results/vllm_gemma3_27b_rung1_trace_20260609T0015JST_summary.md` records the clean
source-overlay rerun. fp8 first-token probes pass; NVFP4-KV reproduces the immediate
quality failure with `0.0` top-logprob overlap for all three probes. The first
`TRACE_LIMIT=8` row was unusable for page-pair proof because warmup/graph-capture consumed
the budget. The high-limit rerun records `558` writes, `234` read-view events, and `195 /
195` matched read samples with `0` mismatches and `0` missing writes. For client requests,
`max_request_num_skipped_tokens=0`. **Conclusion:** Gemma 3 NVFP4-KV is still red, but not
because sampled packed data and scale views point at different physical pages, and not
because these first-token prompts rotate or evict the SWA window. Next vLLM action is a
tensor-level compare: local/global attention output, NVFP4 quant/dequant numerical error,
V-scale deswizzle contents, final hidden state, and logits before sampler preprocessing.

Run packet: `tasks/vllm_gemma3_nvfp4_trace_packet_20260608.md` records the source-overlay
image/env/client packet. Its trace limit has been updated after the low-limit lesson.

Tensor-trace implementation update (2026-06-09): `jethac/vllm@5b67b0ea2`
(`spark/hijinks-021-gemma3-tensor-trace`) adds inactive-by-default
`VLLM_SPARK_GEMMA_TENSOR_TRACE=1` summaries in the Gemma 3 model path and FlashInfer
attention path. It records last-token summaries for Q/K/V, FlashInfer attention input and
output, Gemma layer residual/norm/MLP boundaries, final hidden state, and logits top-20.
Use `tasks/vllm_gemma3_tensor_trace_packet_20260609.md` and
`scripts/vllm_gemma_tensor_trace_compare.py` for the next fp8-vs-NVFP4-KV rerun. The
diagnostic goal is first-divergence localization, not a benchmark row. The first
normal-compile attempt on `bfa123e1f` failed in TorchDynamo during model profiling, so
`5b67b0ea2` disables trace emission while Dynamo is compiling; use `--enforce-eager` for
the live tensor-summary rows.

Live tensor-trace result (2026-06-09):
`results/vllm_gemma3_27b_tensor_trace_20260609T0115JST_summary.md` records the matched
fp8/NVFP4-KV eager diagnostic rows. fp8 returned `spark`, `4`, and `A`; NVFP4-KV returned
` Reigns`, Gujarati text, and `ioane`, with `0.0` top-logprob overlap in all three
first-token probes. The compare matched `561` event/layer keys and localizes the first
strong tensor-level corruption to `flashinfer_attn_output`: NVFP4-KV outputs are
BF16-shaped but become almost entirely nonnegative, with means around `124..126` and max
values exactly `255.0` on many layers. The final hidden-state RMS later looks nearly
identical, but the logits top-20 sets are disjoint. Next vLLM action is a focused
FlashInfer FA2 NVFP4 attention-output probe for Gemma 3 `D=128` local/global shapes:
verify output scaling, dequantization, V-scale deswizzle, and output-buffer
interpretation against a dequantized reference.

Standalone attention-output probe result (2026-06-09):
`results/vllm_flashinfer_gemma3_attention_output_probe_20260609T0134JST_summary.md`
extends `scripts/flashinfer_nvfp4_kv_probe.py` with signed E2M1 values, non-unit
K/V global scales, and actual/expected output stats. The signed swizzled row
(`FLASHINFER_PAGED_V_SF_DESWIZZLE=1`) and signed linear-control row both pass for
NHD/HND decode and prefill at Gemma 3 geometry (`D=128`, `16` KV heads, `32` Q heads):
minimum cosine `0.999997496604919`, max absolute error `0.0001220703125`, and `0 / 4`
byte-like actual outputs in each row. This clears generic standalone FA2 signed-FP4,
non-unit K/V scale, and deswizzle handling for the synthetic case. It does **not** clear
vLLM Gemma. Next vLLM action is to instrument or dump the real wrapper boundary for the
first failing Gemma request: actual model `query`, split packed K/V, K/V scale tensors,
scalar `k_scale`/`v_scale`, output buffer dtype/shape/stride before and after
`wrapper.run(...)`, and whether prefill or decode first produces byte-like output.

Wrapper-boundary trace result (2026-06-09):
`results/vllm_gemma3_27b_wrapper_trace_20260609T0148JST_summary.md` runs
`jethac/vllm@0e07e130d94eddfed209f846ce6c9959c636da02` in the no-downgrade source
overlay and reproduces the same bad first tokens (` Reigns`, Gujarati text, `ioane`).
The real Gemma/FlashInfer FA2 paged prefill wrapper is now the failing boundary. For the
24-token full/global layer call (`window_left=-1`, `num_prefill_tokens=24`,
`num_decode_tokens=0`), the actual model `query_last` and `out_before` are sane signed
BF16 tensors, but immediately after `BatchPrefillWithPagedKVCacheWrapper.run(...)`
`out_after` becomes byte-like BF16: head `[240.0, 1.7265625, 226.0, 137.0, 145.0, 20.0,
186.0, 185.0]`, max `255.0`, mean `129.27871704101562`, RMS
`147.77296447753906`. The subsequent `flashinfer_attn_output` event matches that tensor.
The active layer-5 KV sample has `v_data_head` beginning `[240, 1, 226, 137, 145, 20,
186, 185, ...]`, so the live clue is that the paged prefill path is returning or
interpreting packed V payload bytes as attention output. The same byte-like pattern also
appears in local/SWA prefill calls, while short prompts remain far below the SWA window.
Next vLLM action is no longer a generic FA2 probe; dump/replay the exact active
block-table paged prefill call and compare wrapper output against a dequantized reference.

SWA code-read update (2026-06-08): Gemma 3 local layers are real `SlidingWindowSpec`
groups, while global layers are `FullAttentionSpec`. NVFP4 packed data and FP8 scale
buffers use the same physical page layout for local and global layers; SWA does not rotate
scale buffers separately. The differing path is whole-block lifecycle: skipped blocks are
removed, nulled, and later reused by the sliding-window manager. Therefore the next vLLM
GPU trace should include a prompt longer than `sliding_window + 2 * block_size + 1`, with
one sliding layer and one global layer traced. If write/read bytes match but quality still
fails, run the same prompt with `--disable-hybrid-kv-cache-manager`; if that passes, the
bug is SWA block-table/window reuse rather than NVFP4 encode/decode.

## The Gemma problem, precisely (three intertwined blockers)
Gemma's blocker and the NVFP4-KV blocker are the same blocker: heterogeneous/dual head
dimensions (local layers `D=256`, global layers `D=512`) plus alternating SWA.

Rung -1 config audit update (2026-06-08): this statement applies to the audited Gemma 4
server models, not Gemma 3. `docs/GEMMA_RUNG_MINUS1_CONFIG_AUDIT.md` shows Gemma 3 27B is
uniform `D=128` with SWA/full layer alternation and no `D=512`; Gemma 4 12B, 31B, and
26B-A4B all carry full-attention `D=512`. Therefore the next vLLM rung is Gemma 3 27B to
prove SWA/hybrid KV separately before returning to Gemma 4 mixed-KV.

Ladder order update (2026-06-08): after Gemma 3 27B, do **not** jump to 12B. Use the
operator-provided encoder/modality architecture to preserve one new complication per rung:
Gemma 4 31B text-only is the clean dense `D=512` mixed-KV rung; Gemma 4 26B-A4B text-only
adds MoE; Gemma 4 12B is last because its encoder-free multimodality is fused into the
decoder/KV path and cannot be quarantined by text-only serving. Confirm this from the
running model on every rung.

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
Load Gemma rungs through source overlay on the `cleanfa2-patchedfa2-cutlass` image and
record the *actually selected* attention + MoE backend, quantization path, and whether KV is
bf16/fp8/nvfp4. The immediate sequence is Gemma 3 27B -> Gemma 4 31B text-only -> Gemma 4
26B-A4B text-only -> Gemma 4 12B. Goal: replace the `da1daf40` + Transformers surgery with
a reproducible clean-stack Gemma baseline. This is compatibility, not a speed/KV claim.

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

**Parallel-lane split (read this).** The D=512 diagnosis (register/fragment-shape guard,
not a tile/SMEM tweak) means a *true* full-FP4-KV D=512 kernel is a hard, separate
FlashInfer track — back-burner, benefiting both lanes only if/when it lands. **Mixed KV
(this objective) is the primary near-term Gemma path.** SGLang pursues the *same* mixed-KV
strategy through a *different mechanism* — its hybrid-SWA subpool delegation (see SGLang doc
Objective E) — so the two lanes implement one proven-viable approach through non-overlapping
code and cross-validate it. vLLM's mechanism is per-layer `kv_cache_dtype_skip_layers` +
per-`AttentionSpec` dtype. The only shared surface is the FlashInfer guard itself:
coordinate so neither lane edits `prefill.cuh`'s trait math blindly — any real D=512 kernel
work lands once, in FlashInfer, not twice.

**C. Decouple SWA handling from the dual-head-dim fix using Gemma 3 27B.**
The Rung -1 audit confirms `google/gemma-3-27b-it` has SWA/full layer alternation with
uniform `D=128` and no `global_head_dim=512`. Use it as the first end-to-end
NVFP4-KV-on-Gemma proof; it de-risks the SWA path independently. Do not call it a `D=256`
test.

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
Bring up **Gemma 3 27B Rung 1** through source overlay on the proven Qwen NVFP4-KV stack,
with matched fp8-vs-NVFP4 KV rows. The server log/artifact must include running-model
geometry for every layer: `head_dim`, query heads, KV heads, full/sliding layer map,
window, selected backend/KV dtype, KV page layout, and runtime bytes/token. Gate green only
when both local/sliding and full/global Gemma 3 layers select the intended FA2 NVFP4-KV path,
capacity improves versus fp8 at matched settings, and output quality passes a real
comparator.

After Gemma 3 is green, the next vLLM rung is **Gemma 4 31B text-only**, not 12B. Use it
to prove the per-layer mixed-KV dodge on a dense, encoder-quarantined `D=512` model:
**NVFP4 local + bf16 global** before any full `D=512` NVFP4 claim. Then add MoE on 26B-A4B
text-only. Defer Gemma 4 12B until the final multimodal-KV rung, and defer every image
build until Objective E.
