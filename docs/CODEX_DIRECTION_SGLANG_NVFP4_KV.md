# Direction: SGLang → NVFP4 KV on Spark (convert proven capacity into blessed quality)

> Standing direction for the SGLang lane. SGLang is the lane where NVFP4-KV **capacity**
> is already proven on GB10 (1.779× fp8 pool) — the closest result to the campaign's
> founding goal (memory / context / concurrency). It is **one correctness bug away** from
> being the headline capacity win, and that bug is the highest-leverage thing in the
> whole NVFP4-KV effort.

## ROOT CAUSE LOCALIZED — the radix/prefix cache (2026-06-08) — NEXT FOCUS
The first-token divergence test cornered it
(`results/sglang_qwen_fp4kv_radix_isolation_20260608T2038JST_summary.md`):
- fp8: OpenAI and native endpoints agree (`**`).
- FP4 default: diverge (OpenAI `**`, native `ark`/838).
- **FP4 with `--disable-radix-cache`: agree again (`**`/334).** Skip-warmup alone does NOT
  fix it; radix-off does.

So the SGLang FP4-KV bug is **radix/prefix-cache KV reuse** — a cached FP4 KV prefix is
mishandled on reuse. `--disable-radix-cache` is both the proof and a correctness workaround
(at the cost of prefix caching). **This is a cross-lane pattern:** vLLM Gemma 3 27B fails
FP4 quality too, and its only new variable is **SWA** (sliding-window KV reuse) — see
`docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md`. Both are FP4 breaking in the **KV-reuse
machinery** (prefix reuse / SWA windows), not in plain linear KV. The shared root-cause
class remains "packed FP4 KV plus FP8 scales under reuse/windowing," but the
`f76f80484` write/read trace clears the simplest stale/wrong-page scale-buffer version for
sampled SGLang pages. **SGLang's job:** prove whether the reused cached-prefix contribution
is numerically identical to a recomputed FP4 prefix contribution; meanwhile a
`--disable-radix-cache` row can land a blessed-with-caveat result. Compare with vLLM's SWA
finding.

Instrumentation head: `jethac/sglang@ce1b6d15e` adds inactive-by-default
`SGLANG_FP4_KV_TRACE_RADIX=1` logs through `Req.init_next_round_input`,
`ForwardBatch.init_new`, and FlashInfer prefill/extend path selection. Use it to compare
the default FP4 native request against the radix-off request and prove whether the cached
prefix's packed KV bytes and FP8 scale buffers stay aligned.

Trace result (`results/sglang_qwen_fp4kv_radix_trace_20260608T213052JST_summary.md`):
the default FP4 native request fails (`**` vs `ark`/`838`) while reusing a 55-token prefix
(`prefix_indices_len=55`, `extend_prefix_lens_cpu=[55]`) and running
`forward_extend_merge_paged`; the radix-off row passes (`**` vs `**`) with
`prefix_indices_len=0`, `extend_prefix_lens_cpu=[0]`, and
`forward_extend_ragged_no_prefix`. This narrows the next fix to FP4 cached-prefix merge
page handling, not raw quantizer math, pool layout, prompt serialization, or graph capture.
The next hook must compare reused page IDs with K-data/K-scale/V-data/V-scale page IDs.

Page-pair trace result
(`results/sglang_qwen_fp4kv_page_pair_trace_20260608T214649JST_summary.md`):
`jethac/sglang@839cb7457` adds `SGLANG_FP4_KV_TRACE_PAGE_PAIR=1` and records the
FlashInfer paged plan plus FP4 data/scale view geometry. The default row still fails, but
the paged plan consumes the same 55 logical page IDs as the radix prefix (`4113..4167`),
and all 28 layers report matching first-dimension extents for K data, V data, K scale, and
V scale. So a gross page-list mismatch is not observed. The next hook moves inside
`extend_merge_paged`: sample actual FP4 data/scale bytes at the reused page IDs and log
`o1/s1/o2/s2` before `_safe_merge_state`.

Merge-state trace result
(`results/sglang_qwen_fp4kv_merge_trace_20260608T220823JST_summary.md`):
`jethac/sglang@991ac1e63` adds `SGLANG_FP4_KV_TRACE_MERGE_STATE=1` and records the
layer-0 cached-prefix merge. The default row still fails (`OpenAI **` vs native
`ark`/`838`) with `cached_tokens=55`; radix-off still passes (`**`/`334`) with
`cached_tokens=0` and no paged-prefix merge. The failing layer-0 trace sees readable,
nonzero packed K/V bytes and FP8 K/V scale bytes at pages `4113..4116`, and `o1/s1`,
`o2/s2`, and merged output are all finite. The merged sample matches the paged-prefix
sample, which is plausible for a 55-token cached prefix over a one-token ragged suffix and
is not alone proof of a merge bug. The next hook is now **write/read pairing**: trace
`MHATokenToKVPoolFP4.set_kv_buffer()` for the same physical page IDs and verify the bytes
and scale bytes written during cache fill match those read during radix reuse. If they
match, build a same-prompt no-prefix reference for the paged-prefix contribution and
inspect FlashInfer FA2 paged-prefix numerics / merge weighting.

Write/read trace result
(`results/sglang_qwen_fp4kv_write_read_trace_20260608T222204JST_summary.md`):
`jethac/sglang@f76f80484` adds `SGLANG_FP4_KV_TRACE_WRITE_READ=1`. The default row still
fails (`OpenAI **` vs native `ark`/`838`) with `cached_tokens=55`; radix-off still passes
(`**`/`334`) with `cached_tokens=0`. For layer 0 cached pages `4113..4116`, sampled K
data, V data, K scale, and V scale all match write input bytes = stored bytes = read bytes.
This clears the simple "scale buffer not copied / stale / wrong page" hypothesis for the
sampled pages. The next hook is now numerical: compare the cached FP4 paged-prefix
contribution against an equivalent no-prefix/full-ragged FP4 reference for the same
55-token prefix and query, then inspect FlashInfer FA2 paged-prefix dequant / LSE / merge
weighting if they diverge.

Prefix-reference trace result
(`results/sglang_qwen_fp4kv_prefix_ref_trace_20260608T2306JST_summary.md`):
`jethac/sglang@2a228949a` adds and fixes
`SGLANG_FP4_KV_TRACE_PREFIX_REF=1`. The default row still fails (`OpenAI **` vs native
`ark`/`838`) with `cached_tokens=55`; radix-off still passes (`**`/`334`) with
`cached_tokens=0`. For the sampled failing layer-0 request, cached paged-prefix `o2`
matches the dequantized torch reference (`cosine=0.999997`, `max_abs=0.0078125`), `s2`
matches after converting the reference LSE to FlashInfer's log2 convention
(`max_abs=0.001953125`), and manual `exp2` merge exactly matches `_safe_merge_state` at
BF16 precision (`max_abs=0.0`). This clears FP4 paged-prefix read/layout/dequant, LSE
units, and merge-state math for the sampled failing case. The next hook moves upward:
trace calibration/quantization-error impact at cache fill and request sequencing/state
across the OpenAI/native radix-hit pair.

## Why this, why now
The SGLang FP4 KV row already expands the KV pool ~1.78× over fp8 on GB10. The newest
`d7d931f` matched row improves the evidence: raw `2+2` and chat smoke pass, and backend
trace covers both decode and `extend_merge_paged`. But the standardized FP4 benchmark
text still degenerates, while fp8 produces normal text. The underlying FlashInfer FA2
NVFP4-KV kernel is proven correct standalone (`jethac/flashinfer@e152cf4d`, cosine ≥
0.99999946), and the layout probe ruled out scale-rank as the cause
(`results/sglang_nvfp4_kv_layout_probe_20260608.json`, dequant cosine 0.9999957).
**So the bug is in SGLang's integration or quality-sensitive serving path, not in the
basic kernel math or pool contract.** The job: find and fix that bug, then land a
blessed matched fp8-vs-FP4 serving row.

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
**divide** convention.

**CONFIRMED on hardware (2026-06-08).** The convention bridge
(`results/sglang_nvfp4_kv_convention_probe_20260608.json`) ran all four quantizer/reader
pairings on GB10. The FA2 reader is a **decode-convention reader**; matched pairs are
numerically correct (cosine 0.995 vs source, 0.99999 vs dequant), and the naive
`nvfp4_kv_quantize` + **encode** scale → decode reader is the literal **cosine-0.0**
off-by-`s_enc²` failure:

| quantizer | reader | attn cosine vs source | verdict |
|---|---|---:|---|
| `fp4_quantize` encode | decode | 0.995 | ✓ valid pair |
| `nvfp4_kv_quantize` **encode** | decode | **0.000** | ✗ the crossed garbage case |
| `nvfp4_kv_quantize` decode | decode | 0.995 | ✓ valid pair |
| `nvfp4_kv_quantize` decode | encode | 0.248 | ✗ mismatched |

So the kernel math is **exonerated** and the valid pairings are known: either
`fp4_quantize` + encode scale, or `nvfp4_kv_quantize` + **decode** (inverted) scale.
SGLang's remaining serving corruption means the **full path produces an unmatched pair**
— the bug is now in calibration / V-scale / backend integration (Objective A), not in
the convention at the raw-math level.

**vLLM is now your reference implementation.** vLLM consumes the *same* FlashInfer FA2
reader and, as of 2026-06-08, serves it **cleanly** with normal content and 1.751× fp8
capacity (`results/vllm_qwen_nvfp4_kv_capacity_20260608T1455JST_summary.md`, server log:
`Using FlashInfer FA2 backend for NVFP4 KV cache on SM12x with vLLM V-scale-factor
deswizzle enabled`). vLLM has therefore already produced a correct end-to-end matched
pair on GB10. Diff SGLang's quantize convention + V-scale layout + calibration against
what vLLM's clean row does; the divergence is the bug.

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
- **Convention bridge is DONE** (`sglang_nvfp4_kv_convention_probe_20260608.json`, see
  table above): valid pairings identified, kernel math exonerated, bug localized to
  calibration / V-scale / backend integration. Do not re-run the raw-math bridge; the
  next work is finding where the *serving* path produces an unmatched pair.
- **Pool bridge is DONE** (`results/sglang_fp4_pool_bridge_probe_20260608.json`,
  `results/sglang_fp4_pool_bridge_probe_prefill_20260608.json`):
  `MHATokenToKVPoolFP4.set_kv_buffer()` writes packed K/V plus FP8 scale buffers that
  FlashInfer FA2 can consume directly. The widened probe used real pool getters for both
  `BatchDecodeWithPagedKVCacheWrapper` and `BatchPrefillWithPagedKVCacheWrapper`; both
  passed with `attention_cosine_vs_dequant=0.9999946`, while K/V dequantized back to the
  BF16 source at cosine about `0.9955`. This clears the basic pool layout and
  global-scale application surface for decode and paged prefill; the remaining serving
  corruption is later in backend wrapper/server sequencing, CUDA graph state, or a
  model-path difference not covered by the synthetic pool bridge.
- **Backend trace and matched comparator are now captured**
  (`results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_summary.md`):
  `jethac/sglang@d7d931f` adds opt-in `SGLANG_FP4_KV_TRACE_BACKEND=1` logging. A
  source-overlay Qwen run on the NVIDIA 26.05 image, with
  `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`, reached readiness, allocated `5,517,572`
  FP4 KV tokens versus `3,105,240` fp8 tokens (`1.7769x`), calibrated 28 layers, traced
  all 28 decode layers and all 28 `extend_merge_paged` layers through packed `uint8` K/V
  plus FP8 scale buffers, returned `spark-ok`, and produced sane raw `2+2` text. This
  is still not a blessed row because the FP4 standardized benchmark content remains
  degraded.
- **Logprob quality localization is DONE**
  (`results/sglang_qwen_fp4kv_d7d931f_logprob_quality_20260608T1609JST_summary.md`):
  `scripts/openai_quality_probe.py` compared the degraded FP4 prompts against the fp8
  comparator with generated-token logprobs. FP4 `short_decode` starts with the same
  high-confidence prefix as fp8 (`A local AI workstation`) then drifts into mixed
  Chinese/repetition; FP4 `medium_decode` diverges at token one (`the following code:`
  instead of `**Engineering Note:`) and collapses into repeated `import` text. This
  proves the quality bug is prompt/path-sensitive generation corruption, not just a
  missing backend trace or a capacity-only artifact gap.
- **Native `/generate` divergence-window probe is DONE**
  (`results/sglang_qwen_fp4kv_d7d931f_native_divergence_20260608T1626JST_summary.md`):
  rendering the same `medium_decode` chat prompt with the Qwen tokenizer and calling
  native `/generate` gives a sharper window. fp8 and FP4 match for the first four output
  tokens (`**`, `Engineering`, ` Note`, `:`), then diverge at token index 4: fp8 selects
  ` Valid` while FP4 selects ` Validate`. Both alternatives appear in both top-k lists,
  but FP4 reverses their rank. This is an early decode distribution shift that compounds,
  not a first-token catastrophe under native `/generate`.
- **OpenAI-vs-native prompt reconciliation is DONE**
  (`results/sglang_qwen_fp4kv_prompt_path_reconcile_20260608T173754JST_summary.md`):
  the OpenAI path and local Qwen chat-template render use identical 56-token prompt IDs
  for both fp8 and FP4 (`sha256=5a5d4572e0e3d940a909b85dc4a00350094cbd1d55333c3d4f0a7974a91ee517`).
  Prompt serialization is therefore not the cause. The endpoint split is real: FP4 OpenAI
  Chat Completions still starts plausibly and diverges at token 4, while native
  `/generate` from the same prompt IDs diverges at token 0 (`**` -> `ark`). The next bug
  surface is FP4 endpoint/request metadata or pre-sampling numerics, not chat-template
  tokenization.
- **Endpoint metadata localization packet is DONE**
  (`results/sglang_qwen_fp4kv_endpoint_metadata_20260608T1819JST_summary.md`):
  an offline pass over the prompt-reconciliation artifact records the same 56-token prompt
  hash for FP4 OpenAI and native paths, the OpenAI first token `**`, and the native first
  token `ark` (`838`). Existing backend traces cover decode and `extend_merge_paged`, but
  they are not request-tagged, so they cannot separate OpenAI request state from native
  `/generate` request state. The smallest next hook is now defined:
  `scripts/sglang_fp4_first_token_dump_patch.yaml` patches only `ModelRunner.sample()` to
  dump `next_token_logits` before/after `_preprocess_logits()` plus `ForwardBatch`
  `input_ids`, `positions`, `seq_lens`, and `rids`.

Read `docs/NVFP4_KV_PORTING_MAP.md` (SGLang Reference Map) and the autosafe summary
before starting.

## The SGLang problem, precisely
1. **Quality corruption in eager/no-graph serving** (keystone). The latest trace row
   passes raw/chat smoke, but the standardized benchmark content still degrades with
   CUDA graph and piecewise capture disabled. This is not merely a graph bug, and it is
   subtler than the earlier raw `2+2` failure.
2. **Worse corruption with CUDA graphs.** Graph-enabled decode corrupts output, so the
   fork currently auto-disables capture (`SGLANG_FP4_KV_ENABLE_CUDA_GRAPH=1` opt-in).
   This forces the slow no-graph path (an early variant measured 0.276 tok/s). Separable
   from #1 and likely a calibration-vs-capture state problem.
3. **FlashInfer FP4 decode kernel was force-compiled past errors** (`vec_dtypes.cuh`,
   group6 dtype mismatch, packed head-dim — `fi_fp4_decode_*` probes, commit `fb7f0a1`).
   Confirm the decode path is numerically correct, not merely compiling.

## Objectives, in order
**A. Fix the eager-mode quality corruption (keystone).**
The raw-math convention bridge is done (kernel exonerated; valid pairings known). The
bug is in the **serving path producing an unmatched pair**. Trace where SGLang's
end-to-end path diverges from a valid pairing, suspects in order:
   1. **Global-scale convention in the serving path** — the bridge proved
      `nvfp4_kv_quantize` + **encode** scale → decode reader is cosine-0 garbage. Confirm
      which scale SGLang actually feeds at serving time; switch to a valid pairing
      (`fp4_quantize` + encode, or `nvfp4_kv_quantize` + **decode/inverted**) and check
      raw `2+2`. **Cross-check against vLLM's clean row, which already gets this right.**
   2. **V-scale layout** — SGLang default is **symmetric-linear** V scale-factors;
      vLLM uses **B2 swizzle + in-kernel deswizzle**. SGLang must consume the FlashInfer
      kernel built **without** `FLASHINFER_PAGED_V_SF_DESWIZZLE` (vLLM's clean row builds
      it **with** the macro — so do not copy vLLM's flag, copy its *matched-ness*). A
      layout/macro mismatch corrupts V exactly like a convention mismatch.
   3. **Per-layer calibration application / prompt-path state** — the 28-layer /
      4096-token calibration must
      apply the same global scales at quantize and at in-kernel dequant. This is the most
      likely remaining culprit now that raw convention is understood. The prompt
      reconciliation probe proves OpenAI and native paths can use identical prompt IDs,
      yet FP4 OpenAI and native `/generate` diverge differently. Next compare FP4 request
      metadata, forward-mode/prefill state, and pre-sampling logits/hidden states between
      the two endpoints before changing the quantizer again.
   4. Only then the decode kernel itself (Objective B/C overlap).

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

**E. Gemma via SWA-aware mixed KV (Strategy B) — gated behind the Qwen quality fix.**
The shared Gemma blocker is a FlashInfer register/fragment-shape guard on `D=512` (see the
vLLM doc Objective B — `8*NUM_MMA_D_VO = 256 ≥ 256` before the KV term), so a true
full-FP4-KV global-attention kernel is a hard, separate FlashInfer track, not a quick fix.
The near-term Gemma path on **both** lanes is therefore **mixed KV**: NVFP4 on local
(`D=256`) layers, fp8/bf16 on global (`D=512`) layers — capturing most of Gemma's ~5:1
local:global capacity win while dodging the broken kernel.

SGLang's *mechanism* is its **hybrid-SWA subpool delegation** (`swa_memory_pool.py`,
`mem_cache/`): route local-attention layers to the FP4 subpool, global to fp8/bf16. This is
the SGLang counterpart to vLLM's per-layer `kv_cache_dtype_skip_layers` plumbing — the two
lanes implement the *same strategy through non-overlapping code* and cross-validate it.

**Prerequisite — do NOT start this until SGLang's Qwen FP4-KV quality is blessed
(Objectives A–D).** Gemma's SWA complexity will *mask* whether the long-sequence Qwen
degradation is actually fixed; building Gemma on an unblessed Qwen path is building on sand.
SGLang Gemma also has its own open blockers (issue #14). The shared surface is only the
FlashInfer guard — don't edit `prefill.cuh` trait math here; that lands once, in FlashInfer.
Coordinate attention geometry with the vLLM lane
(`docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md`).

Rung -1 config audit update (2026-06-08): `docs/GEMMA_RUNG_MINUS1_CONFIG_AUDIT.md` shows
Gemma 3 27B is the SWA-only server rung: uniform `D=128`, 52 sliding layers, 10 full layers,
and no `D=512`. Gemma 4 12B/31B/26B-A4B all carry full-attention `D=512`, with 26B-A4B
also adding MoE. Once Qwen FP4-KV quality is blessed, SGLang Gemma should mirror the vLLM
ladder by starting with Gemma 3 27B, not Gemma 4.

Ladder order update (2026-06-08): after Gemma 3 27B, the next SGLang Gemma rung is **Gemma
4 31B text-only**, not 12B. Operator-provided architecture says 31B and 26B-A4B are
encoder-based text+vision models, so text-only serving quarantines vision in the unfired
encoder and isolates attention/KV. Prove dense `D=512` mixed-KV on 31B, then add MoE on
26B-A4B text-only. Gemma 4 12B is last because its encoder-free multimodality is fused into
the decoder/KV path; it is the destination, not the stepping stone.

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
The matched `d7d931f` row, OpenAI logprob probe, native `/generate` divergence-window
probe, OpenAI-vs-native prompt reconciliation, and offline endpoint metadata localization
are done. Do not repeat them as-is. Prompt IDs match; prompt serialization is retired as
the cause, and untagged backend traces are insufficient for the endpoint split. The next
step is a live one-token FP4 run with `scripts/sglang_fp4_first_token_dump_patch.yaml`
enabled, then compare OpenAI versus native `next_token_logits` before and after
`ModelRunner._preprocess_logits()`. No serving image is required until quality passes.
