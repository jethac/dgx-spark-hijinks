# vLLM Gemma 4 Rungs — Modification Plan (Rungs 2–4)

Status: **Phase 0 IMPLEMENTED + GB10 probes answered (2026-06-10).** Branch
`jethac/vllm@spark/hijinks-022-gemma4-mixed-kv` (worktree lane, Codex's checkout
untouched), commits `b815a882b..cbfe86bd5`:

- **M2** `b815a882b` — `"full_attention"` skip literal in `attention.py`.
- **M4** `bcfe2d445` — `unify_kv_cache_spec_page_size` scale-then-pad fallback for
  non-divisible mixed page sizes (+ waste logging); fail-fast guard when the hybrid KV
  manager is disabled for mixed-dtype specs.
- **M3+M5** `0f03d4f12` — `cache_dtype_str` on the base `AttentionSpec` (resolves the MLA
  "less hacky" TODO), populated per layer (incl. TQ spec); FlashInfer builder + reshape
  paths derive per-group dtype strings (tri-state); `kv_cache_dtype_skip_layers` accepts
  `"<matcher>=<dtype>"` overrides (fp8-global). `use_uniform_kv_cache` + the runner's
  global dtype verified no-change-needed by construction.
- **M6** `b91a444c1` + `cbfe86bd5` — Gemma4Config force relaxed under mixed-KV; global
  D>256 layers **explicitly pinned to TRITON_ATTN** via the `Attention(attn_backend=...)`
  override (see probe results below — automatic fallback was falsified on hardware).

**Probe results** (`results/vllm_gemma4_mixed_kv_probes_20260610_summary.md`): open
question **#2 answered — FA2 bf16-D=512 fails the identical, dtype-independent trait
guard** (`prefill.cuh:2615`, `NUM_MMA_D_VO=32`) on GB10, so Triton-global is mandatory.
Open question **#1 split**: two-backend worker machinery is real (per-layer
AttentionGroups; TurboQuant skip-layers is prior art), **but** the selector's
`validate_configuration` over-promises head-512 for FlashInfer, so the fallback had to be
an explicit pin, not automatic (upstream-worthy FlashInfer/vLLM bug, noted).

Not yet run: M1 arch smoke (`gemma4_unified` load), live two-builder coexistence, and the
rungs themselves. Code anchors below were read from the actual fork checkout
`third_party/vllm @ 0c278200e` (v0.13.0rc1-5263), not assumed. This extends the
green Rung 1 result to the Gemma 4 server variants per `docs/GEMMA_COMPATIBILITY_PLAN.md`:

- **Rung 2 — Gemma 4 31B (dense, encoder-based text+vision, D=512)** — text-only, then vision rows
- **Rung 3 — Gemma 4 26B-A4B (MoE, encoder-based text+vision, D=512)** — text-only, then vision rows
- **Rung 4 — Gemma 4 12B (dense, encoder-free text+vision+audio, D=512) — multimodal-in-KV**
- **Rung 5 — Gemma 4 E2B/E4B (dense mobile, PLE, encoder-based text+vision+audio) on vLLM**

Out of scope: the SGLang lane and authoring a true D=512 FP4 kernel (back-burner, §6).
Note on scope vs `docs/GEMMA_COMPATIBILITY_PLAN.md`: that ladder routes E2B/E4B to the
llama.cpp/LiteRT *on-device* side track — that track stands; Rung 5 here additionally
covers **server-side vLLM serving** of the E-variants (images + audio), which the
compatibility plan did not claim.

## Key findings from the code read (what changed vs. earlier assumptions)

Grounding this plan in the actual fork source changed it in four material ways:

1. **The bring-up needs zero vLLM code.** `Attention.__init__` already implements
   per-layer dtype skip (`cache_config.kv_cache_dtype_skip_layers`, matched by explicit
   layer index, falling back to `"auto"`/bf16). NVFP4-local + bf16-global works **today**
   by passing the measured global-layer indices. The "clean patch" is only a ~5-line
   `"full_attention"` literal in the matcher, symmetric with the existing
   `"sliding_window"` literal. (§2-M2)
2. **The allocator problem has an existing lever.** Mixed specs form separate KV groups
   automatically (grouping is by spec equality); the only hard constraint is *cross-group*
   page-size equality, and `AttentionSpec.page_size_padded` exists precisely to pad pages
   to a uniform size. M4 is "compute max page size, pad the smaller group" — not a
   bucketing rewrite. One trap: `unify_hybrid_kv_cache_specs` would crash mixed specs if
   the hybrid KV manager is off → assert it ON. (§2-M4)
3. **The D=512 picture sharpened.** The FlashInfer trait guard is head-dim-driven
   (`NUM_MMA_D_VO = 512/16` regardless of dtype), so bf16-D=512 likely fails FA2 too —
   matching `config.py`'s own "FlashAttention rejects head_size > 256" comment. Global
   layers go to **Triton**. Encouragingly, backend selection is already **per-layer**
   (each `Attention.__init__` calls `get_attn_backend` with its own head size + post-skip
   dtype); the only thing forcing model-wide Triton is the `Gemma4Config` override. The
   open question therefore shifted from "does FA2 do bf16-512" (expected no) to **"can v1
   run two backends' metadata builders in one model"** — now probe #1. (§2-M6, §8)
4. **M5 is three precise sites, not a sweep:** `FlashInferMetadataBuilder.__init__` ~1229
   (derive `is_kvcache_nvfp4` from `self.kv_cache_spec.kv_quant_mode`, not the global
   string — the ~8 downstream reads then fix themselves), `attn_utils._reshape_kv_cache`
   ~340 (take the group's spec, else a global-nvfp4 run silently reshapes bf16 layers as
   packed uint8), and `gpu_model_runner.py` ~446/~7228 (the uniform-cache fast path must
   yield to per-group specs). (§2-M5)

## 0. What we build on (proven, do not redo)

- **Rung 1 green:** Gemma 3 27B NVFP4-KV correct + PPL near-free (+0.005 nats/tok at
  ctx 1024/2048, non-compounding), 1.777× capacity, hybrid-SWA pool (52 local + 10 global).
- **FlashInfer SWA FP4 paged-prefill fixed** (`jethac/flashinfer@0919cdda`→`c3dae30f`).
- **Full NVFP4 K+V prefix reuse proven** (Qwen, `prefix_cache_hits_total=3728`).
- **Quality gate:** `scripts/gemma_nvfp4_kv_quality_gate.py`; geometry hook
  (`VLLM_SPARK_KV_GEOMETRY_LOG`) wired in `attention.py`.
- **Memory guardrails:** single server, `--gpu-memory-utilization ≤ 0.72`, Docker
  `--memory 100g`, sequential comparators (`docs/INCIDENT_20260609_OOM_DEADLOCK.md`).

## 1. The blocker and the strategy

All Gemma 4 server variants carry global-attention `D=512` (rung −1 audit). The FA2 NVFP4
path rejects D=512 at the compile-time trait guard in
`include/flashinfer/attention/prefill.cuh`:
`NUM_MMA_Q * (8*NUM_MMA_D_VO + 2*sizeof(DTypeQKAccum)*NUM_MMA_KV) >= 256` — `NUM_MMA_D_VO`
is a function of head dim alone (512/16 = 32 → `8*32 = 256` at the ceiling before the KV
term). **Note: that makes the guard largely dtype-independent — bf16/fp8 KV at D=512 very
likely trips it too** (consistent with vLLM's own comment that FlashAttention rejects
head_size > 256; see §2-M6). So the working assumption is:

**Per-layer mixed KV + per-layer backend:** NVFP4 KV on local SWA layers (D=256, FlashInfer
FA2 — the proven Rung 1 path), bf16-then-fp8 KV on global layers (D=512, **Triton**
backend, which has no head-size ceiling). Gemma's ~5:1 local:global keeps most of the win.

## 2. Phase 0 — shared foundation, with exact code anchors

### M1 — Gemma 4 arch bring-up
- `Gemma4UnifiedForConditionalGeneration` must load on the fork base (released 0.22.1
  can't; old proof needed source `da1daf40` + Transformers main). Pin
  `transformers >= 5.5.0` in the overlay env; record exact pins.
- Cheap dev vehicle: arch smoke on **12B text-only** (~24 GB bf16) — same `gemma4_unified`
  loader, smallest weights. Not a rung claim.
- Text-only quarantine: `--limit-mm-per-prompt image=0` (RedHatAI card); verify it skips
  encoder memory on our build.

### M2 — per-layer KV dtype: mostly already exists; one small extension

**Finding (big one):** `Attention.__init__` already implements per-layer dtype skip,
`attention.py` ~lines 252–275:

```python
if cache_config is not None and cache_config.kv_cache_dtype_skip_layers:
    skip = False
    if (sliding_window is not None
            and "sliding_window" in cache_config.kv_cache_dtype_skip_layers):
        skip = True
    layer_idx = extract_layer_index(prefix)
    if str(layer_idx) in cache_config.kv_cache_dtype_skip_layers:
        skip = True
    if skip:
        kv_cache_dtype = "auto"   # falls back to model dtype (bf16)
```

Two consequences:
- **Zero-code bring-up path:** NVFP4-local + bf16-global works *today* by passing the
  global layers' **explicit indices** in `kv_cache_dtype_skip_layers` (e.g.
  `--kv-cache-dtype nvfp4` + skip-layers `"5,11,17,..."`). Indices MUST come from the
  measured `config.layer_types` map, not an assumed cadence. Use this for first bring-up
  before writing any code.
- **Clean change (upstreamable):** add a `"full_attention"` literal to the skip matcher,
  symmetric with the existing `"sliding_window"` literal — i.e. skip when
  `sliding_window is None and "full_attention" in skip_layers`. ~5 lines in the block
  quoted above. (Caveat: `sliding_window is None` is the proxy `Attention.__init__` has
  for "global layer"; verify it's reliable for Gemma 4 — see M2b.)

**M2b — layer classification source.** `gemma4.py` already classifies layers and selects
head dims (`Gemma4Attention.__init__` ~434, `Gemma4DecoderLayer.__init__` ~559):

```python
layer_type = config.layer_types[layer_idx]
self.is_sliding = layer_type == "sliding_attention"
sliding_window = config.sliding_window if self.is_sliding else None
...
self.is_full_attention = layer_type == "full_attention"
head_dim = getattr(config, "global_head_dim", config.head_dim) \
    if self.is_full_attention else config.head_dim
```

and constructs `Attention(..., per_layer_sliding_window=sliding_window, ...)`. So:
the dtype split keys off `config.layer_types` (via `sliding_window`/head_size reaching
`Attention.__init__`) — no model-file changes expected. The per-rung geometry measurement
verifies `layer_types` against the running model.

### M3 — fp8-global (second step, needs a real per-layer dtype string)
The skip path can only fall back to `"auto"`/bf16. For fp8-global:
- `vllm/config/cache.py` (CacheConfig): allow `kv_cache_dtype_skip_layers` entries of the
  form `full_attention=fp8_e4m3` (or add a sibling `kv_cache_dtype_overrides` map), so the
  skip block in `attention.py` sets `kv_cache_dtype = "fp8_e4m3"` instead of `"auto"`.
- `vllm/v1/kv_cache_interface.py`: `AttentionSpec` already carries per-layer `dtype:
  torch.dtype` and `kv_quant_mode: KVQuantMode` — fp8 maps via the existing
  `get_kv_quant_mode("fp8...") → FP8_PER_TENSOR`. Likely needs a `cache_dtype_str`-style
  field only if backends must distinguish `fp8_e4m3` vs `fp8_e5m2` per layer (MLA specs
  are the precedent).
Defer M3 entirely until the bf16-global rung is green.

### M4 — allocator: cross-group page-size equality is the real constraint
`kv_cache_utils.py` groups by **spec equality** (dict keyed on the frozen dataclass), so
local NVFP4-D=256 `SlidingWindowSpec` and global bf16-D=512 `FullAttentionSpec` form
separate groups automatically — within-group uniformity is free. The hard constraint is
**across** groups, `_get_kv_cache_groups_uniform_page_size` (~1074):

```
Key assumptions:
1. Physical memory per block: Must be the same across all KV cache groups.
```

So both groups must land on the same `page_size_bytes`. The existing lever is the
`page_size_padded: int | None` field on `AttentionSpec` — its `page_size_bytes` property
already honors it:

```python
if self.page_size_padded is not None:
    assert self.page_size_padded >= real_page_size
    return self.page_size_padded
```

**Change:** compute `max(real_page_size_bytes)` across the model's specs at spec-build
time and set `page_size_padded` on the smaller group(s). Sizing intuition (verify with
measured geometry): NVFP4 packs ~4.5 bits/elem vs bf16's 16, so NVFP4-D=256 pages are
*smaller* than bf16-D=512 pages → the NVFP4 group gets padded → some capacity loss on the
local layers. **Report the padding overhead honestly in the capacity row.**

**Risk to check:** `unify_hybrid_kv_cache_specs` (~1335) converts SlidingWindowSpec →
FullAttentionSpec when the hybrid KV manager is disabled — with mixed dtypes that
conversion would feed `merge()` specs with different `dtype`/`kv_quant_mode` and trip its
assertions. Mixed-KV must require the hybrid manager ON (assert early with a clear error).

### M5 — kill the global-dtype assumptions (three concrete sites)
1. `vllm/v1/attention/backends/flashinfer.py`, `FlashInferMetadataBuilder.__init__`
   (~1229): currently
   ```python
   self.cache_dtype = self.cache_config.cache_dtype
   self.is_kvcache_nvfp4 = self.cache_dtype == "nvfp4"
   ```
   Builders are instantiated per attention group and already hold `self.kv_cache_spec` —
   derive from the spec instead:
   `self.is_kvcache_nvfp4 = self.kv_cache_spec.kv_quant_mode.is_nvfp4`, and resolve
   `self.kv_cache_dtype` from the spec's mode/dtype rather than the global string. The
   SM12x FA2 routing + deswizzle block (~1235–1304, the "Using FlashInfer FA2 backend for
   NVFP4 KV cache on SM12x" log) then applies only to the NVFP4 group. Downstream global
   reads (~1413, 1459, 1580, 1790, 1864, 1877, 1894, 1957 — all on
   `self.kv_cache_dtype`/`self.is_kvcache_nvfp4`) become correct automatically once the
   builder fields are spec-derived.
2. `vllm/v1/worker/gpu/attn_utils.py`, `_reshape_kv_cache` (~340): takes a single
   `cache_dtype: str` and applies the NVFP4 packed-uint8 reshape to every group. Change
   the signature to take the group's `AttentionSpec` and branch on
   `spec.kv_quant_mode.is_nvfp4` — otherwise a global `--kv-cache-dtype nvfp4` run
   reshapes the bf16 global layers as packed NVFP4 (silent corruption).
3. `vllm/v1/worker/gpu_model_runner.py`: ~446 (`self.kv_cache_dtype =
   kv_cache_dtype_str_to_dtype(cache_config.cache_dtype, ...)`) and the allocation branch
   (~7228, `use_uniform_kv_cache(self.attn_groups, cache_dtype)` →
   `_build_kv_cache_for_uniform_attn_groups` / `_reshape_kv_cache_tensors`): the mixed
   case must take the non-uniform path with per-group specs; audit every consumer of the
   global `self.kv_cache_dtype` field in this file and re-source from the group spec.

### M6 — backend: relax the Gemma 4 Triton force into a per-layer split
`vllm/model_executor/models/config.py`, `Gemma4Config.verify_and_update_config` (~57–106)
currently forces one backend for the whole model:

```python
max_head_dim = max(head_dim or 0, global_head_dim or 0)
if (head_dim is not None and global_head_dim is not None
        and head_dim != global_head_dim and max_head_dim > 256
        and vllm_config.attention_config.backend is None):
    vllm_config.attention_config.backend = AttentionBackendEnum.TRITON_ATTN
```

Its own docstring says why: FlashAttention rejects head_size > 256, and it forces Triton
*for all layers* to "prevent mixed-backend numerical divergence."

Key mechanical fact: backend selection is already **per layer** — each
`Attention.__init__` calls `get_attn_backend(head_size, dtype, kv_cache_dtype, ...)` with
that layer's head size and (post-skip) dtype. So with the force removed, local layers
(256, nvfp4) can resolve FlashInfer while global layers (512, auto/bf16) resolve Triton —
mechanically plausible **but unverified** in v1's attn-group machinery (metadata builders
are per (group, backend); confirm two backends' builders coexist in one model).

**Change:** narrow the force — when mixed-KV mode is active, instead of forcing
TRITON_ATTN globally, leave local layers free to resolve FlashInfer and pin only the
global-D=512 layers to Triton (e.g. via per-layer backend hint at the `Attention`
construction site in `gemma4.py`, or a head-size-aware fallback in `get_attn_backend`).
Keep the existing global force as the default for non-mixed runs (it is correct today).

**Probe first (cheap, standalone, decides M6):**
1. FlashInfer FA2 at `D=512, dtype_kv=bf16` on GB10 — expected to FAIL at the same trait
   guard (it's head-dim-driven); confirm rather than assume.
2. A two-backend toy config (FlashInfer + Triton groups in one model) — does v1 build both
   metadata builders and serve? This is open question #1; everything in M6 hangs on it.

### Phase 0 exit gate
- Gemma 3 27B still green after M-changes (it exercises the same hybrid grouping with
  uniform dtype — the natural regression guard), plus one Qwen NVFP4 row.
- A synthetic mixed row (any model, indices-based skip) shows: per-layer dtype in the
  server log, two KV groups with padded-equal page sizes, both backends' builders alive.

## 2.5 Multimodal rows — shared mechanics and gates (applies to Rungs 2b/3b/4/5)

How multimodal tokens meet the KV work: on the **encoder-based** variants (31B, 26B-A4B,
E2B/E4B) the vision/audio encoder output is projected into the decoder sequence as tokens —
so image/audio tokens **occupy decoder KV slots and flow through the mixed-KV machinery
exactly like text tokens.** Text-only quarantines the encoder; a vision row exercises
encoder + projector + multimodal-tokens-in-NVFP4-KV. On the **encoder-free** 12B the
modalities are fused directly (no quarantine possible — that's why it's Rung 4).

Shared requirements for every multimodal row:
- **Batched-token budget:** known trap from `docs/GEMMA4_ON_DGX_SPARK.md` — Gemma 26B
  failed startup with `max_tokens_per_mm_item (2496) > max_num_batched_tokens (2048)`.
  Set `--max-num-batched-tokens 4096` (or measured value ≥ the mm item size) on all
  multimodal rows.
- **Memory re-budget:** the encoder + projector weights now load (text-only rows skipped
  them) — re-measure loaded-model memory per variant and re-size the KV pool; the §7 rule
  (`weights + KV + 20 GiB ≤ 119 GiB`) applies with the encoder counted.
- **Multimodal quality gate:** extend `gemma_nvfp4_kv_quality_gate.py` with an
  image-grounded sanity comparator — same image+prompt through the bf16-KV baseline and
  the mixed-KV candidate; gate on caption/VQA first-token + logprob agreement, not just
  text prompts. (Audio rows: same pattern with an audio clip.)
- **Outlier measurement:** vision/audio token K/V distributions may be more outlier-heavy
  than text. Per modality, capture the KV amax/global-scale stats the pool already logs
  and a modality-specific quality delta — **do not assume the text-row delta transfers.**
- **KV-pressure framing:** multimodal contexts are KV-heavy (hundreds–thousands of tokens
  per image/clip) — these rows are where the capacity win is most visible; record
  tokens-per-image and pool occupancy in the row manifest.

## 3. Rung 2 — Gemma 4 31B (dense D=512 mixed-KV): text-only, then vision

1. **Probes first:** M6 probes + measured geometry from the running model (per-layer
   head_dim map — exactly which indices are D=512 — SWA map, kv_heads, bytes/token per
   group). The measured global-layer indices feed the zero-code skip-layers bring-up.
2. **Memory:** 31B bf16 ≈ 62 GB; 0.72×119 GiB ≈ 86 GB → thin KV pool but workable.
   Bring up with **bf16 weights** (isolates KV variables). The **NVFP4-weights row**
   (`nvidia/Gemma-4-31B-IT-NVFP4` or RedHatAI; attention stays BF16 in those checkpoints)
   is the capacity-demo follow-up (~17–20 GB weights → ~60 GB KV pool). Two rows, labeled.
3. **Sequence:** bf16-KV baseline → mixed via **indices-based skip (zero-code)** → mixed
   via the `"full_attention"` literal (the clean patch) → fp8-global (M3).
4. **Gates:** server log proves per-layer dtype; matched capacity vs full-bf16-KV and
   full-fp8-KV; quality gate measured on 31B (don't inherit Gemma 3's number); one
   prefix-reuse-ON row.
5. **Capacity expectation** (sanity only — compute from measured geometry + padding
   overhead): ~5:1 local:global, NVFP4-local-only ≈ `6/(5·(4.5/8)+1·(16/8)) ≈ 1.25×` vs
   full-fp8 (bf16-global), → `≈ 1.57×` with fp8-global. Minus page-padding loss. **Not
   1.78× — report the honest mixed number.**
6. **Rung 2b — vision rows** (after text-only green): drop `--limit-mm-per-prompt
   image=0`, apply §2.5 (batched-token budget, encoder memory re-budget, image-grounded
   quality gate, vision-token outlier stats). One image-prompt row on the mixed-KV config
   + the bf16-KV comparator. This is the first proof of **multimodal tokens through
   NVFP4 KV** on an encoder-based model.

## 4. Rung 3 — Gemma 4 26B-A4B (adds MoE): text-only, then vision

1. Verify the fork still carries the Qwen-era MoE/hybrid fixes (hybrid KV
   `block_size=None` safety, lazy import, graph-capture alignment — `spark/hijinks-020`).
2. **Checkpoint:** `nvidia/Gemma-4-26B-A4B-NVFP4` (`nvfp4_experts_only`, ~16.5 GB,
   community-proven ~52 tok/s on GB10). Card constraints: **vLLM TP=1 only** (fine), MoE
   backend `VLLM_CUTLASS` or Marlin — record which in evidence.
3. Same gate set as Rung 2; KV machinery should inherit unchanged. Bring up
   `--enforce-eager` first, graphs second (MoE × mixed-KV graph capture is untested).
4. **Rung 3b — vision rows:** same §2.5 pattern as Rung 2b. Architecturally nothing new in
   the KV path beyond 2b (vision × MoE is verification, not new machinery), but note the
   `nvfp4_experts_only` checkpoint quantizes experts only — confirm the vision encoder +
   projector stayed unquantized (checkpoint audit: `nvfp4_checkpoint_audit.py` already
   flags quantized vision/router tensors).

## 5. Rung 4 — Gemma 4 12B (encoder-free multimodal-in-KV) — the destination

1. **Step A — text-only** on Rung 2/3 machinery (audit: 12B configs are unified wrappers
   with `text_config` + vision/audio blocks; whether text-only skips encoder weights must
   be **measured from the loaded model**).
2. **Step B — vision-in-KV:** image rows; extend the quality gate to multimodal sanity vs
   the bf16-KV baseline. Vision-token K/V may be more outlier-heavy than text — **measure
   the multimodal quality delta separately.**
3. **Step C — audio-in-KV**, same pattern.
4. Highest-payoff demo (multimodal contexts are KV-heavy) — feeds the Colab/DevRel
   notebook's final act. Weights: bf16 (~24 GB) or community NVFP4 12B quants
   (AxionML/berkerdooo — community, not NVIDIA-official; label accordingly).

## 5b. Rung 5 — Gemma 4 E2B/E4B on vLLM (PLE + images + audio)

The E-variants are the ex-3n mobile line: dense, **per-layer embeddings (PLE)**,
encoder-based **text+vision+audio**. They are not capacity targets (tiny KV), so this rung
is about **architecture coverage**: proving the whole Gemma 4 family serves on vLLM with
the NVFP4-KV machinery, including the mobile arch. Because the models are small (~2–4 GB
class) and independent of the D=512 work, **this rung can run in parallel any time after
Phase 0** — it never gates the capacity ladder.

1. **Audit first (rung −1 extension):** the existing config audit covered 12B/26B/31B
   only. Audit E2B/E4B: per-layer head_dim map (is D=512 present at all, or uniform?),
   SWA map, **PLE structure**, encoder configs (vision + audio), MatFormer/elastic
   leftovers. If the E-variants are uniform-head, they skip the whole mixed-KV problem
   and can run **full NVFP4 K+V** like Gemma 3 — determine which regime applies before
   planning rows.
2. **Arch support probe:** does the fork load E2B/E4B at all? vLLM grew Gemma 3n support
   (PLE/elastic machinery) in 2025; verify the Gemma 4 E-variants ride that path or the
   `gemma4_unified` one, and that **PLE interacts sanely with the KV-cache spec** (PLE is
   per-layer *embeddings*, not attention KV — expected orthogonal, but verify nothing in
   the spec/grouping path trips on the extra per-layer tensors).
3. **Rows:** text-only → vision → **audio** (§2.5 gates each). Note the audio encoder is
   the first audio-in-KV row on an *encoder-based* variant — simpler than 12B's
   encoder-free audio, so if Rung 4 Step C stalls, this is the cheaper place to debug
   audio-token KV behavior first.
4. **Gates:** standard per-rung set (measured geometry, per-layer dtype log, quality gate
   with image/audio comparators). Capacity comparator is still recorded but is not the
   story here — label these rows "family coverage," not capacity wins.

## 6. Back-burner — true D=512 FP4 kernel (FlashInfer)

Upgrades mixed-KV to full NVFP4 (1.78×) on Gemma 4 later.

**Why back-burner (explicit reasoning):**
- Not on the critical path: mixed-KV greens every rung, and most of mixed-KV already
  exists in vLLM (skip-layers, `page_size_padded`). The kernel only upgrades the ~1-in-6
  global layers fp8→NVFP4: roughly 1.57× → 1.78× pool — the smallest prize on the board.
- Different work class: the SWA-prefill fix completed an existing struct; this is a
  **register-budget wall**. A 512-wide head saturates the per-thread fragment budget
  (`8×NUM_MMA_D_VO = 256` before the KV term), so it needs a new fragment layout — split
  the head into two 256 chunks, and because the QK dot-product sums across the head dim,
  the partial products must be reduced **before** softmax. Cross-fragment reduction
  choreography = real kernel authoring, uncertain timeline.
- Risk asymmetry: if the kernel slips, nothing is blocked; if rungs wait on it,
  everything is hostage to the hardest task.
- Honest counterargument: if this kernel landed, it would **delete most of Phase 0**
  (no dtype split, no padding, no two-backend question — Gemma 4 runs uniform full-NVFP4
  like Gemma 3). One hard fix vs. a pile of plumbing. We pick the plumbing because it is
  bounded and mostly already written; the kernel is unbounded risk.

**Promotion trigger:** if open question **#1 fails** (v1 cannot run two attention
backends in one model), mixed-KV loses its global-layer backend and gets much harder —
this kernel then comes OFF the back-burner and becomes the main path. The two cheap
probes therefore decide the plan's whole shape; run them first.

Own FlashInfer task + issue; never blocks a rung while probes pass. (If the M6 probe
shows bf16 D=512 also fails FA2 — expected — this kernel would unlock FlashInfer-global
for *all* dtypes, which strengthens its upstream value as a standalone contribution.)

## 7. Cross-cutting rules

- One rung at a time; green = quality gate pass + capacity comparator + per-layer dtype
  proof + measured geometry (from the running model — the skip indices route on it).
- Quality measured per variant; Gemma 4's outlier behavior may differ from Gemma 3's.
- Memory: size `weights + KV + 20 GiB headroom ≤ 119 GiB` per run; sequential comparators.
- Regression guard after each phase: Gemma 3 27B + one Qwen NVFP4 row.
- No image bakes until a rung is green; SM120 ride-along policy applies to anything shipped.

## 8. Open questions / risks (resolve in this order)

| # | Question | Decides | How |
|---|---|---|---|
| 1 | Can v1 run **two attention backends** (FlashInfer + Triton) in one model? | M6 design — the biggest unknown | toy two-backend probe; audit attn_groups/builder wiring |
| 2 | Does FA2 reject **bf16 D=512** at the same trait guard? | whether Triton-global is mandatory (expected: yes) | standalone FA2 probe, no serving |
| 3 | Does `gemma4_unified` load on fork base + transformers ≥5.5? | M1 scope | 12B text-only smoke |
| 4 | Does `page_size_padded` cross-group padding work end-to-end with the hybrid manager? | M4 | synthetic mixed-spec unit probe |
| 5 | Does `unify_hybrid_kv_cache_specs` fire and break mixed specs when hybrid manager is off? | M4 guard | code audit + assert |
| 6 | Does text-only (`--limit-mm-per-prompt image=0`) skip encoder memory on 31B/26B? | R2/R3 memory budgets | measure loaded-model memory |
| 7 | Are multimodal-token (vision/audio) KV distributions more outlier-heavy? | R2b/R3b/R4/R5 quality expectations | §2.5 outlier stats on the first vision row |
| 8 | Do E2B/E4B carry D=512 or uniform head dims? PLE structure? | R5 regime (full NVFP4 vs mixed) | rung −1 audit extension |
| 9 | Does the fork load E2B/E4B (PLE path), and does PLE coexist with the KV-spec machinery? | R5 feasibility | E2B load smoke |
| 10 | Does vLLM's Gemma 4 audio path work on the fork (both encoder-based E-variants and encoder-free 12B)? | R4 Step C / R5 audio rows | audio smoke on E2B (cheapest) |

## 9. Sequencing summary

```
Probes #1/#2 (two-backend toy; FA2 bf16 D=512) ─┐
M1 arch smoke (12B text-only)                   ├→ Phase 0 (M2 literal, M4 padding, M5 sites, M6 split)
                                                ┘        → exit gate (Gemma 3 + Qwen regression)
→ Rung 2: 31B   (bf16-KV baseline → indices-skip mixed [zero-code] → literal patch
                  → fp8-global → NVFP4-weights capacity row → 2b vision rows)
→ Rung 3: 26B-A4B (MoE; eager → graphs → 3b vision rows)
→ Rung 4: 12B (text-only → vision-in-KV → audio-in-KV)
[parallel after Phase 0, never gating: Rung 5 E2B/E4B on vLLM (audit → load smoke →
 text → vision → audio); §6 D=512 kernel; SGLang lane; llama.cpp/LiteRT on-device track]
```
