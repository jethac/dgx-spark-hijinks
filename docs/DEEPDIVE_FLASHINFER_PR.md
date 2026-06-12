# Deep-dive: the FlashInfer FA2 head_dim=512 + NVFP4-KV PR

Campaign: `dgx-spark-hijinks`. FlashInfer fork `jethac/flashinfer`, branch
`spark/hijinks-022-fa2-d512`. Diffed against upstream fork point
`a28703432faab15fda7edd71b6c80be0206df973`
(`fix: One sided MOE A2A warp token policy hangs ... (#3371)` — the last upstream
commit before our work). All snippets below are quoted verbatim from
`git diff a2870343..HEAD` with `file:line` anchors.

---

## 1. TL;DR

This PR makes FlashInfer's **FA2 prefill kernel serve `head_dim=512` attention with an
NVFP4 (packed-FP4) KV cache on consumer / Spark Blackwell (CC 12.0 / 12.1)**, which no
shipping kernel could do. Gemma 4's global-attention layers use `head_dim=512`; the FA2
output/value accumulator caps at 256 (register budget), so upstream vLLM/SGLang
**force-route the whole Gemma 4 family to a slow Triton fallback** — and Triton cannot
read a quantized (FP4) KV cache at all. The PR sidesteps the cap with no kernel-math
change: it runs each attention call as **two passes over half of V** (`qk=512, vo=256`),
reusing FlashInfer's existing asymmetric-head-dim machinery (the DeepSeek 192/128
precedent) and concatenating the two output halves exactly. Four supporting fixes make
that path actually dispatch and stay numerically correct on CC 12.x: a smem-aware tile
selector (the `max_mma_kv=0` crash), correct FP4 output sizing from V, independent K/V
strides, and a **linear V-scale-factor mode** so the SF tensor can be sliced along the
head dim.

**Headline evidence** (CC 12.0 *and* 12.1, all green):
- Full Gemma 4 ladder (E2B/E4B/12B/26B-A4B/31B) served NVFP4-KV on a **RTX PRO 6000
  Blackwell (sm_120)** — first non-Spark consumer card — at **3.556x KV capacity**
  with bounded PPL deltas (`results/colab_g4_pro6000_20260612/COLAB_G4_RESULTS.md`).
- Writer-roundtrip cosines `>=0.99999` at `D=512` two-pass VO-split, both bf16 and
  NVFP4 (`>=0.9999983` Block A; `0.99999231` SGLang head-512 roundtrip).
- The `max_mma_kv=0` dispatcher crash fixed and the retirement scorecard green on
  GB10 (sm_121): every FlashInfer cell within +0.028 nats of the Triton comparator,
  31B **better** by 0.040 nats (`results/claude_retirement_scorecard_20260612/`).

This directly answers open upstream issue
[vllm#40677](https://github.com/vllm-project/vllm/issues/40677): an RTX PRO 6000 owner
forcing FLASHINFER for Gemma 4 (incl. the NVFP4 variant) was rejected with
*"head_size not supported"* — exactly the selector-vs-kernel head-512 disagreement this
PR removes.

---

## 2. The problems

### P1 — No FA2 kernel serves `head_dim=512`, so Gemma 4 is stuck on Triton

The FA2 prefill kernel's validity guard counts only the **output/value** MMA tiles
(`NUM_MMA_D_VO`), not the QK width. `include/flashinfer/attention/prefill.cuh:178`,
`KernelTraits::IsInvalid()`:

```cpp
    return ((NUM_MMA_D_VO < 4) || (NUM_MMA_D_VO == 4 && NUM_MMA_KV % 2 == 1) ||
            ...
            (NUM_MMA_Q * (8 * NUM_MMA_D_VO + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >= 256) ||
            ...
```

For `HEAD_DIM_VO=512`, `NUM_MMA_D_VO = 512/16 = 32`, so `8*32 = 256` saturates the
per-thread output-accumulator register budget before the KV term is even added — the
trait is invalid for every config. This is **dtype-independent** (probed identically for
bf16 and FP4 on GB10). Consequence: upstream vLLM logs
`"Gemma4 model has heterogeneous head dimensions ... Forcing TRITON_ATTN backend"` and
routes the entire model to Triton, which (a) is slower (`vllm#38887`: E4B ~9 tok/s on an
RTX 4090) and (b) **cannot read a quantized KV cache** — no NVFP4 capacity win is
reachable on the Triton route.

### P2 — The `max_mma_kv=0` dispatcher crash at (qk=512, vo=256)

Even once you ask for the asymmetric `(512, 256)` trait, the **tile selector** picks a
warp layout whose shared-memory cost is unpayable on CC 12.x. `FA2DetermineCtaTileQ`
chose the `1x4` warp layout (`cta_tile_q=16`, 4 KV-warps) for short prompts. One
`NUM_MMA_KV` step then costs `(qk+vo)*16*4*sizeof(dtype)` = `(512+256)*16*4*2` ≈ 96 KB
plus a 16 KB Q tile — over the ~100 KB/SM smem budget of consumer Blackwell — so the
scheduler computed `max_mma_kv = 0` and **hard-crashed on any short prompt**
(`Unsupported max_mma_kv: 0`). Observed live: SGLang Gemma 4 E4B Rung 0 returned an HTTP
timeout the moment a `decode_as_prefill_vosplit*` request reached
`BatchPrefillWithPagedKVCacheWrapper` (`docs/RESULTS_LEDGER.md` row 39).

### P3 — NVFP4 paged-attention output buffer sized from the wrong tensor

On the packed-FP4 path the Python wrapper sized the output's head dim from **Q**
(`q.shape[-1]`). That silently assumed `head_dim_vo == head_dim_qk`. Under the VO-split
(`vo=256 < qk=512`) the kernel writes `head_dim_vo`-wide rows into a `q.shape[-1]=512`
-wide buffer, garbling the output. (Root cause of the first NVFP4 red:
`FLASHINFER_D512_FA2_KERNEL_PLAN.md`, P0b.)

### P4 — Swizzled V-scale-factor layout cannot be sliced along the head dim

The two-pass VO-split needs to hand each pass **half of V's scale factors** (a head-dim
slice). The default NVFP4 SF layout is a *swizzle* in which a quad of tokens is spread
across the full SF row, so a head-dim slice does not correspond to a contiguous,
sliceable region — it mathematically cannot be cut along the head dim. The fix is a
**linear V-SF mode**, a flag shared by the cache writer (which lays SFs out linearly)
and the kernel reader (which then skips the in-kernel de-swizzle and reads SFs by plain
strides). See `results/d512_vosplit_sf_layout_decision_20260610.md` and the r7-image
incident (a stale writer that swizzled unconditionally, ignoring the knob, produced
deterministic gibberish — `FLASHINFER_D512_FA2_KERNEL_PLAN.md` coherence-hunt verdict).

---

## 3. Does the PR solve them? — yes, with evidence

**The key math fact** (verified in source, `FLASHINFER_D512_FA2_KERNEL_PLAN.md` §2):
attention decomposes **exactly** along the VO dimension. With `V = [V_left | V_right]`
(256+256):

```
S = Q·Kᵀ            (full 512-dim QK — identical in both passes)
P = softmax(S)      (identical)
O = [P·V_left | P·V_right]   (exact concatenation — no online-softmax merge, no LSE games)
```

Unlike a KV-length split (needs online-softmax merge) or a QK-dim split (changes the
logits), a VO split needs **no recombination math**. Two passes, same Q/K, half of V
each, concat outputs.

**Evidence the route is correct and ships:**

| Claim | Number | Artifact |
|---|---|---|
| bf16 (512,256) two-pass matches torch fp32 | cosine **0.9999978** | `results/flashinfer_fa2_vo_split_d512_vo256_probe_20260610...json` |
| NVFP4 KV (512,256) two-pass, linear V-SF | cosine **0.9999986** | `flashinfer_vosplit_p0b_fp4_green_linear_20260610_summary.md` |
| P1 sweep (NHD/HND, batch>1, qo_len=1-as-decode) | cosines **>=0.9999983** | `results/claude_blockA_20260611/BLOCKA_SUMMARY.md` |
| Real 31B full-NVFP4 global geometry (32q/16kv) | cosine **0.9999986** | `results/claude_geomprobe_20260611/GEOMPROBE_SUMMARY.md` |
| SGLang head-512 writer→FA2-reader roundtrip | cosine **0.99999231** | `sglang_gemma4_vosplit_validation_8d85fff9_20260611...md` |

**Evidence it serves end-to-end across both CC 12.x targets:**

- **RTX PRO 6000 Blackwell (sm_120, CC 12.0)** — `results/colab_g4_pro6000_20260612/COLAB_G4_RESULTS.md`.
  All 5 Gemma 4 models green, NVFP4 coherent, **3.556x (= 32/9) KV capacity on every
  model** (format-exact). PPL deltas bounded and model-dependent (better on E2B/E4B/31B,
  worse on the sensitive 12B/26B-A4B pair):

  | model | bf16 KV tok | nvfp4 KV tok | capacity | bf16 PPL | nvfp4 PPL | dPPL |
  |---|---:|---:|---:|---:|---:|---:|
  | E2B-it | 5,616,320 | 19,969,138 | 3.556x | 5.9730 | 5.9335 | -0.0396 |
  | E4B-it | 1,740,455 | 6,188,289 | 3.556x | 4.4416 | 4.4156 | -0.0259 |
  | 31B-it | 27,238 | 96,849 | 3.556x | 5.3439 | 4.9536 | -0.3903 |

- **GB10 / DGX Spark (sm_121, CC 12.1)** — `results/claude_retirement_scorecard_20260612/SCORECARD_SUMMARY.md`.
  The `max_mma_kv=0` fix (head `76af7982`) is validated: every FlashInfer C1 cell within
  +0.028 nats of its Triton comparator (band ±0.05), **31B better by 0.040 nats**, every
  cell double-run bitwise-identical, all 11 servers coherent. FlashInfer rows log
  `TRITON_ATTN selection count = 0` and the `"FA2 VO split"` proof line on D512 rows.

- The **scorecard narrows the "Triton can't read quantized KV" claim** precisely: Triton
  *can* serve fp8 KV, but cannot read NVFP4/packed-FP4 nor run the D512 VO-split — so
  all NVFP4 capacity rows (3.19x–3.57x) are conditional on the FlashInfer route (I2).

Caveat carried (not a kernel issue): `docs/BUG_FLASHINFER_GEMMA3_1B_SERVING_NUMERICS.md`
records a Gemma 3 **1B** serving-numerics deviation at the d256/SWA-512/1-KV-head
geometry — but the same geometry is **CLEAN on sm_120** (FlashInfer vs FLASH_ATTN truth
delta = -0.000757 nats; `COLAB_G4_RESULTS.md` §1B-bisect). The 1B bug is P520-box-specific,
not a property of these kernels.

---

## 4. Code deep-dive

One subsection per change. All snippets quoted from `git diff a2870343..HEAD`.

### 4.1 — head_dim=512 via the two-pass VO-split (output sizing + asymmetric plumbing)

The VO decomposition itself lives in the serving runtimes (vLLM/SGLang issue two passes
over V-halves as strided views). On the FlashInfer side, the PR makes the asymmetric
`(qk, vo)` plan *correct end-to-end*. Two pieces:

**(a) `head_dim_qk` threaded through the scheduler** so the smem-driving QK width reaches
the tile selector. `include/flashinfer/attention/scheduler.cuh` (the `PrefillSplitQOKVIndptr`
signature gains an optional `head_dim_qk`, threaded to `FA2DetermineCtaTileQ` and from
`PrefillPlan`):

```cpp
   const int64_t avg_packed_qo_len = sum_packed_qo_len / batch_size;
-  cta_tile_q = FA2DetermineCtaTileQ(avg_packed_qo_len, head_dim);
+  cta_tile_q = FA2DetermineCtaTileQ(avg_packed_qo_len, head_dim, head_dim_qk);
```

```cpp
   PrefillSplitQOKVIndptr(qo_indptr_h, kv_indptr_h, total_num_rows, batch_size, num_qo_heads,
                          num_kv_heads, head_dim_vo, page_size, max_batch_size_if_split,
-                         enable_cuda_graph, window_left, fixed_split_size, disable_split_kv);
+                         enable_cuda_graph, window_left, fixed_split_size, disable_split_kv,
+                         head_dim_qk);
```

The single-prefill device path passes it too (`prefill.cuh:1985`):

```cpp
-  uint32_t cta_tile_q = FA2DetermineCtaTileQ(packed_qo_len, HEAD_DIM_VO);
+  uint32_t cta_tile_q = FA2DetermineCtaTileQ(packed_qo_len, HEAD_DIM_VO, HEAD_DIM_QK);
```

**(b) Output buffer sized from V, not Q** — the single-prefill, ragged, and paged paths.
`flashinfer/prefill.py:1374` (single) and `:3547` (ragged):

```python
    # NVFP4 packed: unpacked VO width is packed bytes * 2 (supports
    # asymmetric QK/VO; q.shape[-1] assumed QK == VO).
    out_head_dim = v.shape[-1] * 2 if kv_cache_sf is not None else v.shape[-1]
```

`flashinfer/prefill.py:2464` (paged) carries the full rationale:

```python
        # For NVFP4 KV (uint8 packed), v_cache last dim is packed bytes
        # (2 values per byte): the unpacked VO width is v_cache.shape[-1]*2,
        # which equals head_dim_vo even for asymmetric (QK, VO) plans.
        # Using q.shape[-1] here assumed head_dim_vo == head_dim_qk and made
        # the kernel (which writes head_dim_vo-wide rows) garble a too-wide
        # output buffer whenever VO < QK.
        out_head_dim = (
            v_cache.shape[-1] * 2 if kv_cache_sf is not None else v_cache.shape[-1]
        )
```

**Why correct/minimal:** the math is unchanged — the kernel already templates on
`HEAD_DIM_QK` and `HEAD_DIM_VO` independently (DeepSeek 192/128 is shipping precedent).
The only bug was the Python output-sizing assumption `qk == vo`; sizing from V (`*2` to
undo FP4 byte-packing) is the exact unpacked VO width for every plan. (This is P3's fix,
which is what *unblocks* P1 in practice.)

### 4.2 — the `max_mma_kv=0` fix: smem-aware tile selection

`include/flashinfer/utils.cuh:387`. The selector becomes smem-aware: it keeps the
cheap `1x4` layout (`cta_tile_q=16`) only when one MMA_KV step actually fits shared
memory; otherwise it falls back to `cta_tile_q=64` (`4x1`, a 4x-cheaper KV step) — the
same escape hatch the kernel already uses for Turing:

```cpp
-inline uint32_t FA2DetermineCtaTileQ(int64_t avg_packed_qo_len, uint32_t head_dim) {
+inline uint32_t FA2DetermineCtaTileQ(int64_t avg_packed_qo_len, uint32_t head_dim,
+                                     uint32_t head_dim_qk = 0) {
+  // head_dim is the VO dim at the batch-prefill call sites; head_dim_qk (when
+  // nonzero) lets asymmetric (QK != VO) configurations report the dim that
+  // actually drives shared-memory cost.
+  const uint32_t qk = head_dim_qk ? head_dim_qk : head_dim;
```

```cpp
-        // avg_packed_qo_len <= 16
+        // avg_packed_qo_len <= 16: prefer the 1x4 warp layout (cta_tile_q=16),
+        // but ONLY if one NUM_MMA_KV step fits shared memory. With 4 kv-warps a
+        // step costs (qk+vo)*16*4*sizeof(dtype); at (512,256) bf16 that is 96KB
+        // + a 16KB Q tile > the ~100KB/SM budget of CC 12.x consumer Blackwell
+        // (and would yield the unrecoverable "Unsupported max_mma_kv: 0"
+        // dispatch failure). Fall back to cta_tile_q=64 (4x1 layout, 4x
+        // cheaper KV step) exactly like the Turing branch below. ...
+        int dev_id = 0, max_smem_per_sm = 0;
+        cudaGetDevice(&dev_id);
+        cudaDeviceGetAttribute(&max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor,
+                               dev_id);
+        const uint32_t q_tile_smem = 16 * qk * 2;
+        const uint32_t kv_step_smem_1x4 = (qk + head_dim) * 16 * 4 * 2;
+        if (q_tile_smem + kv_step_smem_1x4 > (uint32_t)max_smem_per_sm) {
+          return 64;
+        }
         return 16;
```

**Why correct/minimal:** `sizeof(dtype)` is hard-coded to 2 (worst case), so the guard
only ever *moves configs that could not run at all* — symmetric small-head configs that
fit are untouched, preserving upstream tile choices. The fallback layout (`4x1`,
`cta_tile_q=64`) is the kernel's pre-existing Turing path, so no new code path is
introduced. `head_dim_qk` defaults to 0 → callers that don't pass it behave exactly as
before. Validated: SGLang E4B Rung 0 went from `Unsupported max_mma_kv: 0` (RED) to the
dispatcher wall cleared with FlashInfer `76af7982` (`docs/RESULTS_LEDGER.md` rows 39→40),
and the scorecard's `max_mma_kv` fix carries into the r9 baked image (ledger row 82).

### 4.3 — NVFP4 paged-attention output sizing (the FP4 byte-packing fix)

Same family as 4.1(b) but worth calling out as its own P3 fix: the packed-FP4 buffer is
sized from `v_cache.shape[-1] * 2` (two FP4 values per byte) instead of `q.shape[-1]`.
The diff hunk (`prefill.py:2455`):

```python
-        out_head_dim = q.shape[-1] if kv_cache_sf is not None else v_cache.shape[-1]
+        out_head_dim = (
+            v_cache.shape[-1] * 2 if kv_cache_sf is not None else v_cache.shape[-1]
+        )
```

The JIT module side enforces the FP4 container type so a mis-keyed module fails at
compile, not at runtime — `csrc/batch_prefill_customize_config.jinja`:

```cpp
+{% if require_fp4_kv_cache %}
+#ifndef FLASHINFER_ENABLE_FP4_E2M1
+#error "NVFP4 KV paged prefill compiled without FLASHINFER_ENABLE_FP4_E2M1"
+#endif
+static_assert(std::is_same_v<DTypeKV, __nv_fp4x2_e2m1>,
+              "NVFP4 KV paged prefill must build with the packed FP4 KV container type");
+{% endif %}
```

…and `flashinfer/jit/attention/modules.py` refuses to build an FP4 paged module without
the scale-factor tensors:

```python
+    require_fp4_kv_cache = dtype_map_kv[dtype_kv] == "__nv_fp4x2_e2m1"
+    if require_fp4_kv_cache:
+        missing_sf_tensors = [
+            name
+            for name in ("maybe_k_cache_sf", "maybe_v_cache_sf")
+            if name not in additional_tensor_names
+        ]
+        if missing_sf_tensors:
+            raise ValueError(
+                "NVFP4 KV paged prefill JIT modules require scale-factor tensors "
+                f"{missing_sf_tensors}; pass maybe_k_cache_sf and maybe_v_cache_sf "
+                "as additional tensors."
+            )
```

**Why correct/minimal:** `*2` is the exact inverse of the FP4 2-values-per-byte packing,
so the buffer matches the `head_dim_vo`-wide rows the kernel writes; the `static_assert`
+ `ValueError` turn two previously-silent mis-builds (wrong container, missing SFs) into
loud, early failures. Covered by `tests/jit/test_attention_utils.py`
(`test_batch_prefill_nvfp4_swa_paged_params_declares_sf_strides`,
`test_batch_prefill_nvfp4_requires_sf_tensors`).

### 4.4 — Linear V-scale-factor mode (sliceable SFs + explicit SF strides)

Two coupled changes. **(a)** SF strides are no longer *derived* from KV data strides
(the old `/ SF_CONTAINERS` arithmetic) — they're **passed explicitly**, so a runtime can
hand in linearly-laid-out SF tensors from a separately-allocated pool.
`include/flashinfer/attention/prefill.cuh` `page_produce_kv_sf`:

```cpp
- * SF strides are KV byte strides divided by SF_CONTAINERS (= NVFP4_SF_VEC_SIZE/2 = 8),
- * which is exact because all NVFP4-compatible head_dims are divisible by 16.
+ * SF strides are passed explicitly by the caller instead of being derived from KV data strides.
+ * This lets runtimes pass scale-factor tensors from interleaved or separately allocated KV pools.
```

```cpp
-    const uint32_t kv_head_idx, const uint32_t kv_stride_page, const uint32_t kv_stride_h,
-    const uint32_t kv_stride_n, const uint_fastdiv& page_size, const IdType* indices,
+    const uint32_t kv_head_idx, const uint32_t sf_stride_page, const uint32_t sf_stride_h,
+    const uint32_t sf_stride_n, const uint_fastdiv& page_size, const IdType* indices,
```

**(b)** A **compile-time flag selects linear vs swizzled** SF reads. The default is
**linear** (`FLASHINFER_PAGED_V_SF_DESWIZZLE 0`): the reader takes the plain strided path
and the in-kernel de-swizzle is the opt-in fallback.
`include/flashinfer/attention/prefill.cuh:481`:

```cpp
+#ifndef FLASHINFER_PAGED_V_SF_DESWIZZLE
+#define FLASHINFER_PAGED_V_SF_DESWIZZLE 0
+#endif
```

```cpp
+    if constexpr (produce_v && FLASHINFER_PAGED_V_SF_DESWIZZLE) {
+      static_assert(SF_COLS % 4 == 0,
+                    "Paged V-SF de-swizzle requires HEAD_DIM_VO divisible by 64");
+      uint32_t packed = 0;
+      if (in_bounds) {
+        constexpr uint32_t SF_GROUPS = SF_COLS / 4;
+        const uint32_t a4 = entry_idx & ~3u;
+        const uint32_t e = entry_idx & 3u;
+        uint8_t* packed_bytes = reinterpret_cast<uint8_t*>(&packed);
+#pragma unroll
+        for (uint32_t j = 0; j < 4; ++j) {
+          const uint32_t dcol = sf_smem_col + j;
+          const uint32_t swz_entry = a4 + dcol / SF_GROUPS;
+          const uint32_t swz_sd = (dcol % SF_GROUPS) * 4 + e;
+          packed_bytes[j] =
+              sf_ptr[page_head_base + static_cast<size_t>(swz_entry) * sf_stride_n + swz_sd];
+        }
+      }
+      *reinterpret_cast<uint32_t*>(sf_smem + flat_byte) = packed;
+    } else {
+      const size_t sf_gmem_offset = page_head_base + entry_idx * sf_stride_n + sf_smem_col;
+      ...
+      cp_async::pred_load_32b<fill_mode>(reinterpret_cast<uint32_t*>(sf_smem + flat_byte),
+                                         reinterpret_cast<const uint32_t*>(sf_ptr + sf_gmem_offset),
+                                         in_bounds);
+    }
```

The `else` (default, linear) branch is the one the VO-split uses: a head-dim slice of a
linear SF tensor is just a contiguous byte range, so each pass slices its V-half SFs by
plain `sf_stride_n`/`sf_stride_h`. The swizzle de-interleave (the `swz_entry`/`swz_sd`
arithmetic that spreads a token-quad across the SF row) is exactly the layout that
**cannot** be sliced along the head dim — hence opt-in only. Python/JIT plumbing emits
the SF strides per layout (`flashinfer/jit/attention/utils.py`,
`generate_sf_stride_setter_lines`), and the device entry points read them
(`prefill.cuh`):

```cpp
     uint8_t* maybe_v_cache_sf = nullptr;
+    uint32_t v_cache_sf_stride_n = 0, v_cache_sf_stride_h = 0;
     if constexpr (has_maybe_v_cache_sf_v<Params>) {
       maybe_v_cache_sf = params.maybe_v_cache_sf;
+      v_cache_sf_stride_n = params.maybe_v_cache_sf_stride_n;
+      v_cache_sf_stride_h = params.maybe_v_cache_sf_stride_h;
     }
```

**Why correct/minimal:** explicit SF strides are a strict generalization (the old
derived value is still expressible), the default stays linear so the swizzle path is
purely additive, and the `static_assert` guards the de-swizzle's `SF_COLS % 4`
precondition. The campaign learned the hard way that the *writer* and *reader* must agree
on this flag (the r7-image gibberish: a stale csrc swizzled while the reader expected
linear — `FLASHINFER_D512_FA2_KERNEL_PLAN.md` coherence-hunt verdict;
`results/jit_cache_mode_unsoundness_analysis_20260611.md`). Validated green with linear
V-SF at cosine 0.9999986 (P0b).

#### Supporting: independent K/V strides (enables the asymmetric/strided-V views)

The paged-KV struct gains a third constructor with independent K and V strides, plus
`get_v_elem_offset` / `protective_get_v_*` using V's own strides — so the V-half passes
can be strided VIEWS of the full V without the wrapper's old "k strides == v strides"
assertion. `include/flashinfer/page.cuh:160`:

```cpp
+  /*!
+   * \brief Construct a paged key-value cache with independent K/V strides.
+   */
+  __host__ __forceinline__ paged_kv_t(..., const int64_t* k_strides,
+                                      const int64_t* v_strides, ...) {
+    stride_page = k_strides[0];
+    v_stride_page = v_strides[0];
+    ...
+    v_stride_n = layout == QKVLayout::kHND ? v_strides[2] : v_strides[1];
+    v_stride_h = layout == QKVLayout::kHND ? v_strides[1] : v_strides[2];
+  }
```

`csrc/batch_prefill.cu` drops the symmetric-stride `ICHECK` and passes both stride arrays:

```cpp
-  // get kv_cache_strides
-  const int64_t* kv_cache_strides = paged_k_cache.strides().data();
+  // get kv-cache strides
+  auto k_cache_strides = paged_k_cache.strides();
+  auto v_cache_strides = paged_v_cache.strides();
   TVM_FFI_ICHECK_EQ(paged_k_cache.ndim(), paged_v_cache.ndim());
-  for (int i = 0; i < paged_k_cache.ndim(); ++i) {
-    TVM_FFI_ICHECK_EQ(paged_k_cache.stride(i), paged_v_cache.stride(i))
-        << "k/v strides differs at " << i;
-  }
```

#### Supporting: the mm-prefix `mask_indptr` device one-liner

Also on the branch (part of the multimodal enablement, `docs/MM_PREFIX_MASK_NOTES.md`):
`plan()` moves `mask_indptr` to the custom mask's device before `segment_packbits`, since
the packbits kernel `CHECK_DEVICE`s the indptr on the mask's device but callers (vLLM)
keep `mask_indptr` on CPU. `flashinfer/prefill.py:1994`:

```python
             packed_custom_mask, mask_indptr = segment_packbits(
                 custom_mask.contiguous().view(-1),
-                mask_indptr,
+                mask_indptr.to(custom_mask.device),
                 bitorder="little",
             )
```

---

## 5. What this does NOT change

- **FA2 attention math is untouched.** The VO-split is a host-side decomposition over two
  passes of an already-shipping asymmetric `(qk, vo)` kernel; the per-pass kernel math is
  bit-identical to upstream. The `IsInvalid` guard (`prefill.cuh:178`) is **not relaxed**
  — the `(512,256)` trait passes it as written (`1*(8*16 + 2*4*1) = 136 < 256`).
- **It slots NEXT TO upstream's FA4 unification, not against it.** Upstream is unifying on
  FA4 for Hopper; this route serves **CC 12.x** (consumer/Spark Blackwell, where FA4
  isn't the answer). Single-backend on the model (FLASHINFER for the whole Gemma 4),
  so upstream's mixed-backend divergence concern does not apply.
- **Validated on CC 12.0 AND 12.1** — RTX PRO 6000 (sm_120) and GB10 / DGX Spark
  (sm_121). Same fix, both targets green.
- **Split-dtype module keying is OUT OF SCOPE for this PR.** A `k_data_type != v_data_type`
  shim exists in-tree (`prefill.py` `plan()`, the validated `(fp8_e4m3, uint8)` mixed-KV
  pair) but full split-dtype `dtype_k`/`dtype_v` module keying is descoped here; it is
  noted only so the PR's `plan()` kwargs are understood, not featured.
- The committed `flashinfer/data/*` entries are dev-clone convenience symlinks (so a raw
  clone is self-contained), not PR content.

---

## Appendix — cross-references

- `docs/FLASHINFER_D512_FA2_KERNEL_PLAN.md` — the wall, the VO-split math, the SF-layout
  decision, the coherence-hunt verdict.
- `docs/RESULTS_LEDGER.md` — rows 23–42 (vLLM/SGLang serving), 66–69 (D=512 probes /
  roundtrips), 82–83 (r9 image + E4B baseline).
- `docs/TRITON_RETIREMENT_SCORECARD.md` / `results/claude_retirement_scorecard_20260612/`
  — the GB10 retirement evidence (R1–R5 / I1–I4).
- `results/colab_g4_pro6000_20260612/COLAB_G4_RESULTS.md` — the full sm_120 ladder.
- `docs/BUG_FLASHINFER_GEMMA3_1B_SERVING_NUMERICS.md` — the 1B caveat (P520-box-specific,
  clean on sm_120).
- `docs/MM_PREFIX_MASK_NOTES.md` — the `mask_indptr` device one-liner.
- `docs/ISSUE_TRACKER.md` — upstream [vllm#40677](https://github.com/vllm-project/vllm/issues/40677)
  (RTX PRO 6000 / Gemma 4 NVFP4, "head_size not supported") and friends.
