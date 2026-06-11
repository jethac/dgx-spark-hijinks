# SGLang Gemma 4 Rung Prep

Date: 2026-06-10 JST

Scope: offline implementation map for bringing Gemma 4 text-only rungs to SGLang after the Gemma 3 mixed-KV checkpoint. This is not a serving result.

## Ground Truth

Kernel-side status is `docs/FLASHINFER_D512_FA2_KERNEL_PLAN.md`. The current usable pattern for Gemma 4 global layers is the VO-split route:

- local/sliding layers stay on the ordinary D=256-style path;
- global/full layers with `head_dim_qk=512` run two FlashInfer FA2 passes with `head_dim_vo=256`, then concatenate the two output halves;
- SGLang uses linear V scale-factor layout, so the V-SF slice for VO-split is the simple half-slice case;
- future probes must match real runtime workload, not just head counts: `qo/kv` lengths, batch composition, page size, `o_dtype`, window, split-K flags, graph state, and wrapper plan kwargs.

Claude's latest probe window falsified geometry-only, plan-kwargs-only, and wrapper-backend-only explanations for the vLLM bf16 serving crash. For SGLang this means wrapper construction state must be documented and probed exactly, not inferred from shape-only tests.

2026-06-11 serving checkpoint: `results/sglang_gemma4_e4b_rung0_20260611T141256JST/summary.md`
proves the SGLang wrapper-construction half now behaves as intended for E4B text-only.
Serving dispatch logs show SWA/local layers planned at `head_dim=256, head_dim_vo=256`
and global prefill entering the VO-split path at `head_dim=512, head_dim_vo=256`. The
remaining rung-0 red is decode-side: the D=512 global layer still enters the standard
decode wrapper, which instantiates a symmetric `head_dim_qk=512;head_dim_vo=512` paged
module and fails with `Unsupported max_mma_kv: 0`.

Epoch-2 update: `results/sglang_gemma4_e4b_rung0_chat_20260611T180454JST/summary.md`
is the current rung-0 green. It uses `jethac/sglang@9d78a007f` plus
`jethac/flashinfer@76af7982`; the D=512 dispatcher wall is closed, D=512 global
decode uses `decode_as_prefill_vosplit*`, and chat-formatted serving returns
`The capital of Japan is Tokyo.` The paired raw-vs-chat diagnostic
`results/sglang_gemma4_e4b_chat_compare_20260611T175952JST/summary.md` shows raw
`/generate` repeats separators, so future Gemma 4 IT quality gates should use the
OpenAI chat endpoint unless the task explicitly measures raw completion behavior.

Epoch-2 Rung 1 checkpoint:
`results/sglang_gemma4_e4b_rung1_fullnvfp4_20260611TmanualJST.md` is full-NVFP4
K+V short-green with a fixed allocator denominator. Full NVFP4 (`fp4_e2m1`,
`SGLANG_FP4_KV_MIXED_KV=0`) returns the Tokyo chat answer, routes global D=512
through VO-split for prefill and decode-as-prefill, and passes a short prefix-reuse
PPL pair against bf16/auto (`ctx=512`, `reuse_prefix_len=256`,
`delta_nats_per_token=-0.190174`). `jethac/sglang@96a9ff9ce` fixes the hybrid
full-NVFP4 denominator, raising full NVFP4 capacity to `1,274,008` tokens at the
same memory fraction: `3.5668x` versus bf16/auto and `1.7814x` versus the fp8
allocator row. fp8 serving itself is still red with an internal SGLang request/warmup
timeout plus a missing-fp8-scale warning, so fp8 quality remains an open comparator
issue even though the allocator capacity ratio is now green.

## SGLang Code Surfaces

### Gemma 4 Runtime Geometry

`third_party/sglang/python/sglang/srt/models/gemma4_causal.py` already makes Gemma 4 heterogeneous by layer:

- `Gemma4DecoderLayer` chooses `config.head_dim` for `full_attention` and `config.swa_head_dim` for sliding layers at lines 531-537.
- `Gemma4Attention` chooses full-layer KV heads from `config.num_key_value_heads`, and sliding-layer KV heads from `swa_num_key_value_heads` when present, at lines 296-312.
- `RadixAttention` receives the layer-specific `head_dim`, `num_kv_heads`, and `sliding_window_size` at lines 393-404.

Rung rule: measure this from the running model every time. Config JSON is a hint; the server's per-layer `RadixAttention` objects are the truth.

### KV Pool Allocation

The FP4 pool is `MHATokenToKVPoolFP4` in `third_party/sglang/python/sglang/srt/mem_cache/memory_pool.py`.

- `_create_buffers()` uses `head_dim` and `v_head_dim` to allocate packed NVFP4 buffers and scale buffers at lines 1490-1547.
- Mixed-KV mode stores K as `torch.float8_e4m3fn`, skips K scale buffers, and stores V as packed NVFP4 plus V scale buffers at lines 1497-1547.
- `get_kv_scale_buffer()` returns `(None, v_sf)` in mixed mode and `(k_sf, v_sf)` in full NVFP4 mode at lines 1621-1631.
- `get_kv_global_scale()` returns K scale `1.0` in mixed mode and the calibrated layer scale in full NVFP4 mode at lines 1633-1636.
- `set_kv_buffer()` writes fp8 K in mixed mode, or quantized NVFP4 K plus K-SF in full mode; V is always quantized to NVFP4 at lines 1974-2052.

The current SGLang Gemma 3 row proved the mixed-KV allocator denominator after fixing `DefaultPoolConfigurator` / `HybridSWAPoolConfigurator`. Gemma 4 D=512 globals require the same denominator logic to be layer-dim aware: the full/global subpool must account for the larger global `head_dim`, while sliding layers must keep their smaller local geometry.

### Hybrid SWA Subpool

SWA routing uses the hybrid cache stack:

- `HybridLinearKVPool` wraps the full-attention pool in `memory_pool.py` lines 2107-2195.
- `HybridLinearKVPool.get_kv_buffer()` maps a global layer id to the full-attention local id at lines 2242-2245.
- `HybridLinearKVPool.set_kv_buffer()` forwards the mapped layer id and scalar scales into the inner pool at lines 2261-2280.
- Host/device hybrid attachment and layer maps are assembled in `hybrid_pool_assembler.py`, especially `_swa_layer_mappings()` and `build_hybrid_swa_stack()` around lines 832-878.

Open risk already banked in `tasks/sglang_fp4_kv_hybrid_scale_transfer_20260610.md`: `HybridLinearKVPool.set_kv_buffer()` forwards explicit scalar `k_scale` / `v_scale`, but `MHATokenToKVPoolFP4.set_kv_buffer()` primarily derives authoritative scales from the `layer` object. That was deliberately parked during the Qwen radix hunt. It becomes relevant for Gemma 4 hybrid/SWA rungs and should be fixed or re-probed before full NVFP4 Gemma 4 claims.

### FlashInfer Wrapper Planning

SGLang's FlashInfer backend is `third_party/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py`.

- `FlashInferAttnBackend.__init__()` creates two wrapper slots when `model_runner.sliding_window_size` is present: wrapper 0 for sliding-window attention and wrapper 1 for full attention, lines 724-733.
- Paged prefill wrappers are created at lines 833-848; decode wrappers are created at lines 849-855.
- `FlashInferIndicesUpdaterDecode` currently reads `model_runner.model_config.head_dim` once into `self.head_dim` at lines 2919-2936, then passes that same value into `begin_forward()` at lines 3150-3184.
- `FlashInferIndicesUpdaterPrefill` does the same at lines 3199-3216 and passes the same value to ragged and paged wrappers at lines 3521-3529 and 3570-3583.
- Search result as of 2026-06-10: this SGLang tree does not pass an explicit `jit_args` keyword into FlashInfer wrapper constructors on the FP4 path. Constructor calls pass workspace/layout/backend/use-tensor-core knobs only. That means the exact vLLM bug fixed in `jethac/vllm@e08a6f3ae` is not currently present as a literal `jit_args` override in SGLang, but the same failure class can appear if SGLang adds ctor-level VO-split args or lets cached module state pin symmetric head dims ahead of plan-time `head_dim_vo`.

This is the main Gemma 4 risk. A single `model_config.head_dim` cannot describe both sliding D=256 and global D=512 layers if the wrappers are shared only by window/full wrapper id. The SGLang rung needs per-wrapper or per-layer plan metadata that can express:

- sliding wrapper: local layer geometry, local window, local head dim;
- full wrapper: global layer geometry, `head_dim_qk=512`, and VO-split `head_dim_vo=256`;
- mixed-KV or full NVFP4 K/V dtypes independently.

Do not let constructor/module-cache state override plan-time VO-split. The vLLM 31B crash was fingerprinted to constructor `jit_args` pinning symmetric head dims and defeating the plan-time VO-split. SGLang's current risk is the single `self.head_dim` threaded through `begin_forward()`: the Gemma 4 implementation must make the full-attention wrapper's plan declare `head_dim_qk=512` and `head_dim_vo=256` at the same altitude where FlashInfer chooses/loads the module.

### Extend And Decode Slots

SGLang has three paths that matter:

- Paged-only extend: `forward_extend()` lines 2561-2636 writes cache first, then runs one paged prefill call.
- Ragged no-prefix extend: lines 2659-2684 uses dense ragged K/V and does not read cached K.
- Cached-prefix extend: lines 2686-2817 computes a suffix ragged partial, a cached-prefix paged partial via `_run_paged_native()`, then merges the two partial states with `_safe_merge_state()`.
- Decode: lines 2844-2902 reads the paged cache through the decode wrapper.

For mixed-KV Gemma 4, the cached-prefix merge path is acceptable because K stays fp8. For full NVFP4 K+V Gemma 4, this path is the known SGLang failure mode from Qwen: partial-state LSE sensitivity under FP4 K. The structural route is still the right full-NVFP4 target: make cached+suffix attention look like vLLM's one paged attention over the full cache, instead of splitting into ragged suffix plus paged prefix and merging.

## VO-Split Insertion Plan

For a global Gemma 4 layer with `head_dim_qk=512` and V/output width 512:

1. Keep Q and K intact at width 512.
2. Slice V and V scale factors into two width-256 halves.
3. Run FlashInfer paged prefill twice with identical Q/K/page tables/window/softcap and different V-half views.
4. Concatenate the two output halves before `o_proj`.

Prefill/extend insertion point: the paged part of `forward_extend()` and `_run_paged_native()`. For mixed-KV, K remains fp8 and V halves are packed NVFP4. For full NVFP4, K remains packed NVFP4 and both calls reuse the same K scales.

Decode insertion point: SGLang currently plans decode through `BatchDecodeWithPagedKVCacheWrapper`, whose plan path has no documented `head_dim_vo` slot in this tree. Mirror the vLLM workaround unless FlashInfer decode grows the needed metadata: route global-layer decode through the paged-prefill wrapper with `qo_len=1` / decode-as-prefill under an opt-in flag, while sliding layers keep normal decode.

Implementation checkpoint: `jethac/sglang@9d78a007f` adds this opt-in SGLang route.
When `SGLANG_FLASHINFER_VOSPLIT=1` and a layer reports `head_dim=512`, decode replans
the paged-prefill wrapper with one query row per request, reuses the decode wrapper's
planned page tables, and runs the existing two-pass VO-split reader. This needs the
next E4B rung-0 serving retry before it graduates from scaffold to green row.

## Differences From vLLM To Track

- vLLM's proven path uses packed-cache split views and page-16 assumptions; SGLang uses separate pool tensors, page size 1, and linear V-SF layout.
- vLLM's full NVFP4 prefix-cache hit proof does not use SGLang's ragged-suffix/paged-prefix partial merge. Do not import vLLM's result as proof for SGLang full NVFP4.
- SGLang's mixed-KV path is claim-ready only at the mixed denominator (`~1.28x`), not the full NVFP4 denominator (`~1.78x`).
- SGLang has a graph-write prefix-cache corruption guard. Keep it enabled for mixed-KV rungs until the graph-written radix state bug is fixed.

## Gates Before Serving Gemma 4

1. Static config audit and running-model geometry dump: layer id, layer type, heads, KV heads, `head_dim`, `v_head_dim`, window, page layout, bytes/token.
2. Pool allocation audit: full and SWA subpool classes, token capacities, K/V GB, mixed/full denominator, and scalar scale transfer.
3. FlashInfer plan audit: wrapper id, `qo_len`, `kv_len`, batch, page size, window, softcap, `head_dim_qk`, `head_dim_vo`, `k_data_type`, `v_data_type`, `kv_cache_sf` layout.
4. Wrapper-construction audit: prove no ctor `jit_args`, cached module, `fast_decode_plan`, or wrapper-local `self.head_dim` value pins symmetric dims and overrides the plan-time VO-split.
5. Writer-roundtrip gate at head 512: write via SGLang's real cache writer, read via the real FlashInfer kernel, and compare against dequant/reference output. Probes that fabricate the cache can miss writer/reader disagreements; the new vLLM 31B coherence bug is exactly "writer and reader meet at D=512 for the first time."
6. Standalone FlashInfer probe matching the real SGLang workload signature, not toy head geometry.
7. fp8 comparator row. Current E4B fp8 quality is red with an internal request/warmup
   timeout and missing-scale warning, so it cannot serve as the final quality baseline
   yet. The allocator ratio after `96a9ff9ce` is green at `1.7814x` versus fp8 tokens.
8. mixed-KV serving row first when full NVFP4 is blocked.
9. full NVFP4 K+V: E4B has a short green checkpoint, but the generic SGLang full-NVFP4
   radix structural route remains open for Qwen and other models where the partial-state
   merge is known to fail.
