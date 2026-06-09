# SGLang Qwen FP4-KV Dense-vs-Cached Trace Probe, 2026-06-09

Status: next diagnostic packet; do not run Gemma until this Qwen issue is green.

Purpose: compare the no-prefix/full-prefill FP4 path against the 55-token cached-prefix
FP4 path at the tensor boundary where the first-token distribution becomes disjoint.
The current request-order artifact proves cached-prefix rows have `0 / 20` top-logprob
overlap with full-prefill rows, while no-reuse controls have `20 / 20`.

Driver:

- Use `scripts/sglang_fp4_request_order_probe.py`.
- It already produces the required pairs:
  - dense reference: `openai-first` / `native-first`, `cached_tokens=0`
  - cached failure: `native-second` / `openai-second`, `cached_tokens=55`
  - controls: flush-between and namespace-isolation, both `cached_tokens=0`

Run the cheap no-code matrix before adding new instrumentation:

- Default FP4: expected cached prefix hit, `forward_extend_merge_paged`, known bad.
- `SGLANG_FLASHINFER_USE_PAGED=1`: keep the radix prefix hit, but force full paged
  attention instead of ragged suffix + paged cached prefix + merge.
- `SGLANG_RADIX_FORCE_MISS=1`: force full recompute without disabling the server's radix
  feature path.
- Both env vars: full recompute through full paged attention.

2026-06-09 execution stop point:

- Runner: `scripts/run_sglang_fp4_request_order_matrix.sh`
- Artifact: `results/sglang_qwen_fp4kv_matrix_20260609tmatrix10jst.md`
- Outcome: source overlay builds and installs cleanly from `jethac/sglang@d4fe78078`,
  editable `jethac/flashinfer@4c3c0d99` satisfies SGLang's `flashinfer_python>=0.6.12`
  guard, and `sglang-kernel 0.4.3` satisfies SGLang's version guard. SGLang still exits
  before health because the PyPI `sglang-kernel 0.4.3` wheel tries to load
  `sgl_kernel/sm100/common_ops.abi3.so`, which is ABI-incompatible with the NVIDIA 26.05
  Torch/CUDA stack on GB10.
- Interpretation: no row evidence yet. The active blocker is now `sglang-kernel`
  build-target/ABI compatibility, not FlashInfer. Re-run the same matrix only after
  building `sglang-kernel` from source against the container Torch/CUDA stack, or after
  finding a matching ARM64 CUDA 13.x wheel with SM121-compatible `common_ops`. Do not
  downgrade SGLang, Torch, FlashInfer, or the container to make the guard pass.

2026-06-09 source-build unblock:

- Probe: `scripts/probe_sglang_kernel_source_build.sh`
- Artifact: `results/sglang_kernel_source_build_20260609Tprobe3jst.md`
- Outcome: `jethac/sglang@d96869237` adds selectable `sgl-kernel` build targets, and a
  narrow source build against the NVIDIA 26.05 Torch/CUDA stack succeeds. The installed
  `sglang-kernel 0.4.3` imports
  `/usr/local/lib/python3.12/dist-packages/sgl_kernel/sm100/common_ops.abi3.so` on GB10.
- Interpretation: the `sglang-kernel` ABI blocker is cleared. Rerun the matrix from a
  prepared source-stack image so the four rows do not rebuild `sglang-kernel` independently.

2026-06-09 source-stack matrix:

- Runner: `scripts/run_sglang_fp4_request_order_matrix.sh`
- Prepared image helper: `scripts/prepare_sglang_source_stack_image.sh`
- Artifact: `results/sglang_qwen_fp4kv_matrix_20260609tprep1jst.md`
- Stack: reusable image `sglang-source-stack-20260609tprep1jst` with editable
  `jethac/flashinfer@4c3c0d99`, editable `jethac/sglang@d96869237`, source-built
  `sglang-kernel 0.4.3`, and rebuilt
  `/usr/local/lib/python3.12/dist-packages/sgl_kernel/sm100/common_ops.abi3.so`.
- Outcome: all four rows run. Default cached-prefix reuse remains bad
  (`cached_tokens=55`, `ark`/838); force-miss rows are clean (`cached_tokens=0`,
  identical logprobs); full-paged-with-reuse changes the token class to newline but still
  gives cached-prefix logprobs different from fresh output.
- Interpretation: do not rerun this matrix unless the cache/reuse implementation changes.
  The next probe is dense full-prefill versus FP4 cached-prefix attention/logit comparison.

Decision rule for this matrix:

- Full-paged cached-prefix passes while default fails -> bug is split ragged/paged merge
  interaction.
- Default and full-paged cached-prefix fail while force-miss rows pass -> reused FP4
  prefix state/contribution is the bug.
- Force-miss + full-paged passing is the cleanest full-paged FP4 recompute comparator.

Instrumentation gate:

```bash
SGLANG_FP4_KV_TRACE_DENSE_CACHE=1
SGLANG_FP4_KV_TRACE_DENSE_QUANT_ATTENTION=1
SGLANG_FP4_KV_TRACE_LAYERS=0,1,7,13,20,27
SGLANG_FP4_KV_TRACE_VALUES=64
```

2026-06-09 instrumentation implementation:

- Fork commit: `jethac/sglang@e631a13fd`.
- Branch: `spark/hijinks-018-fp4-e2m1-kv-sm121-serving`.
- Local validation: `python -m py_compile` for the five touched Python files and
  `git diff --check` both passed in the Windows workspace.
- Scope: inactive unless `SGLANG_FP4_KV_TRACE_DENSE_CACHE=1` is set.

2026-06-09 comparator refinement:

- `scripts/sglang_dense_cache_trace_compare.py` canonicalizes the equivalent attention
  output labels (`forward_extend_ragged_no_prefix`, `forward_extend_merge_paged`,
  `forward_extend_paged`) as `attention_output`.
- Dense no-prefix `o_rows` are compared directly against cached-prefix merged
  `merged_rows`. The current-head artifact now localizes the first request-bound
  divergence to layer-0 attention output itself, before the Qwen2 residual/norm/MLP path.

2026-06-09 next diagnostic implementation:

- Fork commit: `jethac/sglang@a8ad6a3ac`.
- Change: `third_party/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py`
  adds inactive `SGLANG_FP4_KV_TRACE_DENSE_QUANT_ATTENTION=1`.
- On the dense no-prefix path, after `set_kv_buffer()` calibrates and writes FP4 KV, the
  trace re-quantizes/dequantizes the same BF16 K/V with `NVFP4KVQuantizeUtil` and computes
  a torch attention reference for the sampled last-token row.
- Logged comparisons:
  - actual ragged FlashInfer output vs BF16 torch reference
  - actual ragged FlashInfer output vs FP4-dequant torch reference
  - FP4-dequant torch reference vs BF16 torch reference
- Decision:
  - If FP4-dequant reference already diverges from BF16 similarly to cached-prefix output,
    the layer-0 split is mostly FP4 KV quantization/scale quality, not radix reuse.
  - If FP4-dequant reference stays close to BF16 while cached-prefix `merged_rows` diverge,
    the bug remains in cached-prefix paged/reuse integration.

2026-06-09 dense-quant and K/V split result:

- Summary: `results/sglang_qwen_fp4kv_kvsplit_trace_20260609_summary.md`
- Fork commits: `jethac/sglang@a8ad6a3ac` added the dense FP4 quant attention split;
  `jethac/sglang@8b95253af` added K-only and V-only references.
- Runner update: `scripts/run_sglang_fp4_dense_cache_trace.sh` now uses
  `PYTHONPATH=/work/third_party/sglang/python:...`, allowing Python-only SGLang
  diagnostics to run from the mounted fork against the prepared source-stack image
  without rebuilding `sglang-kernel`.
- Result: default radix row remains red (`**` dense/no-prefix, `ark` on 55-token prefix
  reuse), but the split moves the diagnosis away from `_safe_merge_state`.
- Main layer-0 row (`row=55`): actual dense FlashInfer vs BF16 reference
  `cosine=0.9999995231628418`; FP4 K+V vs BF16 `0.7876883745193481`; FP4 K-only
  vs BF16 `0.7893718481063843`; FP4 V-only vs BF16 `0.996927797794342`.
- Interpretation: the blocker is K-side attention-logit sensitivity. The paged prefix
  read/LSE and merge are internally correct relative to FP4; the K quantization error is
  enough to move the softmax winner even though direct K reconstruction cosine is high.
- Follow-up direct scale diff: dense/write, dense-quant reference, and cached-prefix
  `extend_merge_paged` all use the same layer-0 global scales
  (`k=0.1197916716337204`, `v=0.0016276042442768812`). FlashInfer's paged wrapper applies
  those scales in the same handedness as the local FP4-dequant reference (`k_scale` folded
  into `sm_scale`, `v_scale` applied to the output). This falsifies a stale or inverted
  cached-prefix global-scale explanation for the captured row.

2026-06-09 K-scale policy probe:

- Fork commits: `jethac/sglang@dfd426442` added the inactive multiplier trace, and
  `jethac/sglang@e4f24bbd3` added the experimental runtime K global-scale multiplier.
- Trace run id: `sglang_qwen_fp4kv_kscale_trace_c3dae30f_dfd426442_20260609T`
- Actual serving run id:
  `sglang_qwen_fp4kv_kscale_actual_c3dae30f_e4f24bbd3_20260609Tactual0125`
- Offline result: K global-scale multiplier `0.125` improves the main row's K-only
  attention reference cosine from `0.7893718481063843` to `0.9584803581237793`, and the
  K+V attention reference from `0.7876883745193481` to `0.9561840295791626`. Direct K
  reconstruction gets worse (`~0.9967` to `0.8765701055526733`), so this is optimizing
  attention-logit behavior rather than raw K reconstruction.
- Actual serving result: still red with radix cache on. Fresh rows still emit `**`
  (`-0.7235294580459595`), but the 55-token radix-hit row emits `To` / token id `1249`
  (`-1.7186779975891113`) instead of `**`. The first layer-0 attention divergence improves
  from `0.006467887232207366` to `0.1657561728524288`, but remains far from a pass.
- Decision: a scalar K multiplier is not a blessable fix. Next: investigate K scale
  quality/policy beyond a single global scalar, including per-head/per-group K scales or
  FP8/BF16 K with FP4 V. Quantify capacity loss before considering a K-not-FP4 fallback
  blessed.

Implemented insertion points:

- `third_party/sglang/python/sglang/srt/model_executor/forward_batch_info.py`
  - Carries `forward_pass_id` on `ForwardBatch` so attention, Qwen2, logits, and sampler
    trace records can be tied back to the same model forward.
- `third_party/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py`
  - Dumps sampled last-token rows for `forward_extend_ragged_no_prefix`,
    `forward_extend_merge_paged`, and forced full-paged `forward_extend_paged`.
  - Captures `q`, dense/ragged output, `o1/s1`, `o2/s2`, merged output, request ids,
    `kind`, `forward_pass_id`, prefix/extend lengths, sequence lengths, cache locations,
    and K/V scale values.
- `third_party/sglang/python/sglang/srt/models/qwen2.py`
  - Dumps selected-layer sampled rows after self-attention, after post-attention norm,
    after MLP, and after final model norm.
  - Start with layers `0,1,7,13,20,27`; expand to all 28 layers only if needed.
- `third_party/sglang/python/sglang/srt/layers/logits_processor.py`
  - Dumps `pruned_states`, `sample_indices`, raw logits, and sampled logits before sampler
    preprocessing. Mirror the same capture for the `extend_return_logprob` path.
  - Propagates request ids, `forward_pass_id`, and prefix/extend lengths through
    `LogitsMetadata` so logits traces no longer fall into the comparator's unknown bucket.
- `third_party/sglang/python/sglang/srt/model_executor/model_runner.py`
  - Dumps `next_token_logits` immediately before and after `_preprocess_logits`.
  - Stamps `ForwardBatch.forward_pass_id` at the start of `ModelRunner.forward` and emits
    sampler traces with normalized CPU-list metadata.

Runner:

```bash
RUN_ID=sglang_qwen_fp4kv_dense_cache_$(date -u +%Y%m%dT%H%M%SZ) \
PREPARE_SOURCE_STACK_IMAGE=1 \
TRACE_LAYERS=0,1,7,13,20,27 \
TRACE_VALUES=64 \
bash scripts/run_sglang_fp4_dense_cache_trace.sh
```

Default `CASES=default` runs the known failing cached-prefix row only. If the default row
localizes the first divergence to attention, run the full-paged comparator without changing
the source stack:

```bash
RUN_ID=sglang_qwen_fp4kv_dense_cache_fullpaged_$(date -u +%Y%m%dT%H%M%SZ) \
SOURCE_STACK_IMAGE=<prepared-source-stack-image> \
PREPARE_SOURCE_STACK_IMAGE=0 \
PREPARE_RUST_IMAGE=0 \
INSTALL_SOURCE_STACK_PER_CASE=0 \
CASES=full_paged \
bash scripts/run_sglang_fp4_dense_cache_trace.sh
```

Capture key:

```text
forward_pass_id, kind, rid/rids, label, layer, mode, extend_prefix_len,
extend_seq_len, sample_rows
```

Capture values:

- last-token hidden vectors as CPU float32 tensors
- full first-token logits before and after preprocessing
- top-20 token ids/values for logits
- stats: cosine, max_abs, RMS, relative RMS, finite count

Decision rule:

- First divergence at layer-0 attention output:
  FP4 cached-prefix attention quality differs from dense full prefill despite internally
  consistent paged read/LSE/merge traces.
- Attention and hidden vectors match, but raw logits diverge:
  LM head/logits path.
- Raw logits match, but post-preprocess logits diverge:
  sampler preprocessing, logits bias, or grammar state.
- All internal tensors match, but API top-logprobs diverge:
  response/logprob packaging path.

Acceptance:

- The probe must produce one artifact comparing dense no-cache rows against cached-prefix
  rows by `rid` and phase. The runner writes
  `results/${RUN_ID}_${case}_dense_cache_compare.json` via
  `scripts/sglang_dense_cache_trace_compare.py`.
- The comparison artifact must include at least one `metric_ok=true` dense/cached row and
  positive `metric_comparison_count`; structural event-key matches without vector/top-k
  metrics are red. For request-bound events, `event_schema_issues`, unknown request/rid
  binding, or non-zero `metricless_comparison_count` are also red, because they mean the
  trace did not bind tensor/logit evidence to the intended dense and cached request rows
  tightly enough. Warmup and health-check forwards may be counted as ignored provenance
  when they do not belong to any request row. The comparator records `first_divergence`
  when the sampled vector or top-k rows differ.
- The run summary must pass `scripts/sglang_dense_cache_trace_summary_audit.py`; a missing,
  unparsable, or red comparison artifact is a red run, not an inconclusive green.
- Do not bless `--disable-radix-cache` or selective no-reuse as the FP4-KV capacity path.
  It remains a diagnostic/emergency workaround only.

Offline verifier:

```bash
python3 scripts/sglang_dense_cache_trace_compare.py --self-test
```

Expected queue artifacts:

- `results/${RUN_ID}_summary.json`
- `results/${RUN_ID}_${case}.json`
- `results/${RUN_ID}_${case}_server.log`
- `results/${RUN_ID}_${case}_dense_cache_compare.json`
- `results/${RUN_ID}_dense_cache_trace_summary_audit.json`
