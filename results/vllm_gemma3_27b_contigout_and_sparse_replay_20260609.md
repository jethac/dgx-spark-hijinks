# vLLM Gemma 3 27B Contiguous Output and Sparse Replay Probe, 2026-06-09

Purpose: test whether the Gemma 3 27B NVFP4-KV byte-like attention output is caused by
vLLM passing a problematic output view to FlashInfer, or by FlashInfer mishandling
nonzero physical page IDs.

## Source

- vLLM source overlay: `third_party/vllm` branch `spark/hijinks-021-gemma3-tensor-trace`
  plus a diagnostic env switch.
- New diagnostic switch: `VLLM_SPARK_NVFP4_PREFILL_CONTIG_OUT=1`.
- Runner: `scripts/run_vllm_gemma3_contigout_probe_container.sh`.
- Replay script: `scripts/vllm_active_page_flashinfer_replay.py`.
- Replay runner: `scripts/run_vllm_active_page_flashinfer_replay_container.sh`.
- Image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`.
- Served model: `google/gemma-3-27b-it`, served as `gemma3-27b-it`.
- Diagnostic mode: `--enforce-eager`; this is not a speed row.

## Artifacts

Local compact artifacts:

- `results/vllm_gemma3_27b_contigout_20260609T0358JST_nvfp4_kv_flashinfer_eager_first_token.json`
- `results/vllm_gemma3_27b_contigout_20260609T0358JST_nvfp4_kv_flashinfer_eager_tensor_trace.jsonl`
- `results/vllm_active_page_flashinfer_replay_layer5_0002_compact_20260609.json`
- `results/vllm_active_page_flashinfer_replay_layer5_0002_sparse_20260609.json`

Remote full run directory:

```text
/home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-jituri-20260609/results/
```

## Result

The missing-FP4-macro hypothesis is falsified.

The failed Gemma 3 JIT rerun already built paged-prefill modules with both
`-DFLASHINFER_ENABLE_FP4_E2M1` and `-gencode=arch=compute_121a,code=sm_121a`. The added
FlashInfer guard remains useful, but the live failure was not caused by a generic raw-uint8
JIT module missing the FP4 macro.

The causal standalone-paged-prefill hypothesis is falsified.

`scripts/flashinfer_nvfp4_kv_probe.py --causal` passed for NHD and HND tuple KV with
swizzled V-scale and Gemma-compatible shape (`D=128`, 32 Q heads, 16 KV heads). Prefill
outputs were signed and non-byte-like, with cosine approximately `1.0` against the
dequantized reference. So causal FA2 NVFP4 paged prefill alone does not reproduce Gemma's
byte-like live output.

The contiguous-output hypothesis is falsified.

The live vLLM run enabled `VLLM_SPARK_NVFP4_PREFILL_CONTIG_OUT=1`, and the trace confirms
`uses_contig_out=true` for the full-attention layer-5 FlashInfer paged-prefill calls. The
`out_arg_view` passed to FlashInfer was dense and contiguous, with shape like
`[23, 32, 128]`, storage offset `0`, and stride `[4096, 128, 1]`.

Quality still failed with the same first-token signatures as the previous red row:

- exact Spark prompt: ` Reigns`
- simple math prompt: Gujarati token
- short decode prompt: `ioane`

The layer-5 `flashinfer_wrapper_prefill_post` output remained byte-like even though it was
written into the fresh scratch buffer:

```text
num_prefill_tokens=23
window_left=-1
out_after head=[240.0, 1.0, 226.0, 137.0, 145.0, 20.0, 186.0, 185.0, ...]
out_after max=255.0
out_after mean=128.9432373046875
out_after rms=147.43911743164062
```

The nonzero physical-page hypothesis is also falsified.

`scripts/vllm_active_page_flashinfer_replay.py` now supports:

- `--page-placement compact`: remap dumped active pages to `[0, 1]`.
- `--page-placement sparse`: place dumped active pages back at their original physical IDs
  and pass original indices, for example `[13, 14]`.

Both replay modes on the useful layer-5 dump produced sane signed output:

```text
compact actual head=[0.0, -4.875, 0.40625, 0.0, 0.8125, -3.25, ...]
sparse  actual head=[0.0, -4.875, 0.40625, 0.0, 0.8125, -3.25, ...]
actual max=15.0
actual min=-13.5
actual rms=1.9469819068908691
actual_vs_original_out_after cosine=-0.005058457143604755
```

The sparse replay used `flashinfer_kv_data_shape=[15, 16, 16, 64]`,
`flashinfer_kv_scale_shape=[15, 16, 16, 8]`, and `flashinfer_kv_indices=[13, 14]`, so
nonzero physical page IDs alone do not reproduce the live wrapper corruption.

## Interpretation

The bug is not:

- output view aliasing/stride in vLLM;
- stale `dtype_kv_u8` JIT URI naming;
- missing FP4 JIT macro;
- causal paged prefill in a standalone synthetic case;
- nonzero physical page IDs by themselves.

The remaining target is the live vLLM wrapper state around `BatchPrefillWithPagedKVCacheWrapper`:
plan reuse, wrapper metadata, split-kv/jit-module selection, or an argument/config value that
differs between the direct replay wrapper and vLLM's long-lived wrapper.
