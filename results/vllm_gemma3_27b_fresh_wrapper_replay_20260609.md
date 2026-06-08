# vLLM Gemma 3 27B Fresh Wrapper Replay Probe, 2026-06-09

Purpose: test whether the Gemma 3 27B NVFP4-KV byte-like FlashInfer prefill output is
caused by long-lived `BatchPrefillWithPagedKVCacheWrapper` state/reuse.

## Source

- vLLM fork: `jethac/vllm@spark/hijinks-021-gemma3-tensor-trace`
- Diagnostic commits:
  - `a10b3da5d` `Add NVFP4 fresh prefill wrapper replay diagnostic`
  - `4a6e4b537` `Use cached dtypes for NVFP4 fresh wrapper replay`
- FlashInfer fork: `jethac/flashinfer@686625b0`
- Image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
- Served model: `google/gemma-3-27b-it`, as `gemma3-27b-it`
- Diagnostic env:
  - `VLLM_SPARK_NVFP4_PREFILL_CONTIG_OUT=1`
  - `VLLM_SPARK_NVFP4_PREFILL_FRESH_WRAPPER_REPLAY=1`
  - `VLLM_SPARK_NVFP4_FRESH_WRAPPER_WORKSPACE_MB=256`
  - `VLLM_SPARK_GEMMA_TENSOR_TRACE=1`
- Mode: `--enforce-eager`; not a speed row.

## Artifacts

- `results/vllm_gemma3_27b_freshreplay_20260609Tfreshreplay3JST_nvfp4_kv_flashinfer_eager_first_token.json`
- `results/vllm_gemma3_27b_freshreplay_20260609Tfreshreplay3JST_nvfp4_kv_flashinfer_eager_tensor_trace.jsonl`
- `results/vllm_gemma3_27b_freshreplay_20260609Tfreshreplay3JST_nvfp4_kv_flashinfer_eager_import_probe.txt`

Remote full run directory:

```text
/home/jethac/spark_tmp/dgx-spark-hijinks-vllm-freshreplay-20260609/results/
```

## Result

The fresh in-process wrapper reproduces the live byte-like output exactly. For the
layer-5 full-attention prefill calls:

- `fresh_wrapper_backend="fa2"` and `live_wrapper_backend="fa2"`
- `q_data_type=torch.bfloat16`, `kv_data_type=torch.uint8`, `o_data_type=torch.bfloat16`
- `fixed_split_size=-1`, `disable_split_kv=false`
- fresh-vs-live `max_abs_diff=0.0`, `mean_abs_diff=0.0`, `rms_diff=0.0`

Representative bad call:

```text
shape=[23, 32, 128]
fresh_out head=[240.0, 1.0, 226.0, 137.0, 145.0, 20.0, 186.0, 185.0, ...]
fresh_out max=255.0
fresh_out mean=128.9432373046875
fresh_out rms=147.43911743164062
fresh_vs_live cosine=1.0000001192092896
```

The first-token probe still reproduces the known red Gemma 3 signatures:

```text
exact_spark_ok -> " Reigns"
simple_math    -> Gujarati token
short_decode   -> "ioane"
```

## Interpretation

This falsifies the long-lived wrapper-state hypothesis. A brand-new
`BatchPrefillWithPagedKVCacheWrapper`, planned inside the same live vLLM process from the
same wrapper buffers and run on the same live tensors/scales, returns the same packed-byte
output as the reused wrapper.

The remaining difference versus the offline compact/sparse replay is therefore not wrapper
lifetime, output contiguity, active page IDs, or sampled page/scale pairing in isolation.
The next target is the exact live-process FlashInfer module/plan path versus the offline
replay path: module cache identity, generated module arguments, plan fields, and any
argument/layout difference between `scripts/vllm_active_page_flashinfer_replay.py` and the
live vLLM call.
