# vLLM Gemma 3 27B Plan-Signature Trace, 2026-06-09

Purpose: compare the live FlashInfer FA2 NVFP4-KV paged-prefill wrapper plan/run
signature against the offline active-page replay path after fresh-wrapper replay
falsified long-lived wrapper state.

## Source

- vLLM fork: `jethac/vllm@spark/hijinks-021-gemma3-tensor-trace`
- Diagnostic commit: `1fabc6649` `Trace NVFP4 prefill wrapper plan signatures`
- FlashInfer fork: `jethac/flashinfer@686625b0`
- Image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
- Served model: `google/gemma-3-27b-it`, as `gemma3-27b-it`
- Mode: `--kv-cache-dtype nvfp4`, FlashInfer, `--enforce-eager`; not a speed row.
- Install policy: source overlay with `--no-deps`; no Torch/FlashInfer/FA2 stack downgrade.

## Artifacts

- `results/vllm_gemma3_27b_plantrace_20260609TplantraceJST_nvfp4_kv_flashinfer_eager_first_token.json`
- `results/vllm_gemma3_27b_plantrace_20260609TplantraceJST_nvfp4_kv_flashinfer_eager_tensor_trace.jsonl`
- `results/vllm_gemma3_27b_plantrace_20260609TplantraceJST_nvfp4_kv_flashinfer_eager_import_probe.txt`
- `results/vllm_active_page_flashinfer_replay_layer5_0002_plantrace_schema_20260609.json`

Remote run directory:

```text
/home/jethac/spark_tmp/dgx-spark-hijinks-vllm-plantrace-20260609/results/
```

## Result

The known red first-token signatures reproduce:

```text
exact_spark_ok -> " Reigns"
simple_math    -> Gujarati token
short_decode   -> "ioane"
```

For the failing full-attention layer-5 prefill calls, live vLLM resolves the wrapper as:

```text
backend=fa2
kv_layout=NHD
causal=True
q_data_type=torch.bfloat16
kv_data_type=torch.uint8
o_data_type=torch.bfloat16
fixed_split_size=-1
disable_split_kv=False
FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1
```

Representative failing 18-token call:

```text
qo_indptr=[0, 18]
paged_kv_indptr=[0, 2]
paged_kv_indices=[13, 14]
paged_kv_last_page_len=[2]
plan_info=(1, 18, 0, 64, 0, 16, 32, 0, 48, 56, 0, 0, 0, 0, 0)
```

The fresh in-process wrapper still matches the live reused wrapper exactly:

```text
fresh_vs_live max_abs_diff=0.0
fresh_vs_live mean_abs_diff=0.0
fresh_vs_live rms_diff=0.0
fresh_vs_live cosine=0.99999994
```

The updated offline replay was run against the previously captured failing 18-token
layer-5 active-page dump with the same explicit plan fields: `window_left=-1`,
`head_dim_vo=128`, `o_data_type=torch.bfloat16`, `fixed_split_size=-1`, and
`disable_split_kv=False`. Its resolved wrapper signature matches the live signature above,
but the output is sane signed BF16 while the original live output is byte-like:

```text
offline actual:     byte_like=False, max=15.0, min=-13.5, rms=1.94698
live original out:  byte_like=True,  max=255.0, min=0.0,   rms=147.44287
actual_vs_original_out_after cosine=-0.00506
actual_vs_active_v_bytes_repeated cosine=-0.01126
```

## Interpretation

This narrows the vLLM Gemma 3 NVFP4-KV failure again. The remaining difference is not the
Python-visible wrapper plan/run signature: dtypes, layout, causal/window settings, page
tables, split-K flags, deswizzle flag, and `plan_info` all match the offline replay shape.

The next target is below or beside that Python signature: the live server process'
FlashInfer module/JIT cache identity, generated source/compiled object, or the C++ `paged_run`
argument interpretation. The next probe should stamp the generated FlashInfer module URI,
JIT directory/source path, compile flags, and `paged_run` function identity inside
FlashInfer itself, then compare live server process versus offline replay process.
