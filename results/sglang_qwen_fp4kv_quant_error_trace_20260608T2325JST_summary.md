# SGLang Qwen FP4-KV Quant-Error Trace, 2026-06-08 23:25 JST

Status: red serving path; quantization/calibration is not the distinguishing factor.

Purpose: rerun the known SGLang Qwen FP4-KV default-vs-radix-off first-token probe with
`jethac/sglang@d4fe78078633e70fde968c58032a675c72f13bc1` and
`SGLANG_FP4_KV_TRACE_QUANT_ERROR=1`. The probe compares dense K/V at cache fill against
the same tensors after NVFP4 quantize/dequantize.

Runtime:

- image: `nvcr.io/nvidia/sglang:26.05-py3`
- source overlay: `jethac/sglang@d4fe78078633e70fde968c58032a675c72f13bc1`
- model: `Qwen/Qwen2.5-1.5B-Instruct`
- KV dtype: `fp4_e2m1`
- attention backend: `flashinfer`
- CUDA graphs: disabled
- page size: `1`
- probe: `medium_decode`, `max_new_tokens=1`, `temperature=0`

Rows:

| row | OpenAI first token | native first token | match | native cached tokens |
|---|---|---|---|---:|
| default FP4 | `**` | `ark` / `838` | fail | 55 |
| radix-off FP4 | `**` | `**` / `334` | pass | 0 |

Quant-error result for layer 0:

| row | loc_len | K cosine | K max abs | K RMS | V cosine | V max abs | V RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| default | 4096 | 0.9967067 | 42.0 | 6.0214 | 0.9956598 | 0.203125 | 0.03473 |
| radix-off | 4096 | 0.9967067 | 42.0 | 6.0214 | 0.9956598 | 0.203125 | 0.03473 |
| default | 56 | 0.9966692 | 42.0 | 6.0341 | 0.9957151 | 0.199219 | 0.03569 |
| radix-off | 56 | 0.9966692 | 42.0 | 6.0341 | 0.9957151 | 0.199219 | 0.03569 |

Both rows use the same layer-0 auto-calibrated globals:

- K global scale: `0.1197916716337204`
- V global scale: `0.0016276042442768812`

Interpretation:

- FP4 quantization loss is real, but it is not the switch that separates the failing row
  from the passing row. The default and radix-off rows have the same sampled quant/dequant
  profile, while only the default row reuses a 55-token radix prefix and fails.
- Together with the prior prefix-reference trace, this clears calibration/global-scale
  selection and simple quant/dequant error as primary explanations for the OpenAI/native
  first-token split.
- The next discriminator is request ordering/cache state: run native first after
  `flush_cache`, OpenAI second after native, flush-between isolation, and cache namespace
  isolation if `extra_key`/`cache_salt` is accepted.

Artifacts:

- parsed analysis: `results/sglang_qwen_fp4kv_quant_error_trace_20260608T2325JST_analysis.json`
- endpoint JSON:
  `results/sglang_qwen_fp4kv_quant_error_trace_20260608T2325JST_default.json`,
  `results/sglang_qwen_fp4kv_quant_error_trace_20260608T2325JST_radixoff.json`
- quant-error lines:
  `results/sglang_qwen_fp4kv_quant_error_trace_20260608T2325JST_default_quant_error_lines.txt`,
  `results/sglang_qwen_fp4kv_quant_error_trace_20260608T2325JST_radixoff_quant_error_lines.txt`
- server logs:
  `results/sglang_qwen_fp4kv_quant_error_trace_20260608T2325JST_default_fp4_server.log`,
  `results/sglang_qwen_fp4kv_quant_error_trace_20260608T2325JST_radixoff_fp4_server.log`
- container inspect:
  `results/sglang_qwen_fp4kv_quant_error_trace_20260608T2325JST_default_fp4_container_inspect.json`,
  `results/sglang_qwen_fp4kv_quant_error_trace_20260608T2325JST_radixoff_fp4_container_inspect.json`

Runner note: the remote script packaged complete artifacts, then exited nonzero on a CRLF
line ending after the tarball path was printed. No server/container was left running.
