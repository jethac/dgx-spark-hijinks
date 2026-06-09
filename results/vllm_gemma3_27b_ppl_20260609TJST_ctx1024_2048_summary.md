# vLLM Gemma 3 NVFP4-KV PPL Pair

Run: vllm_gemma3_27b_ppl_20260609TJST_ctx1024_2048

Status: green

Scope: sequential fp8-vs-NVFP4 supplied-token prompt-logprob PPL at 1024 and 2048
tokens. This extends the earlier 512-token gate toward and across Gemma 3's 1024-token
SWA window, but it is still not a full long-context, throughput, or concurrency blessing.

Stack:

- model: `google/gemma-3-27b-it`
- served model: `gemma3-27b-it`
- image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
- memory guardrails: sequential servers, `--gpu-memory-utilization 0.72`, Docker
  `--memory 100g --memory-swap 100g`, no concurrent fp8/NVFP4 comparator
- prefix caching: disabled
- max model length: `4096`
- max batched tokens: `4096`
- hardware key: `NVIDIA_GB10:sm_121:sms_48`

Run root: /home/jethac/spark_tmp/vllm_gemma3_27b_ppl_20260609TJST_ctx1024_2048

Compare artifact: /home/jethac/spark_tmp/vllm_gemma3_27b_ppl_20260609TJST_ctx1024_2048/results/vllm_gemma3_27b_ppl_20260609TJST_ctx1024_2048_compare.json

Result:

| ctx | PPL fp8 | PPL NVFP4 | delta PPL | fp8 nats/token | NVFP4 nats/token | delta nats/token |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 35.0895 | 35.2563 | +0.1667 | 3.5579 | 3.5626 | +0.0047 |
| 2048 | 20.5861 | 20.4757 | -0.1104 | 3.0246 | 3.0192 | -0.0054 |

Both rows recovered all supplied prompt-token logprobs:

- fp8 1024: `1023 / 1023` scored, `0` missing
- fp8 2048: `2047 / 2047` scored, `0` missing
- NVFP4 1024: `1023 / 1023` scored, `0` missing
- NVFP4 2048: `2047 / 2047` scored, `0` missing

Artifacts:

- `results/vllm_gemma3_27b_ppl_20260609TJST_ctx1024_2048_summary.json`
- `results/vllm_gemma3_27b_ppl_20260609TJST_ctx1024_2048_compare.json`
- `results/vllm_gemma3_27b_ppl_20260609TJST_ctx1024_2048_fp8_ppl.json`
- `results/vllm_gemma3_27b_ppl_20260609TJST_ctx1024_2048_nvfp4_ppl.json`
- `results/vllm_gemma3_27b_ppl_20260609TJST_ctx1024_2048_corpus_manifest.json`

Interpretation:

The fixed Gemma 3 NVFP4-KV path remains quality-sane at 1024 and 2048 prompt tokens. The
measured deltas are tiny in both directions (`+0.0047` and `-0.0054` nats/token), so this
does not show context-length compounding through the 1024-token SWA window in this corpus
slice.

Next:

- Run a larger context row if memory allows, or switch to a capacity/concurrency row for
  Gemma 3.
- Add throughput under the same stack before calling Gemma 3 fully blessed.
