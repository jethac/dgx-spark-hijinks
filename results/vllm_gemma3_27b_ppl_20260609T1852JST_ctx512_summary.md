# vLLM Gemma 3 NVFP4-KV PPL Pair

Run: vllm_gemma3_27b_ppl_20260609T1852JST_ctx512

Status: green

Scope: short 512-token supplied-token prompt-logprob PPL gate. This is stronger than the
first-token smoke, but it is not a long-context/SWA-window stress test or throughput row.

Stack:

- model: `google/gemma-3-27b-it`
- served model: `gemma3-27b-it`
- image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
- repo checkout: `dad4468`
- FlashInfer submodule: `c3dae30f`
- vLLM submodule: `0c278200e`
- memory guardrails: sequential servers, `--gpu-memory-utilization 0.72`, Docker
  `--memory 100g --memory-swap 100g`, no concurrent fp8/NVFP4 comparator
- prefix caching: disabled
- max model length: `4096`
- context measured: `512`

Result:

| ctx | PPL fp8 | PPL NVFP4 | delta PPL | fp8 nats/token | NVFP4 nats/token | delta nats/token |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 115.4583 | 119.8578 | +4.3995 | 4.7489 | 4.7863 | +0.0374 |

Both rows recovered all supplied prompt-token logprobs:

- fp8: `511 / 511` scored, `0` missing
- NVFP4: `511 / 511` scored, `0` missing

Artifacts:

- `results/vllm_gemma3_27b_ppl_20260609T1852JST_ctx512_summary.json`
- `results/vllm_gemma3_27b_ppl_20260609T1852JST_ctx512_compare.json`
- `results/vllm_gemma3_27b_ppl_20260609T1852JST_ctx512_fp8_ppl.json`
- `results/vllm_gemma3_27b_ppl_20260609T1852JST_ctx512_nvfp4_ppl.json`

Interpretation:

The short PPL gate supports the earlier first-token result: Gemma 3 NVFP4 KV is no longer
obviously corrupt on short text-only prompts after the FlashInfer paged-prefill fix. The
measured short-context quality cost is small in this artifact (`+0.0374` nats/token).

Next:

- Repeat at larger contexts that cross or approach Gemma 3's SWA window behavior.
- Record capacity/concurrency and throughput in the same row before calling Gemma 3 fully
  blessed.
