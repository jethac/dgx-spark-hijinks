# vLLM Gemma 3 27B Rung 1 fp8 Comparator, 2026-06-08 19:24 JST

Purpose: establish the fp8 KV baseline for Gemma 3 27B text-only Rung 1 before testing
FlashInfer FA2 NVFP4 KV.

Remote context:

- run checkout: `/home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608`
- run id: `vllm_gemma3_27b_rung1_20260608T1924JST_fp8_flashinfer`
- model: `google/gemma-3-27b-it`
- served model: `gemma3-27b-it`
- vLLM overlay: `jethac/vllm@25ab073ef87f4443616fbaf00a2f6f09a9087c1f`
- precompiled wheel base: `4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa`
- image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
- serving flags: `--dtype bfloat16 --kv-cache-dtype fp8 --attention-backend flashinfer --max-model-len 131072 --gpu-memory-utilization 0.85 --max-num-batched-tokens 4096`

Artifacts on the remote checkout:

- `results/vllm_gemma3_27b_rung1_20260608T1924JST_fp8_flashinfer_server.log`
- `results/vllm_gemma3_27b_rung1_20260608T1924JST_fp8_flashinfer_openai_benchmark.json`
- `results/vllm_gemma3_27b_rung1_20260608T1924JST_fp8_flashinfer_quality.json`
- `results/vllm_gemma3_27b_rung1_20260608T1924JST_fp8_flashinfer_row_manifest.json`
- `results/vllm_gemma3_27b_rung1_20260608T1924JST_fp8_flashinfer_runtime_probe.json`
- `results/vllm_gemma3_27b_rung1_20260608T1924JST_fp8_flashinfer_build_target_audit.json`
- `results/vllm_gemma3_27b_rung1_20260608T1925JST_fp8_driver.log`

Result:

- The fp8 comparator row served successfully.
- The server selected fp8 KV and FlashInfer decoder attention:
  - `Using fp8 data type to store kv cache`
  - `Using AttentionBackendEnum.FLASHINFER backend`
- vLLM reported `GPU KV cache size: 882,851 tokens`.
- vLLM reported maximum concurrency `6.74x` for `131,072` tokens per request.
- The OpenAI benchmark completed all three cases with `ok=true`.
- Quality probe output was nonempty and unflagged for all three cases.

Measured running-model geometry:

- Decoder layers logged: `62`.
- Local/sliding-window layers: `52`.
- Full/global layers: `10`.
- Pattern: layers `0-4` local with `sliding_window=1024`, layer `5` full, repeated every six layers through layer `59`, with layers `60-61` local.
- All logged decoder layers used `heads=32`, `kv_heads=16`, `head_dim=128`, `head_dim_v=128`.
- fp8 KV spec logged `dtype=torch.uint8`, `kv_quant_mode=FP8_PER_TENSOR`, `page_size_bytes=65536`, `bytes_per_token=4096.000` per layer.

Benchmark rows:

| case | prompt tokens | output tokens | decode tok/s | status |
|---|---:|---:|---:|---|
| `short_decode` | 24 | 64 | `4.2391` | `ok=true` |
| `medium_decode` | 36 | 192 | `4.1616` | `ok=true` |
| `long_prefill` | 2266 | 64 | `4.2057` | `ok=true` |

Harness caveats:

- The `_runtime_probe.json` package section was collected outside the serving container and
  reports missing Python packages. Do not use that field as package-state evidence for this
  row.
- This row is a comparator only. It does not prove NVFP4 KV routing, capacity gain, or
  output correctness.
- The packet was later tightened to use `pip install --no-build-isolation --no-deps -e .`
  and copy the ABI-matched FA2 extension from `/opt/jethac-vllm`; dependency downgrades are
  rejected for this lane.

Next gate:

Run the matching `nvfp4` row with `RUN_FP8=0 RUN_NVFP4=1` against the same prefix and compare
KV cache size, concurrency, backend selection, geometry, and quality against this fp8 row.
