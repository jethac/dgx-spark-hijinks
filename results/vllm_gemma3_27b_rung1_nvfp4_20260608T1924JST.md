# vLLM Gemma 3 27B Rung 1 NVFP4 Candidate, 2026-06-08 19:24 JST

Purpose: test FlashInfer FA2 NVFP4 KV on Gemma 3 27B text-only Rung 1 against the captured
fp8 comparator.

Remote context:

- run checkout: `/home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608`
- run id: `vllm_gemma3_27b_rung1_20260608T1924JST_nvfp4_kv_flashinfer`
- comparator: `results/vllm_gemma3_27b_rung1_fp8_20260608T1924JST.md`
- vLLM overlay: `jethac/vllm@25ab073ef87f4443616fbaf00a2f6f09a9087c1f`
- FlashInfer overlay: `jethac/flashinfer@e152cf4da4ab2a9d093b7d9d4b499198b0211c61`
- command: `RUN_FP8=0 RUN_NVFP4=1 bash docs/results/vllm_gemma3_27b_rung1_20260608TCHECKOUTJST_command_packet.sh`
- driver log: `results/vllm_gemma3_27b_rung1_20260608T1948JST_nvfp4_driver.log`

Artifacts on the remote checkout:

- `results/vllm_gemma3_27b_rung1_20260608T1924JST_nvfp4_kv_flashinfer_server.log`
- `results/vllm_gemma3_27b_rung1_20260608T1924JST_nvfp4_kv_flashinfer_chat_smoke.json`
- `results/vllm_gemma3_27b_rung1_20260608T1924JST_nvfp4_kv_flashinfer_openai_benchmark.json`
- `results/vllm_gemma3_27b_rung1_20260608T1924JST_nvfp4_kv_flashinfer_row_manifest.json`
- `results/vllm_gemma3_27b_rung1_20260608T1924JST_nvfp4_kv_flashinfer_runtime_probe.json`
- `results/vllm_gemma3_27b_rung1_20260608T1924JST_nvfp4_kv_flashinfer_build_target_audit.json`
- `results/vllm_gemma3_27b_rung1_20260608T1924JST_nvfp4_kv_flashinfer_quality.json`
- `results/vllm_gemma3_27b_rung1_20260608T1924JST_quality_compare.json`

Result: not green.

- Capacity/routing: passed.
- Benchmark request completion: passed.
- Output correctness/quality: failed.
- Manifest: `ok=false` because the strict `spark-ok` chat smoke failed.

Routing and geometry evidence:

- Server log selected NVFP4 KV:
  - `Using nvfp4 data type to store kv cache`
  - `Using AttentionBackendEnum.FLASHINFER backend`
  - `Using FlashInfer FA2 backend for NVFP4 KV cache on SM12x with vLLM V-scale-factor deswizzle enabled.`
- Decoder geometry matched the fp8 row:
  - `62` decoder layers
  - `52` local SWA layers
  - `10` full/global layers
  - uniform `heads=32`, `kv_heads=16`, `head_dim=128`, `head_dim_v=128`
- NVFP4 KV spec logged `dtype=torch.uint8`, `kv_quant_mode=NVFP4`,
  `page_size_bytes=36864`, `bytes_per_token=2304.000` per layer.

Capacity:

| row | KV tokens | max concurrency at 131,072 tokens/request |
|---|---:|---:|
| fp8 comparator | `882,851` | `6.74x` |
| NVFP4 candidate | `1,568,861` | `11.97x` |
| ratio | `1.7770x` | `1.7760x` |

Throughput:

| case | fp8 tok/s | NVFP4 tok/s | NVFP4/fp8 |
|---|---:|---:|---:|
| `short_decode` | `4.2391` | `4.2475` | `1.0020x` |
| `medium_decode` | `4.1616` | `4.1003` | `0.9853x` |
| `long_prefill` | `4.2057` | `4.2213` | `1.0037x` |

Quality failure:

- Strict chat smoke expected exactly `spark-ok` and instead returned nonsensical mixed-script text.
- The three benchmark cases completed, but generated nonsensical mixed-script text.
- The simple quality heuristic did not flag this, so it is insufficient for this rung.
- The fp8-vs-NVFP4 text similarity was extremely low:
  - `short_decode`: `0.0066`
  - `medium_decode`: `0.0140`
  - `long_prefill`: `0.0213`

Interpretation:

The FlashInfer FA2 NVFP4 KV path is routing and sizing correctly for Gemma 3 27B hybrid SWA,
and it delivers the expected `~1.78x` KV capacity gain at decode-speed parity. It is not
serving-correct: the output corruption makes this row a red compatibility result, not a
blessed Gemma NVFP4-KV path.

Next debugging direction:

- Treat the failure like the SGLang FP4-KV lane: capacity and raw routing are not enough.
- Add a request-tagged first-token/logits or fp8-vs-NVFP4 comparator for vLLM Gemma 3 to
  localize whether corruption appears before attention output, after logits preprocessing,
  or during sampling.
- Check whether Gemma 3 hybrid SWA needs a per-layer or per-attention-type scale/layout
  handling difference even though the page-size math and FA2 routing are correct.
