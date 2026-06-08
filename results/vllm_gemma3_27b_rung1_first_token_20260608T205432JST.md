# vLLM Gemma 3 27B Rung 1 First-Token Diagnostic, 2026-06-08 20:54 JST

Status: red NVFP4-KV serving path, stronger localization.

Purpose: rerun the Gemma 3 27B Rung 1 fp8-vs-NVFP4 packet with the first-token probes
enabled. The earlier `20260608T1924JST` row proved NVFP4 routing and capacity but stopped
before first-token diagnostics.

Remote context:

- checkout: `/home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608`
- packet: `docs/results/vllm_gemma3_27b_rung1_20260608T205432JST_command_packet.sh`
- model: `google/gemma-3-27b-it`
- served model: `gemma3-27b-it`
- vLLM overlay: `jethac/vllm@25ab073ef87f4443616fbaf00a2f6f09a9087c1f`
- FlashInfer overlay: `jethac/flashinfer@e152cf4da4ab2a9d093b7d9d4b499198b0211c61`
- image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
- dependency policy: editable vLLM install uses `--no-deps`; dependency downgrades are
  rejected.

Artifacts:

- `results/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_first_token.json`
- `results/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_first_token.json`
- `results/vllm_gemma3_27b_rung1_20260608T205432JST_first_token_compare.json`
- `results/vllm_gemma3_27b_rung1_20260608T205432JST_fp8_flashinfer_server.log`
- `results/vllm_gemma3_27b_rung1_20260608T205432JST_nvfp4_kv_flashinfer_server.log`

Routing and capacity:

- fp8 row served and passed manifest.
- NVFP4 row served but failed manifest because strict output quality is still red.
- NVFP4 server log selected the intended path:
  `Using FlashInfer FA2 backend for NVFP4 KV cache on SM12x with vLLM V-scale-factor
  deswizzle enabled.`
- NVFP4 KV cache size: `1,595,236` tokens.
- NVFP4 max concurrency at `131,072` tokens/request: `12.17x`.
- NVFP4 KV spec: `SlidingWindowSpec` and `FullAttentionSpec` both report
  `kv_quant_mode=NVFP4`, `real_page_size_bytes=36864`, and `bytes_per_token=2304.000`.

First-token comparison:

| case | fp8 first token | NVFP4 first token | top-logprob overlap |
|---|---|---|---:|
| `exact_spark_ok` | `spark` | Cyrillic token, see JSON | `0.0` |
| `simple_math` | `4` | CJK token, see JSON | `0.0` |
| `short_decode` | `A` | CJK phrase token, see JSON | `0.0` |

Interpretation:

- The corruption is present on the first generated token, not only after decode state
  compounds.
- Candidate sets are disjoint, not just rank-flipped. This points to attention/KV state or
  logits before sampling, not sampling noise.
- The cross-lane FP4-KV reuse hypothesis remains live: Qwen standard-attention FP4 KV is
  clean, while Gemma 3 adds hybrid SWA/local-window reuse and fails immediately.
- The next vLLM diagnostic should trace SWA block lifecycle, slot mapping, NVFP4 split/view
  offsets, and FlashInfer read-side page IDs under an env-gated audit. The invariant is that
  packed FP4 data and FP8 scale views must be derived from the same physical page/block on
  both write and read.
