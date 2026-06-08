# SGLang Qwen FP4 KV Autosafe Row

Date: 2026-06-08

Run id: `sglang_qwen_fp4kv_autosafe_20260608T1315JST`

This row records a matched no-graph SGLang fp8-vs-FP4-KV comparison for `Qwen/Qwen2.5-1.5B-Instruct` on a Spark-class GB10 system. It proves the FP4 KV capacity path, but it does not prove usable FP4 KV serving quality or speed.

## Setup

- image: `nvcr.io/nvidia/sglang:26.05-py3`
- runtime: NVIDIA SGLang container plus `jethac/sglang` clean source overlay
- attention backend: `flashinfer`
- model: `Qwen/Qwen2.5-1.5B-Instruct`
- memory policy: `--page-size 1 --mem-fraction-static 0.40`
- fp8 comparator graph policy: `--disable-cuda-graph --disable-piecewise-cuda-graph`
- FP4 row graph policy: fork auto-disables CUDA graph and piecewise graph capture for native FP4 KV

The scratch source overlay did not preserve a `.git` directory, so treat this as source-overlay evidence, not a clean wheel or pinned source checkout proof.

## Artifacts

- FP4 manifest: `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_row_manifest.json`
- FP4 benchmark: `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_openai_benchmark.json`
- FP4 raw sanity: `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_raw_2plus2.json`
- FP4 server log: `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_server.log`
- fp8 manifest: `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_fp8_row_manifest.json`
- fp8 benchmark: `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_fp8_openai_benchmark.json`
- fp8 raw sanity: `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_fp8_raw_2plus2.json`
- fp8 server log: `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_fp8_server.log`

## Result

| row | KV dtype | max tokens | graph policy | decode rows | quality |
|---|---:|---:|---|---|---|
| fp8 comparator | `torch.float8_e4m3fn` | `3,101,822` | disabled to match FP4 policy | `56.73`, `56.81`, `57.10 tok/s` | raw `2+2` returns `4` |
| FP4 KV | `torch.float4_e2m1fn_x2` | `5,519,481` | auto-safe no-graph | mechanically `90.95`, `55.42`, `55.98 tok/s` | raw `2+2` fails; benchmark text degenerates |

FP4 KV capacity ratio versus fp8: `5,519,481 / 3,101,822 = 1.779x`.

The FP4 server log records:

- `NVFP4 KV cache calibrated 28 layers from 4096 eager prefill tokens`
- `Disabling CUDA graph capture for native FP4 KV cache`
- `cuda graph: False`

## Interpretation

This is a real capacity proof for SGLang FP4 KV on GB10, but not a blessed serving row. The recorder marked the OpenAI benchmark mechanically `ok=true` because requests completed and returned non-empty text, but the standardized raw sanity artifact answers `2+2` with malformed text and the benchmark content is repetitive or incoherent.

Do not use the FP4 decode token rates as a speed claim. The usable comparison today is that fp8 remains the quality-passing SGLang Qwen path at about `57 tok/s`, while FP4 KV expands the KV pool by about `1.78x` but still corrupts output on this small Qwen row.

The next useful SGLang work is a quality fix or a model/shape where the FP4 KV row passes deterministic sanity and a stronger quality comparator. Until then, the counterpart evidence audit should keep this requirement partial.
