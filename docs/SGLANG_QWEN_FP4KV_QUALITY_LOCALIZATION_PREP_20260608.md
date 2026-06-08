# SGLang Qwen FP4-KV Quality Localization Prep

Date: 2026-06-08 JST

Scope: SGLang Qwen prep only. Do not serve Gemma until Qwen FP4-KV quality is blessed.
This checkpoint does not start SGLang, build images, or run GPU work.

## Next Live Step

Run one matched fp8/FP4 Qwen probe against already-started SGLang servers to answer two
questions:

1. Does OpenAI Chat Completions serialize the `medium_decode` prompt to the same token
   ids as the native `/generate` Qwen chat-template path?
2. At the first native divergence, token index 4 (`fp8: " Valid"`, FP4: `" Validate"`),
   what calibration/global-scale, V-scale layout, and prefill/decode metadata did FP4
   actually use?

Client-side probe prepared:

```powershell
python scripts/sglang_openai_native_reconcile.py `
  --fp8-url http://127.0.0.1:30012 `
  --fp4-url http://127.0.0.1:30013 `
  --model Qwen/Qwen2.5-1.5B-Instruct `
  --model-path Qwen/Qwen2.5-1.5B-Instruct `
  --case medium_decode `
  --run-id sglang_qwen_fp4kv_prompt_path_reconcile_YYYYMMDDTHHMMJST `
  --output results/sglang_qwen_fp4kv_prompt_path_reconcile_YYYYMMDDTHHMMJST.json
```

The script requests `return_prompt_token_ids=true` from `/v1/chat/completions`, then
replays native `/generate` from those exact ids and from the locally rendered Qwen chat
template. The decisive fields are:

- `fp8.prompt_id_comparison.first_prompt_id_diff`
- `fp4.prompt_id_comparison.first_prompt_id_diff`
- `comparison.fp8_vs_fp4_native_from_openai_prompt_ids.divergence_window.token_index`
- `comparison.fp8_vs_fp4_native_from_local_render_text.divergence_window.token_index`

Expected artifact:

- `results/sglang_qwen_fp4kv_prompt_path_reconcile_YYYYMMDDTHHMMJST.json`

Expected result if prompt serialization is not the cause:

- OpenAI prompt ids match the local rendered prompt ids on both fp8 and FP4.
- Native replay from OpenAI prompt ids reproduces the known token-index-4 rank reversal.

If OpenAI prompt ids differ, stop there and fix/align serialization before touching KV
math.

## Server Env For The Live Run

Use the same no-graph Qwen setup as the `d7d931f` evidence row. Required environment:

```bash
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
SGLANG_FP4_KV_TRACE_BACKEND=1
SGLANG_FP4_KV_AUTOCALIB=1
SGLANG_FP4_KV_ENABLE_CUDA_GRAPH=0
```

Do not set vLLM's `FLASHINFER_PAGED_V_SF_DESWIZZLE=1` in the SGLang run. SGLang must stay
on symmetric-linear V scale factors unless the FlashInfer/SGLang pair is changed
together.

Expected existing log lines:

```text
NVFP4 KV cache calibrated 28 layers from 4096 eager prefill tokens
Disabling CUDA graph capture for native FP4 KV cache. Current FlashInfer FA2 NVFP4 KV graph capture can produce corrupt decode output; set SGLANG_FP4_KV_ENABLE_CUDA_GRAPH=1 only for graph-safety experiments.
NVFP4 KV backend trace label=extend_merge_paged layer=0 wrapper_idx=0 metadata=...
NVFP4 KV backend trace label=decode layer=0 wrapper_idx=0 metadata=...
```

Expected artifacts copied from the live run:

- `results/sglang_qwen_fp4kv_prompt_path_reconcile_YYYYMMDDTHHMMJST_fp8_server.log`
- `results/sglang_qwen_fp4kv_prompt_path_reconcile_YYYYMMDDTHHMMJST_fp4_server.log`
- `results/sglang_qwen_fp4kv_prompt_path_reconcile_YYYYMMDDTHHMMJST_fp4_trace_excerpt.txt`
- `results/sglang_qwen_fp4kv_prompt_path_reconcile_YYYYMMDDTHHMMJST_summary.md`

## Temporary Instrumentation Hooks

The current client probe can expose prompt ids and output top-k distributions. It cannot
expose internal scale/layout state. Add temporary env-gated logs in the source overlay
only if the existing backend trace is insufficient.

### Prompt Serialization

Hook: `third_party/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py`

Location: `_apply_jinja_template`, immediately after `rendered_prompt` is encoded into
`prompt_ids`.

Proposed env flag: `SGLANG_TRACE_PROMPT_SERIALIZATION=1`

Expected line:

```text
SGLang prompt serialization trace path=openai_chat template=jinja token_count=56 token_sha256=<sha256> rendered_sha256=<sha256> add_special_tokens=False prompt_ids_head=[...] prompt_ids_tail=[...]
```

Hook: `third_party/sglang/python/sglang/srt/managers/tokenizer_manager.py`

Location: `_tokenize_one_request`, after text or `input_ids` are resolved and before
`_create_tokenized_object`.

Expected lines:

```text
SGLang prompt serialization trace path=native_generate source=text token_count=56 token_sha256=<sha256> prompt_ids_head=[...] prompt_ids_tail=[...]
SGLang prompt serialization trace path=native_generate source=input_ids token_count=56 token_sha256=<sha256> prompt_ids_head=[...] prompt_ids_tail=[...]
```

### Calibration And Global Scale

Hook: `third_party/sglang/python/sglang/srt/model_executor/model_runner.py`

Location: `_calibrate_nvfp4_kv_cache`, before and after the explicit eager prefill.

Proposed env flag: `SGLANG_FP4_KV_TRACE_CALIBRATION=1`

Expected lines:

```text
NVFP4 KV calibration trace phase=start tokens=4096 page_size=1 available_slots=<n> forward_mode=EXTEND graph_capture=False piecewise_graph=False
NVFP4 KV calibration trace phase=done calibrated_layers=28 total_layers=28 k_global_min=<f> k_global_max=<f> v_global_min=<f> v_global_max=<f>
```

Hook: `third_party/sglang/python/sglang/srt/mem_cache/memory_pool.py`

Location: `MHATokenToKVPoolFP4._maybe_calibrate_global_scales` and
`MHATokenToKVPoolFP4.set_kv_buffer`, after `k_global`/`v_global` are selected and after
`NVFP4KVQuantizeUtil.quantize`.

Expected lines:

```text
NVFP4 KV scale trace layer=<i> phase=calibrate n_tokens=4096 k_amax=<f> v_amax=<f> k_global=<f> v_global=<f> convention=decode_global_scale
NVFP4 KV scale trace layer=<i> phase=set_kv_buffer forward_path=extend loc_min=<n> loc_max=<n> k_global=<f> v_global=<f> k_sf_shape=(...) v_sf_shape=(...) v_sf_layout=symmetric_linear
```

### First Decode Perturbation

Hook: `third_party/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py`

Location: `_trace_nvfp4_native_call`. For the focused run, allow repeated `decode` trace
for decode steps `0..4` instead of the current once-per-label/layer suppression.

Proposed env flags:

```bash
SGLANG_FP4_KV_TRACE_BACKEND=1
SGLANG_FP4_KV_TRACE_DECODE_STEPS=0,1,2,3,4
SGLANG_FP4_KV_TRACE_LAYERS=0,13,27
```

Expected lines:

```text
NVFP4 KV decode-step trace step=4 layer=0 label=decode wrapper_idx=0 metadata={'num_decode_tokens': 1, ...} q=(...) kv_cache=(...) k_sf=(...) v_sf=(...) k_scale=<f> v_scale=<f> v_sf_layout=symmetric_linear prefill_path=extend_merge_paged
NVFP4 KV decode-step trace step=4 layer=13 label=decode ...
NVFP4 KV decode-step trace step=4 layer=27 label=decode ...
```

The client JSON should still be the authority for the selected-token/top-k divergence;
the server logs should explain which KV state and path produced that step.

## Stop Condition

Stop after the prompt-path JSON, fp4/fp8 logs, trace excerpt, and short summary are
captured. Do not start Gemma, do not run a capacity row, and do not change quantizer math
until the OpenAI-vs-native serialization result is known.
