# vLLM Qwen NVFP4-KV Prefix-Cache Reuse Probe

Date: 2026-06-10 JST

## Verdict

The vLLM server successfully ran Qwen with full NVFP4 KV and prefix caching enabled, and two repeated long-prefix chat requests returned the same first token.

This is a useful negative-corruption smoke result, but it is not yet decisive proof that vLLM read reused FP4 K from the prefix cache. The OpenAI-compatible response did not report cached-token counts, and the server log does not include an explicit prefix-cache hit metric for these two requests.

## Run Configuration

- Runner: `scripts/run_vllm_qwen_prefix_cache_probe.sh`
- Probe: `scripts/vllm_prefix_cache_reuse_probe.py`
- Run id: `vllm_qwen_nvfp4_prefixcache_on_retry2_20260610TmanualJST`
- Image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv`
- Model: `/home/jethac/models/aeon/qwen36-nvfp4`
- Served model alias: `qwen36-fast`
- KV cache dtype: `nvfp4`
- Prefix caching: enabled
- Attention backend: FlashInfer
- `--gpu-memory-utilization`: `0.72`
- `--max-model-len`: `262144`
- Docker memory guardrail: `--memory 100g --memory-swap 100g`
- Readiness timeout: `1800s`

## Startup Evidence

The 30-minute readiness window was necessary. The server reached readiness only after long load/compile/warmup:

- weights loaded in `171.60s`
- `torch.compile` took `70.99s`
- initial profile/create-KV-cache/warmup took `389.35s`
- CUDA graph capture completed
- server started on `http://0.0.0.0:8001`

The server selected the intended NVFP4-KV path:

- `Using nvfp4 data type to store kv cache`
- `Using FlashInfer FA2 backend for NVFP4 KV cache on SM12x with vLLM V-scale-factor deswizzle enabled`
- available KV cache memory: `54.9 GiB`
- GPU KV cache size: `9,345,263 tokens`
- maximum concurrency at 262,144-token request length: `35.65x`

The log also reports Qwen hybrid-cache caveats:

- Mamba cache mode was set to `align` because prefix caching is enabled.
- vLLM warns that prefix caching in Mamba `align` mode is experimental.

## Probe Result

Probe case: `long_shared_prefix`

- Prompt length: `10,738` characters
- Prompt tokens: `2,054`
- Requests: `2`
- `max_tokens`: `1`
- First tokens: `["*", "*"]`
- `all_first_tokens_same`: `true`
- Probe verdict: `ok=true`
- Request latencies:
  - request 0: `0.9747s`
  - request 1: `0.3405s`
- First-token logprobs:
  - request 0: `*`, logprob `-0.06476`
  - request 1: `*`, logprob `-0.11844`

The response usage fields were:

- `prompt_tokens_details: null`
- `cached_tokens_from_usage: [null, null]`
- `any_usage_reports_cache_hit: false`

The server log shows both chat requests completed with HTTP 200, but does not expose a prefix-cache hit count:

- `POST /v1/chat/completions HTTP/1.1" 200 OK`
- `POST /v1/chat/completions HTTP/1.1" 200 OK`

## Interpretation

This run does not reproduce the SGLang full-NVFP4 radix failure at the HTTP first-token level: with vLLM prefix caching enabled, two identical long-prefix requests produced the same first token and similar top-logprob ordering.

However, the central question remains partially open: did the second request actually hit and read FP4 K from the prefix cache? This run cannot prove that because neither the response usage nor the captured server log exposes cache-hit accounting.

The honest status is:

- vLLM full NVFP4 KV + prefix caching enabled: serves the smoke prompt without obvious first-token corruption.
- vLLM full NVFP4 KV + proven prefix-cache reuse of FP4 K: not yet proven.
- Previous vLLM clean-path PPL with prefix caching disabled still should not be quoted as proof of reused-FP4-K quality.

## Follow-Up Needed

Before using this as the decisive cross-check against SGLang:

1. Capture `/metrics` after the probe, before container cleanup, and check for vLLM prefix-cache hit counters if exported.
2. If metrics are absent or insufficient, add targeted vLLM instrumentation around prefix-cache hit/miss and KV block reuse.
3. Re-run the same two-request probe and require both:
   - explicit cache-hit evidence on request 2;
   - first-token/logprob stability under that hit.

## Artifacts

- `results/vllm_qwen_nvfp4_prefixcache_on_retry2_20260610TmanualJST_prefix_reuse_probe.json`
- `results/vllm_qwen_nvfp4_prefixcache_on_retry2_20260610TmanualJST_server_before_probe.log`
- `results/vllm_qwen_nvfp4_prefixcache_on_retry2_20260610TmanualJST_server_after_probe.log`
- `results/vllm_qwen_nvfp4_prefixcache_on_retry2_20260610TmanualJST_container_inspect.json`
