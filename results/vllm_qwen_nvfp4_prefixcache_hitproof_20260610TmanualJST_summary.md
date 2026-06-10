# vLLM Qwen NVFP4-KV Prefix-Cache Hit Proof

Date: 2026-06-10 JST

## Verdict

vLLM successfully served Qwen with full NVFP4 KV while actually reusing local prefix-cache tokens.

This locks the vLLM side of the cross-check: FP4-K cache reuse can serve the probe without a first-token flip in this stack. That makes the SGLang full-NVFP4 radix failure more likely to be a feed/scale bug in SGLang's FP4-K attention path, not inherent FP4-K attention loss.

## Run Configuration

- Runner: `scripts/run_vllm_qwen_prefix_cache_probe.sh`
- Probe: `scripts/vllm_prefix_cache_reuse_probe.py`
- Run id: `vllm_qwen_nvfp4_prefixcache_hitproof_20260610TmanualJST`
- Image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv`
- Model: `/home/jethac/models/aeon/qwen36-nvfp4`
- Served model alias: `qwen36-fast`
- KV cache dtype: `nvfp4`
- Prefix caching: enabled
- Attention backend: FlashInfer
- Docker memory guardrail: `--memory 100g --memory-swap 100g`
- Prompt size: `31,698` chars / `6,054` prompt tokens per request
- vLLM selected attention block size: `3,728` tokens

## Runtime Evidence

The server selected the intended path:

- `Using nvfp4 data type to store kv cache`
- `Using FlashInfer FA2 backend for NVFP4 KV cache on SM12x with vLLM V-scale-factor deswizzle enabled`
- available KV cache memory: `55.72 GiB`
- GPU KV cache size: `9,484,846 tokens`

The probe sent two identical long-prefix requests:

- Request 0: first token `*`, first-token logprob `-0.3430597782`, latency `1.6706s`
- Request 1: first token `*`, first-token logprob `-0.1285892278`, latency `0.4326s`
- `all_first_tokens_same: true`
- `ok: true`

The OpenAI-compatible usage field still did not expose cached-token details:

- `prompt_tokens_details: null`
- `cached_tokens_from_usage: [null, null]`

But Prometheus metrics prove local prefix-cache reuse:

```text
vllm:prefix_cache_queries_total 12108.0
vllm:prefix_cache_hits_total 3728.0
vllm:prompt_tokens_by_source_total{source="local_compute"} 8380.0
vllm:prompt_tokens_by_source_total{source="local_cache_hit"} 3728.0
vllm:prompt_tokens_cached_total 3728.0
```

This lines up with the selected attention block size: request 2 hit one full `3728`-token local cache block.

## Interpretation

This closes the missing proof from the previous shorter-prompt run. The earlier 2054-token prompt could not hit because it was below the selected 3728-token cache block size; metrics from that run showed `prefix_cache_hits_total = 0`.

With a >block-size prompt, vLLM did prove a local prefix-cache hit and did not flip the first token. Therefore:

- vLLM full NVFP4 K+V with prefix reuse: green for this smoke gate.
- FP4-K reuse is not categorically broken across runtimes.
- SGLang's full-NVFP4 radix failure should be treated as a SGLang FP4-K feed/scale bug until proven otherwise.
- The next target is SGLang's cached-prefix FP4-K read into FlashInfer attention: global scale, block scale, inverse/handedness, and layout/order relative to vLLM's working path.

## Caveats

- This is a first-token/logprob smoke gate, not a full PPL or long-form quality sweep.
- Qwen's hybrid cache path logs a Mamba `align` prefix-caching caveat, so later full-quality claims should keep model architecture in scope.
- One prior vLLM run with the shorter prompt returned `**` then `*`, but metrics showed no prefix-cache hit; do not attribute that mismatch to reused FP4 K.

## Artifacts

- `results/vllm_qwen_nvfp4_prefixcache_hitproof_20260610TmanualJST_prefix_reuse_probe.json`
- `results/vllm_qwen_nvfp4_prefixcache_hitproof_20260610TmanualJST_metrics_after_probe.txt`
- `results/vllm_qwen_nvfp4_prefixcache_hitproof_20260610TmanualJST_server_after_probe.log`
- `results/vllm_qwen_nvfp4_prefixcache_hitproof_20260610TmanualJST_container_inspect.json`
