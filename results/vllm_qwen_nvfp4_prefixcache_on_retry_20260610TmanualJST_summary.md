# vLLM Qwen NVFP4-KV Prefix-Cache Reuse Probe - Startup Blocker

Date: 2026-06-10 JST

## Verdict

The decisive vLLM cross-check is still unanswered. This run did not reach API readiness before the harness timeout, so it produced no prefix-cache reuse probe and no evidence for or against full NVFP4 K+V correctness under cache reuse.

This is a startup/readiness blocker, not a model-quality result.

## Why This Run Exists

The SGLang full-NVFP4 radix investigation showed that:

- SGLang dense/no-prefix serving attends over raw current K/V while writing FP4 cache.
- SGLang cached-prefix serving eventually reads FP4 K from cache, and full NVFP4 K+V collapses on the Qwen first-token gate.
- The merge arithmetic itself is correct; regime-matched FP4 partials merge back to the all-FP4 recompute.

The remaining campaign question is whether vLLM with prefix caching enabled can serve correctly while actually reusing full NVFP4 K+V cache. If vLLM works, SGLang likely has a feed/scale bug. If vLLM also breaks, the earlier 1.78x NVFP4-KV headline must be scoped more narrowly because the previous vLLM PPL smoke disabled prefix caching.

## Run Configuration

- Runner: `scripts/run_vllm_qwen_prefix_cache_probe.sh`
- Probe: `scripts/vllm_prefix_cache_reuse_probe.py`
- Run id: `vllm_qwen_nvfp4_prefixcache_on_retry_20260610TmanualJST`
- Host: Spark/GB10 over tailnet
- Image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv`
- Model: `/home/jethac/models/aeon/qwen36-nvfp4`
- Served model alias: `qwen36-fast`
- Port: `8001`
- KV cache dtype: `nvfp4`
- Prefix caching: enabled
- Attention backend: FlashInfer
- `--gpu-memory-utilization`: `0.72`
- `--max-model-len`: `262144`
- `--max-num-batched-tokens`: `32768`
- `--max-num-seqs`: `64`
- Docker cgroup: `--memory 100g --memory-swap 100g`
- Readiness timeout: `420s` for this failed run; the runner default has since been raised to `1800s`

## Observed Startup Evidence

The server log confirms the intended test configuration:

- `kv_cache_dtype='nvfp4'`
- `enable_prefix_caching=True`
- `Using nvfp4 data type to store kv cache`
- FlashInfer attention backend selected
- `Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM`
- `Using 'MARLIN' NvFp4 MoE backend`

The container inspect captured:

- `OOMKilled=false`
- cgroup memory limit: `107374182400` bytes
- command line includes `--kv-cache-dtype nvfp4` and `--enable-prefix-caching`
- labels identify FlashInfer source revision `e152cf4da4ab2a9d093b7d9d4b499198b0211c61`
- environment includes `TORCH_CUDA_ARCH_LIST=12.1a` and `FLASHINFER_CUDA_ARCH_LIST=12.1a`

The log reached cold compile/warmup:

- weights loaded in about `171.38s`
- model loading took `21.88 GiB`
- `torch.compile` graph compile took `68.98s`
- AOT compiled function saved under the vLLM torch compile cache

The API still was not listening at `http://127.0.0.1:8001/v1/models` before the 420s readiness timeout. The runner then exited and removed the container via its cleanup trap.

## Artifacts

- `results/vllm_qwen_nvfp4_prefixcache_on_retry_20260610TmanualJST_server_before_probe.log`
- `results/vllm_qwen_nvfp4_prefixcache_on_retry_20260610TmanualJST_server_after_probe.log`
- `results/vllm_qwen_nvfp4_prefixcache_on_retry_20260610TmanualJST_container_inspect.json`

There is no `*_prefix_reuse_probe.json` for this run because the server never reached readiness.

## Interpretation

Do not use this run to claim that vLLM passes or fails full NVFP4 K+V prefix-cache reuse. It only proves that the harness launched the intended server configuration and then timed out during cold startup/warmup.

The timeout was likely too short for this image after cold compile on GB10. The runner default has been raised to 30 minutes; the next retry should keep the same single-server memory guardrails:

```bash
RUN_ID=vllm_qwen_nvfp4_prefixcache_on_retry2_20260610TmanualJST \
PORT=8001 \
bash scripts/run_vllm_qwen_prefix_cache_probe.sh
```

If the next run reaches readiness, the gate remains:

- prove a real prefix-cache hit, via response usage fields or server log metrics;
- verify deterministic first-token/logprob behavior under cache reuse;
- if stable, run an fp8 comparator or equivalent quality check before restoring the public 1.78x NVFP4-KV claim.
