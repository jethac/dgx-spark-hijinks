# vLLM Qwen Clean-Path NVFP4-KV PPL Sweep

Status: staged; 8k live smoke attempted on 2026-06-09 JST but not accepted as a PPL row.

Goal: measure the quality cost of Qwen3.6 NVFP4 KV versus fp8 KV on the clean full-attention path, excluding the known broken reuse paths (Gemma/SWA and SGLang radix/prefix-cache reuse).

## Scope

- Model: `/home/jethac/models/aeon/qwen36-nvfp4`
- Served model: `qwen36-fast`
- Image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv`
- Runtime ref: `jethac/vllm@a919d635d + jethac/flashinfer@e152cf4d + jethac/flash-attention@7d53245 + cutlass-dsl==4.5.2`
- Attention backend: `flashinfer`
- Contexts: `8192`, `32768`, `131072`
- Prefix caching: disabled with `--no-enable-prefix-caching`
- Measurement API: `/v1/completions` with token-ID prompt input, `prompt_logprobs=1`, `max_tokens=1`, `return_token_ids=true`
- Harness: `scripts/vllm_prompt_ppl_sweep.py`

The harness intentionally uses raw completions with token IDs instead of chat completions. This gives exact context lengths and avoids measuring the chat template. If a chat-formatted PPL row is needed later, run it as a separate row with `chat_template_kwargs={"enable_thinking": false}`.

## Acceptance Gate

For each KV dtype, the server log must prove:

- `enable_prefix_caching=False`
- `kv_cache_dtype=fp8` or `kv_cache_dtype=nvfp4`
- `Using AttentionBackendEnum.FLASHINFER backend`
- for NVFP4 KV: `Using FlashInfer FA2 backend for NVFP4 KV cache on SM12x with vLLM V-scale-factor deswizzle enabled.`
- the weight/MoE backend is unchanged between fp8 and NVFP4 rows

The row is invalid if a non-KV backend changes between fp8 and NVFP4, because the PPL delta would no longer isolate KV dtype.

## Commands

Important: the commands below are the launch skeleton. The 2026-06-09 smoke showed
that this skeleton can select `VLLM_CUTLASS` MoE, while the accepted capacity row
used `MARLIN`. Before accepting any PPL row, add and verify the exact env/flag that
freezes the non-KV backend to match the accepted capacity row.

Prepare an isolated run directory:

```bash
RUN=vllm_qwen_clean_ppl_$(date -u +%Y%m%dT%H%M%SZ)
ROOT=/home/jethac/spark_tmp/${RUN}
mkdir -p ${ROOT}/scripts ${ROOT}/docs ${ROOT}/results
cp /home/jethac/dgx-spark-hijinks/scripts/vllm_prompt_ppl_sweep.py ${ROOT}/scripts/
cp /home/jethac/dgx-spark-hijinks/scripts/spark_hardware.py ${ROOT}/scripts/
cp /home/jethac/dgx-spark-hijinks/docs/CAMPAIGN_LOG.md ${ROOT}/docs/
```

Start the fp8 server:

```bash
docker run -d --name ${RUN}_fp8 --gpus all --net host --ipc host \
  -v /home/jethac/models/aeon/qwen36-nvfp4:/models/target:ro \
  -v ${ROOT}:/work -w /work \
  jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv \
  vllm serve /models/target \
    --served-model-name qwen36-35b-heretic qwen36-fast qwen36-deep \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code --max-model-len 262144 \
    --quantization compressed-tensors --load-format safetensors \
    --attention-backend flashinfer --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.85 --max-num-batched-tokens 65536 --max-num-seqs 128 \
    --enable-chunked-prefill --no-enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

After `/v1/models` is healthy and the server log passes the acceptance gate:

```bash
docker exec ${RUN}_fp8 python3 scripts/vllm_prompt_ppl_sweep.py \
  --url http://127.0.0.1:8000 \
  --model qwen36-fast \
  --tokenizer /models/target \
  --text-file docs/CAMPAIGN_LOG.md \
  --ctx 8192 --ctx 32768 --ctx 131072 \
  --run-id ${RUN}_fp8 \
  --kv-cache-dtype fp8 \
  --runtime-ref 'jethac/vllm@a919d635d + jethac/flashinfer@e152cf4d clean full-attention PPL' \
  --container-image jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv \
  --output results/${RUN}_fp8_ppl.json
docker logs ${RUN}_fp8 > ${ROOT}/results/${RUN}_fp8_server.log 2>&1
docker rm -f ${RUN}_fp8
```

Repeat with `--kv-cache-dtype nvfp4`:

```bash
docker run -d --name ${RUN}_nvfp4 --gpus all --net host --ipc host \
  -v /home/jethac/models/aeon/qwen36-nvfp4:/models/target:ro \
  -v ${ROOT}:/work -w /work \
  jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv \
  vllm serve /models/target \
    --served-model-name qwen36-35b-heretic qwen36-fast qwen36-deep \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code --max-model-len 262144 \
    --quantization compressed-tensors --load-format safetensors \
    --attention-backend flashinfer --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization 0.85 --max-num-batched-tokens 65536 --max-num-seqs 128 \
    --enable-chunked-prefill --no-enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3

docker exec ${RUN}_nvfp4 python3 scripts/vllm_prompt_ppl_sweep.py \
  --url http://127.0.0.1:8000 \
  --model qwen36-fast \
  --tokenizer /models/target \
  --text-file docs/CAMPAIGN_LOG.md \
  --ctx 8192 --ctx 32768 --ctx 131072 \
  --run-id ${RUN}_nvfp4 \
  --kv-cache-dtype nvfp4 \
  --runtime-ref 'jethac/vllm@a919d635d + jethac/flashinfer@e152cf4d clean full-attention PPL' \
  --container-image jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv \
  --output results/${RUN}_nvfp4_ppl.json
docker logs ${RUN}_nvfp4 > ${ROOT}/results/${RUN}_nvfp4_server.log 2>&1
docker rm -f ${RUN}_nvfp4
```

Compare:

```bash
python3 scripts/vllm_prompt_ppl_sweep.py \
  --compare-fp8 results/${RUN}_fp8_ppl.json \
  --compare-nvfp4 results/${RUN}_nvfp4_ppl.json \
  --output results/${RUN}_comparison.json
```

## 2026-06-09 Smoke Attempt

Artifacts:

- `results/vllm_qwen_ppl_smoke_20260609Tlocal_fp8_server.log`
- `results/vllm_qwen_ppl_smoke_20260609Tlocal_fp8_retry_server.log`

Result: no PPL number accepted.

Findings:

- Both fp8 smoke launches proved `enable_prefix_caching=False`, but both selected `VLLM_CUTLASS` NvFp4 MoE while the accepted Qwen NVFP4-KV capacity row used `MARLIN`. That would make a fp8-vs-NVFP4 PPL delta non-isolated.
- The first fp8 launch reached `GPU KV cache size: 6,076,050 tokens`, but startup took about `466.81 s` through profile/KV/cache/warmup and the readiness wait missed the API becoming healthy.
- GPU was returned idle and no PPL containers were left running.
