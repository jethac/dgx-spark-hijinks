# vLLM Gemma 3 NVFP4-KV Trace Packet, 2026-06-08

Run this on the GB10 host only when the GPU is free.

Purpose: rerun the Gemma 3 27B fp8/NVFP4 first-token comparison with
`jethac/vllm@e2a8197a9` trace hooks enabled. This is a diagnostic packet, not a benchmark
row.

## Assumptions

```bash
cd /home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608
export HF_TOKEN="$(tr -d '\r' < /home/jethac/.cache/huggingface/token)"

IMAGE=jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass
VLLM_SRC=/home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608/third_party/vllm
FLASHINFER_SRC=/home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608/third_party/flashinfer
HF_CACHE=/home/jethac/.cache/huggingface
RESULTS_DIR=results
MODEL=google/gemma-3-27b-it
SERVED_MODEL=gemma3-27b-it
STAMP=20260608TTRACEJST
```

Expected source pins:

- `third_party/vllm`: `e2a8197a9c8b67172aa909463c58f6e447ad2bba`
- `third_party/flashinfer`: `e41016fcd121986aea923d5c7e68fc9f152d2a07`

Common server args:

```bash
COMMON_ARGS="--served-model-name ${SERVED_MODEL} --dtype bfloat16 \
  --attention-backend flashinfer --max-model-len 131072 \
  --gpu-memory-utilization 0.85 --max-num-batched-tokens 4096 \
  --host 0.0.0.0 --port 8000"
```

Trace envs for both rows:

```bash
VLLM_SPARK_KV_TRACE=1
VLLM_SPARK_KV_TRACE_LIMIT=512
VLLM_SPARK_KV_TRACE_VALUES=16
VLLM_SPARK_KV_TRACE_LAYERS=layers.0.self_attn.attn,layers.5.self_attn.attn,layers.6.self_attn.attn
```

## Server Row Template

Use this once with `KV_DTYPE=fp8` and once with `KV_DTYPE=nvfp4`.

```bash
KV_DTYPE=fp8
RUN=vllm_gemma3_27b_rung1_${STAMP}_${KV_DTYPE}_flashinfer

docker run -d --gpus all --ipc=host --network=host --name ${RUN} \
  -e HF_TOKEN \
  -e VLLM_USE_V1=1 \
  -e VLLM_LOGGING_LEVEL=DEBUG \
  -e VLLM_SPARK_KV_GEOMETRY_LOG=1 \
  -e VLLM_SPARK_KV_TRACE=1 \
  -e VLLM_SPARK_KV_TRACE_FILE=/results/${RUN}_kv_trace.jsonl \
  -e VLLM_SPARK_KV_TRACE_LIMIT=512 \
  -e VLLM_SPARK_KV_TRACE_VALUES=16 \
  -e VLLM_SPARK_KV_TRACE_LAYERS=layers.0.self_attn.attn,layers.5.self_attn.attn,layers.6.self_attn.attn \
  -e SPARK_FLASHINFER_SOURCE_ROOT=/flashinfer-src \
  -e FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e CUDA_MODULE_LOADING=LAZY \
  -v "${VLLM_SRC}:/vllm-src" \
  -v "${FLASHINFER_SRC}:/flashinfer-src" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "$(pwd)/${RESULTS_DIR}:/results" \
  -v "$(pwd):/workspace/dgx-spark-hijinks" \
  --entrypoint bash "${IMAGE}" -lc '
set -euo pipefail
mkdir -p /tmp/spark-sitecustomize
cp /workspace/dgx-spark-hijinks/scripts/flashinfer_source_sitecustomize.py /tmp/spark-sitecustomize/sitecustomize.py
export PYTHONPATH="/tmp/spark-sitecustomize:${PYTHONPATH:-}"
cd /vllm-src
VLLM_USE_PRECOMPILED=1 \
VLLM_MAIN_CUDA_VERSION=13.0 \
VLLM_PRECOMPILED_WHEEL_COMMIT=4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa \
VLLM_PRECOMPILED_SKIP_FLASH_ATTN=1 \
VLLM_VERSION_OVERRIDE=0.1.dev1+ge2a8197a9 \
python3 -m pip install --no-build-isolation --no-deps -e . > /results/'${RUN}'_editable_install.log 2>&1
cp /opt/jethac-vllm/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so /vllm-src/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
exec vllm serve '${MODEL}' --kv-cache-dtype '${KV_DTYPE}' '${COMMON_ARGS}'
'

docker logs -f ${RUN} > ${RESULTS_DIR}/${RUN}_server.log 2>&1 &
```

For NVFP4, use:

```bash
KV_DTYPE=nvfp4
RUN=vllm_gemma3_27b_rung1_${STAMP}_nvfp4_kv_flashinfer
```

## Client Probes

Run after `/v1/models` is ready, before any benchmark or row manifest traffic.

```bash
python3 scripts/openai_first_token_probe.py \
  --url http://127.0.0.1:8000 \
  --model gemma3-27b-it \
  --backend vllm \
  --phase before \
  --run-id vllm_gemma3_27b_rung1_${STAMP}_fp8_flashinfer_first_token \
  --output results/vllm_gemma3_27b_rung1_${STAMP}_fp8_flashinfer_first_token.json

python3 scripts/openai_first_token_probe.py \
  --url http://127.0.0.1:8000 \
  --model gemma3-27b-it \
  --backend vllm \
  --phase after \
  --run-id vllm_gemma3_27b_rung1_${STAMP}_nvfp4_kv_flashinfer_first_token \
  --output results/vllm_gemma3_27b_rung1_${STAMP}_nvfp4_kv_flashinfer_first_token.json

python3 scripts/openai_first_token_probe.py \
  --input-report results/vllm_gemma3_27b_rung1_${STAMP}_nvfp4_kv_flashinfer_first_token.json \
  --compare-to results/vllm_gemma3_27b_rung1_${STAMP}_fp8_flashinfer_first_token.json \
  --run-id vllm_gemma3_27b_rung1_${STAMP}_first_token_compare \
  --output results/vllm_gemma3_27b_rung1_${STAMP}_first_token_compare.json
```

## Expected Artifacts

- `results/vllm_gemma3_27b_rung1_${STAMP}_fp8_flashinfer_server.log`
- `results/vllm_gemma3_27b_rung1_${STAMP}_fp8_flashinfer_kv_trace.jsonl`
- `results/vllm_gemma3_27b_rung1_${STAMP}_fp8_flashinfer_first_token.json`
- `results/vllm_gemma3_27b_rung1_${STAMP}_fp8_flashinfer_editable_install.log`
- `results/vllm_gemma3_27b_rung1_${STAMP}_nvfp4_kv_flashinfer_server.log`
- `results/vllm_gemma3_27b_rung1_${STAMP}_nvfp4_kv_flashinfer_kv_trace.jsonl`
- `results/vllm_gemma3_27b_rung1_${STAMP}_nvfp4_kv_flashinfer_first_token.json`
- `results/vllm_gemma3_27b_rung1_${STAMP}_nvfp4_kv_flashinfer_editable_install.log`
- `results/vllm_gemma3_27b_rung1_${STAMP}_first_token_compare.json`

## Pass/Fail

Pass:

- editable install reports vLLM `0.1.dev1+ge2a8197a9`
- fp8 first-token probe succeeds for all cases
- NVFP4 server log shows FlashInfer FA2 NVFP4 SM12x path
- NVFP4 trace JSONL contains `fi_metadata`, `kv_write_pre`, `kv_write_post_nvfp4`,
  `kv_read_views_nvfp4`, and preferably `swa_skip`
- trace layer names include layers `0`, `5`, and `6`
- sampled NVFP4 write/read slots show consistent page/offset and sane data/scale views

Fail:

- no `*_kv_trace.jsonl`
- trace limit is consumed before client probes (the first live `LIMIT=8` run hit this; use
  `512` or higher unless warmup tracing is disabled)
- NVFP4 falls back away from FlashInfer FA2
- fp8 comparator fails
- NVFP4 trace shows page or scale/data view mismatch for the same sampled slot

Risk:

- `swa_skip` may be absent on short prompts that do not cross the `1024`-token sliding
  window boundary. That absence is not itself a failure; the metadata and NVFP4 split-view
  events are the first gate.
