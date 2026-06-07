# Single-Spark SGLang Recipe

Status: draft, not blessed.

SGLang is tracked as a first-class Spark runtime alongside vLLM, llama.cpp, and LiteRT-LM.

## Why It Matters

The `hikarioyama/sglang-nvfp4-kv-sm120` repo shows an SGLang NVFP4 KV path for RTX Blackwell-class systems:

- `--kv-cache-dtype fp4_e2m1`
- FlashInfer FA2 kernel patches
- native FP4 KV memory pool
- hybrid-SWA wiring
- per-layer global-scale auto-calibration before CUDA graph capture

That is relevant to Spark because it attacks the same missing NVFP4 plumbing problem we saw on the vLLM side. It still needs `sm_121` validation on our GB10 unit.

## Preflight

```bash
RUN_ID=sglang_$(date -u +%Y%m%dT%H%M%SZ)
IMAGE=nvcr.io/nvidia/sglang:26.05-py3
MODEL=Qwen/Qwen2.5-1.5B-Instruct

python3 scripts/spark_doctor.py --json > results/spark_doctor_before_${RUN_ID}.json
python3 scripts/spark_doctor.py > results/spark_doctor_before_${RUN_ID}.md
docker manifest inspect $IMAGE > results/${RUN_ID}_manifest.json
docker pull $IMAGE
docker image inspect $IMAGE > results/${RUN_ID}_image_inspect.json
docker run --rm --gpus all $IMAGE nvidia-smi > results/${RUN_ID}_container_nvidia_smi.txt
```

## Smoke

Start an SGLang OpenAI-compatible server:

```bash
docker run -d --rm --name sglang-smoke \
  --gpus all \
  --ipc=host \
  --shm-size 32g \
  -p 30000:30000 \
  -v "$PWD:/workspace/dgx-spark-hijinks" \
  -w /workspace/dgx-spark-hijinks \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN \
  $IMAGE \
  python3 -m sglang.launch_server \
    --model-path $MODEL \
    --host 0.0.0.0 \
    --port 30000 \
    --tp 1 \
    --dtype bfloat16 \
    --mem-fraction-static 0.75

until curl -fsS http://127.0.0.1:30000/health; do sleep 5; done
curl -s http://127.0.0.1:30000/v1/models > results/${RUN_ID}_models.json

python3 scripts/openai_chat_smoke.py \
  --url http://127.0.0.1:30000 \
  --model $MODEL \
  --output results/${RUN_ID}_chat_smoke.json

python3 scripts/runtime_process_probe.py \
  --url http://127.0.0.1:30000 \
  --match sglang \
  --output results/${RUN_ID}_runtime_probe.json

docker exec sglang-smoke python3 -c "import platform, torch, importlib.metadata as m; print('machine', platform.machine()); print('torch', torch.__version__, torch.version.cuda); print('cuda', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print('arch_list', torch.cuda.get_arch_list()); [print(p, m.version(p)) for p in ['sglang','sglang-kernel','flashinfer-python','torch','triton'] if m.distribution(p)]" \
  > results/${RUN_ID}_python_versions.txt 2>&1

docker exec -w /workspace/dgx-spark-hijinks sglang-smoke \
  python3 scripts/cuda_so_audit.py \
    --package sglang \
    --package sgl_kernel \
    --package flashinfer \
    --output results/${RUN_ID}_cuda_so_audit_sglang.json

docker logs sglang-smoke > results/${RUN_ID}_server.log 2>&1
docker rm -f sglang-smoke
```

If `nvcr.io/nvidia/sglang:26.05-py3` fails due to CUDA 13.2 / driver compatibility, retry the same smoke with:

```bash
IMAGE=lmsysorg/sglang:latest-cu130-runtime
```

## Result Requirements

A blessed SGLang result must record:

- SGLang version or container
- model id and revision
- quantization mode
- KV cache dtype
- attention backend
- CUDA graph enabled/disabled
- Spark doctor snapshot
- output quality check against fp8 KV or another reference path
- prompt/generation throughput and memory state

## Current Rule

Do not bless SGLang NVFP4 KV on Spark until it passes a single-Spark smoke test and a quality check. For small models, prefer fp8 KV unless NVFP4 quality is proven on that model.

## Gemma 4 Status

`nvcr.io/nvidia/sglang:26.05-py3` passed a Qwen BF16 smoke on GB10, but failed `google/gemma-4-E2B-it-qat-w4a16-ct` before health:

- default launch: Gemma4 multimodal/audio setup crashed with `MergedColumnParallelLinear` missing `.weight`
- `--language-only` retry: failed validation because SGLang expects `--encoder-urls`

Do not present SGLang as a Gemma 4 path until this is fixed or a different image/commit passes.

## Fork Rule

If SGLang needs source changes, fork `sgl-project/sglang` to `jethac/sglang`, add it as `third_party/sglang`, and do the patch in a worktree named for the GitHub Issue.
