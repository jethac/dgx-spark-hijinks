# Single-Spark SGLang Recipe

Status: draft, not blessed.

SGLang is tracked as a first-class Spark runtime alongside vLLM and llama.cpp. LiteRT-LM is optional side-runtime coverage.

## Why It Matters

The `hikarioyama/sglang-nvfp4-kv-sm120` repo shows an SGLang NVFP4 KV path for RTX Blackwell-class systems:

- `--kv-cache-dtype fp4_e2m1`
- FlashInfer FA2 kernel patches
- native FP4 KV memory pool
- hybrid-SWA wiring
- per-layer global-scale auto-calibration before CUDA graph capture
- reported 1.778x KV capacity versus fp8 on Step3.7-Flash
- reported fp4 decode near fp8 on large-model tests, with small-model quality warnings

That is relevant to Spark because it attacks the same missing NVFP4 plumbing problem we saw on the vLLM side. It still needs `sm_121` validation on our GB10 unit. Treat the fork as prior art to build from, not as a Spark-blessed result.

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
    --mem-fraction-static 0.40

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

python3 scripts/cuda_build_target_audit.py \
  --log results/${RUN_ID}_server.log \
  --output results/${RUN_ID}_build_target_audit_sglang.json

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
- build/JIT target audit path
- CUDA shared-object audit path
- output quality check against fp8 KV or another reference path
- prompt/generation throughput and memory state

## Current Rule

Use BF16 as the proven local SGLang baseline. Do not bless SGLang NVFP4 KV on Spark until it passes a single-Spark smoke test and a quality check. fp8 is the required comparator; for small models, incoherence under fp4 is an expected negative-control result per the SM120 fork, not a Spark blessing and not necessarily a kernel failure.

## Qwen Speed And KV Probe

Qwen should be the first SGLang before/after target for `fp4_e2m1` KV:

- before row: public Qwen model with BF16 or fp8 KV
- after row: same model with `--kv-cache-dtype fp4_e2m1 --attention-backend flashinfer --page-size 1`
- for NVFP4-weight Qwen checkpoints, attach `scripts/nvfp4_checkpoint_audit.py` output before treating speed as meaningful
- preferred first shape: standard-attention Qwen2.5 7B-class model before Qwen3.6 hybrid/MoE
- required comparator: deterministic output sanity plus fp8-vs-fp4 quality check
- required metrics: KV pool tokens, maximum concurrency, TTFT, warmed decode tok/s, memory state, and selected backend logs

The existing local `Qwen/Qwen2.5-1.5B-Instruct` BF16 smoke is a runtime proof, not a meaningful NVFP4 KV quality proof. Small Qwen models may be negative controls for fp4 KV quality.

Capture each Qwen row with the shared manifest wrapper:

```bash
python3 scripts/record_openai_serving_row.py \
  --backend sglang \
  --phase before \
  --run-id "$RUN_ID" \
  --url http://127.0.0.1:30000 \
  --model "$MODEL" \
  --container-image "$IMAGE" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --attention-backend "$ATTENTION_BACKEND" \
  --server-log "results/${RUN_ID}_server.log" \
  --process-match sglang \
  --cuda-so-package sglang \
  --cuda-so-package sgl_kernel \
  --cuda-so-package flashinfer
```

## Qwen DFlash Probe

AEON's DFlash evidence is vLLM-first, but the current `jethac/sglang` fork already has DFlash-specific SGLang surfaces:

- `python/sglang/srt/arg_groups/speculative_hook.py`
- `python/sglang/srt/models/dflash.py`
- Qwen model `set_dflash_layers_to_capture` hooks
- speculative metrics for accepted drafts

Treat this as a candidate SGLang counterpart, not a proven port. Only run this after ordinary Qwen serving is stable:

```bash
python3 -m sglang.launch_server \
  --model-path QWEN_TARGET_MODEL \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path QWEN_DFLASH_DRAFTER \
  --speculative-num-draft-tokens 15 \
  --attention-backend flashinfer \
  --host 0.0.0.0 \
  --port 30000
```

Record acceptance metrics, decode tok/s, CUDA graph/overlap scheduler state, output quality, and the same non-DFlash Qwen row for comparison. Do not call this an AEON-equivalent result until it runs on GB10 with artifacts.

## AEON-Style Gemma NVFP4 Weight Probe

AEON's Gemma result is useful SGLang prior art for NVFP4 weights, but it does not prove FP4 KV. If testing Gemma 4 in SGLang, start with ordinary KV:

```bash
python3 scripts/nvfp4_checkpoint_audit.py \
  --model-dir /path/to/Gemma-4-26B-A4B-it-Uncensored-NVFP4 \
  --output results/${RUN_ID}_nvfp4_checkpoint_audit.json \
  --strict
```

```bash
python3 -m sglang.launch_server \
  --model-path AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4 \
  --attention-backend triton \
  --dtype bfloat16 \
  --fp4-gemm-backend flashinfer_cutlass
```

If that fails because of backend selection, try `--fp4-gemm-backend marlin` and, if the SGLang build exposes it, the closest MoE runner backend flag. Do not test SGLang DFlash or `fp4_e2m1` KV on Gemma until non-speculative BF16/fp8 KV serving works.

## Experimental NVFP4 KV Fork Probe

If testing the hikarioyama design on Spark, keep it separate from the BF16 smoke and record:

- `hikarioyama/sglang-nvfp4-kv-sm120` source commit, currently audited at `9b2160f0fb8e11dbbb5171a57f06a02b0e9ba6e2`
- SGLang upstream base commit
- FlashInfer patch source and upstream base commit
- `--kv-cache-dtype fp4_e2m1 --attention-backend flashinfer --page-size 1`
- `SGLANG_FP4_KV_AUTOCALIB` value
- checkpoint `k_scale`/`v_scale` presence
- fresh FlashInfer JIT cache path
- build/JIT target audit showing `sm_121`, `sm_121a`, or an explicitly documented compatible SM12x family target
- paired fp8 and fp4 runs on the same model and prompts
- deterministic prompt output and a quality comparison, not just throughput

## Gemma 4 Status

`nvcr.io/nvidia/sglang:26.05-py3` passed a Qwen BF16 smoke on GB10, but failed `google/gemma-4-E2B-it-qat-w4a16-ct` before health:

- default launch: Gemma4 multimodal/audio setup crashed with `MergedColumnParallelLinear` missing `.weight`
- `--language-only` retry: failed validation because SGLang expects `--encoder-urls`

Do not present SGLang as a Gemma 4 path until this is fixed or a different image/commit passes.

## Fork Rule

If SGLang needs source changes, fork `sgl-project/sglang` to `jethac/sglang`, add it as `third_party/sglang`, and do the patch in a worktree named for the GitHub Issue. Build on the hikarioyama implementation unless a GB10-specific test shows it is the wrong path.
