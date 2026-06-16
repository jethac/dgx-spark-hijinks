# 26B-A4B Mixed Whole-Layer KV Ladder

Scope: Vast/sm120 vLLM offline supplied-token discriminator for Gemma 4 26B-A4B. This is a distribution and PPL ladder, not a broad support claim by itself.

## Run Info

```text
schema=vllm-26b-a4b-mixed-layer-ladder/v1
model=google/gemma-4-26B-A4B-it
ctx=512
prefix=256
max_model_len=1024
max_num_batched_tokens=4096
gpu_memory_utilization=0.82
prompt_logprobs=5
keep_topk=5
position_stride=64
dense_prefix_positions=16
first_block=0 1 2 3 4
first_plus_active=0 1 2 3 4 5 6 7
all_sliding=0 1 2 3 4 6 7 8 9 10 12 13 14 15 16 18 19 20 21 22 24 25 26 27 28
all_global=5 11 17 23 29
rows=bf16 fp8_0_4
wheel_ref=sm120a-wheels-4fcbf4c48
wheel_sha256=2e92cc1a0139b3b8e24a881a363098f921cd15efbeed0ab51562a1265d9fa919
vllm_ref=4fcbf4c48f1a90136ca562d61d4b12241f621473
purpose=whole-layer fp8 override ladder for 26B-A4B NVFP4 KV; distribution/top-k evidence, not NLL-only
torch=2.12.0+cu130
cuda=13.0
device=NVIDIA RTX PRO 6000 Blackwell Workstation Edition
capability=(12, 0)
MAX_JOBS=
```

## Row Status

```text
label	dtype	override_dtype	layers	status
bf16	auto			ok
fp8_0_4	nvfp4	fp8_e4m3	0 1 2 3 4	failed_rc_0
```

## Distribution Metrics

| row | delta vs bf16 | delta vs base NVFP4 | bf16 top1 | bf16 Jaccard@5 | base top1 | base Jaccard@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bf16` | `+0.000000000` | `+nan` | `1.000000000` | `1.000000000` | `nan` | `nan` |

Interpretation to be filled after reviewing logs and row failures.
