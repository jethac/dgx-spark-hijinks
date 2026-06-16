# 26B-A4B Mixed Whole-Layer KV Ladder

Scope: Vast/sm120 vLLM offline supplied-token discriminator for Gemma 4 26B-A4B. This is a distribution and PPL ladder, not a broad support claim by itself.

## Run Info

```text
schema=vllm-26b-a4b-mixed-layer-ladder/v1
model=google/gemma-4-26B-A4B-it
ctx=8185
prefix=4096
max_model_len=8192
max_num_batched_tokens=4096
gpu_memory_utilization=0.82
prompt_logprobs=20
keep_topk=20
position_stride=128
dense_prefix_positions=256
first_block=0 1 2 3 4
first_plus_active=0 1 2 3 4 5 6 7
all_sliding=0 1 2 3 4 6 7 8 9 10 12 13 14 15 16 18 19 20 21 22 24 25 26 27 28
all_global=5 11 17 23 29
rows=bf16 base_k100 bf16_0_4 fp8_0_4 fp8_0_7 fp8_all_sliding fp8_all_global
wheel_ref=sm120a-wheels-4fcbf4c48
wheel_sha256=2e92cc1a0139b3b8e24a881a363098f921cd15efbeed0ab51562a1265d9fa919
vllm_ref=4fcbf4c48f1a90136ca562d61d4b12241f621473
purpose=whole-layer fp8 override ladder for 26B-A4B NVFP4 KV; distribution/top-k evidence, not NLL-only
torch=2.12.0+cu130
cuda=13.0
device=NVIDIA RTX PRO 6000 Blackwell Workstation Edition
capability=(12, 0)
MAX_JOBS=4
```

Patch note: the run used the published `sm120a-wheels-4fcbf4c48` wheel plus the
`jethac/vllm@spark/hijinks-e3-vllm` source patch `d0f6221` applied to installed
`vllm/v1/core/kv_cache_utils.py`. See `PATCH_INFO.txt`.

## Row Status

```text
label	dtype	override_dtype	layers	status
bf16	auto			ok
base_k100	nvfp4			ok
bf16_0_4	nvfp4	auto	0 1 2 3 4	failed_rc_1
fp8_0_4	nvfp4	fp8_e4m3	0 1 2 3 4	ok
fp8_0_7	nvfp4	fp8_e4m3	0 1 2 3 4 5 6 7	failed_rc_1
fp8_all_sliding	nvfp4	fp8_e4m3	0 1 2 3 4 6 7 8 9 10 12 13 14 15 16 18 19 20 21 22 24 25 26 27 28	ok
fp8_all_global	nvfp4	fp8_e4m3	5 11 17 23 29	failed_rc_1
```

## Distribution Metrics

| row | delta vs bf16 | delta vs base NVFP4 | bf16 top1 | bf16 Jaccard@5 | base top1 | base Jaccard@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_k100` | `-0.117964257` | `+0.000000000` | `0.674825175` | `0.569111444` | `1.000000000` | `1.000000000` |
| `bf16` | `+0.000000000` | `+0.117964257` | `1.000000000` | `1.000000000` | `0.674825175` | `0.569111444` |
| `fp8_0_4` | `-0.670869492` | `-0.552905235` | `0.622377622` | `0.515318015` | `0.618881119` | `0.481962482` |
| `fp8_all_sliding` | `-0.577708249` | `-0.459743992` | `0.674825175` | `0.580364080` | `0.678321678` | `0.534257409` |

## Interpretation

The `d0f6221` padding fix scales from the short smoke to long context for mixed rows that keep Gemma 4
26B-A4B's D=512 global layers in NVFP4. The prior storage/stride crash is gone for the successful mixed
rows.

Successful rows:

- `fp8_0_4`: global NVFP4 with layers `0..4=fp8_e4m3`; `322,723` KV tokens; mean NLL `7.262490918`;
  delta `-0.670869492` vs bf16 and `-0.552905235` vs `base_k100`.
- `fp8_all_sliding`: all sliding layers fp8, global layers NVFP4; `322,723` KV tokens; mean NLL
  `7.355652161`; delta `-0.577708249` vs bf16 and `-0.459743992` vs `base_k100`.

Failed rows:

- `bf16_0_4`: per-layer `auto` override is still on the old shape path and tries to view the first five
  auto/bf16 layers as NVFP4 packed shape `[..., 144]`, failing with
  `RuntimeError: shape '[41699, 2, 16, 8, 144]' is invalid for input of size 2732785664`.
- `fp8_0_7` and `fp8_all_global`: any fp8 override that includes Gemma 4's D=512 global layers reaches
  FlashInfer's explicit sm120 shared-memory reject for 1-byte KV:
  `head_dim_qk=512 head_dim_vo=256 ... minimum valid tile needs 122880 B but only 102400 B/threadblock is available`.

This is a discriminator/checkpoint, not a claim-grade serving row. Whole-layer fp8 on sliding layers can
run at long context after the padding fix, but the distribution metrics remain lower-NLL/overconfident
rather than parity with bf16. D=512 global layers cannot be fp8 on sm120 with the current FlashInfer prefill
kernel; they must remain NVFP4 or use a different kernel strategy.
