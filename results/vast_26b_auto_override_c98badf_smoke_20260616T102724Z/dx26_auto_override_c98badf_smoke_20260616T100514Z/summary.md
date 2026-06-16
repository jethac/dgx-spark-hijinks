# 26B-A4B Mixed Whole-Layer KV Auto-Override Smoke

Scope: Vast/sm120 vLLM offline supplied-token diagnostic for Gemma 4 26B-A4B. This is a narrow smoke for the `kv_cache_dtype_skip_layers=auto` shape fix, not a serving or quality claim.

Verdict: GREEN for the shape bug. The row `--kv-cache-dtype nvfp4 --kv-cache-dtype-skip-layers 0=auto ... 4=auto` initialized KV cache and completed supplied-token scoring. The earlier failure mode was a packed-shape crash while treating `auto` override groups as global `nvfp4`; this run instead allocated mixed groups successfully and reported `GPU KV cache size: 109,707 tokens`.

Important caveat: this was run at `ctx=512` / `prefix=256` only to prove the runtime shape path. It does not compare quality against bf16/base rows because only the diagnostic row was run.

## Run Info

```text
schema=vllm-26b-a4b-mixed-layer-ladder/v1
model=google/gemma-4-26B-A4B-it
ctx=512
prefix=256
max_model_len=1024
max_num_batched_tokens=4096
gpu_memory_utilization=0.82
prompt_logprobs=20
keep_topk=20
position_stride=64
dense_prefix_positions=128
first_block=0 1 2 3 4
first_plus_active=0 1 2 3 4 5 6 7
all_sliding=0 1 2 3 4 6 7 8 9 10 12 13 14 15 16 18 19 20 21 22 24 25 26 27 28
all_global=5 11 17 23 29
rows=bf16_0_4
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

Runtime source overlay:

- base wheel: `sm120a-wheels-4fcbf4c48`
- source branch: `jethac/vllm@spark/hijinks-e3-vllm`
- semantic patch head: `c98badf Fix mixed KV auto override shape resolution`
- preceding required branch fixes: `d0f6221` padded mixed KV standard attention pages; `505513a` keep `calculate_kv_scales` for quantized overrides
- overlaid installed files: `vllm/v1/core/kv_cache_utils.py`, `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/kv_cache_interface.py`, `vllm/model_executor/layers/attention/attention.py`
- FlashInfer source ref: `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`
- Vast instance: RTX PRO 6000 WS sm120, Ubuntu 22.04 CUDA 13.0 image; instance destroyed after artifact pull

## Row Status

```text
label	dtype	override_dtype	layers	status
bf16_0_4	nvfp4	auto	0 1 2 3 4	ok
```

## Distribution Metrics

| row | delta vs bf16 | delta vs base NVFP4 | bf16 top1 | bf16 Jaccard@5 | base top1 | base Jaccard@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bf16_0_4` | `+nan` | `+nan` | `nan` | `nan` | `nan` | `nan` |

## Key Log Evidence

- The row used `kv_cache_dtype=nvfp4` with `kv_cache_dtype_skip_layers=['0=auto', '1=auto', '2=auto', '3=auto', '4=auto']`.
- The engine initialized with FlashInfer and linear NVFP4 V scale factors.
- KV allocation completed and logged `GPU KV cache size: 109,707 tokens`.
- The row completed prompt-logprob scoring with `mean_nll_nats=5.784226017409545`, `ppl=325.1302975131`, and `sampled_positions=130`.

Interpretation: `c98badf` fixes the per-group cache dtype resolution for unquantized `auto` overrides. This unblocks the bf16-control ladder rows that previously crashed before quality measurement. It does not make the whole-layer mixed policy claim-grade; the prior long-context ladder still stands until the newly unblocked bf16 controls are run at the full context.
