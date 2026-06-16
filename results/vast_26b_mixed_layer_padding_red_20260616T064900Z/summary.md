# Vast 26B-A4B Mixed Whole-Layer KV Padding Red

## Verdict

RED before quality. The `sm120a-wheels-4fcbf4c48` wheel accepts `kv_cache_dtype_skip_layers` and reaches mixed per-layer KV-cache planning, but the first mixed row (`0..4=fp8_e4m3`, rest `nvfp4`) fails during KV-cache tensor materialization.

This is a vLLM mixed-page padding/reshape blocker, not a Gemma 4 quality result.

## Stack

- Model: `google/gemma-4-26B-A4B-it`
- Hardware: Vast RTX PRO 6000 Blackwell Workstation Edition, `sm_120`, Ubuntu 22.04 CUDA 13 container
- vLLM: `0.1.dev1+g4fcbf4c48.sm120a`
- Wheel: `sm120a-wheels-4fcbf4c48`
- Wheel SHA256: `2e92cc1a0139b3b8e24a881a363098f921cd15efbeed0ab51562a1265d9fa919`
- FlashInfer source overlay: `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`
- Harness: `docs/vast_anchor/run_26b_mixed_layer_ladder.sh`
- Scorer: `docs/vast_anchor/vllm_toplogprob_attribution.py`

The Vast instance created for this run was destroyed after artifacts were pulled.

## Rows

Smoke context only: `ctx=512`, `prefix=256`, `max_model_len=1024`, `max_num_batched_tokens=4096`, `prompt_logprobs=5`, `keep_topk=5`.

| row | KV config | status | note |
| --- | --- | --- | --- |
| `bf16` | `kv_cache_dtype=auto` | GREEN control | Mean NLL `3.7926521906438246`, PPL `44.373932476731696`, `119,607` KV-cache tokens |
| `fp8_0_4` | global `nvfp4`, layers `0 1 2 3 4 = fp8_e4m3` | RED before quality | Engine init fails in `_reshape_kv_cache_tensors()` |

## Failure

The mixed row command reaches vLLM with the expected per-layer override:

```text
kv_cache_dtype='nvfp4'
kv_cache_dtype_skip_layers=['0=fp8_e4m3', '1=fp8_e4m3', '2=fp8_e4m3', '3=fp8_e4m3', '4=fp8_e4m3']
```

vLLM then unifies mixed per-layer page sizes to `65536` bytes. The first failing layer is layer 5:

```text
Padding KV cache pages of layer language_model.model.layers.5.self_attn.attn from 55296 to 65536 bytes (block_size 16 -> 48, 15.62% of this layer's pool wasted)
```

Focused debug instrumentation captured the allocation/reshape mismatch:

```text
CODEX_PAD_DEBUG language_model.model.layers.5.self_attn.attn raw 5465636864 spec_page 65536 real 55296 padded 65536 num_blocks 83399 shape (250197, 2, 16, 2, 288) order (0, 1, 2, 3, 4) inv [0, 1, 2, 3, 4] dtype torch.uint8 cache_dtype nvfp4
RuntimeError: setStorage: sizes [250197, 2, 16, 2, 288], strides [65536, 9216, 576, 288, 1], storage offset 0, and itemsize 1 requiring a storage size of 16396863488 are out of bounds for storage of size 5465636864
```

Interpretation: storage is allocated for `83399` padded pages, but the standard-attention reshape expands those pages into `250197` kernel-block rows (`3x`) and then applies the padded page stride to each kernel-block row. That requires `3x` the allocated storage. Mixed non-divisible per-layer page padding is therefore not materialized correctly for standard attention.

## Artifacts

- Clean smoke: `dx26_mixed_script_smoke3_20260616T062811Z/`
- Focused debug rerun: `dx26_mixed_fp8_debug2_20260616T064444Z/`
- Bundle: `results/dx26_mixed_smoke_and_padding_red_20260616T064900Z.tgz`

## Next

Fix vLLM's padded-page standard-attention materialization for mixed per-layer KV specs before rerunning the whole-layer fp8 ladder. Do not run the full quality ladder against this wheel: the mixed rows fail before serving.
