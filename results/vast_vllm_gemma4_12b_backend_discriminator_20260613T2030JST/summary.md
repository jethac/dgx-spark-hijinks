# Vast vLLM Gemma 4 12B Backend Discriminator

Date: 2026-06-13 JST

Purpose: follow-up to the matched vLLM Gemma 4 12B KV anchor, checking whether the observed NVFP4 quality delta could be explained by the FlashInfer attention backend rather than NVFP4 KV itself.

## Context

Previous anchor:

- `results/vast_vllm_gemma4_12b_matched_kv_anchor_20260613T1935JST/summary.md`
- Model: `google/gemma-4-12B-it`
- Device: RTX PRO 6000 Blackwell Workstation Edition, sm_120
- vLLM wheel: `vllm-0.1.dev1+ge32459eea.sm120a`
- Torch/CUDA: `2.12.0+cu130` / CUDA 13.0
- Prior FlashInfer bf16: mean NLL `4.555228970943781`, PPL `95.12853441238842`
- Prior FlashInfer NVFP4 K+V: mean NLL `4.836664469623121`, PPL `126.04821211811733`
- Prior NVFP4 delta: `+0.2814354986793397` nats/token

This run used the same prompt-scoring harness after adding an explicit `--attention-backend` argument.

## Results

| Cell | Result | Notes |
| --- | --- | --- |
| bf16 + `TRITON_ATTN` | green | mean NLL `4.534544480957847`, PPL `93.1810598883856`, scored `4088` tokens, chat smoke `Tokyo` |
| NVFP4 K+V + `TRITON_ATTN` | unsupported | vLLM rejects the configuration: `kv_cache_dtype not supported` |
| bf16 + `FLASH_ATTN` | unsupported | vLLM rejects the Gemma 4 config: partial multimodal token full attention unsupported; mm-prefix requires FlashAttention v4 for this head size |
| bf16 + default FlashInfer rerun | setup/provenance failure | packaged path hit `Unsupported max_mma_kv: 0`; source-shim path hit a CUDA 13 compile failure in `fastdiv.cuh` |

The only completed alternate-backend quality row is bf16 + Triton:

```json
{
  "kv_cache_dtype": "bfloat16",
  "score": {
    "mean_nll_nats": 4.534544480957847,
    "ppl": 93.1810598883856,
    "num_scored_tokens": 4088
  },
  "chat_smoke": {
    "generated": "Tokyo",
    "contains_tokyo_or_paris": true
  }
}
```

## Interpretation

The Triton bf16 result argues against FlashInfer bf16 attention being the source of the large NVFP4 delta: Triton bf16 is slightly better than the prior FlashInfer bf16 row by about `0.02068449` nats/token.

This run does **not** prove the NVFP4 delta is inherent to NVFP4 KV, because vLLM cannot run the matching NVFP4 cell on Triton or FlashAttention for this Gemma 4 configuration. It only narrows the interpretation:

- The baseline bf16 backend choice is not hiding a `+0.281` nats/token penalty.
- The large delta remains associated with the NVFP4-on-FlashInfer path.
- There is no alternate-backend NVFP4 discriminator available in this wheel/config.

## Artifacts

- `bf16_triton_attn.json`
- `bf16_triton_attn.log`
- `nvfp4_triton_attn.log`
- `bf16_flash_attn_arg.log`
- `bf16_flash_attn.log`
- `bf16_flashinfer.log`
- `bf16_flashinfer_shim.log`

