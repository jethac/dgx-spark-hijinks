# Codex -> Claude: vLLM 12B backend discriminator result

Artifact:

- `results/vast_vllm_gemma4_12b_backend_discriminator_20260613T2030JST/summary.md`

Short result:

- bf16 + `TRITON_ATTN` is green: mean NLL `4.534544480957847`, PPL `93.1810598883856`, chat smoke `Tokyo`.
- bf16 + `FLASH_ATTN` is unsupported for this Gemma 4 config: partial multimodal token full attention unsupported; mm-prefix needs FlashAttention v4 for this head size.
- NVFP4 K+V + `TRITON_ATTN` is unsupported: `kv_cache_dtype not supported`.
- Same-machine FlashInfer rerun was setup/provenance-red, not a quality row.

Interpretation:

The alternate bf16 backend does not explain the large NVFP4 delta. Triton bf16 is slightly better than the earlier FlashInfer bf16 row by about `0.02068449` nats/token, while the NVFP4 delta in the matched FlashInfer anchor was `+0.281435499`.

This still does not prove the delta is inherent NVFP4 loss, because the current wheel/config has no alternate backend that can run the matching NVFP4 cell. It does narrow the confound: bf16 FlashInfer baseline quality is not hiding the whole effect.

