# DiffusionGemma DG-R2 Prerequisite: DG-R1 Weight-Warning Audit

Date: 2026-06-11 JST

Status: GREEN for proceeding to a text-only DG-R2 quality baseline. NOT green
for multimodal/image quality claims.

Input artifact:

- `results/sglang_dgemma_dgr1_stock_smoke_20260611T2340JST/server.log`
- SGLang commit: `651d55cd2e6a3d90de0eb65af643d0aa4ee7fca2`

## Warning Summary

The DG-R1 live server log reports:

```text
Some weights were not initialized from checkpoint: [...]
```

Parsed count: `742` names.

Breakdown:

| Group | Count | Classification |
|---|---:|---|
| `model.layers.*.router.norm.weight` | 30 | Expected SGLang-derived router field |
| `model.layers.*.router.root_size` | 30 | Expected SGLang-derived constant |
| `model.layers.*.self_attn.v_norm.weight` | 30 | Expected no-scale RMSNorm buffer |
| `model.layers.*.self_attn.rotary_emb.cos_sin_cache` | 2 | Expected runtime RoPE cache buffer |
| `model.self_conditioning.post_norm.weight` | 1 | Expected no-scale RMSNorm buffer |
| `embed_vision.embedding_pre_projection_norm.weight` | 1 | Vision-path no-scale RMSNorm buffer |
| `vision_tower.encoder.layers.*.layer_scalar` | 27 | Vision-path scalar, not exercised by text-only smoke |
| `vision_tower.encoder.layers.*.self_attn.v_norm.weight` | 27 | Vision-path no-scale RMSNorm buffer |
| `vision_tower.encoder.layers.*.{input,output}_{min,max}` | 594 | Vision-path quant/stat buffers, not audited for image prompts |

## Code Evidence

Text path:

- `Gemma4Router` creates `root_size` as a derived `hidden_size**-0.5`
  constant and fuses `scale * root_size` into `norm.weight` on first forward.
  The checkpoint stores the learned router scale/projection values; the derived
  fused field is not an independent checkpoint tensor.
- `DiffusionGemmaAttention` constructs V normalization as
  `Gemma4RMSNorm(..., with_scale=False)`. `Gemma4RMSNorm` registers that weight
  as a non-persistent ones buffer, so no checkpoint tensor is expected.
- `DiffusionGemmaSelfConditioning` documents `post_norm` as no-scale, so no
  checkpoint tensor is expected.
- `rotary_emb.cos_sin_cache` is a runtime cache buffer, not a checkpoint
  tensor.

Vision path:

- `DiffusionGemmaForBlockDiffusion` injects image soft tokens only when
  `forward_batch.dllm_is_encoder` and `forward_batch.contains_image_inputs()`
  are both true.
- The DG-R1 smoke and the planned first DG-R2 baseline are text-only, so the
  vision tower and `embed_vision` path are not exercised by those claims.
- The 594 vision quant/stat buffers are not classified as harmless for image
  serving. They are simply out of scope for text-only DG-R2.

## Decision

Proceed to DG-R2 text-only quality baseline. The live warning does not indicate
missing text backbone, MoE, self-conditioning projection, or LM-head weights.

Do not make a multimodal/image quality claim from DG-R1. Before image prompts
are claim-grade, add a separate vision-load/vision-forward audit that checks the
vision tower scalar/stat buffers against the official implementation.
