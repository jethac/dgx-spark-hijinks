# DiffusionGemma Vision Warning Static Audit

Date: 2026-06-12 JST

Status: GREEN for classifying the DG-R1 vision-path uninitialized-weight
warnings as SGLang-created defaults rather than missing checkpoint payload.
This is NOT an image-quality or multimodal serving claim.

## Inputs

- Runtime warning source:
  `results/sglang_dgemma_dgr1_stock_smoke_20260611T2340JST/server.log`
- Prior text-only audit:
  `results/sglang_dgemma_dgr2_weight_warning_audit_20260611TmanualJST.md`
- Checkpoint index:
  `B:\workshop\hf_cache\huggingface\hub\models--google--diffusiongemma-26B-A4B-it\snapshots\0f28bc42f588fbd8f71e08102b1c3960298a1358\model.safetensors.index.json`
- Code inspected:
  - `third_party/sglang/python/sglang/srt/models/gemma4_diffusion.py`
  - `third_party/sglang/python/sglang/srt/models/gemma4_mm.py`
  - `third_party/sglang/python/sglang/srt/models/gemma4_vision.py`
  - `third_party/sglang/python/sglang/srt/layers/clippable_linear.py`
  - `third_party/sglang/python/sglang/srt/layers/layernorm.py`

No model download or Spark serving run was performed for this audit.

## Checkpoint Cross-Check

The official BF16 checkpoint index has `1047` keys total:

| Group | Count |
|---|---:|
| `model.encoder.vision_tower.*` | 355 |
| `model.encoder.embed_vision.*` | 1 |
| combined vision/embed payload | 356 |

The combined payload contains the real vision weights: layer norms, q/k/v/o
projection weights, MLP projection weights, patch embedding, `std_bias`,
`std_scale`, and `embed_vision.embedding_projection.weight`.

The checkpoint contains zero keys matching these DG-R1 warning-only names:

| Name pattern | Checkpoint count |
|---|---:|
| `*embedding_pre_projection_norm.weight` | 0 |
| `*self_attn.v_norm.weight` | 0 |
| `*layer_scalar` | 0 |
| `*input_min`, `*input_max`, `*output_min`, `*output_max` | 0 |

So the warning group is not hidden skipped checkpoint payload. It is model
state that SGLang creates locally and that the checkpoint does not provide.

## Warning Classification

DG-R1 emitted `742` uninitialized-weight names. The vision/image portion is:

| Warning group | Count | Static classification |
|---|---:|---|
| `embed_vision.embedding_pre_projection_norm.weight` | 1 | no-scale RMSNorm ones buffer |
| `vision_tower.encoder.layers.*.self_attn.v_norm.weight` | 27 | no-scale RMSNorm ones buffer |
| `vision_tower.encoder.layers.*.layer_scalar` | 27 | default scalar-one buffer |
| `vision_tower.encoder.layers.*` clip bounds | 594 | non-persistent clippable-linear bounds, default no-op infinities |

Code anchors:

- `Gemma4MultimodalEmbedder` constructs
  `embedding_pre_projection_norm = Gemma4RMSNorm(..., with_scale=False)`.
- `Gemma4VisionAttention` constructs
  `v_norm = Gemma4RMSNorm(..., with_scale=False)`.
- `Gemma4RMSNorm(with_scale=False)` registers `weight` as a non-persistent
  ones buffer and applies a normal RMSNorm without learned scale.
- `Gemma4VisionEncoderLayer` registers `layer_scalar` as `torch.ones(())`;
  the checkpoint has no corresponding learned scalar payload.
- `Clippable*Linear` wrappers register clip bounds as non-persistent buffers
  initialized to `-inf` / `+inf`. With no checkpoint clip-stat payload, the
  clamps are no-ops.

## Decision

The prior text-only DG-R2 warning audit is strengthened: there is no evidence
of missing vision checkpoint payload in the DG-R1 warning list. The warning
names are SGLang defaults or no-op runtime buffers.

This clears the load-warning ambiguity for the image path, but it still does
not prove multimodal output quality. Image prompts remain outside the current
DiffusionGemma serving claims until a live vision-forward/image prompt gate
runs against the real processor and OpenAI multimodal endpoint.
