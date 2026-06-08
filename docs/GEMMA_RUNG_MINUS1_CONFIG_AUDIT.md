# Gemma Rung -1 Config Audit

Status: config-only audit complete on 2026-06-08. This is a planning artifact, not a
serving proof. Every rung still has to re-measure geometry from the running runtime before
it can be marked green.

Primary artifact:

- `results/gemma_rung_minus1_config_audit_20260608.json`
- Strict variant artifact: `results/gemma_rung_minus1_config_audit_strict_20260608.json`

## Decision

`D=512` does **not** live only in Gemma 4 26B-A4B. It appears in every audited Gemma 4
server model:

- Gemma 4 12B: full-attention layers use `global_head_dim=512`.
- Gemma 4 31B: full-attention layers use `global_head_dim=512`, and the model is dense.
- Gemma 4 26B-A4B: full-attention layers use `global_head_dim=512`, and the model is MoE.

This means Gemma 4 31B isolates dense `D=512` before the 26B-A4B MoE rung. The ladder is
therefore ordered by two surfaces:

- Config-confirmed attention geometry: Gemma 3 27B -> Gemma 4 31B -> Gemma 4 26B-A4B.
- Operator-provided encoder/modality architecture: Gemma 4 12B is last, because its
  encoder-free multimodality is fused into the decoder/KV path and cannot be quarantined
  by text-only serving.

The strict variant artifact records wrapper/config fields, but those fields do not settle
whether modality is quarantined in an unfired encoder or fused into the decoder. That is a
runtime measurement gate. Treat Gemma 4 31B and 26B-A4B as text-only encoder-quarantine
rungs through Rung 3; treat Gemma 4 12B as the final multimodal-KV rung.

## Model Findings

| model | config source | attention geometry | dense/MoE | encoder/config notes |
|---|---|---|---|---|
| `google/gemma-3-27b-it` | HF config snapshot `005ad3404e59d6023443cb575daa05336842228a` normalized by Transformers | 62 layers: 52 sliding + 10 full; full layers `[5, 11, 17, 23, 29, 35, 41, 47, 53, 59]`; uniform `head_dim=128`; `num_attention_heads=32`, `num_key_value_heads=16`; `sliding_window=1024`; no `global_head_dim` | dense | vision config present; no audio config; the old `uniform head dim 256` plan note was wrong |
| `google/gemma-4-12B-it` | HF config snapshot `5926caa4ec0cac5cbfadaf4077420520de1d5205` | 48 layers: 40 sliding `D=256` + 8 full `D=512`; full layers `[5, 11, 17, 23, 29, 35, 41, 47]`; `num_attention_heads=16`, sliding KV heads `8`, global KV heads `1`; `sliding_window=1024` | dense | operator-provided: encoder-free multimodal; vision+audio fused into decoder/KV; destination rung |
| `google/gemma-4-31B-it` | HF config snapshot `3548789868c5356dbf307c98e6f609007b82b3eb` | 60 layers: 50 sliding `D=256` + 10 full `D=512`; full layers `[5, 11, 17, 23, 29, 35, 41, 47, 53, 59]`; `num_attention_heads=32`, sliding KV heads `16`, global KV heads `4`; `sliding_window=1024` | dense | vision config present; no audio config; this is the dense `D=512` isolation rung |
| `google/gemma-4-26B-A4B-it` | HF config snapshot `20da991ab4afab98e8f910c4a2e8f4fbefc404ad` | 30 layers: 25 sliding `D=256` + 5 full `D=512`; full layers `[5, 11, 17, 23, 29]`; `num_attention_heads=16`, sliding KV heads `8`, global KV heads `2`; `sliding_window=1024` | MoE: `num_experts=128`, `top_k_experts=8` | vision config present; no audio config |

## Variant Findings

The cached QAT/server variants preserve the base attention geometry, so they do not change
the rung ordering:

| model | base | attention geometry | dense/MoE | encoder/config notes |
|---|---|---|---|---|
| `google/gemma-4-12B-it-qat-q4_0-unquantized` | 12B | same as 12B: 40 sliding `D=256` + 8 full `D=512`; BF16 raw KV config hint `344,064` bytes/token | dense | same final-rung operator architecture as base 12B |
| `google/gemma-4-12B-it-qat-w4a16-ct` | 12B | same as 12B: 40 sliding `D=256` + 8 full `D=512`; BF16 raw KV config hint `344,064` bytes/token | dense | same final-rung operator architecture as base 12B |
| `google/gemma-4-31B-it-qat-w4a16-ct` | 31B | same as 31B: 50 sliding `D=256` + 10 full `D=512`; BF16 raw KV config hint `901,120` bytes/token | dense | unified wrapper; vision config present; no audio config |
| `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized` | 26B-A4B | same as 26B-A4B: 25 sliding `D=256` + 5 full `D=512`; BF16 raw KV config hint `225,280` bytes/token | MoE: `num_experts=128`, `top_k_experts=8` | unified wrapper; vision config present; no audio config |

The BF16 raw KV byte counts are config-derived planning hints only. Serving rows must
record actual runtime page layout and KV bytes/token from vLLM/SGLang.

## Ladder Correction

The corrected server ladder is:

1. Rung 0: Qwen standard attention, already proven for vLLM NVFP4 KV.
2. Rung 1: Gemma 3 27B, SWA/hybrid local-global KV with uniform `D=128`, no `D=512`.
3. Rung 2: Gemma 4 31B text-only, dense `D=512` mixed-KV with vision quarantined in the
   unfired encoder.
4. Rung 3: Gemma 4 26B-A4B text-only, MoE on a solved dense `D=512` base.
5. Rung 4: Gemma 4 12B, encoder-free multimodal through the decoder/KV path.

The next vLLM live step remains Gemma 3 27B Rung 1. The next SGLang live step remains the
Qwen FP4-KV quality fix; do not start SGLang Gemma until Qwen is quality-blessed.

## Runtime Gate

For every live rung, record from the running runtime, not just config:

- per-layer `head_dim`
- per-layer query heads and KV heads
- full/sliding layer map and window
- selected per-layer KV dtype/backend
- KV page layout and runtime KV bytes/token
- capacity delta versus a matched fp8 row
- output correctness with a real comparator
