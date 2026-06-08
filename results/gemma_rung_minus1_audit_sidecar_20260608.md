# Gemma Rung -1 Audit Sidecar, 2026-06-08

Scope: read-only audit of existing local docs, scripts, and result artifacts for
Gemma 3 27B, Gemma 4 12B, Gemma 4 31B, and Gemma 4 26B-A4B. No model downloads were
started. This file is the only audit artifact added by this sidecar.

Primary sources read:

- `docs/GEMMA_COMPATIBILITY_PLAN.md`
- `docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md`
- `docs/CODEX_DIRECTION_SGLANG_NVFP4_KV.md`
- `docs/GEMMA_RUNG_MINUS1_CONFIG_AUDIT.md`
- `results/gemma_rung_minus1_config_audit_20260608.json`
- `results/gemma_rung_minus1_config_audit_strict_20260608.json`
- `results/vllm_gemma3_27b_rung1_*`
- `results/vllm-gemma4-12b-unified-tfmain-cleanjit-da1daf4-20260607T152639Z_*`
- `results/vllm_gemma4_26b_a4b_*`
- `results/aeon_gemma26_dflash_20260608T0436JST_*`
- `results/flashinfer_nvfp4_kv_probe_gemma4_26b_*`

## Rung Ordering Status

Current rung ordering should be:

1. Rung 1: Gemma 3 27B, SWA/hybrid KV only, uniform `D=128`, no `D=512`.
2. Rung 2: Gemma 4 31B text-only, dense `D=512` mixed-KV isolation.
3. Rung 3: Gemma 4 26B-A4B text-only, MoE on the already-solved `D=512` base.
4. Rung 4: Gemma 4 12B, encoder-free multimodal through decoder/KV.

Reason: the strict config audit and markdown audit show `D=512` appears in Gemma 4
12B, 31B, and 26B-A4B, and 31B is dense. That preserves a clean dense-`D=512` rung
before the MoE rung. The older `results/gemma_rung_minus1_config_audit_20260608.json`
contains a stale recommended ladder putting 12B before 31B; treat the strict artifact and
`docs/GEMMA_RUNG_MINUS1_CONFIG_AUDIT.md` as the corrected rung order.

Status: Rung 1 has live vLLM geometry and capacity evidence but is red on NVFP4-KV
quality. Do not climb to Gemma 4 31B until the Gemma 3 SWA/NVFP4-KV first-token failure is
understood.

## Proven vs Config-Derived

| model | proven from running artifacts | config-derived / operator-provided only | running geometry gate |
|---|---|---|---|
| Gemma 3 27B | vLLM fp8 and NVFP4 Rung 1 logs measure 62 decoder layers, 52 local/SWA + 10 full layers, layers `5,11,17,23,29,35,41,47,53,59` full, uniform `heads=32`, `kv_heads=16`, `head_dim=128`, `head_dim_v=128`; fp8 comparator serves; NVFP4 routes through FlashInfer FA2 and gives about `1.777x` KV capacity. | Vision config / encoder quarantine is still from config and operator plan, not explicitly measured as an unfired encoder under the live rung. | Partially measured. Attention geometry and KV spec measured for vLLM. Missing explicit encoder/quarantine confirmation and passing NVFP4 quality. |
| Gemma 4 12B | vLLM exploratory row serves text with `Gemma4UnifiedForConditionalGeneration`, auto/bf16 KV, Triton attention, about `7.7 tok/s`; log proves heterogeneous `head_dim=256, global_head_dim=512` at model level and a multimodal warmup failure. | Per-layer 40 sliding `D=256` + 8 full `D=512`, full layer indices, dense status, audio/vision layout, and encoder-free multimodal-in-decoder/KV are config/operator-derived. | Not measured per rung. No per-layer running geometry, no KV page bytes by layer, no NVFP4/mixed-KV row, no runtime proof that multimodality is fused into decoder/KV. |
| Gemma 4 31B | No local serving row found. | 60 layers, 50 sliding `D=256` + 10 full `D=512`, full layers `5,11,17,23,29,35,41,47,53,59`, dense, vision config/no audio, and text-only encoder quarantine are config/operator-derived. | Missing entirely. This is the next intended dense `D=512` measurement row after Gemma 3 is green. |
| Gemma 4 26B-A4B | vLLM BF16/unquantized and AEON NVFP4-weight/DFlash rows serve locally; logs prove `Gemma4ForConditionalGeneration`, model-level heterogeneous `head_dim=256, global_head_dim=512`, Triton target attention, MoE backend selection, encoder cache initialization, and multimodal warmup completion. AEON row proves NVFP4 weights + DFlash/MoE throughput, not NVFP4 KV. FlashInfer synthetic 26B-like probes prove local `D=256` shape is viable and global `D=512` NVFP4-KV shape hits the `prefill.cuh:3215` invalid-configuration guard. | Exact per-layer 25 sliding `D=256` + 5 full `D=512`, full layers `5,11,17,23,29`, `num_experts=128`, `top_k=8`, and text-only encoder quarantine remain config/operator-derived for the rung gate. | Not measured for NVFP4-KV/mixed-KV rung. Existing serving rows are ordinary KV or NVFP4 weights; synthetic FlashInfer shape probes are not running-model geometry. |

## Exact Missing Measurement Rows

These are the missing rows needed to turn planning claims into rung evidence:

| rung | model | runtime | required missing measurement row |
|---|---|---|---|
| 1 closeout | `google/gemma-3-27b-it` | vLLM | Passing NVFP4-KV row with the already-measured 62-layer geometry plus request-tagged trace of write/read page pairing, scale views, and SWA/block lifecycle; must include output correctness/quality passing. |
| 1 quarantine | `google/gemma-3-27b-it` | vLLM | Runtime confirmation that text-only serving leaves vision/encoder path unfired or otherwise quarantined; current row measures attention geometry but not this modality boundary. |
| 1 counterpart | `google/gemma-3-27b-it` | SGLang | Not started. After Qwen FP4-KV quality is blessed, measure per-layer head_dim, Q/KV heads, SWA map/window, subpool/page layout, per-layer KV dtype/backend, bytes/token, fp8 comparator, and output quality. |
| 2 primary | `google/gemma-4-31B-it` | vLLM | First live dense `D=512` text-only mixed-KV row: measure 60-layer map, 50 local `D=256`, 10 full `D=512`, Q/KV heads by layer, selected per-layer KV dtype/backend, page layout, bytes/token, fp8/bf16 comparator, capacity, and quality. |
| 2 quarantine | `google/gemma-4-31B-it` | vLLM | Runtime proof that text-only serving quarantines vision in an unfired encoder; current claim is operator-provided/config-derived. |
| 2 counterpart | `google/gemma-4-31B-it` | SGLang | Same dense `D=512` mixed-KV measurement after SGLang Qwen and Gemma 3 gates are green; no local 31B SGLang row exists. |
| 3 primary | `google/gemma-4-26B-A4B-it` | vLLM | Text-only MoE-on-`D=512` mixed-KV row after 31B: measure 30-layer map, 25 local `D=256`, 5 full `D=512`, Q/KV heads by layer, MoE routing/backend, per-layer KV dtype/backend, page layout, bytes/token, matched comparator, capacity, and quality. |
| 3 quarantine | `google/gemma-4-26B-A4B-it` | vLLM | Runtime proof that text-only serving quarantines vision encoder; existing AEON and BF16 rows initialize/warm multimodal paths but do not prove quarantine for the rung. |
| 3 counterpart | `google/gemma-4-26B-A4B-it` | SGLang | Same MoE mixed-KV measurement after SGLang earlier gates; existing SGLang Gemma artifacts are E-variant/mobile-related, not this server model. |
| 4 primary | `google/gemma-4-12B-it` | vLLM | Final encoder-free multimodal-KV row: measure per-layer 48-layer geometry, 40 local `D=256`, 8 full `D=512`, Q/KV heads, selected mixed/NVFP4 KV dtype/backend, page layout, bytes/token, and quality with multimodal path in scope. |
| 4 architecture | `google/gemma-4-12B-it` | vLLM | Runtime proof that vision/audio are fused into decoder/KV and cannot be quarantined by text-only serving; current ordering relies on operator-provided architecture. |
| 4 counterpart | `google/gemma-4-12B-it` | SGLang | Same final multimodal-KV measurement after SGLang lower rungs; no existing SGLang 12B server-rung row found. |

## Bottom Line

Only Gemma 3 27B has measured running-model attention geometry, and that row is red for
NVFP4-KV correctness. Gemma 4 12B and 26B-A4B have useful exploratory serving evidence,
but not rung-grade running geometry. Gemma 4 31B is config-only locally. The current ladder
is still correct, but it is a plan until the missing measurement rows above are captured.
