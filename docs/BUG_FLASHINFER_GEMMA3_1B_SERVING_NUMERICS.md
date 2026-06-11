# BUG: FlashInfer serving-path numerics wrong at Gemma 3 1B geometry (d256 / SWA-512 / 1 KV head)

Status: OPEN, root cause unknown. Observed 2026-06-12 on sm_120; platform
attribution PENDING (1B has never been served on any other platform - "sm_120
bug" is provisional naming).
Found by: P520 Gemma 3 1B serving verification (zero-bug diagnostics).
Severity: FI-bf16 quality silently wrong (+0.22 to +1.38 nats); FI-nvfp4
unusable (deterministic gibberish). Coherent short-prompt chat MASKS the bug.

## Environment

- P520: RTX 5060 Ti, CC 12.0 (sm_120), WSL2 Ubuntu 24.04, CUDA 13.0 (nvcc
  V13.0.88), torch 2.12.0+cu130.
- vLLM: jethac/vllm spark/hijinks-022-gemma4-mixed-kv @ 9759e3b06 (exact
  r9-image code), editable build, TORCH_CUDA_ARCH_LIST=12.0a, 42 sm_120a
  cubins confirmed, NVFP4 linear-latch diag PASS (head 128 and 256).
- FlashInfer: jethac/flashinfer spark/hijinks-022-fa2-d512 @ 7d5d477b,
  JIT-compiled on box (NOT the Spark AOT path).
- Model: google/gemma-3-1b-it - uniform head_dim 256, GQA 4q/1kv (the only
  Gemma with a single KV head), sliding window 512 (smaller than the 1024 of
  the larger Gemma 3 sizes), 5:1 sliding:global.

## Evidence (results/p520_gemma3_1b_serving_20260612/, ctx 8191, 8190 scored)

Truth references, agreeing to <0.001 nats on all corpora:
- HF transformers eager bf16: C1/C2/C3 = 2.35778 / 3.21392 / 1.42429
- vLLM FLASH_ATTN-backend bf16 serving row: matches HF on all three.

Deviations from truth (nats, C1/C2/C3):
| FI row | delta vs truth | chat smoke |
|---|---|---|
| bf16 | +0.221 / +1.243 / +1.380 | coherent ("Tokyo") |
| fp8_e4m3 | +0.006 / +0.159 / +0.494 | coherent |
| nvfp4 (+linear V-SF) | +1.592 / +2.436 / +2.752 | GIBBERISH, deterministic |

All FI rows internally deterministic. nvfp4 gibberish reproduced
byte-identical on a VIRGIN FlashInfer JIT cache (not stale kernels); writer
latch clean, so suspicion is on the read path. Pre-diagnostic JIT cache
preserved at WSL ~/.cache/flashinfer_prediag_070355 for forensics.

## Diagnostic structure (three tells)

1. Short-prompt chat coherent while long-ctx PPL inflated: chat prompts stay
   inside the 512-token sliding window; ctx-8191 scoring crosses it
   constantly. Points at sliding-window boundary handling in the FI serving
   path (paged KV at depth), which single-call kernel probes never
   reproduced (6/6 probes passed at 31B/E4B geometries).
2. fp8 closer to truth than bf16 on the SAME backend: dtype-conditional
   kernel templates / tile dispatch differ, so the defect is likely in a
   path-conditional spot, not shared mask math.
3. Novel geometry axes vs everything previously tested: window 512 (not
   1024) AND kv_heads=1. Either could be the unexercised path.

## What is and is not contaminated

- NOT contaminated: all sm_121 (GB10) results - Triton-vs-FlashInfer pairs
  within 0.04 nats across 5 sizes (12B-31B geometries) corroborate each
  other; G3-12B/27B FI rows checked against FLASH_ATTN/Triton baselines.
- Contaminated/blocked: any Gemma 3 1B FI claim; Gemma 3 bf16 retirement
  flip (scoped back to Gemma 4-only, see TRITON_RETIREMENT_SCORECARD
  adjudication log); Gemma 3 1B nvfp4 support claim (row banked RED).

## Bisect + investigation plan

1. Gemma 3 1B rerun on sm_121 Spark (cheap; next available window): same
   deviation -> geometry bug (window-512 / 1-kv-head path, platform-
   independent, simply never reached); clean -> sm_120-specific (JIT codegen
   or arch-conditional path). 
2. Logit-level diff probe FI vs FLASH_ATTN at exact 1B geometry on P520:
   find the first divergent layer/position; check position vs window
   boundary (expect divergence onset near token 512 if tell #1 is right).
3. Geometry ablation in the probe harness: window 512 vs 1024, kv_heads 1
   vs 2, d256 fixed - isolate the axis.
4. Once root-caused: fix on jethac/flashinfer branch + upstream filing with
   minimal repro (this doc is the filing draft skeleton).

## Cross-references

- results/p520_gemma3_1b_serving_20260612/ (full artifacts incl. token
  dumps on the P520 side: B:\workshop\wsl_sm120\results\gemma3_1b_serving_20260612\)
- docs/RESULTS_LEDGER.md row (2026-06-12); mail/0044.
- docs/TRITON_RETIREMENT_SCORECARD.md adjudication log (flip scoping).
