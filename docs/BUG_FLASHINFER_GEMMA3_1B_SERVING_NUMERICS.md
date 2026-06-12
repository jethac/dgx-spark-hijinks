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

## Bisect result 2026-06-12 (Spark / GB10, sm_121): PLATFORM / sm_120-specific

Bisect step 1 of the plan above, run on the Spark (GB10, sm_121) with the
baked r9 image `jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9`
(id 8c37bdbc4fdb), `google/gemma-3-1b-it`, C1 ctx 8191, each cell run TWICE
bitwise. Backend forced via `--attention-backend FLASH_ATTN|FLASHINFER`,
engaged backend verified from the `Using AttentionBackendEnum.<X> backend.`
proof lines (not the flag).

| row | backend (proof) | kv dtype | C1 ×2 (bitwise) | smoke |
|---|---|---|---|---|
| FLASH_ATTN bf16 (truth) | FLASH_ATTN | bf16 | 2.356493110435786 (IDENTICAL) | "Tokyo" COHERENT |
| FLASHINFER bf16 (suspect) | FLASHINFER | bf16 | 2.359283581557766 (IDENTICAL) | "Tokyo" COHERENT |
| FLASHINFER nvfp4 (+LINEAR_V_SF) | FLASHINFER | nvfp4 | 2.400746942552027 (IDENTICAL) | "Tokyo" COHERENT |

**FI-bf16 − FLASH_ATTN-bf16 on the Spark = +0.00279 nats** (P520 was +0.221).
That is well under the 0.01 threshold → FlashInfer MATCHES FLASH_ATTN at the
exact 1B geometry on sm_121. **Verdict: PLATFORM / sm_120-specific. The bug
does NOT reproduce on the Spark — the GEOMETRY hypothesis is REFUTED.** NVFP4 KV
is COHERENT and deterministic on the Spark (+0.044 vs truth) where the P520 gave
deterministic gibberish (+1.59). The Spark FLASH_ATTN truth row (2.35649) is
within 0.0014 nats of both the P520 FLASH_ATTN row (2.35785) and the HF eager
truth (2.35778), confirming the setup is sound.

Implication: the defect is scoped to sm_120 (the P520 editable / source-tree
FlashInfer build — JIT codegen or an arch-conditional path), NOT the
d256/SWA-512/1-kv-head geometry. The Spark/sm_121 Gemma 3 1B path (bf16 AND
NVFP4 KV) is fine. The Gemma 3 1B NVFP4 row banked RED on the P520 is
GREEN-class on the Spark. Bug stays OPEN but re-scoped to sm_120; investigation
steps 2-3 (P520 logit-diff probe + window/kv-head ablation) remain the way to
localize the sm_120 root cause. Artifacts:
results/claude_1b_bug_bisect_20260612/ (BISECT_SUMMARY.md, ppl JSONs ×2/row,
smoke transcripts, proof lines); Spark master copy + server logs + token dumps
at /home/jethac/spark_tmp/claude_1b_bug_bisect_20260612/.

## Cross-references

- results/claude_1b_bug_bisect_20260612/ (Spark sm_121 bisect — PLATFORM verdict)
- results/p520_gemma3_1b_serving_20260612/ (full artifacts incl. token
  dumps on the P520 side: B:\workshop\wsl_sm120\results\gemma3_1b_serving_20260612\)
- docs/RESULTS_LEDGER.md row (2026-06-12); mail/0044, mail/0056.
- docs/TRITON_RETIREMENT_SCORECARD.md adjudication log (flip scoping).
