# The +0.40 long-context NVFP4-KV red — reproduced on vLLM, localized

**Verdict: the +0.40 reproduces on vLLM. It is a GENERAL NVFP4 long-context attention
issue, NOT SGLang-radix/partial-state-merge-specific. The prefix-reuse / merge path is
exonerated.**

## Setup
- Box: vast.ai RTX PRO 6000 S (sm_120, 96 GB), wheel `vllm 0.1.dev1+g6adc00f70.sm120a`,
  torch 2.12.0+cu130, FlashInfer `jethac@7d5d477b`.
- Model `google/gemma-4-12b-it`, corpus wikitext-2 test (raw), `--enforce-eager`.
- nvfp4 env: `VLLM_NVFP4_KV_VOSPLIT=1 VLLM_NVFP4_KV_LINEAR_V_SF=1 VLLM_FLASHINFER_MM_PREFIX=1`
  (12B has 512-wide global heads; FlashInfer rejects head_size>256 on CC 12.x without the
  two-pass VO split, and the split needs linear — not swizzled — V scale factors).
- Harness `docs/vast_anchor/vllm_matched_kv_anchor.py` + `pfx_reuse_matrix.sh`. Supplied-token
  NLL over the IDENTICAL scored suffix (positions 4097..8184, 4088 tokens — matches the SGLang
  row) under three attention paths; only the path differs, bf16 vs nvfp4 each.
- Raw artifacts: `docs/vast_anchor/pfx_results/*.json`.

## The matrix (ctx 8185, prefix 4096, scored 4088 tokens)

| path | what it exercises | bf16 NLL | nvfp4 NLL | Δ (nvfp4−bf16) |
| --- | --- | ---: | ---: | ---: |
| **single**  | one prefill over full 8185 ctx, NO merge (`max_nbt=8192`) | 8.2816 | 8.7031 | **+0.4215** |
| **chunked** | in-pass chunked prefill, paged+ragged merge (`max_nbt=4096`, no reuse) | 8.2764 | 8.4670 | **+0.1906** |
| **reuse**   | warm 4096 prefix → score (cross-request radix / partial-state merge) | 8.2764 | 8.4670 | **+0.1906** |

## Context-length control (single-chunk path, same model + VO-split)

| ctx | bf16 NLL | nvfp4 NLL | Δ |
| ---: | ---: | ---: | ---: |
| 4096 | 8.4143 | 8.4502 | **+0.0359** |
| 8185 | 8.2816 | 8.7031 | **+0.4215** |

## Findings
1. **The +0.40 reproduces on vLLM** (single-chunk nvfp4 long-ctx, +0.4215 — magnitude-matches
   Codex's SGLang +0.403). It is therefore **NOT SGLang-radix-specific**; it is a general NVFP4
   long-context attention phenomenon shared by both runtimes.
2. **The prefix-reuse / partial-state-merge / radix path is EXONERATED.** It is the *better*
   path here (+0.1906, vs +0.4215 single). `chunked == reuse` to four decimals (8.4670)
   confirms vLLM's chunked-prefill and prefix-cache-reuse take the identical paged+ragged
   merge path, and that path *reduces* the error rather than causing it. The original
   hypothesis (merge causes the red) is inverted by the data.
3. **It scales with context length:** Δ grows +0.0359 (ctx 4096) → +0.4215 (ctx 8185) on the
   same model, same VO-split, same single-chunk path. This is long-context accumulation of
   NVFP4 dequant noise in the attention weighted-sum, not a fixed cost.
4. **VO-split is NOT a fixed tax** — near-lossless at ctx 4096 (+0.0359) with the same
   `VOSPLIT=1 LINEAR_V_SF=1` knobs. So the red is not the two-pass split per se.
5. **Mechanistic lead:** single-prefill (one running online-softmax/LSE accumulation over all
   8185 KV) is worse than chunked (partial-state renormalization between chunks). The error is
   concentrated in long single-pass accumulation; chunking roughly halves it. Suspect: NVFP4
   dequant-noise accumulation in the unnormalized attention sum / LSE at long context, in the
   FlashInfer prefill — common to both runtimes.

## Implication for the lanes
- **Codex (SGLang):** do NOT rewrite the radix / partial-state-merge path — it is not the cause.
  The fix lives in the NVFP4 long-context attention numerics (FlashInfer / quant), Claude's lane.
- **Claude (vLLM/FlashInfer):** localize the long-context accumulation (single vs chunked kernel
  numerics; LSE/online-softmax in NVFP4 dequant) and prototype a fix. The chunked-is-better
  result is the lever — understand why the merge renormalization helps and whether the single
  path can adopt it.

## Process note (the 0130 lesson, applied)
Every number here is the real vLLM serving path (not a reference sim). The context-length
control was added specifically to rule out a VO-split fixed tax / -it artifact before declaring
the verdict.
