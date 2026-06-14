# The +0.40 long-context NVFP4-KV red — reproduced on vLLM, localized to a kernel artifact

**Verdict (refined by ground-truth reference): the +0.40 is a FlashInfer SINGLE-PREFILL
online-softmax accumulation artifact, NOT the true NVFP4 cost. Exact-softmax ground truth
and vLLM's chunked/merge path agree the true long-ctx cost is ≈ +0.19. The single-prefill
kernel inflates it to +0.42; SGLang's +0.403 is the same inflation. Not SGLang-radix; the
prefix-reuse/merge path is exonerated AND is the numerically correct path.**

## Ground-truth disambiguation (the decisive arm)
Exact HF eager SDPA with nvfp4-qdq K+V (no FlashInfer kernel, no tiling, no VO split), same
12B-it model, same scored suffix [4097..8184] @ ctx 8185 (`docs/vast_anchor/refsim_longctx.py`,
log `pfx_results/refsim_longctx_ctx8185.log`):

| path | Δ (nvfp4 − bf16) |
| --- | ---: |
| **exact SDPA reference (ground truth)** | **+0.1932** |
| vLLM chunked / reuse (paged+ragged merge) | +0.1906  ← matches truth |
| vLLM single-prefill (one online-softmax pass) | +0.4215  ← inflated kernel artifact |
| SGLang extend/merge (Codex) | +0.403   ← same inflation |

Exact penalty binned by position is ~flat (+0.15–0.23/token beyond the first 1K) — it does
**NOT** grow with context. So the serving "+0.036 @4096 → +0.42 @8185 growth" is the kernel
inflating with KV-tile count, not the format. The true nvfp4 long-ctx cost is ~+0.19, flat.

## Chunk-size sweep (confirms the kernel mechanism)
Single 8185-token nvfp4 request, varying `max_num_batched_tokens` (the prefill chunk size),
same scored suffix; bf16 ≈ 8.28 flat across nbt (`docs/vast_anchor/nbt_sweep.sh`):

| max_num_batched_tokens | nvfp4 NLL | Δ vs bf16 |
| ---: | ---: | ---: |
| 4096 | 8.467 | **+0.19** (= ground truth) |
| 6144 | 8.661 | +0.385 |
| 8192 (single chunk) | 8.703 | +0.42 |

nvfp4 error rises monotonically with prefill chunk size; the smallest chunk matches the exact
reference. (nbt 2048 not run — model mm items need ≥2496 batched tokens.) → the inflation is
driven by the FlashInfer per-kernel-call online-softmax accumulation over more KV/query tiles,
not by the NVFP4 format.

## Refined conclusion & fix target
- **True NVFP4 long-ctx cost ≈ +0.19** (exact reference + chunked serving agree). It is a real
  format cost, **not** near-lossless and **not** zero — the headline must say +0.19, not "free".
- **The +0.40 is a FlashInfer single-/large-chunk-prefill accumulation artifact** (adds ~+0.23
  on top of the true +0.19). vLLM's chunked-prefill default avoids it; SGLang's extend path
  hits it (+0.403).
- **Fix target:** the single/large-prefill online-softmax accumulation in `prefill.cuh`
  (`update_mdo_states` o_frag rescaling / nvfp4 V dequant accumulation) — make a single large
  prefill pass match the chunked/exact result. Stopgap for SGLang: smaller extend chunks.
- **Goal-premise correction:** "return long-ctx nvfp4 to near-lossless" is not achievable — the
  format floor at ctx 8185 is +0.19. The achievable fix recovers the single-prefill inflation
  (+0.42 → +0.19), i.e. a kernel-correctness fix, not a format improvement.

## Kernel localization progress (open — needs a focused session)
Findings from `VLLM_SPARK_KV_TRACE` + an instrumented FlashInfer build on vast PRO6000
(artifacts `pfx_results/trace_multi_*.log`, `disp_*` rerun reproduced single 8.7031 / chunked 8.467):
1. **single = one 8185-token prefill call; chunked = 4096 + 4089 calls.** Same model/ctx, only the
   scheduler batching differs.
2. **Layer 0 is a sliding-window layer** (`window_left=1023`, window 1024) — there a query attends
   ≤1024 recent KV, identical in both paths. The gap must live in the **global (full-causal)
   layers**.
3. **The nvfp4 VO-split attention does NOT go through the stock
   `BatchPrefillWithPagedKVCacheDispatched`** — an env-gated log inserted at that dispatch (3 sites)
   never fired during the nvfp4 run. The VO-split uses a separate FA2-nvfp4 kernel entry
   (`flashinfer/prefill.py: get_batch_prefill_module(backend="fa2", ...)` / custom two-pass), which
   is where the fix must go. **Next: find and bisect that kernel.**
4. **Cross-runtime (Codex mail 0142):** SGLang `--chunked-prefill-size 2048` only moved the red
   +0.403 → +0.355 (−0.048), nowhere near vLLM-chunked's +0.19. So vLLM-chunked's recovery is
   **not merely smaller chunks** — vLLM and SGLang route chunked/extend attention differently and
   only vLLM's path recovers. The fix is not a one-line chunk-size change; it is in the
   FA2-nvfp4 accumulation / how the recent-KV block is attended.

**Status:** diagnosis (true cost +0.19; +0.40 = single/large-prefill FA2-nvfp4 artifact) is
delivered and banked; the kernel-correctness fix is localized to the FA2-nvfp4 prefill path but
not yet pinned to the exact precision/tiling site — a focused follow-up session.

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
