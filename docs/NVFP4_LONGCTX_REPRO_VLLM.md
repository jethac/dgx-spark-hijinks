# The +0.40 long-context NVFP4-KV red — reproduced on vLLM, localized to a kernel artifact

**Verdict (refined by ground-truth reference): the +0.40 is a FlashInfer SINGLE-PREFILL
online-softmax accumulation artifact, NOT the true NVFP4 cost. Exact-softmax ground truth
and vLLM's chunked/merge path agree the true long-ctx cost is ≈ +0.19. The single-prefill
kernel inflates it to +0.42; SGLang's +0.403 is the same inflation. Not SGLang-radix; the
prefix-reuse/merge path is exonerated AND is the numerically correct path.**

## ⚠️ Cross-stack reference fragility (2026-06-14, Spark/GB10) — read this before quoting +0.19
The exact-reference delta is **NOT hardware/stack-invariant**, which weakens any single-number
"true cost" claim. Same `refsim_longctx.py`, suffix-matched 12B-it ctx 8185:
- **vast (sm_120, Torch 2.12):** +0.1932 — and it *agrees with* vast vLLM-chunked serving (+0.1906).
- **Spark (sm_121/GB10, Torch 2.11):** +0.6949 (Codex 0145, reproduced).

Ruled out as the cause (`docs/vast_anchor/refsim_disc.py` on the Spark): the fp8-e4m3 block-scale
conversion is **bit-identical** to a manual e4m3 table (selfcheck maxabs 0); **tf32-off is
unchanged** (+0.6949); forcing the **MATH SDPA backend** moves it only +0.69→+0.60. So it is a
deeper Torch-2.11/GB10 numerical difference, plausibly the recursive q/dq path amplifying
stack-level noise across 48 layers (bf16 baseline matches to ~0.02; only the q/dq side diverges).

**Interpretation:** the Spark refsim (+0.60–0.69) is *worse than SGLang serving on the same box*
(+0.40), which is impossible for a true exact reference — so the **Spark refsim is pathological**
on that stack, not a revised cost. **+0.19 remains the best estimate** because on the vast stack the
exact reference and the chunked serving path independently agree. BUT: do not present +0.19 as a
hardware-invariant "true cost." The defensible claim is the *within-stack* one (below): on a given
stack, single-prefill inflates nvfp4 error vs chunked. The absolute long-ctx nvfp4 cost is
stack-sensitive and the refsim is fragile on some stacks — quote it with the stack, and treat
SGLang's +0.40 on GB10 as possibly closer to that box's real cost than +0.19.

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

### Source map of the nvfp4 prefill paths (next-session starting point)
vLLM's FlashInfer backend has MULTIPLE prefill paths; my +0.42/+0.19 anchor's exact path is not
yet runtime-confirmed (the instrumented stock `BatchPrefillWithPagedKVCacheDispatched` never fired,
so it's not that one). Candidates in
`vllm/v1/attention/backends/flashinfer.py`:
- `BatchDCPPrefillWrapper` (L1031): a **cascade** — `_context` = `BatchPrefillWithPagedKVCacheWrapper`
  (nvfp4 paged), `_new_tokens` = `BatchPrefillWithRaggedKVCacheWrapper` run on **fresh bf16
  key/value**, merged by `merge_attn_states` (LSE). Recent KV is bf16 here. BUT this requires a DCP
  group (`get_dcp_group()`), so single-GPU anchor runs likely do NOT use it.
- `use_cascade = common_prefix_len > 0` (L2237) — a separate cascade mode for shared prefixes.
- The FA2-nvfp4 custom entry via `get_batch_prefill_module(backend="fa2", ...)` + the
  `_fa2_nvfp4_prefill_jit_args` (L825) two-pass VO split — the prime suspect for the monolithic
  single path.
- `TRTLLMPrefill` (L1417) / `trtllm_batch_context_with_kv_cache`.

**Decisive next step (no kernel rebuild):** add Python-level prints in each `*.run()` path in the
backend, run single (nbt 8192) vs chunked (nbt 4096) nvfp4, and log which path + which kernel module
each takes. That pins whether the +0.19-vs-+0.42 split is (a) different paths (cascade/bf16-recent
vs monolithic-nvfp4) or (b) same path different tiling — then target that kernel. Cross-runtime: if
vLLM-chunked routes recent KV through bf16 (cascade) while SGLang reads it nvfp4, that alone
explains Codex's 0142 (SGLang chunk-2048 stays at +0.355).

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

## Localization update (2026-06-14, vast PRO6000, wget-wheel box) — split_kv ruled out
`fipath_sitecustomize.py` path logging on single (nbt8192) vs chunked (nbt4096) nvfp4:
- single: ONE `BatchPrefillWithPagedKVCacheWrapper` call, qo=8185 → NLL 8.7031 (+0.42)
- chunked: TWO paged calls, qo=4096 + 4089 → NLL 8.467 (+0.19)
- **Both pure PAGED nvfp4 — no bf16/RAGGED cascade.** (Refutes the "chunked attends recent KV in
  bf16" idea.)

`split_kv` hypothesis TESTED and RULED OUT: the nvfp4 path's `disable_split_kv` is gated on
`VLLM_BATCH_INVARIANT` (unset in my runs → `disable_split_kv=False`, binary-search). Forcing
`prefill_fixed_split_size=4096` on the single prefill (so it splits KV + fp32 partial-state merges)
left it at **8.7031, unchanged** — a KV-split merge is numerically equivalent to the single pass, so
the fp32 merge is NOT the corrector.

**Refined mechanism:** the inflation tracks the **query-batch size in one paged-prefill kernel call**
(8185 queries with long KV → +0.42; splitting the *query* dimension into separate prefill calls →
+0.19; nbt sweep monotonic 4096→+0.19, 6144→+0.385, 8192→+0.42). So it's the kernel's
online-softmax / `o_frag` accumulation across the large query-tile × long-KV grid for nvfp4 — not
split_kv, not a cascade. **Practical mitigation: chunked prefill (nbt ≤ 4096), which is vLLM's
default for long sequences → +0.19.** The exact one-line kernel fix needs per-tile accumulation
instrumentation (next).

## Parallel kernel-fix session (2026-06-14) — it's a structural FA2-nvfp4 BUG, not precision
Per-position NLL (single vs chunked): inflation is **uniform** across all scored positions
(+0.12..+0.35), not concentrated at the chunk boundary. So every query inflates systematically in
the 8185-query call vs the 4089-query call — a **batch-size-dependent** difference. +0.23 nats is
**far too large for reduction-order rounding** (~1e-5), so the single large-prefill is computing
nvfp4 attention *incorrectly* above some query-count/KV threshold; chunked (= exact reference) is
correct. This is a genuine FA2-nvfp4 kernel bug.

Ruled out (all tested on real serving):
- bf16 cascade — fipath: single + chunked are BOTH pure `BatchPrefillWithPagedKVCacheWrapper`.
- split_kv / fp32 partial-state merge — forcing `prefill_fixed_split_size=4096` on single: 8.7031,
  unchanged.
- `VLLM_BATCH_INVARIANT` — selector rejects ("FLASHINFER: batch invariance not supported").
- per-call quantization — the paged nvfp4 KV is nbt-independent (constant global scale + per-16
  block SF; same bytes regardless of batch).
- per-element V precision — fp32 V scale-factor application (`__hmul2`→fp32, 8 sites): 8.7031,
  unchanged.

**Open (needs a dedicated deep session): the exact bug site** — a batch-size-dependent
miscomputation in the FA2-nvfp4 large-prefill (suspects: the VO-split two-pass merge, or a
smem/register reuse across query tiles that leaks for nvfp4 above a threshold). Pinning it needs
per-tile attention-output dumps (single vs chunked for the same query) — not random patches.

**Practical mitigation (shippable, verified): chunked prefill (`max_num_batched_tokens ≤ 4096`) →
+0.19**, which is vLLM's default for long sequences. The +0.42 only occurs when chunking is
disabled (nbt ≥ seqlen). For SGLang (which can't recover via vLLM-chunked), the fix waits on the
kernel bug.

**Infra win this session:** the wheel is a public GitHub release (`sm120a-wheels-6adc00f70`); boxes
`wget` it at 3 Gbps instead of scp through the throttled vast proxy (`docs/vast_anchor/e2e_setup_wget.sh`).
