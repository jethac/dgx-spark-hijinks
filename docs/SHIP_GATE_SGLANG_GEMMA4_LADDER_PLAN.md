# Ship Gate: SGLang Gemma 4 AR ladder (12B / 26B-A4B / 31B) — execute-only plan

**Date:** 2026-06-13 · **Gate:** Task #40 — bring the SGLang Gemma 4 text AR ladder to
claim-grade, matching the vLLM lane. This doc is the **claim methodology + per-rung
run/claim spec + critical path**, so the next GPU windows are execute-only. It
complements `docs/SGLANG_GEMMA4_RUNG_PREP.md` (the code-surface implementation map);
this doc is the *run matrix and claim criteria*.

Authored from the vLLM lane (the ground-truth side of the gate) + the FlashInfer
dispatcher root-cause just landed (`docs/FI_DISPATCHER_UPSTREAM_REPRO_VERDICT.md`).

---

## 1. What makes a row claim-grade (the matched-delta rule)

Absolute PPL on our deterministic repo-markdown corpus is **not** a claim. The claim is
the **matched bf16-vs-NVFP4 (or fp8) delta**, where *both* arms differ in exactly one
variable — the KV dtype — and share everything else:

- same **image/commit** (flip only `--kv-cache-dtype` / the KV-quant flag),
- same **corpus slice** (identical bytes, identical tokenization),
- same **shape**: ctx, reused prefix, page size, batch, window, softcap, `o_dtype`,
- same **graph state** (graphs off for these rungs),
- same **backend/wrapper plan** (FlashInfer FA2; VO-split for D=512 globals).

Report: `mean NLL_bf16`, `mean NLL_nvfp4`, `Δ = NLL_nvfp4 − NLL_bf16` (nats/token), and
the per-position sweep. **A cross-artifact delta (bf16 from image A, NVFP4 from image B)
is contaminated and must be labeled non-claim-grade.** The SGLang 12B row is now a matched
red (`+0.402969` nats/token, `results/sglang_gemma4_12b_ar_matched_bf16_fullnvfp4_ctx8185_prefix4096_20260613T153712JST/STOP_SUMMARY.md`).
Claude's ground-truth discriminator refined the diagnosis in mail 0140: exact SDPA and vLLM
chunked/reuse put the true NVFP4 cost near `+0.19`, while large single-prefill FlashInfer
inflates to `+0.40`.

Capacity is a separate, independent claim: KV-token ratio at matched K+V byte budget. Full
NVFP4 K+V reports the expected `~3.556x` raw cache-token denominator versus bf16 on the
short E4B checkpoints; long-context quality is the open blocker. Mixed-KV remains a separate
fallback path and must not be conflated with full-NVFP4 rows.

---

## 2. Per-rung status & gap (SGLang lane, against vLLM anchor)

| Rung | SGLang state (ledger) | Gap to claim-grade | GPU? |
|---|---|---|---|
| **12B** | matched bf16-vs-full-NVFP4 row is RED by `+0.402969` nats/token at ctx 8185 / prefix 4096; multimodal short smoke is scoped green | Wait for Claude's FlashInfer large-prefill accumulation fix, then rerun the matched row; expected corrected delta is near `+0.19`. Do not chase SGLang radix/merge or global-scale calibration for this red; mail 0140 exonerates them | yes, after fix |
| **26B-A4B** | not yet claim-grade in SGLang AR ladder; same D=512 global VO-split path as E4B/31B plus MoE | run only after the long-context quality fix and current package image are ready, unless doing an explicitly scoped bring-up diagnostic | yes |
| **31B** | no SGLang serving row banked; D=512 VO-split scaffolding/probes exist, but serving must be proven with the packaged SGLang path | first SGLang serving bring-up + matched delta after the shared quality/dispatcher blockers are resolved | yes |
| **E4B scoped checkpoint** | bf16 and full-NVFP4 short rows are green; baked mm-prefix image row is green; fp8 comparator is red in FlashInfer dispatcher | hold fp8 comparator until D512/VO256 1-byte-KV dispatcher fix lands | yes, after fix |

**vLLM anchor caveat (confirm before quoting "matches vLLM"):** the vLLM lane has
Gemma 3 + Qwen matched-PPL claim rows and the DG-V DiffusionGemma green, **but no banked
claim-grade vLLM Gemma 4 *text* AR ladder row** (only image builds, probes, the AEON-26B
Triton smoke, and the YAK capacity/throughput notebook). So the gate's reference rows on
the vLLM side may need to be produced too — the smallest tractable anchor is **vLLM
Gemma 4 12B matched bf16-vs-NVFP4** (my lane, P520/Spark). Flagging rather than asserting.

---

## 3. Critical path (priority order, lane + GPU mapped)

1. **Claude FlashInfer large-prefill accumulation fix** — mail 0140 proves the
   12B `+0.40` class is a single-/large-prefill kernel artifact. The true NVFP4
   long-context cost is about `+0.19` by exact SDPA and vLLM chunked/reuse.
2. **SGLang 12B matched rerun** *(Codex)* — same ctx 8185 / prefix 4096 shape, same image
   and corpus, flip only KV dtype. This is the gate that turns the current scoped red into a
   claim-grade pass or a new blocker. A deliberately scoped chunked/merge
   diagnostic may run earlier if it directly checks whether SGLang can avoid the
   large-prefill artifact and recover the `+0.19` reference path.
3. **FlashInfer dispatcher fix for E4B fp8 D512/VO256 1-byte KV** *(Claude/FI)* — then rerun
   the SGLang E4B fp8 comparator so the comparison matrix is complete.
4. **SGLang 26B-A4B and 31B serving bring-up / matched rows** *(Codex)* — after the shared
   quality/dispatcher blockers clear, using the current packaged-image path.

Do not spend Spark windows repeating known reds unless a dependency changed or the row is
explicitly a scoped diagnostic.

---

## 4. Diagnostic aid now available

The FlashInfer dispatcher fix (3 sites, `docs/flashinfer_pr/`) converts both
`Unsupported max_mma_kv: 0` **and** `NUM_MMA_KV=1 … please file an issue` into a clear
*"insufficient shared memory … head_dim_qk=… head_dim_vo=… needs NUM_MMA_KV>=N, only M
fits"* message. **Recommended:** carry the 3-line guard onto the SGLang/campaign
FlashInfer branch (`jethac/flashinfer@8d85fff9`) so any VO-split routing misfire in the
E4B rerun reads cleanly instead of opaquely. (Claude's parked FlashInfer task; does not
touch SGLang serving code.) It does **not** enable any config — enablement is the VO-split
route already staged.

---

## 5. Banking template (per rung)

```
results/sglang_gemma4_<rung>_<dtype>_matched_<ctx>_<prefix>_<ts>/STOP_SUMMARY.md
  - provenance: image sha256, sglang commit, flashinfer commit, FlashInfer src paths
  - geometry dump from the RUNNING model: per-layer id/type/heads/kv_heads/head_dim/
    v_head_dim/window/page/bytes-per-token (config JSON is a hint, not truth)
  - plan audit: wrapper id, qo_len, kv_len, batch, page, window, softcap,
    head_dim_qk, head_dim_vo, k_data_type, v_data_type, SF layout
  - wrapper-ctor audit: prove no jit_args/cached-module/self.head_dim pins symmetric dims
  - readiness + chat smoke (token-in/token-out)
  - MATCHED arms: NLL_bf16, NLL_nvfp4 (same corpus/shape/image), Δ nats/token + sweep
  - capacity (if claimed): KV tokens at matched K+V byte budget, denominator stated
  - RED policy: verbatim error + the failing plan signature
```

---

## 6. Open decisions (Jetha)
- File the upstream FlashInfer diagnostic PR now, or bundle with the campaign carrot?
- Should Claude produce the vLLM Gemma 4 12B anchor row (item 2) on the next P520/Spark
  window, or is there already a claim-grade vLLM Gemma 4 ladder row I should cite instead?
