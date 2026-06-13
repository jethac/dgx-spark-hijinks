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
is contaminated and must be labeled non-claim-grade.** This is exactly the open caveat on
the SGLang 12B full-NVFP4 row (Codex 0105: `Δ=+0.403` nats/token is cross-artifact).

Capacity is a separate, independent claim: KV-token ratio at matched K+V byte budget
(`~1.28x` mixed denominator for SGLang today; `~1.78x`/`3.556x` full-NVFP4 only after the
structural route lands — do **not** quote the pre-fix allocator-token `1.78x`).

---

## 2. Per-rung status & gap (SGLang lane, against vLLM anchor)

| Rung | SGLang state (ledger) | Gap to claim-grade | GPU? |
|---|---|---|---|
| **12B** | full-NVFP4 GREEN as dispatcher diagnostic (mail 0105); PPL 144.74 / NLL 4.975 | **Matched bf16-vs-NVFP4 rerun** in one image/harness (current Δ is cross-artifact) | yes |
| **26B-A4B (E4B)** | Rung-0 RED: D=512 decode hits `Unsupported max_mma_kv: 0`; VO-split decode routing staged (`SGLANG_FLASHINFER_VOSPLIT=1`, `jethac/sglang@9d78a007f`), static-checked only | **Rerun with VO-split decode** → first coherent serving row; then matched mixed-KV delta | yes |
| **31B** | Writer-roundtrip GREEN at D=512 VO-split (`8d85fff9`); real-geometry probes GREEN; no serving row | **First serving bring-up** + matched delta; watch the 31B coherence open bug | yes |

**vLLM anchor caveat (confirm before quoting "matches vLLM"):** the vLLM lane has
Gemma 3 + Qwen matched-PPL claim rows and the DG-V DiffusionGemma green, **but no banked
claim-grade vLLM Gemma 4 *text* AR ladder row** (only image builds, probes, the AEON-26B
Triton smoke, and the YAK capacity/throughput notebook). So the gate's reference rows on
the vLLM side may need to be produced too — the smallest tractable anchor is **vLLM
Gemma 4 12B matched bf16-vs-NVFP4** (my lane, P520/Spark). Flagging rather than asserting.

---

## 3. Critical path (priority order, lane + GPU mapped)

1. **E4B rung-0 VO-split decode rerun** *(Codex / SGLang GPU window)* — graduate the staged
   `SGLANG_FLASHINFER_VOSPLIT=1` route from scaffold to green. This is the immediate
   blocker and the kernel+routing are already staged; it needs only the GPU window.
   Success = D=512 global decode no longer hits `max_mma_kv: 0`; coherent generation.
2. **vLLM Gemma 4 12B matched bf16-vs-NVFP4** *(Claude / vLLM lane, P520 or Spark)* —
   establish the ground-truth anchor the SGLang 12B must match. Smallest tractable rung.
3. **SGLang 12B matched rerun** *(Codex)* — paired bf16-vs-NVFP4 in one image → claim-grade.
4. **31B serving bring-up** *(both lanes)* — gated on the 31B coherence bug being understood.

Items 1 and 2 are independent → can run in parallel across the two GPUs (coordinate via
the `agent_bus/p520_gpu_queue.md` marker; Spark is bandwidth-bound, P520 is the sm_120
kernel bench).

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
