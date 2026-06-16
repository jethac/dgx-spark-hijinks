# PR-readiness report — FlashInfer + vLLM NVFP4-KV branches (2026-06-17)

Goal: clean PR branches for the NVFP4 KV-cache-on-consumer-Blackwell work. **No PRs submitted.**
Approach chosen by Jetha: **new clean `*-pr` branches**; the working `spark/hijinks-e3-*` branches are
untouched (Codex's lane refs + the built wheels stay intact).

---

## 1. FlashInfer — `spark/hijinks-e3-flashinfer-pr`  ✅ READY
Pushed to `origin` (github.com/jethac/flashinfer). Base = upstream merge-base `c15ac84c922f`.
**17 commits, 23 files, +699/-137.** Built by cherry-picking the genuine kernel commits and **dropping**:
- the two debug-logging commits (`Add paged prefill JIT debug logging`, `Bind prefill debug tensor logs`)
- the `data/` JIT-source-symlink commit (clone convenience, not upstream)

Verified: **0 debug residue** (`SparkPrint*`/`FLASHINFER_PREFILL_DEBUG` gone), 0 data symlinks, all changed
`.py` parse. Cherry-picks applied with **no conflicts**. Content: FA2 NVFP4-KV paged prefill (E2M1 casts,
scale-stride plumbing, output-sized-from-V, JIT module disambiguation), SM121 FP4 dispatch + regression
test, the `max_mma_kv=0` smem-aware dispatcher fix, the validated fp8+NVFP4 mixed-KV `plan()` pair, and the
clean real-smem rejection for infeasible KV tiles.

**Remaining before submission (optional polish):** squash the `plan()` add-kwargs → use → remove-descoped
trio (`30b58e5b`+`4fceed4d`+`609c8858`) into one commit; confirm the SM121 CI-arch-list + installation.rst
note belong in the same PR or a separate infra PR. Otherwise submittable as-is once rebased onto live
upstream/main (small surface, low conflict risk).

---

## 2. vLLM — `spark/hijinks-e3-vllm-pr`  ⚠️ STRONG START, needs 3 manual steps
Pushed to `origin` (github.com/jethac/vllm). Base = our dev merge-base `967c5c3bc388`.
**29 commits, 80 files, +8463/-665** — the genuine NVFP4-KV feature. Built by cherry-picking our source
commits and **dropping**: 5 upstream ROCm/CPU/Docker bugfixes (not ours), all CI wheel-workflow commits,
all Spark image-build commits, and the `Add Gemma tensor trace hooks`/`Disable…` churn pair. Cherry-picks
applied with **no conflicts**. No `.github`/notebook/`results/`/`.md` leakage.

### Why it is not yet submittable (3 steps, each needs a human + a build env to verify)
1. **Strip `spark_tensor_trace` debug instrumentation.** Rode in via the foundation squash. Clean and
   mechanically strippable in `gemma3.py` (10 isolated `if should_emit(): trace(...)` blocks) but
   **interleaved in `flashinfer.py`** (29 `_spark*` helpers + 16 call sites, some the *sole body* of
   non-trace `if` blocks → naive removal leaves dangling empty blocks). Needs `pass`-insertion judgment +
   a build to verify. Files: `vllm/utils/spark_tensor_trace.py` (delete), `gemma3.py`, `flashinfer.py`.
2. **Split the foundation commit** `spark-hijinks epoch-2 patch set …` (≈2801 lines: `flashinfer.py`
   +1609, the CUDA `nvfp4_kv_cache_kernels.cu`, `attention.py` integration, `envs.py`, model changes,
   `kv_cache_utils`, the routing test) into reviewable logical commits (kernels / FI backend / attention
   integration / calibration / model wiring). Also fold the **amendment churn** (BF16_GEMMA Amendment 3/4/5
   + the Triton-retirement re-flips that cancel) into final-state commits.
3. **Rebase onto LIVE upstream/main.** This branch sits on a **stale** dev-base; our `upstream/main` ref
   (`c621af169`) is itself behind. Critically the branch *depends on* upstream's own DiffusionGemma PR
   `7caa13add (#45163)` as a base — once rebased onto current upstream that PR is already there and our
   DG pieces (`build_attn_metadata(causal)`, DG-2 FI routing) reduce to a small delta. Expect real
   conflicts from upstream drift; resolve against the validated tree in `spark/hijinks-e3-vllm`.

### Genuine, self-contained source fixes in this branch (good first small PRs on their own)
- `build_attn_metadata: accept per-request causal flag` — enables DiffusionGemma block-diffusion (backward-compat).
- `gemma4_mm` + `gemma4_unified: stack variable-length audio clips` — fixes multi-clip audio crash.
- `flash_attn: guard mm_prefix_range_tensor with getattr` — fixes text-on-FA-backend crash.
- The 3 mixed-dtype KV plumbing fixes (`per-group cache_dtype_str`, `padded VO-split`, `calculate_kv_scales`).
- `Support layer-aware NVFP4 KV calibration`.

---

## Status
- Working branches `spark/hijinks-e3-{vllm,flashinfer}` UNCHANGED (Codex refs + wheels safe).
- `*-pr` branches pushed to origin for review. **No PRs opened.**
- FlashInfer is review-ready; vLLM is a strong curated base needing trace-strip + foundation-split +
  live-upstream rebase (all build-gated) before it is a clean upstream PR.
