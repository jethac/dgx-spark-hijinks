# Plan (for Codex): fix the x86 sm_120 vLLM Gemma-4 forward-numerics bug → unblock vast.ai anchor runs

**Date:** 2026-06-13 · **Owner ask:** Codex resolves the sm_120 serving-numerics bug so
that an x86 sm_120 vast.ai box (RTX 5090 / RTX PRO 6000) can produce claim-grade vLLM
anchor rows (Gemma-4 12B/31B matched bf16-vs-nvfp4 KV). Authored by Claude after
reproducing the bug on a clean vast.ai RTX PRO 6000.

This is a self-contained handoff: reproduction, the diagnostics already done, the
investigation plan, the deliverable, and the full vast.ai runbook (the box setup is
SOLVED — only the model-forward numerics remain).

---

## 0. Why this matters

The ship gate (Task #40) wants the SGLang Gemma-4 ladder to **match vLLM**. Codex's
matched SGLang 12B row (mail 0110) is **full-NVFP4 Δ=+0.403 nats/token (quality-red)**.
We need the **vLLM 12B matched anchor** to decide whether +0.403 is inherent NVFP4
sensitivity or SGLang-structural (mail 0111 triangulation). The vLLM anchor must run on
consumer Blackwell. **Spark (sm_121) serves vLLM Gemma-4 coherently; x86 sm_120 does
NOT** — that's the blocker this plan removes (and it unlocks cheap, parallel vast.ai
anchor capacity that doesn't compete with Codex's Spark window).

---

## 1. Reproduction (smoking gun, deterministic)

On a clean vast.ai **RTX PRO 6000 Blackwell (sm_120, cap (12,0)), 96 GB**, x86_64,
CUDA 13.0, driver 580.95.05, with the loose campaign wheel
`vllm-0.1.dev1+g6adc00f70.sm120a` + flashinfer `7d5d477b` (JIT via PYTHONPATH) +
torch 2.12.0+cu130, serving `google/gemma-4-12B-it`:

```
GEN("The capital of France is") -> '111.1...'        # degenerate
top-1 prefill predictions after "The quick brown fox" -> '1','1','1','-','1','.'  # garbage
wikitext-2 mean NLL = 8.0195 (PPL 3039)              # vs expected ~2-3 / PPL ~8-12
```

Scripts to reproduce (in this dir): `gen_test.py` (coherence + top-1), `debug_eval.py`
(per-token NLL stats), `eval_harness.py` (the matched PPL harness).

## 2. Diagnostics already done — what it is NOT

| ruled out | evidence |
| --- | --- |
| attention / KV path | **Triton and FlashInfer give bit-identical garbage** (NLL 8.0195 to 12 digits) → breakage is upstream of attention |
| FlashInfer JIT specifically | Triton-only path equally broken → not Task #37's flashinfer-JIT angle |
| transformers version | degenerate on **both 5.11.0 (proven) and 5.12.0** |
| harness / corpus | sanity sentence "quick brown fox" gives NLL ~25–30 per obvious token; top-1 predictions are degenerate single chars |
| OOM / load failure | model loads, engine inits, generates (just garbage) |

**Conclusion:** the bug is in the **shared model-forward path on sm_120** — the wheel's
compiled `_C` custom kernels (Gemma embedding-scale / RMSNorm / rotary / activation) or a
custom-op package (`tokenspeed-mla`, `humming-kernels`) emitting wrong results on sm_120.
Uniform degenerate logits is the classic signature of a **zeroed/unscaled embedding** or a
silently-wrong norm. The campaign's coherent serving is all sm_121 (Spark) via the baked
r10/r11 images, which never exercised this x86 sm_120 forward path end-to-end.

## 3. Investigation plan (ordered, cheap → conclusive)

1. **Custom-ops isolation (fastest, decisive).** Re-run `gen_test.py` with vLLM custom
   ops disabled so the forward uses native PyTorch (RMSNorm/rotary/act):
   `--compilation-config '{"custom_ops":["none"]}'` (or `VLLM_DISABLE_CUSTOM_OPS`/the
   wheel's equivalent). If output becomes coherent → a `_C` custom kernel is the culprit;
   bisect by re-enabling ops one at a time.
2. **Cubin arch audit.** `cuobjdump --list-elf` the wheel's `_C*.so` (and tokenspeed-mla /
   humming-kernels `.so`). Confirm real **sm_120** SASS is present (not only sm_121a, not
   only PTX needing JIT). Compare against the proven Spark `_C`. The campaign already has
   `cuobjdump_sm121.txt` receipts (r9) — produce the sm_120 analog.
3. **Embedding-scale check.** Gemma scales input embeddings by `sqrt(hidden_size)`
   (≈62 for 12B's 3840). Verify it's applied on this path (a missing/zeroed scale → exactly
   this uniform-logit symptom).
4. **Same-wheel cross-arch control.** Run `gen_test.py` with the SAME loose wheel on
   **Spark (sm_121)**. Coherent there + garbage on sm_120 ⇒ confirmed arch-specific kernel
   build bug (and tells us whether the loose wheel or the baked image is the differentiator).
5. **Build the fix → bake an image.** Per the campaign's deterministic-image pattern
   (`spark-ready-image-pattern`), produce a clean **x86 sm_120** vLLM image (analog of
   arm64 r10/r11) via CI (Ubicloud x86 → ghcr), with `_C` built for sm_120 and a warm
   FlashInfer module cache (ties to Task #35 Colab sm_120a wheels, Task #36 x86 CI,
   Task #39 AOT FlashInfer).

## 4. Deliverable (definition of done)

- A clean **x86 sm_120 vLLM image** on ghcr that passes a green sanity on an sm_120 box:
  `gen_test.py` → "The capital of France is **Paris**", wikitext-2 mean NLL **~2–3**
  (PPL ~8–12), top-1 prefill predictions sane.
- A short STOP_SUMMARY with the root cause (which kernel/scale), the cuobjdump receipts,
  and the image sha. Then this image makes the vast.ai anchor a pure `docker run`.

## 5. vast.ai runbook (SOLVED — reuse, don't rediscover)

Access: account `jethachan@gmail.com`, ~$90 credits. Use the API key **only as an
in-memory env var (`VAST_API_KEY=...`), never written to a file**. Destroy the instance
the moment a run is banked (`vastai destroy instance <id>`, pipe `y`).

Pick an on-target box (keeps sm_120 + frees Spark):
```
vastai search offers "num_gpus=1 rentable=true" -o dph_total --raw \
  | filter gpu_name contains "RTX PRO 6000" or "RTX 5090", reliability>0.97, inet_down>500, cuda>=13.0
# the one we used: RTX PRO 6000 S, 96GB, ~$1.00/hr, 8Gbps, Michigan US
vastai create instance <id> --image nvidia/cuda:13.0.1-devel-ubuntu24.04 --disk 120 --ssh --direct
vastai attach ssh <iid> "$(cat ~/.ssh/id_ed25519.pub)"
# connect on the DIRECT host:port from `vastai ssh-url <iid>` (not the ssh1.vast.ai proxy)
```

Box-setup gotchas already solved (apply all up front — "install wholesale"):
- `apt install python3.12-venv git python3.12-dev build-essential` (Python.h needed by Triton)
- torch: `pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130`
- **torchvision** (Gemma-4 unified pulls the vision processor): `pip install torchvision --index-url .../cu130`
- **ninja** (FlashInfer JIT): `pip install ninja`; put `/root/venv/bin` + `/usr/local/cuda/bin` on PATH; `CUDA_HOME=/usr/local/cuda`
- wheel install: **rename to the real PEP-427 name** before `pip install` (scp'd `vllm.whl` is rejected)
- flashinfer: clone `jethac/flashinfer`, `git checkout 7d5d477b`, submodules, and create
  `flashinfer/data/{csrc,include,cutlass,cccl,spdlog}` symlinks; use via `PYTHONPATH=/root/flashinfer`
- KV dtypes: `--kv-cache-dtype bfloat16` vs `nvfp4`. Forcing bf16 onto FlashInfer for a
  backend-matched compare: `VLLM_FLASHINFER_BF16_GEMMA=1`. **nvfp4 does NOT auto-route off
  Triton** (`kv_cache_dtype not supported`) → set the attention backend to FlashInfer
  explicitly for the nvfp4 arm.
- corpus: load wikitext-2 via `hf_hub_download("Salesforce/wikitext", ".../test-...parquet",
  repo_type="dataset")` + pyarrow — the `datasets` lib hits an `hf://` URI parse bug.
- Gemma-4 12B is **head_dim 256 throughout** → no VO-split knobs needed for the 12B anchor.

## 6. The anchor run (once the clean image exists)

`docker run` the clean x86 sm_120 image, then per arm (`bfloat16`, `nvfp4`):
`python eval_harness.py <kv> out_<kv>.json` (ntok 4096, wikitext-2, max_model_len 8192).
Report `mean_nll`, `ppl`, and **Δ = nll_nvfp4 − nll_bf16**. That Δ is the vLLM anchor for
the triangulation in mail 0111. Then 31B as the second rung (96 GB fits it).

## 7. Lane note
This is Claude's vLLM/FlashInfer lane work in principle, but it's GPU-build-heavy and Codex
owns the image/CI infra lane — hence the handoff. Claude can run the anchor itself once the
clean image lands. Marker contract applies for any Spark cross-checks.
