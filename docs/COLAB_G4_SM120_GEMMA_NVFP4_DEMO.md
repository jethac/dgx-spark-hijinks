# Colab G4 (RTX PRO 6000 / sm_120) — Gemma NVFP4-KV Before/After Demo

A Gemma-DevRel-quality Colab notebook that (1) shows NVFP4 KV cache is **unavailable for the
whole Gemma family on consumer Blackwell** today, then (2) shows our patch making it work —
on hardware any Gemma developer can rent (Colab Pro "G4" = RTX PRO 6000 Blackwell, 96 GB,
`sm_120`).

> Context: parting deliverable, author leaves the Gemma team 2026-06-25. The "before" half
> is valid and shippable NOW; the "after" half drops in once the FlashInfer SWA-prefill fix
> lands and passes sm_120 validation. If the fix slips past 6/25, before + root-cause +
> linked in-flight PR is still a strong standalone artifact.

## Why Colab G4
- It's `sm_120` (RTX PRO 6000), the **sibling** of the DGX Spark's `sm_121` — same consumer
  Blackwell family, the audience that actually buys these cards (vllm #31085, llama.cpp
  #18250 are SM120 demand).
- It closes the one gap the campaign couldn't: SM120 was **compiled-but-unclaimed** because
  we had no RTX PRO 6000. Colab makes it **validated-on-sm_120** for ~$10-50/mo.
- Honest scope to state in the notebook: sm_120 (this G4) and sm_121 (DGX Spark) are
  validated **separately** — non-portable `a` cubins. The G4 has **96 GB discrete** VRAM
  (not the Spark's 128 GB unified), so this demo is for **arch + correctness**, while the
  Spark remains the **capacity hero**. A GPU-OOM here just kills the process (no unified-mem
  deadlock — that's Spark-specific, see `docs/INCIDENT_20260609_OOM_DEADLOCK.md`).

## One arch-agnostic notebook (runs on BOTH Colab G4 `sm_120` and DGX Spark `sm_121`)
Ship a **single** `.ipynb`, not two. It detects the GPU and adapts — same before/after,
arch-labeled output. This is the stronger deliverable: "run it on whatever consumer
Blackwell you have." Design rules:
- **Detect + assert CC 12.x.** `torch.cuda.get_device_capability()` → `(12,0)` (RTX PRO 6000
  / G4) or `(12,1)` (GB10 / Spark). Assert consumer Blackwell; print the arch. The
  FlashInfer probe JIT-compiles for `120a`/`121a` automatically — the *same probe is the
  dual-arch validator*.
- **Host-arch-aware install (the main complexity).** Colab is **x86_64**, the Spark is
  **aarch64** — wheels differ. Branch the install cell: on x86 pip-install the pinned
  stock/fork wheels; on aarch64 prefer the **already-built forks** in the Spark env (or the
  proven container's Python) rather than building from scratch. The GPU kernels are
  arch-targeted and JIT, so kernel behavior transfers; only the package install branches.
- **Memory-model-aware (CRITICAL on the Spark).** G4 = 96 GB **discrete** (a GPU-OOM just
  kills the process). Spark = 128 GB **unified** — apply the incident guardrails so the demo
  does NOT re-trigger the OOM-deadlock: cap `gpu_memory_utilization` (≤0.7), **one model at
  a time, never concurrent fp8+nvfp4**, leave ≥15-20 GiB OS headroom
  (`docs/INCIDENT_20260609_OOM_DEADLOCK.md`). A DevRel notebook must never brick the
  reader's box. The capacity-demo numbers also differ per box (discrete 96 vs unified 128) —
  compute them from the detected total, don't hardcode.
- **Serving-cell asymmetry (minor).** The standalone FlashInfer kernel probe is trivially
  dual-arch-native. The vLLM/SGLang *serving* cells use the offline `LLM` API (no Docker):
  trivial on Colab (pip install fork); on the Spark, either a native fork install or run the
  notebook's kernel **inside the proven container**. Note this in the install cell.

## Vehicle / build notes
- **No Docker** (Colab is itself a container). Use vLLM's **offline `LLM` API** in-notebook
  (`LLM(model=..., kv_cache_dtype="nvfp4", ...)` + `.generate()`), and the repo's standalone
  harnesses (`scripts/flashinfer_nvfp4_kv_probe.py`, `gemma_nvfp4_kv_quality_gate.py`,
  the convention bridge) — they pip-install/JIT fine in a notebook.
- FlashInfer JIT-compiles for `sm_120a` on first use → the SWA-prefill kernel either fails
  to build (before) or builds + passes (after). That IS the demonstration.
- Pin exact commits/wheels (stock vs `jethac/vllm` + `jethac/flashinfer`) for repro.
- Model matrix: a parameterized list with a representative default and a "run all Gemma"
  toggle. All variants fit individually on 96 GB (NVFP4 weights for 27B/31B/26B-A4B).

## Notebook structure

### 0. Hardware check (assert sm_120)
- `nvidia-smi` + `torch.cuda.get_device_capability()` → assert `(12, 0)` ("G4" = RTX PRO
  6000). If not a G4, stop — the demo is meaningless on other GPUs.
- One-paragraph DevRel framing: consumer Blackwell, FP4 tensor cores, why KV cache is the
  interesting lever.

### 1. The promise (explainer)
- NVFP4 KV cache = 4-bit KV → ~1.78× more context / concurrency at the **same** decode
  speed, on a memory-bound box. Why Gemma cares: long context + (Gemma 4 12B) multimodal =
  KV-heavy. Frame as **capacity, not speed** (the roofline truth).

### 2. BEFORE — NVFP4 KV is unavailable for Gemma on sm_120
- **2a. Stock path fails.** Stock vLLM/FlashInfer with `kv_cache_dtype="nvfp4"` on a Gemma
  model → routes to trtllm-gen → no consumer-Blackwell cubins → error. Capture it verbatim.
  Honest line: "the datacenter NVFP4-KV path has no sm_120/sm_121 cubins (TRT-LLM #11799)."
- **2b. Root cause, isolated.** Run the standalone FlashInfer NVFP4-KV probe on sm_120 for
  the Gemma **SWA-prefill** shape → 54 compile errors: `PagedParams` missing
  `maybe_k/v_cache_sf_stride_*` (`prefill.cuh`). Contrast with the **decode** shape and a
  **non-SWA (Qwen)** shape → those work. This proves it's the **SWA-prefill kernel**, i.e.
  the *whole Gemma family* (all SWA), not one model.
- **2c. The Gemma family sweep (the "before" table).** Loop Gemma 3 (4B, 27B), Gemma 3n
  (E4B), Gemma 4 (E4B, 12B, 26B-A4B, 31B) attempting NVFP4 KV → all FAIL. One non-SWA
  control (Qwen2.5) → PASS. Output table: `model | SWA? | NVFP4-KV result`. Headline:
  **"NVFP4 KV is unavailable for the entire Gemma family on consumer Blackwell."**

### 3. The fix (explainer + install)
- DevRel explanation: the FP4 SWA-prefill kernel's params struct was missing its
  scale-factor strides → it couldn't compile → silent fallback to raw-byte KV → garbage.
  Show the error + the conceptual one-liner ("complete the struct, mirror the decode path").
- Install the patched forks (`jethac/flashinfer` + `jethac/vllm`, pinned). Clear stale JIT
  cache. (This is the cell that changes between before/after.)

### 4. AFTER — NVFP4 KV working for Gemma on sm_120 (our patch)
- **4a. Kernel correctness.** Re-run the standalone probe on sm_120 → SWA-prefill module
  **compiles**, cosine ≥ 0.9999 vs bf16 reference for the Gemma SWA shapes.
- **4b. Family sweep, now green.** Re-run the 2c sweep → Gemma models serve with NVFP4 KV
  and pass `gemma_nvfp4_kv_quality_gate.py` (first token sane, logprob comparator). Same
  table, now PASS.
- **4c. The payoff — capacity.** Per Gemma model: fp8 KV vs NVFP4 KV → KV pool tokens / max
  context that fits at matched memory, showing **~1.78× at decode parity.** The headline
  number, with a correctness comparator so it's not a hollow capacity claim.
- **4d. The "wow".** Pick one Gemma model; show a long-context prompt that **OOMs under fp8
  KV but fits under NVFP4 KV** at the same memory budget — the capacity unlock made visceral.
- **4e. Honest quality.** Measure the NVFP4-KV PPL delta on Gemma (not assumed from Qwen) —
  Gemma's outlier-heavy attention is *why* NVFP4 weight releases keep attention in BF16, so
  the KV quality cost is a real, measured number here, reported plainly.

### 5. Scope & honesty
- Validated on `sm_120` (this G4) and separately `sm_121` (DGX Spark) — both consumer
  Blackwell; non-portable cubins. Capacity profile differs (96 GB discrete vs 128 GB
  unified). Capacity, not speed. Gemma quality measured, not assumed.

### 6. Try it yourself (DevRel CTA)
- One-cell repro on your own RTX PRO 6000 or Spark. Links: the repo, the FlashInfer
  contribution, vllm #31085 / llama.cpp #18250. "Compiled + validated on both sm_120 and
  sm_121."

## Phasing vs the 6/25 deadline
1. **Now:** build Sections 0-2 (BEFORE) — valid today, demonstrates the gap + the diagnosed
   root cause. Independently shippable.
2. **When the SWA-prefill fix lands + passes sm_120 validation:** drop in Sections 3-4.
3. **If the fix slips past 6/25:** ship BEFORE + Section 3 explainer + "fix in flight, PR
   linked" — still a strong, honest DevRel artifact (problem found, root-caused, fix in
   review). Do not fake an AFTER that isn't green.

## Reuse from the repo
`scripts/flashinfer_nvfp4_kv_probe.py`, `scripts/gemma_nvfp4_kv_quality_gate.py`,
`scripts/sglang_nvfp4_kv_convention_probe`-style harness, `scripts/openai_serving_benchmark.py`,
`docs/GEMMA_COMPATIBILITY_PLAN.md` (model matrix + geometry), `docs/GEMMA_RUNG_MINUS1_CONFIG_AUDIT.md`.
