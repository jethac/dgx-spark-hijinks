# Upstream-overlap audit (task 23) - 2026-06-11

Base: our vLLM branch merge-base with upstream/main = 4dcd10eb0d (recent -
early June). Upstream head audited: 6e64c1bab1.

## Findings
1. **NVFP4 KV kernels: ZERO upstream drift.** csrc/libtorch_stable/
   nvfp4_kv_cache_kernels.cu + cache_kernels.cu untouched since our base.
   The KV-cache lane has no collision and no competition.
2. **flashinfer.py backend: ZERO upstream commits** since merge-base (same
   path, no follow-renames). Our 190+ lines of VO-split orchestration sit
   on a quiet file. Rebase cost ~zero here.
3. **THE BIG ONE - upstream Gemma4Config changed 2026-06-10**
   (6deb05e0e4, "Unified FA4 for all layers + FlashAttention mm_prefix"):
   when `is_fa_version_supported(4)` and max_head_dim <= 512, force FA4 for
   all layers; OTHERWISE STILL FORCE TRITON_ATTN. Decisive detail in
   fa_utils.py: the FA4 TMEM gate rejects head_size > 128 (except 192) on
   ALL major >= 10 devices - which includes CC 12.x. So upstream's fix
   serves Hopper; consumer Blackwell (GB10, RTX PRO 6000) STILL lands on
   forced Triton. CONSEQUENCES: (a) vllm#38887/#40677 remain unfixed for
   the consumer-Blackwell audience - our VO-split/FlashInfer route remains
   the only path there, receipts intact and sharpened ("upstream's own fix
   explicitly leaves CC 12.x behind"); (b) config.py is a CERTAIN textual
   conflict on rebase - semantic resolution: layer our knob branches onto
   their new structure (their FA4 path for capable devices; our FLASHINFER
   paths under VLLM_FLASHINFER_VOSPLIT / NVFP4 knobs for sm_12x; their
   Triton fallback last).
4. gemma4.py: two minor upstream commits (MoE-refactor mechanical churn +
   an error-message fix) - trivial rebase.
5. **DiffusionGemma is NOT in upstream main** (no model file, no registry
   entry). Day-zero vLLM support evidently ships via NVIDIA's playbook
   stack (their container/recipe) or the transformers-backend route - NOT
   upstream vllm. DG-0 prep must locate the actual serving stack from the
   playbook; this also means our rebased branch won't magically serve it.
6. vLLM PR #40082 (FlashInfer b12x MoE/FP4 GEMM for SM120/121): weights-
   side; adjacent to dispatch we already carry via the hikari-derived work;
   no KV overlap. Track for the rebase, don't block on it.

## Decision
REBASE: yes, cheap, do it before DG-0 (one real conflict: config.py; plan
the semantic merge above). Our four patch areas: flashinfer.py (clean),
config.py (conflict, planned), gemma4.py (trivial), envs.py + kernels
(clean). After rebase: revalidate with the latch diag + probe slice before
any serving claim (provenance rules).
