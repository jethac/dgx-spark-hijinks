# Campaign plan, epoch 2: NVFP4 KV everywhere + DiffusionGemma

Date: 2026-06-11. Branch: `epoch2` is THE working branch for both agents
(replaces the 022/docs split; both lanes commit here; `git pull --rebase`
before every push). Blog/Colab/upstream filing remain gated on completing
the FULL ladder (Jetha's standing call - no publication-date sequencing).

## Why epoch 2
DeepMind released DiffusionGemma (2026-06-10): block-autoregressive discrete
diffusion on the Gemma 4 26B-A4B base. Day-zero NVFP4 support is
WEIGHTS-ONLY (Model Optimizer checkpoint + vLLM playbooks). Its decoding
re-reads the cached prefix KV 12-48x per canvas - KV bandwidth becomes the
dominant decode cost - so NVFP4 KV (our territory, still nobody else's) is
worth MORE here than on autoregressive models. Our assets map directly:
VO-split + dispatcher fix (76af7982) if it inherits D=512 globals;
bidirectional-canvas masking = the mm-prefix custom-mask problem;
harnesses/provenance gates carry over unchanged.

## Rungs (additions; the existing Gemma ladder rungs all remain)
- DG-0 baseline: stock day-zero DiffusionGemma NVFP4-weights on Spark vLLM.
  Measured: tok/s vs context length, KV pool, attention backend + per-layer
  geometry from SERVING DISPATCH, canvas/prefix cache implementation notes.
- DG-1 cache analysis: how vLLM's DiffusionGemma manages prefix KV + canvas
  attention; whether reshape_and_cache + our NVFP4 writer/reader engage
  unmodified; what masking the denoising steps demand.
- DG-2 full NVFP4 KV: knobs on; canvas masking enablement if required
  (FlashInfer custom-mask = generalization of task 21's deferred half).
- DG-3 the killer benchmark: decode tok/s vs context length, NVFP4-KV vs
  fp8/bf16-KV - the KV-read-amplification thesis measured. Capacity +
  quality rows per campaign standards.
- DG-4: AR Gemma 4 26B-A4B rung inherits everything from DG-2/3.

## Lane split
**Claude (vLLM + FlashInfer kernel lane):**
1. Upstream-overlap audit (vLLM PR #40082, 0.19 diffs near our files) ->
   rebase decision + plan.
2. Dispatcher-fix validation window (rt-base/rt5 flip green + regression
   slice) -> unblocks bf16 anchor -> quality table -> Triton-retirement
   benchmarks (existing ladder debt, stays scheduled).
3. DG-0/DG-1 on vLLM.
4. FlashInfer custom-mask (canvas/mm-prefix) enablement; split-dtype module
   keying (task 22, Codex's graph-gate consumer).
**Codex (SGLang + images/infra lane):**
1. Finish SGLang Gemma 4 rung 0/1 per existing goal (mid-flight).
2. r9 image: r8 + flashinfer 76af7982 (+ latch diag + cache hygiene gates
   as in r8).
3. SGLang DiffusionGemma feasibility: does SGLang have/want day-zero
   support? Scope what serving it takes (its diffusion scheduling is
   nontrivial in any engine).
4. Inherits split-dtype keying when Claude lands it -> mixed-KV graph gate.

## Protocols
- Box: CLAUDE_WINDOW_OPEN marker unchanged (present = Claude holds Spark).
- Messages: mail/ on epoch2 (see mail/README.md). Jetha no longer relays.
- Evidence/provenance/memory guardrails: unchanged from epoch 1.

## Post-ladder capstone (gated; Jetha 2026-06-11)
llama.cpp NVFP4 KV cache contribution starts ONLY after vLLM + SGLang are
verified across Gemma 3, all Gemma 4 sizes, and DiffusionGemma. Approach:
extend ggml's existing GGML_TYPE_NVFP4 to KV (writer kernel + fattn
converter + plumbing), contributed upstream with our layout/provenance
lessons and the cross-implementation quality evidence - NOT a FlashInfer
port. Details in task 28.
