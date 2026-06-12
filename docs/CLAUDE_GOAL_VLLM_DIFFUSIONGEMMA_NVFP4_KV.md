# GOAL (Claude / vLLM lane): DiffusionGemma NVFP4 KV-cache on vLLM (DG-V rungs)

**One-liner:** bring full-NVFP4 K+V KV cache (3.556x, format-exact) to DiffusionGemma
26B-A4B on vLLM/Blackwell (sm_120 + sm_121), to parity with SGLang's DG-R5/R6 receipts.
Closes the gap that the engine with the *official* DiffusionGemma recipe serves it bf16-only.

## Why this exists
vLLM supports DiffusionGemma (official recipe: https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it)
but in bf16 only -- no NVFP4 KV, no Blackwell, no FlashInfer. We routed ALL DiffusionGemma
work to Codex's SGLang lane and never applied the NVFP4-KV campaign to it on vLLM. DiffusionGemma
is the strongest 4-bit story we have (the decoder re-reads the full prefix every denoise step,
so 3.556x KV capacity compounds harder than anywhere in the AR ladder) -- so it should NOT be
missing from the more-visible vLLM path.

## Reference (do NOT re-derive -- match confirmed-good)
- vLLM AR NVFP4-KV impl (`spark/hijinks-e2-vllm`): NVFP4 writer, `config.py` per-layer
  routing, `flashinfer.py` VO-split. Proven across E2B->31B on Pro 6000 + Spark.
- SGLang DG-R5/R6 (Codex): proves the NVFP4 read path tolerates block-diffusion attention.
  Format is geometry-independent; these are the parity target.
- Upstream vLLM DiffusionGemma recipe: the model class + serving constraints
  (`--max-num-seqs 4`, `--gpu-memory-utilization 0.85`, entropy-bound sampling overrides,
  256-token block denoise loop).

## Gate 0 -- PROBE FIRST, decide before building (the one real unknown)
Does vLLM's DiffusionGemma attention backend expose a hook to the FlashInfer NVFP4 path
the way the AR models do?
- (a) Does the block-diffusion decoder route through the same paged-prefill/decode wrappers
  our VO-split patches?
- (b) Does the bidirectional-within-block mask survive the VO-split two-pass over V halves?

**If no clean hook -> STOP, write the blocker up, escalate. Do not force it.**
If yes -> proceed to Build.

## Build (only past Gate 0 -- wiring, not new kernels)
1. Wire vLLM's DiffusionGemma model class into the existing NVFP4-KV config routing
   (`kv_data_type=uint8`, D=512 VO-split, linear-V-SF knob).
2. DG-specific allocator accounting: the 256-token-block denoise loop re-reads the full
   prefix every step -- confirm the NVFP4 page budget (`9*head/16`) holds under that
   re-read pattern with `--max-num-seqs 4`.

## Per-rung green bar (zero-bug)
- **DG-V5** (= SGLang DG-R5): full-NVFP4 K+V serves coherent generations, **>=3.5x KV
  capacity** vs bf16 denominator, **double-run bitwise-deterministic**.
- **DG-V6** (= DG-R6): perf pair (NVFP4 vs bf16 throughput/latency at matched batch);
  quantify the compounding win from per-denoise-step prefix re-read.
- Both on **sm_120 (Colab Pro 6000) AND sm_121 (Spark)**; bf16 denominator captured each.

## Done
DG-V5 + DG-V6 green on both silicon, receipts banked to `results/`, the two vLLM
DiffusionGemma cells flip (gap -> green). DiffusionGemma is then cross-engine complete and
the FlashInfer surface is proven on the diffusion path in BOTH stacks -- clearing the last
DG-shaped item under the "surface stable across all Gemma variants before filing" PR gate.

## Gate 0 VERDICT (2026-06-12): GO -- lift smaller than assumed
Probe done. DiffusionGemma routes through machinery we already own and proved:
- **Model id** `google/diffusiongemma-26B-A4B-it`; arch `DiffusionGemmaForBlockDiffusion`,
  `model_type: diffusion_gemma`. Block-diffusion MoE (128 experts top-8) + vision,
  30 layers sliding/full alternating, `max-model-len 262144` (256k -> NVFP4 KV pays off hard).
- **It is NOT mainline-plain:** the recipe needs "a vLLM build with diffusion support in the
  Gemma docker image." Upstream vLLM main DOES have `vllm/model_executor/models/diffusion_gemma.py`;
  **our fork (e2-vllm @ e32459eea) predates it** -> port it in, don't author it.
- **diffusion_gemma.py delegates its text decoder to `Gemma4Model`** -- the exact backbone our
  NVFP4 writer + config routing + VO-split already cover. Attention goes through vLLM's standard
  `Attention` + `build_attn_metadata` + `kv_cache_dtype` (NOT a bespoke attention call).
- **The block-diffusion mask is a per-request `causal` flag** in attn_metadata: encoder phase=causal,
  decoder phase=bidirectional (`causal=False`). Standard non-causal prefill -- FlashInfer supports it;
  SGLang DG-R5 already proved the NVFP4 read path tolerates the bidirectional decoder phase.
- **`attention_k_eq_v` wrinkle is ALREADY HANDLED in our backbone:** DiffusionGemma full-attn layers
  have no v_proj (V=K pre-k_norm). Our `gemma4.py` (use_k_eq_v, lines ~431/462/570/626) loads K into
  both K and V qkv_proj slots, so V==K flows automatically and the NVFP4 V-writer sees a normal V.
- **head_dim 256 (16 heads / 8 KV) per config** -> VO-split likely NOT on the critical path here;
  whatever the gemma4 text_config sets, our head-dim-conditional VO-split engages or no-ops itself.

**Revised lift (wiring, confirmed):** (1) port `diffusion_gemma.py` + its config plumbing + the
block-diffusion scheduling (diffusion_states, per-req causal buf, denoise loop) from upstream main
onto our NVFP4-KV branch; (2) register the `DiffusionGemmaForBlockDiffusion` arch; (3) NVFP4 routing
is automatic (Gemma4Model underneath) -- confirm it engages for the new arch name; (4) empirically
confirm the bidirectional (causal=False) decoder phase reads NVFP4 correctly on vLLM (the one genuine
remaining unknown, de-risked by SGLang DG-R5). No new kernel work identified.

## Coordination
vLLM = my lane. No Spark/P520 GPU touch while another agent holds the marker. Mail Codex
the DG-V plan so SGLang DG-R5/R6 receipts are the agreed parity target.
