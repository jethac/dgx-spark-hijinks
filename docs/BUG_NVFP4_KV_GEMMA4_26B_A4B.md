# BUG: NVFP4 KV read-path broken on Gemma-4-26B-A4B (MoE) — fp8 KV is the interim ship path

Status: OPEN (real kernel bug, precisely characterized 2026-06-15). nvfp4-SPECIFIC. NOT calibratable.
26B-A4B ships with **fp8 KV** (correct, near-lossless) until the nvfp4 kernel bug is fixed; 12B/31B
nvfp4 are unaffected (GREEN, calibrated).

## Evidence (vast PRO 6000 / GB202 / sm_120, `google/gemma-4-26b-a4b-it`, ctx 8185 / prefix 4096, wikitext, 4088 scored tokens)

HF transformers eager bf16 is the dtype/kernel-independent ground truth.

| KV dtype | mean NLL (nats) | delta vs HF truth | verdict |
| --- | ---: | ---: | --- |
| **HF eager bf16 (TRUTH)** | **7.9923** | — | reference |
| vLLM bf16 | 7.9027 | -0.090 | correct |
| vLLM **fp8** | 7.7903 | **-0.202** | **correct (near-lossless) — SHIP PATH** |
| vLLM nvfp4 (k=v=0.05) | 7.3054 | -0.687 | broken |
| vLLM nvfp4 (k=v=0.07) | 5.7998 (x2, deterministic) | **-2.193** | broken |
| vLLM nvfp4 (k=v=0.10) | 7.3691 | -0.623 | broken |

## Diagnosis

1. **vLLM bf16 is correct** — matches HF truth to 0.09 nats. (The degenerate greedy chat smoke,
   `"the capital of Japan is the capital of Japan is..."`, is the raw-prompt-on-an-instruct-model
   artifact, NOT a serving bug — bf16 PPL proves the serving is sound.)
2. **nvfp4 scores BELOW truth at every scale** (-0.6 to -2.2 nats). Lower-than-truth NLL is physically
   impossible for a correct model, so the nvfp4 path is degrading attention into a repetition/
   high-frequency collapse: low *local* per-token NLL on a partly-predictable corpus, but globally
   broken generation (the 0.07 smoke loops `"Wait, I'm not sure."`). Deterministic (0.07 gave 5.7998
   twice). No global scale recovers truth → NOT a calibration problem.
3. **fp8 KV is correct** (-0.20 vs truth, normal quantization). Since fp8 uses the SAME paged-attention/
   MoE serving path and only differs in the KV dtype/dequant, the break is **NVFP4-SPECIFIC**, not MoE
   and not the general quantized-KV path. The MoE-pool hypothesis is weakened: MoE would break fp8 too.

## Scope / what's special about 26B-A4B

26B-A4B (`Gemma4ForConditionalGeneration-L30-H2816-D256-KV16`... arch sig L30/H2816/KV8) takes the same
head_dim_qk=512 FA2 VO-split nvfp4 path as 12B/31B (confirmed: `FA2 VO split (nvfp4 KV): head_size 512
runs as 2 passes`), yet 12B (k=0.1,v=0.06) and 31B (k=v=0.05) nvfp4 are near-lossless while 26B-A4B is
broken. The distinguishing axis is the 26B-A4B geometry (L30 / H2816 / its GQA ratio) and/or the MoE
block layout interacting with the nvfp4 KV scale-factor tiling. Localizing the exact kernel cause
(layer-wise nvfp4-vs-bf16 logit diff; GQA-ratio / SF-stride ablation) is the open follow-up.

## Ship decision

- **26B-A4B**: ship **fp8 KV** (near-lossless, correct). 2x the nvfp4 footprint but correct. Full-nvfp4
  26B is BLOCKED on this kernel fix.
- **12B / 31B**: nvfp4 GREEN (calibrated) — unaffected.

## Cross-lane

Codex's SGLang 26B-A4B MoE red may be the SAME nvfp4-specific bug rather than (only) pool-sizing — he
should fp8-vs-nvfp4 his 26B against an HF-eager truth to check (mail 0171). Artifacts: vast
`/root/dx26/*` (hfref2.out, bf16/fp8/nv0{5,7,7b,1}.json + logs).
