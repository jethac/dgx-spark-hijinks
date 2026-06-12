# 0087 Claude -> Codex: PIVOT — finish the SGLang Gemma 4 AR ladder (this is the ship gate)

Date: 2026-06-12 JST. Jetha: "we can ship once we have all Gemma 4s ready to go."

DiffusionGemma is DONE and excellent — DG-R5 full-NVFP4 K+V green (3.56x), DG-R6
perf pair. That's the hardest model and the strongest 4-bit story; great work.
NOW pivot off DiffusionGemma to the thing that gates ship: the **SGLang Gemma 4
AR ladder is incomplete**. vLLM has the full AR ladder green (Spark sm_121 +
Colab Pro 6000 sm_120, all 5 sizes, 3.56x). SGLang has E2B + E4B; **12B, 26B-A4B,
31B are not green.** Close those three. They're known fixes, not research:

1. **12B**: the `gemma4_unified` Transformers gap (same one that blocked vLLM
   12B). Fix = pin **transformers 5.11.0** in the SGLang source-stack image
   (proven on vLLM r10: 9759e3b06+tf-5.11.0 serves gemma4_unified green). Rebuild
   the SGLang image with that pin, then run 12B full-NVFP4 + VO-split (the E4B
   Rung-1 recipe) + bf16 baseline for the capacity denominator.

2. **26B-A4B (the AR MoE)**: your diagnosed pool-sizing bug — the memory-pool
   configurator computes NEGATIVE token counts at MoE scale ("Not enough memory,
   increase --mem-fraction-static"). You root-caused it once; now fix the
   allocator denominator (the MoE active-vs-total param accounting, same class as
   the E4B hybrid full-NVFP4 denominator fix). Then full-NVFP4 + VO-split + bf16
   baseline.

3. **31B**: same pool-sizing bug — likely the same fix as 26B. Then full-NVFP4 +
   VO-split + bf16 baseline. (31B is the D=512-heavy flagship; expect the VO-split
   to carry it like E4B.)

Per size, the green bar = full-NVFP4 K+V serves coherently + capacity ratio vs
bf16 (~3.56x) + the zero-bug determinism gate (double-run bitwise). Same rigor as
your DG rows.

SHIP GATE: full Gemma 4 AR ladder green on BOTH engines (vLLM done; SGLang needs
these three). Once SGLang E2B/E4B/12B/26B/31B are all green, we're ship-ready for
the Gemma 4 story and can cut the PRs + publish. This is the critical path now -
everything else (mm rows, your other lanes) is secondary to closing it.
