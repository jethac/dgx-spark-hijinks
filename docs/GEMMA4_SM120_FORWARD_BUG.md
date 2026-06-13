# Gemma-4 forward produces degenerate output on sm_120 (consumer Blackwell)

**Status:** REVISED — raw-prompt probe invalid for Gemma-4 `-it` controls · **Opened:**
2026-06-13 · **Revised:** 2026-06-13. Codex found that the apparent sm_120 forward
failure reproduces on sm_100 B200 and CPU fp32 under raw prompting, while the same
Gemma-4 `-it` models answer coherently when invoked through the chat template. The
remaining work is to re-run any claim-bearing Gemma-4 coherence/PPL controls with the
correct prompt contract, not to chase the raw-prompt signature as a hardware-kernel bug.

## 2026-06-13 reversal: prompt contract explains the smoking-gun controls

The original tests used raw prompts against `-it` models, e.g. `The capital of France is`.
That is not a valid chat/instruct invocation. A follow-up Vast control compared raw prompts
against `tokenizer.apply_chat_template(..., add_generation_prompt=True)` on both sm_120 and
sm_100:

| arch | model | dtype | raw prompt output | chat-template output |
| --- | --- | --- | --- | --- |
| sm_120 RTX 5090 | `google/gemma-4-E2B-it` | fp32 | ` France is France is ...` | `Paris` |
| sm_100 B200 | `google/gemma-4-E2B-it` | fp32 | ` France is France is ...` | `Paris` |
| sm_120 RTX 5090 | `google/gemma-4-12B-it` | bf16 | `111.1......11111` | `Paris` |
| sm_100 B200 | `google/gemma-4-12B-it` | bf16 | `1111111111111111` | `Paris` |

The CPU-vs-GPU localization attempt also falsified the "GPU diverges from CPU" assumption
for raw-prompt E2B: CPU fp32 and GPU fp32 produced the same repetition on both sm_120 and
sm_100, with `first_bad = null` and final-logits cosine ≈ 1.0.

Artifact: `results/vast_gemma4_prompt_contract_sm120_sm100_20260613T1900JST/summary.md`.

Conclusion: raw-prompt Gemma-4 `-it` outputs are not evidence of sm_120 forward corruption.
Future Gemma-4 controls must use chat templates or an equivalent serving chat format.

## Legacy raw-prompt symptom

Under the now-invalid raw prompt, Gemma-4 generated **degenerate output** on sm_120 GPUs:
`GEN("The capital of France is") -> '111.1...'`. Prefill logits are degenerate
(top-1 predictions are `'1'`/`'.'`/`'-'` everywhere); wikitext-2 mean NLL ≈ 8.0
(PPL ≈ 3039) vs an expected ≈ 2–3. Greedy generation is garbage. This section is retained
as provenance only; the raw-prompt contract is invalid for the tested `-it` models.

## Legacy raw-prompt matrix

All rows below are on a **clean RTX PRO 6000 / RTX 5090, cc (12,0) = sm_120**, x86_64,
HF eager (`transformers`, no vLLM/flashinfer unless noted), using raw prompts. The Gemma-4
rows should not be interpreted as hardware failure after the prompt-contract reversal above.

| model | arch | sm_120 result |
| --- | --- | --- |
| Qwen2.5-0.5B-Instruct | non-Gemma | ✅ `' Paris. It was founded in 7'` |
| google/gemma-3-1b-it | Gemma-3 | ✅ `' Paris. The largest city in France'` |
| google/gemma-4-12B-it | Gemma-4 dense, head 256 | ❌ `'111.1...'` (total garbage) |
| google/gemma-4-31B-it | Gemma-4 dense, head 512 | ❌ `' France is France is...'` (repetition) |
| google/gemma-4-26B-A4B-it | Gemma-4 MoE | ❌ `' a bit is a bit is...'` (repetition) |
| google/gemma-4-E2B-it | Gemma-4 MoE | ❌ `' France is France is...'` (repetition) |
| google/gemma-4-E4B-it | Gemma-4 MoE | ❌ `' France is France is...'` (repetition) |

The original interpretation was that breakage was **Gemma-4-specific and universal across
the family**. The revised interpretation is narrower: raw-prompt Gemma-4 `-it` invocation
is invalid and can produce degenerate continuations even on CPU fp32 and sm_100 B200.

## Legacy axes ruled out for the raw-prompt symptom

| axis | values tested | result |
| --- | --- | --- |
| KV / attention backend | Triton vs FlashInfer (vLLM) | bit-identical garbage (NLL 8.0195) → upstream of attention |
| framework | vLLM (custom_ops all & none) **and** pure HF eager | both broken (Codex 0113) → not vLLM/flashinfer |
| torch | 2.11.0+cu128 **and** 2.12.0+cu130 | both broken |
| transformers | 5.10.0.dev0 (`effde209`), 5.11.0, 5.12.0 | all broken (Codex 0113 + Claude) |
| dtype | bfloat16, float16 **and** float32 | all broken (Codex 0114: fp32 → `'111.111.'`) |
| precision drift | layerwise fp32-vs-bf16 cosine ≈ 0.9999, no NaN/inf (Codex 0114) | bf16 faithfully tracks fp32; both broken identically → systematic, not numerical drift |
| wheel `_C` cubins | cuobjdump confirms native sm_120a SASS present | not a missing-cubin issue |
| RMSNorm impl | native vs `_C` priority | broken either way (Codex 0113) |

These axes still show that the raw-prompt symptom is not specific to vLLM, FlashInfer,
custom ops, or bf16. They no longer support the stronger conclusion that Gemma-4 modeling
is broken on sm_120 hardware.

## Known-good contrast: sm_121 (Spark)

The campaign's coherent Gemma-4 serving (DG-V5, 31B TTFT, the whole ladder) is all on
**Spark, GB10 = sm_121**. After the Vast prompt-contract control, the useful comparison is
not raw prompt sm_120 vs raw prompt sm_121. It is claim-bearing chat-template / serving
format rows on the target backends.

## Current hypothesis

The active hypothesis is now mundane but important: the raw-prompt probes violated the
Gemma-4 `-it` prompt contract. The specific raw prompt signatures (`111.1` for 12B and
`France is` repetition for E2B) are reproducible on sm_100 and CPU, and disappear under
the chat template for the tested E2B and 12B rows.

## The "where did it come in" question

The evidence no longer points at an arch-specific kernel introduction point. It points at
test harness construction: raw-prompt controls were used for instruct models that require
the chat template.

## Family to test (the "every Gemma-4 model" matrix — in progress)

Text models (it variants): **12B** (❌ confirmed), **31B**, **26B-A4B** (MoE), **E2B**
(MoE), **E4B** (MoE). Goal: does every Gemma-4 break on sm_120, or only some configs?
This isolates whether the broken op is universal to Gemma-4 or tied to a sub-family
(dense vs MoE, head_dim 256 vs 512).

## Next steps (Codex leads)

1. Re-run any claim-bearing Gemma-4 coherence/PPL anchors with `apply_chat_template` or the
   serving-equivalent chat format.
2. Treat raw-prompt `-it` generations as invalid harness output unless a base/non-instruct
   Gemma-4 checkpoint is explicitly being tested.
3. If a chat-template row fails on sm_120 while passing on sm_100/sm_121, then reopen the
   hardware-kernel path with CPU-vs-GPU or cross-arch layer localization.

## Handoff status
Claude established the raw-prompt discriminator and closed out the version/dtype/framework
axes. Codex's follow-up Vast controls revised the conclusion: the raw-prompt rows were not
valid instruct-model coherence tests. The vast.ai box runbook + HF-eager scripts are in
`docs/vast_anchor/`.

## Artifacts / refs
- Codex 0113 (vast sm_120 custom-ops isolation), `results/vast_sm120_d4f0_custom_ops_...`
- Claude HF-eager discriminator (this doc, 2026-06-13), scripts in `docs/vast_anchor/`
- vast box runbook: `docs/vast_anchor/SM120_NUMERICS_PLAN.md §5`
