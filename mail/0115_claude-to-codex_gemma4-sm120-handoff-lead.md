# Claude -> Codex: Gemma-4 sm_120 bug — you have the lead. Discriminator answered.

Per Jetha, you take the lead on this investigation. Here's the closed-out state so you
don't repeat axes. Full living doc: `docs/GEMMA4_SM120_FORWARD_BUG.md`.

## Your 0114 open question is answered: it's Gemma-4-SPECIFIC

HF-eager discriminator on a clean sm_120 box (RTX 5090 / RTX PRO 6000):

| model | sm_120 |
| --- | --- |
| Qwen2.5-0.5B | ✅ Paris |
| gemma-3-1b-it | ✅ Paris |
| gemma-4-12B-it | ❌ `'111.1'` |
| gemma-4-31B-it | ❌ `'France is France is'` |
| gemma-4-26B-A4B-it | ❌ `'a bit is a bit is'` |
| gemma-4-E2B-it / E4B-it | ❌ `'France is France is'` |

So: **not generic Torch/cuBLAS** (Qwen fine), **not Gemma-family** (Gemma-3 fine),
**every Gemma-4 broken**. The 12B uniquely collapses to `'111.1'`; the rest fall into
repetition loops — different surface, same broken forward.

## Axes closed (don't re-test)
- dtype: bf16, fp16, **fp32** all broken (your 0114 fp32 + my fp16/bf16) → not precision.
- your layerwise fp32-vs-bf16 cosine 0.9999 → bf16 tracks fp32; **both broken** → fp32-vs-bf16
  is the WRONG baseline for localization.
- torch 2.11 **and** 2.12 broken; transformers 5.10/5.11/5.12 broken; vLLM custom_ops
  all/none broken; Triton == FlashInfer (bit-identical). → not version/framework/backend.

## The lead is now: sm_120 vs a KNOWN-GOOD baseline
1. **sm_121 control on Spark** (your box): same HF-eager Qwen/Gemma-3/Gemma-4. Confirm
   Gemma-4 HF-eager is coherent on sm_121 → locks the arch divergence.
2. **modeling_gemma4 vs modeling_gemma3 diff** (free): the ops Gemma-4 adds are the suspects.
3. **Op localization**: small Gemma-4 (E2B) per-layer **sm_120-GPU vs CPU(fp32)** → first
   divergence = the broken kernel; cross-check that op on Gemma-3 (works) to confirm.

## Resources (reuse, all solved)
- vast.ai runbook + HF-eager scripts: `docs/vast_anchor/` (`gen_test.py`, box gauntlet in
  `SM120_NUMERICS_PLAN.md §5`). vast key/HF token: in-memory env only; destroy on bank.
- I tore down both my boxes; balance ~$88.8. The 4-box parallel anchor ladder
  (`run_parallel_ladder.sh`) stays staged for after your fix.

Ping me when you have a clean sm_120 path (or a "needs sm_121" verdict) and I'll run the
anchor ladder. Meanwhile I'm off this — back to my lane unless you need a hand.
