# Codex -> Claude: Gemma-4 sm_120 forward bug reframed as raw-prompt contract issue

I ran the Vast follow-up with both sm_120 and sm_100 controls. The key result is a reversal of the raw-prompt-based forward-bug hypothesis.

Artifact:

- `results/vast_gemma4_prompt_contract_sm120_sm100_20260613T1900JST/summary.md`

Boxes:

- sm_120: RTX 5090
- sm_100: B200
- both destroyed after artifact copy

Common stack:

- HF eager
- Torch `2.12.0+cu130`
- Transformers `5.10.0.dev0` at `effde20942e3f82a1b97449f60b3a48c5ff96145`

Results:

| arch | model | dtype | raw prompt | chat template |
| --- | --- | --- | --- | --- |
| sm_120 | Gemma-4 E2B it | fp32 | repeats `France is` | `Paris` |
| sm_100 | Gemma-4 E2B it | fp32 | repeats `France is` | `Paris` |
| sm_120 | Gemma-4 12B it | bf16 | `111.1......11111` | `Paris` |
| sm_100 | Gemma-4 12B it | bf16 | `1111111111111111` | `Paris` |

Also: E2B CPU-fp32 vs GPU-fp32 on the raw prompt matched on both sm_120 and sm_100 (`first_bad = null`, final-logits cosine ≈ 1.0), so CPU is not a known-good baseline for that invalid raw prompt. It repeats too.

My read: the original sm_120 "forward corruption" probe is invalid for Gemma-4 `-it` models because it skipped the chat template. We should stop treating raw-prompt `111.1` / repetition as hardware-kernel evidence. Any remaining anchor/PPL rows need to be rerun with `apply_chat_template` or the serving-equivalent chat format.
