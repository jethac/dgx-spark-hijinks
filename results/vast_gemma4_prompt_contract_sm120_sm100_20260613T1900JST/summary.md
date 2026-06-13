# Vast Gemma 4 Prompt-Contract Control: sm_120 vs sm_100

Date: 2026-06-13 JST

Scope: pure Hugging Face eager controls on fresh Vast instances, testing whether the apparent Gemma 4 sm_120 forward failure survives correct `-it` chat-template prompting.

Instances:

- sm_120: RTX 5090, compute capability 12.0
- sm_100: B200, compute capability 10.0

Both instances were destroyed after artifacts were copied back.

Common environment:

- Image: `nvidia/cuda:13.0.2-devel-ubuntu22.04`
- Torch: `2.12.0+cu130`
- CUDA: `13.0`
- Transformers: `5.10.0.dev0` from `effde20942e3f82a1b97449f60b3a48c5ff96145`

## Results

### Raw Prompt vs Chat Template

Prompt contract:

- Raw prompt: `The capital of France is`
- Chat prompt: `What is the capital of France? Answer in one word.`
- Chat prompt is rendered with `tokenizer.apply_chat_template(..., add_generation_prompt=True)`.

| arch | model | dtype | raw prompt output | chat-template output |
| --- | --- | --- | --- | --- |
| sm_120 RTX 5090 | `google/gemma-4-E2B-it` | fp32 | ` France is France is ...` | `Paris` |
| sm_100 B200 | `google/gemma-4-E2B-it` | fp32 | ` France is France is ...` | `Paris` |
| sm_120 RTX 5090 | `google/gemma-4-12B-it` | bf16 | `111.1......11111` | `Paris` |
| sm_100 B200 | `google/gemma-4-12B-it` | bf16 | `1111111111111111` | `Paris` |

### CPU-vs-GPU Localization Attempt

For `google/gemma-4-E2B-it` with the raw prompt, CPU fp32 and GPU fp32 match on both architectures:

| arch | CPU output | GPU output | first_bad | final-logits cosine |
| --- | --- | --- | --- | --- |
| sm_120 RTX 5090 | ` France is France is ...` | ` France is France is ...` | `null` | 1.00000012 |
| sm_100 B200 | ` France is France is ...` | ` France is France is ...` | `null` | 1.0 |

This falsifies the attempted CPU-vs-GPU localization for the raw-prompt E2B case: CPU is not a known-good baseline for that invalid prompt contract; it reproduces the same repetition.

## Interpretation

The raw-prompt Gemma 4 `-it` probes are not valid coherence evidence by themselves. The exact degenerate signatures that looked like forward corruption under raw prompting also appear on sm_100 B200 and on CPU fp32, while the same models answer correctly when invoked through the chat template.

This does not by itself prove all Gemma 4 serving rows are green. It does retire the specific "Gemma 4 forward is broken on sm_120" hypothesis for the tested HF-eager controls. Future coherence/PPL probes for Gemma 4 `-it` models must use `apply_chat_template` or an equivalent serving chat template; raw prompt completions should be labeled invalid for these models.

## Artifacts

- `sm120_e2b_cpu_gpu/`
- `sm100_e2b_cpu_gpu/`
- `sm120_e2b_chat_control/`
- `sm100_e2b_chat_control/`
- `sm120_12b_bf16_chat_control/`
- `sm100_12b_bf16_chat_control/`
