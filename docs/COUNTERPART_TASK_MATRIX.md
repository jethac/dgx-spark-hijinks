# Counterpart Task Matrix

Status: live tasks defined; vLLM Qwen3.6 DFlash row accepted; vLLM Qwen NVFP4-KV capacity recorded; SGLang FP4 KV capacity recorded but quality-failing; six counterpart rows still missing or partial.

`scripts/counterpart_evidence_audit.py` says which AEON-derived SGLang and llama.cpp counterpart rows are still missing. This file points to the runnable task definitions for those rows.

Validate the matrix:

```bash
python3 scripts/counterpart_task_matrix.py \
  --tasks tasks/counterpart_evidence_tasks.jsonl \
  --audit results/counterpart_evidence_audit_20260608.json \
  --output results/counterpart_task_matrix_20260608.json
```

Current validation artifact: `results/counterpart_task_matrix_20260609.json`.

## Rules

- These tasks are not proof by themselves. They are command contracts for live GB10 sessions and stay in the matrix even after a row becomes claim-ready.
- A task is not claim-ready until the expected artifacts exist and the counterpart evidence audit accepts them.
- Debug overlays, `_patched_`, `_nograph`, `_nographs`, `_autosafe`, quality-failing, and startup-only artifacts do not satisfy the clean SGLang FP4-KV row.
- Qwen speed remains separate from Gemma compatibility. Do not generalize one model family to the other.
- llama.cpp Q4_0/Q4_K practical serving does not prove native NVFP4/MXFP4 GGUF dispatch.

## Defined Rows

| requirement | runtime | first useful proof |
|---|---|---|
| `sglang_gemma_nvfp4_ordinary_kv` | SGLang | Gemma NVFP4-weight serving with ordinary BF16/fp8 KV and checkpoint audit |
| `sglang_clean_fp4_kv_after_row` | SGLang | clean fp8-vs-fp4 KV Qwen row with quality and graph policy; current autosafe row is capacity-only because quality fails |
| `sglang_dflash_or_eagle_qwen` | SGLang | Qwen speculative row with accepted-draft metrics and non-speculative comparator |
| `vllm_qwen36_nvfp4_dflash` | vLLM | claim-ready for AEON and derived fork serving rows; clean fork packaging and native-target proof remain separate |
| `vllm_qwen_nvfp4_kv_capacity` | vLLM | recorded: Qwen fp8-vs-`nvfp4` KV capacity row shows `1.751x` KV pool/concurrency and decode parity; Gemma still requires the separate vLLM direction work |
| `llamacpp_larger_qwen_gguf` | llama.cpp | Qwen3/Qwen3.6 GGUF practical serving row |
| `llamacpp_native_fp4_gguf` | llama.cpp | NVFP4/MXFP4 GGUF row with native FP4 dispatch evidence |
| `llamacpp_live_loglikelihood` | llama.cpp | direct/full-vocabulary native logprob path after both the live top-512 task and pinned-b9536 OpenAI echo probe failed to expose supplied-token logprobs |
| `llamacpp_newer_echo_logprobs` | llama.cpp | newer-stock `llama-server` echo probe over every smoke row via `scripts/llamacpp_echo_logprobs_contract_runner.py` before accepting or forking a native endpoint |

The source of truth for commands is `tasks/counterpart_evidence_tasks.jsonl`.
