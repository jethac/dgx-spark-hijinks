# AEON Prior-Art Port Map

Date: 2026-06-08 JST

Purpose: track what from `B:/workshop/CODEX_REPORT_AEON7.md` has been ported, what still needs a counterpart in SGLang or llama.cpp, and what should not be blindly copied across runtimes.

AEON-7's public work is primarily vLLM work. Treat it as high-value Spark evidence, not as proof that SGLang or llama.cpp are fixed.

Primary sources checked:

- `B:/workshop/CODEX_REPORT_AEON7.md`
- `https://github.com/AEON-7/Qwen3.6-NVFP4-DFlash/tree/main/patches`
- `https://github.com/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4`
- `https://github.com/AEON-7/vllm-dflash`
- current forks: `jethac/vllm@6804e1b`, `jethac/sglang@98ad46961`, `jethac/flashinfer@e152cf4d`

## Rules

- Direct vLLM patches stay in `jethac/vllm`; do not port them to SGLang or llama.cpp unless the same failure is reproduced there.
- SGLang and llama.cpp need counterpart experiments, not string-for-string vLLM patch copies.
- AEON Gemma proves NVFP4 weights plus ordinary KV plus Triton target attention plus DFlash, not FA2 NVFP4 KV.
- A runtime is not broadly blessed from a Gemma-only or Qwen-only row.

## Audit Command

Use the local audit to verify that the submodules still match this map:

```bash
python3 scripts/aeon_prior_art_audit.py \
  --output results/aeon_prior_art_audit_YYYYMMDD.json
```

Current validation artifact: `results/aeon_prior_art_audit_20260608.json`.

Use the counterpart evidence audit to keep source-marker checks separate from live proof:

```bash
python3 scripts/counterpart_evidence_audit.py \
  --output results/counterpart_evidence_audit_YYYYMMDD.json
```

Current validation artifact: `results/counterpart_evidence_audit_20260608.json`. It intentionally reports all seven counterpart proof rows as missing, partial, or blocked; AEON source coverage is not the same thing as SGLang/llama.cpp serving evidence.

The command contracts for those seven rows live in `tasks/counterpart_evidence_tasks.jsonl` and are validated by:

```bash
python3 scripts/counterpart_task_matrix.py \
  --tasks tasks/counterpart_evidence_tasks.jsonl \
  --audit results/counterpart_evidence_audit_20260608.json \
  --output results/counterpart_task_matrix_20260608.json
```

Current validation artifact: `results/counterpart_task_matrix_20260608.json`.

## Literal AEON Qwen Patch Inventory

| AEON patch | purpose | current vLLM status | SGLang counterpart | llama.cpp counterpart |
|---|---|---|---|---|
| `register_qwen3_5_text.py` | register text-only `Qwen3_5*ForCausalLM` classes so Qwen3.5/3.6 does not fall into the multimodal path | ported in `jethac/vllm@6804e1b`; evidence: `third_party/vllm/vllm/model_executor/models/registry.py`, `tests/model_executor/test_qwen3_5_registry.py` | no direct port; SGLang has its own model registry and must be tested with Qwen3.6 | no direct port; Qwen3.6 GGUF conversion/loading is a separate experiment |
| `patch_cuda_optional_import.py` | make `_C_stable_libtorch` import lazy when unused SM100 MXFP4 symbols are missing on SM12x | ported in `jethac/vllm@6804e1b`; evidence: `third_party/vllm/vllm/platforms/cuda.py` | reject direct port; SGLang does not import vLLM's extension | reject direct port |
| `patch_kv_cache_utils.py` | avoid `block_size=None` crashes for hybrid linear-attention/Mamba groups | ported in `jethac/vllm@6804e1b`; evidence: `third_party/vllm/vllm/v1/engine/core.py`, `third_party/vllm/vllm/v1/worker/gpu_model_runner.py`, `third_party/vllm/vllm/model_executor/layers/mamba/abstract.py` | candidate only if Qwen3.6/SGLang reproduces a `block_size=None` failure; current SGLang already has hybrid/spec paths to audit | reject direct port |
| `patch_mrope_text_fallback.py` | text-only M-RoPE fallback for Qwen3.6 when the model class lacks `SupportsMRoPE` | ported in `jethac/vllm@6804e1b`; evidence: `third_party/vllm/vllm/v1/worker/gpu_model_runner.py` | candidate only if SGLang Qwen3.6 text-only load shows an M-RoPE mismatch | no direct port; GGUF RoPE metadata must be verified instead |
| `patch_cudagraph_align.py` | align speculative-decode CUDA graph capture sizes for non-`NONE` graph modes, not only `FULL` | ported in `jethac/vllm@6804e1b`; evidence: `third_party/vllm/vllm/config/compilation.py`, `tests/compile/test_config.py` | do not copy literally; SGLang has its own speculative graph machinery and must be tested with DFlash/EAGLE on GB10 | no direct port; llama.cpp graph/spec-decode behavior is separate |
| `strip_language_model_prefix.py` | checkpoint-conversion helper for one Qwen source model layout | not a vLLM source patch; current AEON v2 weights do not require this as a fork change | maybe useful as a one-off conversion check if SGLang loads a text-only Qwen checkpoint with multimodal prefixes | no direct source patch; any GGUF converter must handle names during conversion |

## AEON Gemma Lessons

| lesson | vLLM status | SGLang action | llama.cpp action |
|---|---|---|---|
| NVFP4 weights are the proven Gemma speed lever | locally reproduced through AEON image; about `48-54 tok/s` short/medium decode and `98 tok/s` long-prefill | test Gemma 4 NVFP4 weights with ordinary KV before FP4 KV; current SGLang Gemma path is not blessed | current GGUF Q4_0 path is practical; native NVFP4/MXFP4 GGUF remains a separate experiment |
| Gemma 4 target attention is Triton because local/global head dimensions differ | documented in vLLM recipe and local logs | do not make Gemma FP4 KV the first SGLang goal; prove ordinary KV serving first | no FA2 KV implication |
| Routers, vision tower, and vision embeddings must not be blindly NVFP4-quantized | use AEON weights unless re-quantizing; run `scripts/nvfp4_checkpoint_audit.py` before treating a checkpoint as safe | if building SGLang-loadable Gemma NVFP4 checkpoints, preserve BF16 routers and vision tensors and attach the checkpoint audit artifact | if making GGUFs from AEON or our own quantization, record whether routers/vision stayed high precision before conversion |
| EOS/control-token cleanup matters | AEON weights/config sidestep this for the reproduced row | include deterministic output sanity in Gemma SGLang row | include chat smoke and template checks for Gemma GGUF |

## DFlash And Speculative Decode

AEON's DFlash result is the main single-stream multiplier, but the port story is runtime-specific.

| runtime | current evidence | decision |
|---|---|---|
| vLLM | AEON Gemma DFlash row is locally reproduced; Qwen36 vLLM image acquisition is still blocked | continue with AEON Qwen36 reproduction, then matched `jethac/vllm` fork row |
| SGLang | current `jethac/sglang@98ad46961` tree contains DFlash surfaces including `python/sglang/srt/arg_groups/speculative_hook.py`, `python/sglang/srt/models/dflash.py`, Qwen model `set_dflash_layers_to_capture` hooks, and metrics for accepted drafts | candidate, not proven; add a SGLang Qwen/Gemma DFlash smoke only after ordinary Qwen/Gemma serving is stable |
| llama.cpp | no `third_party/llama.cpp` submodule exists because no llama.cpp code change has been needed yet; current proof is practical GGUF serving and native loglikelihood harness work | do not port DFlash literally; evaluate llama.cpp-native speculative/draft-model support only when a GGUF drafter/model pair exists |

## Still Needed Counterparts

1. SGLang Gemma NVFP4-weight serving with ordinary KV.
2. SGLang graph-compatible FP4 KV after-row with quality checks; current overlay row proves capacity but not usable speed.
3. SGLang DFlash or EAGLE row on a Qwen-class model if ordinary serving is stable.
4. vLLM Qwen3.6 NVFP4+DFlash local serving row once image acquisition or rebuild is fixed.
5. llama.cpp larger Qwen3/Qwen3.6 GGUF rows.
6. llama.cpp native NVFP4/MXFP4 GGUF tensor-core proof, separate from Q4_0/Q4_K serving.
7. llama.cpp live native loglikelihood task proof before paper-comparable GGUF accuracy claims.

These seven rows are now mechanically tracked by `scripts/counterpart_evidence_audit.py`.

## Explicit Non-Ports

Do not spend time porting these directly:

- vLLM `_C_stable_libtorch` lazy import into SGLang or llama.cpp.
- vLLM registry changes into SGLang or llama.cpp.
- vLLM M-RoPE fallback into llama.cpp.
- vLLM CUDA graph capture-size code into SGLang without a reproduced SGLang graph failure.
- AEON Gemma FA2 NVFP4-KV claims; AEON Gemma is not an FA2 NVFP4-KV result.
- vLLM env knobs such as `VLLM_NVFP4_GEMM_BACKEND`, `VLLM_USE_FLASHINFER_MOE_FP4`, and `VLLM_TEST_FORCE_FP8_MARLIN` as if they were cross-runtime fixes.

## Next Proof Order

1. Reconnect to the GB10 host and finish AEON Qwen36 image acquisition or rebuild.
2. Run `scripts/nvfp4_checkpoint_audit.py` on any NVFP4 Qwen/Gemma checkpoint before using it as speed evidence or conversion input.
3. Run the Qwen speed lane for vLLM/SGLang/llama.cpp with `scripts/qwen_speed_lane.py`.
4. Attempt SGLang Gemma NVFP4-weight serving with ordinary KV.
5. Only then decide whether a SGLang DFlash or FP4-KV code change is justified.
6. Create `third_party/llama.cpp` only when a real llama.cpp source change is required.
