# vLLM AEON Qwen Patch Port

Date: 2026-06-08 JST

Branch: `jethac/vllm@spark/hijinks-020-aeon-qwen-dflash-sm121a`

Commit: `6804e1b81e6ea2ca53bb5021151bdad0f201b11d3`

## Scope

This is a source-level Qwen compatibility stop point. It is not a GB10 serving or speed result.

The patch set ports the AEON Qwen3.6/DFlash source fixes that apply to the current `jethac/vllm` branch:

- keep the earlier lazy fallback import for `_C_stable_libtorch` on SM12x builds
- keep the earlier speculative-decode CUDA graph capture-size alignment for all non-`NONE` graph modes
- register text-only `Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`
- tolerate hybrid KV cache groups whose spec has `block_size=None`
- give Mamba cache specs a conservative fallback when `mamba_block_size` is unset
- add text-only M-RoPE fallback positions when the model does not implement multimodal M-RoPE

The branch still inherits the SM12x NVFP4 KV FA2 routing/deswizzle work from `jethac/vllm@spark/hijinks-007-nvfp4-kv-sm121`.

## AEON Patch Triage

AEON source patch mapping:

| AEON patch | local disposition |
|---|---|
| `patch_cuda_optional_import.py` | already ported before this stop point as a source-shaped lazy import fallback |
| `patch_cudagraph_align.py` | already ported before this stop point with a regression test for pure `PIECEWISE` graph mode |
| `patch_kv_cache_utils.py` | partially ported where it applies to the current branch: `EngineCore`, `GPUModelRunner`, and Mamba block-size fallbacks |
| `patch_mrope_text_fallback.py` | ported as text-only fallback positions for non-multimodal requests |
| `register_qwen3_5_text.py` | ported into model registry and registry examples |
| `strip_language_model_prefix.py` | not ported; this is a checkpoint-conversion helper, not a vLLM runtime source patch |

## Validation

```powershell
python -m py_compile third_party/vllm/vllm/model_executor/models/registry.py third_party/vllm/tests/models/registry.py third_party/vllm/vllm/v1/engine/core.py third_party/vllm/vllm/model_executor/layers/mamba/abstract.py third_party/vllm/vllm/v1/worker/gpu_model_runner.py third_party/vllm/tests/model_executor/test_qwen3_5_registry.py
```

Result: passed.

```powershell
git -C third_party/vllm diff --check
```

Result: passed, with line-ending warnings only.

```powershell
python -m pytest third_party/vllm/tests/model_executor/test_qwen3_5_registry.py -q
```

Result: blocked by the local Windows checkout missing vLLM test dependency `tblib`:

```text
ModuleNotFoundError: No module named 'tblib'
```

GB10 live validation did not run because the host was unreachable:

```text
ssh: connect to host 192.168.68.112 port 22: Connection timed out
```

## Remaining Proof

- Finish the AEON Qwen3.6 NVFP4+DFlash image acquisition.
- Run `scripts/run_aeon_vllm_reproduction.sh qwen36-dflash ...` with `RECORD=1`.
- Run the same Qwen serving row on the `jethac/vllm` fork plus matching `jethac/flashinfer` fork.
- Record DFlash acceptance, CUDA graph mode, backend selection, TTFT, warmed decode, throughput, and zero-error soak behavior.
- Do not claim a Qwen fork speedup until matched before/after rows exist.
