# vast.ai sm120 d4f0 custom-ops isolation

Date: 2026-06-13 JST

Scope: vLLM Gemma 4 12B forward-numerics diagnostic on a rented RTX PRO 6000
sm_120 host. This is a red diagnostic artifact, not a serving or capacity row.

## Runtime

- Host: vast.ai RTX PRO 6000 Blackwell Workstation Edition, compute capability
  `(12, 0)`, Ubuntu 22.04.
- vLLM wheel:
  `vllm-0.1.dev1+gd4f0f79c3.sm120a-cp312-cp312-linux_x86_64.whl`
- vLLM runtime version: `0.1.dev1+gd4f0f79c3.sm120a`
- Torch: `2.12.0+cu130`
- FlashInfer: `jethac/flashinfer@7d5d477b` source overlay via
  `PYTHONPATH=/root/flashinfer`
- Transformers:
  - Initial auto-resolved wheel: `5.12.0`
  - Rechecked with recorded snapshot:
    `git+https://github.com/huggingface/transformers.git@effde20942e3f82a1b97449f60b3a48c5ff96145`
    (`5.10.0.dev0`)
- Model: `google/gemma-4-12B-it`
- KV dtype for vLLM diagnostics: `bfloat16`

The vast instance was destroyed after artifact collection.

## Result

The custom-ops isolation is red. Disabling vLLM compilation custom ops did not
recover coherent output:

- `CUSTOM_OPS=all`: `GEN: '111.1...'`
- `CUSTOM_OPS=none`: `GEN: '111.1...'`
- `CUSTOM_OPS=none NATIVE_RMS=1`: `GEN: '111.111.'`

Forcing native RMS changed the configured priority to
`rms_norm=['native', 'vllm_c']`, but did not change the failure signature.

The pure Transformers control is also red on this host:

- Transformers `5.12.0`: `GEN: '111.1...'`
- Transformers `5.10.0.dev0` at `effde209`: `GEN: '111.1...'`

This means the d4f0 vLLM custom-op toggle is not sufficient to explain the
failure in this environment. The failure exists under HF eager model forward as
well.

## Scalar

`debug_eval_vllm_effde209.log`:

- sanity sentence: `mean_nll 18.419`, `ppl 99833250.15`
- Wikitext-2 4095 scored tokens: `mean 8.04`, `median 7.641`, `max 34.142`

Expected green sanity remains roughly `Paris` generation and Wikitext mean NLL
in the `2-3` range.

## Binary Provenance

`cuobjdump_summary.txt`:

- `_C.abi3.so`: `26 sm_120a`, `4 sm_80`
- `_C_stable_libtorch.abi3.so`: `42 sm_120a`, `1 sm_89`, `1 sm_90`
- `_moe_C_stable_libtorch.abi3.so`: `26 sm_120a`, `1 sm_80`

Runtime imports:

- `vllm._C`: OK
- `vllm._C_stable_libtorch`: OK
- `vllm._moe_C`: missing import name, but `_moe_C_stable_libtorch.abi3.so`
  exists and contains sm_120a cubins
- `humming`: OK
- `tokenspeed_mla`: OK

## Artifacts

- `gen_custom_ops_all.log`
- `gen_custom_ops_none.log`
- `gen_custom_ops_none_native_rms.log`
- `transformers_control.log`
- `transformers_control_effde209.log`
- `debug_eval_vllm_effde209.log`
- `runtime_provenance_source_overlay.txt`
- `cuobjdump_summary.txt`
- `vllm_so_files.txt`
- `aux_kernel_files.txt`
