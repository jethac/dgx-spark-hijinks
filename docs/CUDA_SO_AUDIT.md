# CUDA Target And Shared Object Audit

Date: 2026-06-07

## Two-Stage CUDA Target Evidence

Spark-native claims need two kinds of evidence:

1. Build or JIT logs should show the intended target, such as `sm_121`, `compute_121`, `121-real`, or a documented SM12x family target.
2. Resulting CUDA extension objects should be inspected with `cuobjdump` where a `.so` exists.

Use the build/JIT log checker first:

```bash
python3 scripts/cuda_build_target_audit.py \
  --log results/RUN_ID_build.log \
  --output results/RUN_ID_build_target_audit.json
```

If a project intentionally uses a family-compatible target such as `120f`, make that explicit:

```bash
python3 scripts/cuda_build_target_audit.py \
  --log results/RUN_ID_jit.log \
  --allow-family-target 120f \
  --output results/RUN_ID_build_target_audit.json
```

Then inspect compiled shared objects:

Historical audit command:

```bash
/home/jethac/gemma4-evals/.venv/bin/python /home/jethac/cuda_so_audit.py \
  --package vllm \
  --package flashinfer \
  --max-files 120 \
  --output /home/jethac/dgx-spark-hijinks-results/cuda_so_audit_vllm_flashinfer_20260607T111023Z.json
```

Result snapshot:

- `results/cuda_so_audit_vllm_flashinfer_20260607T111023Z.json`

## Summary

The audit inspected 14 CUDA extension shared objects from the benchmark venv.

Architecture counts:

| architecture | object count |
|---|---:|
| `sm_100` | 5 |
| `sm_110` | 3 |
| `sm_120` | 3 |
| `sm_75` | 1 |
| `sm_80` | 4 |
| `sm_87` | 3 |
| `sm_89` | 3 |
| `sm_90` | 3 |
| `sm_90a` | 6 |
| `sm_121` | 0 |

Important objects:

| object | architectures |
|---|---|
| `vllm/_C.abi3.so` | `sm_100`, `sm_110`, `sm_120`, `sm_75`, `sm_80`, `sm_87`, `sm_89`, `sm_90`, `sm_90a` |
| `vllm/_C_stable_libtorch.abi3.so` | `sm_100`, `sm_110`, `sm_120`, `sm_80`, `sm_87`, `sm_89`, `sm_90`, `sm_90a` |
| `vllm/_moe_C.abi3.so` | `sm_100`, `sm_110`, `sm_120`, `sm_80`, `sm_87`, `sm_89`, `sm_90`, `sm_90a` |
| `vllm/_flashmla_C.abi3.so` | `sm_100`, `sm_90a` |
| `vllm/_flashmla_extension_C.abi3.so` | `sm_100`, `sm_90a` |
| `vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so` | `sm_80` |
| `vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so` | `sm_90a` |

## Interpretation

This does not prove the current vLLM stack is broken on Spark. The initial personal benchmark run did run several vLLM safetensors rows successfully on GB10.

It does prove that the current installed extension set does not carry explicit `sm_121` SASS in the inspected objects. The general vLLM extensions carry `sm_120`, while some attention/MLA objects are clearly older or datacenter-specific targets.

That is exactly why Spark validation cannot stop at "the package imports" or "the GPU is busy." We need runtime backend logging and per-recipe validation showing which kernels are actually selected on GB10.

## SGLang Follow-Up

This audit covered the installed vLLM/FlashInfer objects in the benchmark venv. It did not audit SGLang because SGLang was not part of the first benchmark stack.

After reviewing `hikarioyama/sglang-nvfp4-kv-sm120`, SGLang should be audited separately once installed or containerized. That implementation is relevant because it routes SGLang NVFP4 (`fp4_e2m1`) KV through FlashInfer FA2 patches and adds native FP4 KV pools, hybrid-SWA wiring, and per-layer global-scale auto-calibration before CUDA graph capture.
