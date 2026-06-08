# llama.cpp NVFP4 Runtime Gate, 2026-06-08

Status: first runtime NVFP4 GGUF smoke passed on GB10 `sm_121`; broader correctness and
speed remain unproven.

## Scope

This stop point continues the llama.cpp native FP4 lane after
`results/llamacpp_native_fp4_arch_20260608T164917JST_summary.md`.

The previous artifact proved that `jethac/llama.cpp@19bba67c1` builds/emits native
`mxf4nvf4.block_scale` PTX for `sm_121a`. This run moved from build proof to an actual
NVFP4 GGUF:

- source: `/home/jethac/src/llama.cpp-native-fp4-sm121-19bba67`
- source commit: `19bba67c1f4db723c60a0d421aa0788bf4ddc699`
- binary: `build-native-fp4-121a-20260608T164933JST/bin/llama-server`
- model source: `/home/jethac/models/aeon/qwen36-nvfp4`
- generated GGUF: `/home/jethac/spark_tmp/llamacpp_nvfp4_runtime_gate_20260608T1748JST/qwen36-nvfp4-nvfp4.gguf`
- host: `thinkstationpgx-00b4`, `NVIDIA GB10`, compute capability `12.1`

## Conversion

The cached AEON Qwen3.6 NVFP4 checkpoint was local already; no model download was run.

Dry run:

```bash
/home/jethac/gemma4-evals/.venv/bin/python \
  /home/jethac/src/llama.cpp-native-fp4-sm121-19bba67/convert_hf_to_gguf.py \
  /home/jethac/models/aeon/qwen36-nvfp4 \
  --outtype bf16 \
  --dry-run \
  --no-mtp \
  --outfile /home/jethac/spark_tmp/llamacpp_nvfp4_runtime_gate_20260608T1748JST/qwen36-nvfp4-{ftype}.gguf
```

Result: planned `qwen36-nvfp4-nvfp4.gguf`, `n_tensors = 1293`, `total_size = 22.5G`.

Full conversion:

```bash
/usr/bin/time -f "elapsed=%E maxrss_kb=%M" \
  /home/jethac/gemma4-evals/.venv/bin/python \
  /home/jethac/src/llama.cpp-native-fp4-sm121-19bba67/convert_hf_to_gguf.py \
  /home/jethac/models/aeon/qwen36-nvfp4 \
  --outtype bf16 \
  --no-mtp \
  --outfile /home/jethac/spark_tmp/llamacpp_nvfp4_runtime_gate_20260608T1748JST/qwen36-nvfp4-{ftype}.gguf
```

Result: exported a 21 GiB local GGUF in `elapsed=1:36.23`, `maxrss_kb=19366788`.

## Runtime Smoke

Command shape:

```bash
/home/jethac/src/llama.cpp-native-fp4-sm121-19bba67/build-native-fp4-121a-20260608T164933JST/bin/llama-server \
  --model /home/jethac/spark_tmp/llamacpp_nvfp4_runtime_gate_20260608T1748JST/qwen36-nvfp4-nvfp4.gguf \
  --alias qwen36-nvfp4-gguf \
  --host 127.0.0.1 \
  --port 18087 \
  -ngl 999 \
  -c 2048 \
  --threads 8 \
  --no-webui \
  --reasoning off \
  --chat-template-kwargs '{"enable_thinking":false}'
```

Server log evidence:

- `CUDA : ARCHS = 1210`
- `BLACKWELL_NATIVE_FP4 = 1`
- loaded `/home/jethac/spark_tmp/llamacpp_nvfp4_runtime_gate_20260608T1748JST/qwen36-nvfp4-nvfp4.gguf`
- `chat template, thinking = 0`
- `model loaded`

Chat smoke:

```json
{"role":"assistant","content":"The capital of Japan is Tokyo."}
```

Timing from this tiny smoke:

- prompt: `25` tokens, `59.77 tok/s`
- decode: `8` tokens, `52.69 tok/s`

These timings are not a benchmark row.

## Profiler Dispatch Evidence

`nsys profile --trace=cuda,nvtx` captured one reasoning-off request:

- remote report: `/home/jethac/spark_tmp/llamacpp_nvfp4_runtime_gate_20260608T1748JST/nsys_llamacpp_nvfp4.nsys-rep`
- compact summary copied here: `results/llamacpp_nvfp4_runtime_gate_20260608T1748JST/nsys_cuda_gpu_kern_sum.txt`

Key kernel-summary lines:

- `void mul_mat_q<(ggml_type)40, (int)24, (bool)0>`: `280` instances, `19.7%` GPU kernel time
- `quantize_mmq_nvfp4(...)`: `280` instances
- `nvjet_sm121_tst_mma_96x64x64_4_16x64x64_tmaAB_bz_TNNN`: `30` instances
- `nvjet_sm121_tst_mma_32x64x64_8_16x32x64_tmaAB_splitK_TNNN`: `60` instances

In this llama.cpp pin, `GGML_TYPE_NVFP4 = 40`, and the source gate in
`ggml/src/ggml-cuda/mmq.cu` selects native FP4 MMQ when:

```text
blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4)
```

The profiler therefore proves the runtime executed NVFP4 (`ggml_type 40`) quantized
matmul kernels on the `sm_121` host. Treat the `nvjet_sm121_tst_mma...` entries as
supporting evidence of SM121 tensor-core MMA activity, not as a standalone mapping to a
specific llama.cpp source function.

## Copied Artifacts

- `convert_dry_run.stdout`
- `convert.stderr`
- `server_reasoning_off.log`
- `chat_reasoning_off.json`
- `nsys_status.txt`
- `nsys_chat.json`
- `nsys_cuda_gpu_kern_sum.txt`

Large remote-only artifacts intentionally not copied:

- `qwen36-nvfp4-nvfp4.gguf` (21 GiB)
- `nsys_llamacpp_nvfp4.nsys-rep` (836 KiB, available remotely if needed)
- `nsys_llamacpp_nvfp4.sqlite`

## Claim Boundary

Proven:

- A cached HF NVFP4 checkpoint can be converted locally to an NVFP4 GGUF with the pinned
  `jethac/llama.cpp@19bba67c1` converter.
- The pinned `sm_121a` llama-server loads that NVFP4 GGUF on GB10.
- A small reasoning-off chat request returns correct content.
- Nsight Systems sees `GGML_TYPE_NVFP4` (`ggml_type 40`) quantized matmul kernels and
  `quantize_mmq_nvfp4` during the request.

Still unproven:

- Accuracy versus BF16/Q8 reference logits or an eval task.
- Prefill/decode speed versus Q4_K_M, Q8, or BF16 under matched prompts and contexts.
- Whether every NVFP4 matmul uses native block-scale MMA rather than mixed/fallback kernels.
- Long-context behavior, MoE routing quality, and repeatability beyond one short smoke.
