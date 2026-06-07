# NVFP4 KV Porting Map

Status: FlashInfer FA2 KV stride/page patch and vLLM SM12x NVFP4 FA2 routing patch are pushed; SGLang integration and GB10 runtime proof pending.

This map turns the SM120 reference repos into an upstream-shaped port plan. The working priority is:

1. NVFP4 KV cache capacity/concurrency.
2. FP4 weight/prefill GEMM dispatch and packaging.

The KV lane is higher priority because it targets Spark's bottleneck directly: usable memory, context length, and concurrency. Decode speed is still measured, but the first expected win is KV pool size.

## Active Fork Lanes

| lane | fork branch | worktree | base/current commit | reference |
|---|---|---|---|---|
| FlashInfer FA2 NVFP4 KV | `jethac/flashinfer:spark/hijinks-007-fa2-nvfp4-kv-sm121` | `B:/workshop/worktrees/flashinfer/spark-hijinks-007-fa2-nvfp4-kv-sm121` | current `jethac/flashinfer@e152cf4da4ab2a9d093b7d9d4b499198b0211c61`; based on `a42c8f0751c70a2f69596f063170e284710c94ac` | `hikarioyama/vllm-nvfp4-kv-sm120`, `hikarioyama/sglang-nvfp4-kv-sm120` |
| vLLM NVFP4 KV | `jethac/vllm:spark/hijinks-007-nvfp4-kv-sm121` | `B:/workshop/worktrees/vllm/spark-hijinks-007-nvfp4-kv-sm121` | current `jethac/vllm@2c1405dd129d873d268b8baea78c5739cd384951`; based on `vllm-project/vllm@4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa` | `hikarioyama/vllm-nvfp4-kv-sm120@f6156ee3b22b24885a52c02bdafb34a9c201fe86` |
| SGLang FP4 KV | `jethac/sglang:spark/hijinks-018-fp4-e2m1-kv-sm121` | `B:/workshop/worktrees/sglang/spark-hijinks-018-fp4-e2m1-kv-sm121` | current `jethac/sglang@67c7967a1c1b6145a8c9d26a7b941258735ebd8d`; based on `sgl-project/sglang@02be2e71899491b7aaf2849dce6431f61fc190b6` | `hikarioyama/sglang-nvfp4-kv-sm120@9b2160f0fb8e11dbbb5171a57f06a02b0e9ba6e2` |

Reference clones are local scratch only:

- `B:/workshop/scratch/vllm-nvfp4-kv-sm120-reference`
- `B:/workshop/scratch/sglang-nvfp4-kv-sm120-reference`

Do not vendor those overlay trees into production images.

The FlashInfer KV branch intentionally starts from `a42c8f07`, the earlier `spark/hijinks-004-sm121-flashinfer` branch tip. That keeps the SM121 `mm_fp4` dispatch and `121a` JIT-cache work in the ancestry while leaving FA2 NVFP4 KV as a separate, issue-scoped change.

## Current FlashInfer Patch State

Branch `jethac/flashinfer:spark/hijinks-007-fa2-nvfp4-kv-sm121` is pushed at `e152cf4da4ab2a9d093b7d9d4b499198b0211c61`.

It adds the first FA2 NVFP4 KV port layer:

- explicit JIT/codegen scale-factor stride fields for `maybe_k_cache_sf` and `maybe_v_cache_sf`
- FA2 prefill and persistent attention reads using those explicit scale-factor strides instead of deriving them from packed data strides
- independent K/V page strides and V offset helpers in `paged_kv_t`
- separate K and V local offset arrays where K and V storage widths differ
- optional compile-time `FLASHINFER_PAGED_V_SF_DESWIZZLE` support for vLLM-style B2 in-kernel V scale-factor de-swizzle
- default symmetric-linear V scale-factor behavior for SGLang/reference compatibility when the de-swizzle macro is not enabled

Touched files:

- `csrc/batch_decode.cu`
- `csrc/batch_prefill.cu`
- `flashinfer/jit/attention/modules.py`
- `flashinfer/jit/attention/utils.py`
- `include/flashinfer/attention/persistent.cuh`
- `include/flashinfer/attention/prefill.cuh`
- `include/flashinfer/page.cuh`
- `tests/jit/test_attention_utils.py`

Local verification:

- `python -m py_compile flashinfer/jit/attention/utils.py flashinfer/jit/attention/modules.py tests/jit/test_attention_utils.py` passed.
- `git diff --check` passed.
- `python -m pytest tests/jit/test_attention_utils.py` is blocked in this Windows workspace because FlashInfer's test conftest imports `tvm_ffi`, which is not installed.

Missing verification:

- no clean FlashInfer wheel/container build has been run yet for GB10
- no GB10 FA2 NVFP4 KV harness run has passed yet
- no vLLM/SGLang serving proof has selected this path yet

## vLLM Reference Map

Current `jethac/vllm:spark/hijinks-007-nvfp4-kv-sm121` patch:

- commit: `2c1405dd129d873d268b8baea78c5739cd384951`
- branch URL: https://github.com/jethac/vllm/tree/spark/hijinks-007-nvfp4-kv-sm121
- routes SM12x `--kv-cache-dtype nvfp4` through FlashInfer FA2 instead of `trtllm-gen`
- keeps SM100 NVFP4 on the existing TRTLLM path
- uses model dtype query/output on the SM12x FA2 path instead of the TRTLLM FP8-query/FP8-output workaround
- adds a one-time runtime log: `Using FlashInfer FA2 backend for NVFP4 KV cache on SM12x.`
- explicitly rejects DCP for this first SM12x NVFP4 FA2 path
- adds a wrapper-routing regression test for FA2/TRTLLM/auto backend selection
- updates vLLM's attention backend design doc to mention the SM12x FA2 NVFP4 KV route

Local verification:

- `python -m py_compile vllm/v1/attention/backends/flashinfer.py tests/kernels/attention/test_flashinfer_nvfp4_sm12x_routing.py` passed.
- `git diff --cached --check` passed before commit.
- `python -m ruff check ...` is blocked in this Windows workspace because `ruff` is not installed.
- `python -m pytest tests/kernels/attention/test_flashinfer_nvfp4_sm12x_routing.py -q` is blocked in this Windows workspace because vLLM's `tests/conftest.py` imports missing `tblib`.
- PGX source-file routing probe passed on real GB10/SM121:
  - result: `results/vllm_nvfp4_sm12x_routing_probe_20260607T165144Z.json`
  - source revision: `2c1405dd129d873d268b8baea78c5739cd384951`
  - GPU: `NVIDIA GB10`
  - Torch CUDA capability: `[12, 1]`
  - vLLM platform capability: `[12, 1]`
  - SM12x NVFP4 KV prefill/decode wrapper backend: `fa2`
  - SM100-style NVFP4 fallback case still selects `trtllm-gen`
  - non-NVFP4 case still selects `auto`

Missing verification:

- no vLLM wheel/container build has been run yet with `jethac/flashinfer@e152cf4d`
- no GB10 server log has selected the FA2 NVFP4 KV path yet
- no fp8-vs-NVFP4 KV correctness, capacity, quality, or throughput row exists yet

Reference files:

- `src/vllm/v1/attention/backends/flashinfer.py`
- `src/flashinfer/data/include/flashinfer/page.cuh`
- `src/flashinfer/data/include/flashinfer/attention/prefill.cuh`
- `src/flashinfer/jit/attention/utils.py`
- `src/flashinfer/data/csrc/batch_prefill*.cu/.jinja`
- `src/flashinfer/data/csrc/batch_decode*.cu/.jinja`
- `harness/h_layout_b2.py`
- `harness/h_layout_mixed.py`

Likely `jethac/vllm` work:

- Ported in `2c1405d`: route `--kv-cache-dtype nvfp4` to FlashInfer FA2 on SM12x instead of forcing `trtllm-gen`.
- Ported in `2c1405d`: preserve SM100 TRTLLM NVFP4 while adding an explicit SM12x FA2 gate.
- Already present upstream and retained: split NVFP4 data and scale views are passed through `nvfp4_kv_cache_split_views()` and `kv_cache_sf`.
- Ported in `2c1405d`: runtime logging that proves FA2 native NVFP4 KV routing was selected.
- Add serving proof scripts/docs for fp8-vs-NVFP4 KV pool tokens, maximum concurrency, quality, TTFT, and warmed decode.

Likely `jethac/flashinfer` work:

- Ported in `e152cf4d`: explicit scale-factor stride plumbing for `maybe_k_cache_sf` and `maybe_v_cache_sf`.
- Ported in `e152cf4d`: FA2 page layout support for independent K/V page strides and offsets.
- Ported in `e152cf4d`: compile-time gated B2-style V scale-factor de-swizzle; the vLLM branch must enable this path when using the interleaved vLLM cache layout.
- Still pending: GB10 harness coverage for target `H_q/H_kv/D/page` shapes before serving.
- Still pending: clean wheel/container build proof and runtime logs proving native FA2 NVFP4 KV selection.

Patch ownership from the SM120 vLLM reference:

- `01-flashinfer-page-split-kv-offsets.patch`: FlashInfer `paged_kv_t`, independent K/V page strides, and V offset helpers.
- `02-flashinfer-prefill-inkernel-vsf-deswizzle.patch`: FlashInfer prefill path and B2 in-kernel V scale-factor de-swizzle.
- `03-flashinfer-codegen-sf-stride.patch`: FlashInfer JIT/codegen explicit scale-factor stride parameters.
- `04-vllm-flashinfer-sm120-nvfp4-kv.patch`: vLLM backend routing and tensor/view plumbing.
- `harness/h_layout_b2.py`: proof harness only; not runtime code.

First port scope: symmetric NVFP4 B2 only. Exclude the older scratch-cache path and mixed K/V experiments until the symmetric path passes GB10 harness and serving proof.

Current upstream vLLM risk:

- Current main already has NVFP4 KV scaffolding in `vllm/v1/attention/backends/flashinfer.py`, but it still documents that FA2/FA3 do not support NVFP4 and forces `trtllm-gen`.
- The reference repo was pinned to vLLM `0.1.dev16944` and FlashInfer `0.6.11.post2`; current main has drifted. Re-derive the logic; do not patch-apply blindly.

## SGLang Reference Map

Current `jethac/sglang:spark/hijinks-018-fp4-e2m1-kv-sm121` patch:

- commit: `67c7967a1c1b6145a8c9d26a7b941258735ebd8d`
- branch URL: https://github.com/jethac/sglang/tree/spark/hijinks-018-fp4-e2m1-kv-sm121
- allows `flashinfer` in MHA `fp4_e2m1` KV compatibility gates only when `is_sm120_supported()` is true, which covers SM120 and GB10/SM121 in SGLang's helper naming
- lets `NVFP4KVQuantizeUtil.quantize()` run on SM120-family devices and route SM100/SM120 through `flashinfer.nvfp4_kv_quantize`; SM90 keeps the existing `fp4_quantize` fallback
- adds server-args unit coverage for SM12x FlashInfer MHA KV4 acceptance, non-SM12x rejection, and FA4-prefill plus FlashInfer-decode acceptance

Local verification:

- `python -m py_compile python/sglang/srt/server_args.py python/sglang/srt/layers/quantization/kvfp4_tensor.py test/registered/unit/server_args/test_server_args.py` passed.
- `git diff --check` passed.
- `python -m ruff check ...` is blocked in this Windows workspace because `ruff` is not installed.
- `PYTHONPATH=python python -m pytest test/registered/unit/server_args/test_server_args.py -k KV4Compatibility -q` is blocked in this Windows workspace because SGLang imports the POSIX-only `resource` module.

Missing verification:

- no Linux/PGX SGLang unit test run has passed yet for this branch
- no SGLang native FP4 KV memory-pool or FlashInfer backend wrapper patch has landed yet
- no GB10 `fp4_e2m1` serving proof exists yet

Reference files:

- `sglang-src/sglang/srt/server_args.py`
- `sglang-src/sglang/srt/layers/attention/flashinfer_backend.py`
- `sglang-src/sglang/srt/layers/quantization/kvfp4_tensor.py`
- `sglang-src/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`
- `sglang-src/sglang/srt/mem_cache/memory_pool.py`
- `sglang-src/sglang/srt/mem_cache/swa_memory_pool.py`
- `sglang-src/sglang/srt/model_executor/model_runner.py`
- `sglang-src/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `rebase/flashinfer/data/include/flashinfer/page.cuh`
- `rebase/flashinfer/data/include/flashinfer/attention/prefill.cuh`
- `rebase/flashinfer/jit/attention/utils.py`

Likely `jethac/sglang` work:

- Ported in `67c7967`: allow `--kv-cache-dtype fp4_e2m1` with FlashInfer MHA only on SM12x where the reference path is relevant.
- Still pending: wire native FP4 KV pools, separate data and scale buffers, global scales, and scale getters.
- Add or preserve pre-CUDA-graph calibration for NVFP4 KV global scales.
- Delegate FP4 subpools through hybrid-SWA.
- Route FlashInfer backend calls with `kv_cache_sf` and page-size-specific handling.
- Keep SGLang Gemma blockers separate from NVFP4 KV work; Qwen/Step-like models are better first probes.

Likely `jethac/flashinfer` work:

- Same FA2/page/stride work as the vLLM lane, with SGLang layout differences.
- The SGLang reference includes a FlashInfer patch for symmetric linear V scale behavior; it belongs in FlashInfer, not SGLang.

Patch ownership from the SM120 SGLang reference:

- SGLang owns server args, FlashInfer backend wrapper, `kvfp4_tensor`, FP4 KV quant method, MHA KV memory pools, SWA pool delegation, model-runner calibration, and model-runner KV-cache mixin changes.
- FlashInfer owns `page.cuh`, `prefill.cuh`, `jit/attention/utils.py`, and the symmetric linear V scale behavior even when a patch filename is under the SGLang overlay.

Current upstream SGLang risk:

- Current main already has partial `fp4_e2m1` scaffolding in KV-pool setup and quantization modules.
- Missing or unproven pieces are native FlashInfer `kv_cache_sf`, SWA FP4 delegation, startup calibration, and GB10 proof.
- The reference overlay was based on a nightly/dev image, not stable release code. Re-derive against the current fork branch.

## Smallest GB10 Proof Sequence

1. Keep existing NVIDIA SGLang 26.05 Qwen BF16 and vLLM 26B/12B rows as baselines; do not merge them into NVFP4-KV claims.
2. Port FlashInfer FA2 stride/page work into a clean `jethac/flashinfer` branch.
3. Run the reference layout harness for the target model shape on GB10 with a fresh JIT cache.
4. Start vLLM with `--kv-cache-dtype nvfp4`; logs must prove FlashInfer FA2 native NVFP4 KV, not fp8/bf16 fallback or trtllm-gen.
5. Compare fp8 and NVFP4 on the same model, prompts, memory utilization, CUDA graph mode, and concurrency.
6. Record KV pool tokens, maximum concurrency, memory telemetry, deterministic output, quality/PPL or retrieval sanity, TTFT, and warmed decode tok/s.
7. Only after vLLM proof, port or validate the SGLang-specific pool/calibration path.

Passing harness-only evidence is not enough for a PR. A Spark NVFP4 KV PR needs clean wheel/container evidence and a serving proof packet.
