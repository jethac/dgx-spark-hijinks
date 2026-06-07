# Campaign Log

## 2026-06-07

- Created this public record repository.
- Imported the first Gemma 4 benchmark reports.
- Added the initial diagnosis and solution plan.
- Added first-pass tools:
  - `scripts/spark_doctor.py`
  - `scripts/gguf_logprobs_probe.py`
  - `scripts/openai_chat_smoke.py`
- Added SGLang as a first-class runtime workstream after reviewing `hikarioyama/sglang-nvfp4-kv-sm120`.
- Added practical llama.cpp serving as a first-class runtime workstream and LiteRT-LM as optional side-runtime coverage.
- Added the upstream fork/submodule/worktree policy for patched libraries.
- Added before/after GB10 benchmark protocol for measuring campaign impact.
- Added NVFP4 dependency map from the FlashInfer/vLLM/SGLang subagent investigation.
- Added a remediation matrix clarifying that the FlashInfer SM121 `mm_fp4` patch is only one workstream; remaining work includes packaging, vLLM/SGLang integration, Gemma 4 12B, NVFP4 KV, llama.cpp/lm-eval, optional LiteRT-LM, HF fallback, observability, benchmark proof, and upstream coordination.
- Added a compact OpenAI-compatible serving benchmark harness and captured the first vLLM before row.
- Added root-readable runtime process probe evidence for the live vLLM server.
- Added `scripts/spark_smoke_suite.py` as the compact before/after orchestrator. Core tracks are vLLM, SGLang, llama.cpp, HF fallback telemetry, MTP/spec decode, and NVFP4; LiteRT-LM is opt-in.
- Reproduced the llama.cpp GGUF logprobs incompatibility with a temporary `llama-server` on port `18081`; server was stopped after the probe.
- Captured runtime availability matrix showing vLLM/FlashInfer/PyTorch present, SGLang and LiteRT-LM absent, Docker available, and llama.cpp built but not on `PATH`.
- Started tracking work through GitHub Issues.
- Ran `spark_doctor` on `thinkstationpgx-00b4` using `/home/jethac/gemma4-evals/.venv/bin/python`.
  - GPU: `NVIDIA GB10`
  - compute capability: `12.1` / `sm_121`
  - host: `aarch64`
  - driver: `580.159.03`
  - CUDA runtime reported by `nvidia-smi`: `13.0`
  - `nvcc`: CUDA `13.0`, found through `/usr/local/cuda`
  - `cuobjdump`: CUDA `13.0`, found through `/usr/local/cuda`
  - PyTorch: `2.11.0+cu130`
  - vLLM: `0.22.1`
  - FlashInfer: `0.6.11.post2`
  - PyTorch arch list: `sm_80`, `sm_90`, `sm_100`, `sm_110`, `sm_120`; no explicit `sm_121`
  - snapshot: `results/spark_doctor_20260607T110833Z.md`
- Ran `cuda_so_audit` against vLLM and FlashInfer package roots in the benchmark venv.
  - inspected objects: 14
  - objects with explicit `sm_121`: 0
  - objects with `sm_120`: 3
  - vLLM FA2 extension: `sm_80`
  - vLLM FA3 extension: `sm_90a`
  - vLLM FlashMLA extensions: `sm_100`, `sm_90a`
  - snapshot: `results/cuda_so_audit_vllm_flashinfer_20260607T111023Z.json`
- Proved llama.cpp `b9536` as a practical Gemma 4 26B Q4_0 serving path with `--reasoning off`.
- Evaluated LiteRT-LM `0.13.1` on Linux `aarch64`: CPU chat works for Gemma 4 E2B, CPU/GPU benchmark commands work, GPU chat still exits `returncode=-11` after producing `spark-ok`.
- Ran SGLang 26.05 model-shaped FlashInfer `mm_fp4` proxy microbenchmarks against installed and patched source/JIT FlashInfer.
  - patched source selected `b12x`, `cutlass`, `cudnn` and built an SM121a-targeted FP4 GEMM path.
  - patched dense-decode proxies were mixed; patched MoE-shaped proxies were slower on all tested shapes.
  - conclusion: the FlashInfer SM121 predicate patch remains dispatch enablement, not a proven speedup.
- Served Gemma 4 12B through a vLLM source/precompiled probe at upstream commit `da1daf40` plus Transformers main after removing stale FlashInfer JIT-cache files.
  - the run proved `Gemma4UnifiedForConditionalGeneration` can start on GB10, but compact decode was only about 7.7 tok/s and vLLM forced `TRITON_ATTN`.
  - conclusion: this is a compatibility and packaging proof, not a clean blessed container or performance win.
- Audited hikarioyama's vLLM and SGLang NVFP4-KV SM120 reference repos.
  - vLLM reference HEAD: `f6156ee3b22b24885a52c02bdafb34a9c201fe86`.
  - SGLang reference HEAD: `9b2160f0fb8e11dbbb5171a57f06a02b0e9ba6e2`.
  - conclusion: build on them as prior art through `jethac` forks, but do not vendor overlays or call them Spark validation until GB10 `sm_121` fp8-vs-NVFP4 proof exists.
  - priority shift: vLLM NVFP4 KV is a capacity/concurrency lane first. Measure KV pool tokens, maximum concurrency, quality, and hidden allocations before chasing decode tok/s.
- Tightened the smoke suite:
  - `run_with_telemetry.py` now captures `pre_memory` before launching the child command.
  - `spark_smoke_suite.py` wraps MTP/spec-decode commands with telemetry and supports `--mtp-model`.
- Created `jethac/vllm` and `jethac/sglang` forks, added them as submodules, and pushed issue-named worktree branches for NVFP4 KV work.
  - vLLM branch: `spark/hijinks-007-nvfp4-kv-sm121` at `4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa`.
  - SGLang branch: `spark/hijinks-018-fp4-e2m1-kv-sm121` at `02be2e71899491b7aaf2849dce6431f61fc190b6`.
- Created the FlashInfer FA2 NVFP4 KV branch `spark/hijinks-007-fa2-nvfp4-kv-sm121` from `a42c8f0751c70a2f69596f063170e284710c94ac`, so the KV lane inherits the earlier SM121 `mm_fp4` dispatch and `121a` JIT-cache work.
- Recorded the NVFP4 KV porting map from the vLLM and SGLang subagent audits:
  - build on hikarioyama's SM120 work as prior art, but re-derive minimal upstream-shaped patches in `jethac` forks.
  - keep FlashInfer kernel/page/stride changes in FlashInfer, vLLM routing/tensor plumbing in vLLM, and SGLang memory-pool/calibration/backend-wrapper changes in SGLang.
  - first port scope is symmetric NVFP4 B2 only; no hidden V scale-factor scratch cache and no mixed K/V experiments until GB10 proof exists.
- Pushed FlashInfer FA2 NVFP4 KV patch branch `spark/hijinks-007-fa2-nvfp4-kv-sm121` at `e152cf4da4ab2a9d093b7d9d4b499198b0211c61`.
  - added explicit scale-factor stride plumbing for `maybe_k_cache_sf` and `maybe_v_cache_sf`.
  - added independent K/V page strides and V offset helpers in `paged_kv_t`.
  - updated FA2 prefill and persistent attention to consume explicit scale-factor strides.
  - added compile-time gated vLLM B2 V scale-factor de-swizzle while preserving symmetric-linear V scale-factor behavior by default for SGLang/reference compatibility.
  - verification: Python syntax compile and `git diff --check` passed.
  - limitation: targeted pytest collection is blocked in this Windows workspace by missing `tvm_ffi`; no clean GB10 build, harness proof, or serving proof yet.
- Pushed vLLM SM12x NVFP4 KV routing patch branch `spark/hijinks-007-nvfp4-kv-sm121` at `2c1405dd129d873d268b8baea78c5739cd384951`.
  - routes SM12x `--kv-cache-dtype nvfp4` through FlashInfer FA2 instead of `trtllm-gen`.
  - keeps SM100 NVFP4 on the existing TRTLLM path.
  - uses model dtype query/output on the SM12x FA2 path instead of the TRTLLM FP8-query/FP8-output workaround.
  - adds a one-time runtime log for FA2 NVFP4 KV selection.
  - adds wrapper-routing regression coverage and updates vLLM's attention backend design doc.
  - verification: Python syntax compile and staged `git diff --check` passed.
  - limitations: local `ruff` is unavailable and pytest collection is blocked by missing `tblib`; no clean GB10 vLLM plus FlashInfer build or serving proof yet.
- Ran a PGX source-file routing probe for `jethac/vllm@2c1405dd129d873d268b8baea78c5739cd384951`.
  - result: `results/vllm_nvfp4_sm12x_routing_probe_20260607T165144Z.json`
  - host/GPU: `thinkstationpgx-00b4`, `NVIDIA GB10`, capability `[12, 1]`
  - installed dependency context: vLLM `0.22.1`, Torch `2.11.0+cu130`, CUDA `13.0`
  - outcome: SM12x NVFP4 KV wrapper routing selects FlashInfer `fa2`; SM100-style NVFP4 still selects `trtllm-gen`; non-NVFP4 still selects `auto`
  - limitation: this loads the forked `flashinfer.py` source file against installed compiled dependencies. It does not install the full fork, build FlashInfer kernels, start a server, or prove correctness/capacity/performance.
- Pushed vLLM V-scale-factor deswizzle follow-up at `8916796bc50926fd61e606718b194a71e2e31a24`.
  - reason: vLLM's native NVFP4 cache writer stores V scale factors in a swizzled layout for the old SM100 TRTLLM path, so the SM12x FA2 path must enable FlashInfer's in-kernel V-SF deswizzle variant.
  - scope: still keyed only on `kv_cache_dtype == "nvfp4"` and the SM12x consumer-Blackwell family helper; fp8/auto routing remains unchanged.
  - family note: vLLM reporting GB10 as capability family `120` is correct for FA2 NVFP4 KV routing, but native FP4/MXFP4 MMA work still needs Spark-appropriate `sm_121a` or validated compatible targets.
- Ran the scripted PGX routing/deswizzle probe for `jethac/vllm@8916796bc50926fd61e606718b194a71e2e31a24`.
  - result: `results/vllm_nvfp4_sm12x_routing_probe_20260607T171227Z.json`
  - outcome: SM12x NVFP4 KV wrapper routing selects FlashInfer `fa2`, SM100-style NVFP4 still selects `trtllm-gen`, non-NVFP4 still selects `auto`, and the deswizzle helper sets `-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1`.
  - limitation: this is still routing/JIT-flag evidence only. It does not replace hikari-style NHD/HND cosine checks or an end-to-end serving proof.
- Pushed SGLang SM12x FP4 KV gate patch branch `spark/hijinks-018-fp4-e2m1-kv-sm121` at `67c7967a1c1b6145a8c9d26a7b941258735ebd8d`.
  - allows FlashInfer MHA in `fp4_e2m1` KV compatibility gates only when SGLang's `is_sm120_supported()` helper is true.
  - allows `NVFP4KVQuantizeUtil.quantize()` on SM120-family devices and routes SM100/SM120 through `flashinfer.nvfp4_kv_quantize`.
  - adds server-args unit coverage for SM12x FlashInfer MHA KV4 gates.
  - verification: Python syntax compile and `git diff --check` passed.
  - limitations: local `ruff` is unavailable and pytest collection is blocked on Windows by missing POSIX `resource`; native FP4 KV memory-pool/backend wrapper work and GB10 serving proof are still pending.
- Ran a PGX/Linux verification pass for `jethac/sglang@67c7967a1913960055e64c49c26c5f622c1f1ff1`.
  - result: `results/sglang_fp4_kv_sm121_pgx_verify_20260608T0205JST.md`
  - outcome: Linux/aarch64 branch fetch and detached worktree checkout passed; `python3 -m py_compile` passed for the touched SGLang files.
  - limitation: targeted `KV4Compatibility` pytest was not run because PGX has no `python` shim and `python3` does not have `pytest` installed.

## First Benchmark Campaign Summary

The initial personal Gemma 4 benchmark run was run on `thinkstationpgx-00b4` in `/home/jethac/gemma4-evals`.

At the last local sync:

- smoke rows: 152 complete
- smoke `ok`: 21
- smoke `eval_failed`: 11
- smoke `loader_failed`: 120
- full eval records: 70
- full eval `ok`: 65
- full eval failed: 5
- throughput rows observed: 2
- MTP rows observed: 2

That personal benchmark run was still in full accuracy when monitoring stopped. It was not killed.
