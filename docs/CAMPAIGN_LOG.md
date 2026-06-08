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
- Ran a GB10 source-file routing probe for `jethac/vllm@2c1405dd129d873d268b8baea78c5739cd384951`.
  - result: `results/vllm_nvfp4_sm12x_routing_probe_20260607T165144Z.json`
  - host/GPU: `thinkstationpgx-00b4`, `NVIDIA GB10`, capability `[12, 1]`
  - installed dependency context: vLLM `0.22.1`, Torch `2.11.0+cu130`, CUDA `13.0`
  - outcome: SM12x NVFP4 KV wrapper routing selects FlashInfer `fa2`; SM100-style NVFP4 still selects `trtllm-gen`; non-NVFP4 still selects `auto`
  - limitation: this loads the forked `flashinfer.py` source file against installed compiled dependencies. It does not install the full fork, build FlashInfer kernels, start a server, or prove correctness/capacity/performance.
- Pushed vLLM V-scale-factor deswizzle follow-up at `8916796bc50926fd61e606718b194a71e2e31a24`.
  - reason: vLLM's native NVFP4 cache writer stores V scale factors in a swizzled layout for the old SM100 TRTLLM path, so the SM12x FA2 path must enable FlashInfer's in-kernel V-SF deswizzle variant.
  - scope: still keyed only on `kv_cache_dtype == "nvfp4"` and the SM12x consumer-Blackwell family helper; fp8/auto routing remains unchanged.
  - family note: vLLM reporting GB10 as capability family `120` is correct for FA2 NVFP4 KV routing, but native FP4/MXFP4 MMA work still needs Spark-appropriate `sm_121a` or validated compatible targets.
- Ran the scripted GB10 routing/deswizzle probe for `jethac/vllm@8916796bc50926fd61e606718b194a71e2e31a24`.
  - result: `results/vllm_nvfp4_sm12x_routing_probe_20260607T171227Z.json`
  - outcome: SM12x NVFP4 KV wrapper routing selects FlashInfer `fa2`, SM100-style NVFP4 still selects `trtllm-gen`, non-NVFP4 still selects `auto`, and the deswizzle helper sets `-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1`.
  - limitation: this is still routing/JIT-flag evidence only. It does not replace hikari-style NHD/HND cosine checks or an end-to-end serving proof.
- Pushed SGLang SM12x FP4 KV gate patch branch `spark/hijinks-018-fp4-e2m1-kv-sm121` at `67c7967a1c1b6145a8c9d26a7b941258735ebd8d`.
  - allows FlashInfer MHA in `fp4_e2m1` KV compatibility gates only when SGLang's `is_sm120_supported()` helper is true.
  - allows `NVFP4KVQuantizeUtil.quantize()` on SM120-family devices and routes SM100/SM120 through `flashinfer.nvfp4_kv_quantize`.
  - adds server-args unit coverage for SM12x FlashInfer MHA KV4 gates.
  - verification: Python syntax compile and `git diff --check` passed.
  - limitations: local `ruff` is unavailable and pytest collection is blocked on Windows by missing POSIX `resource`; native FP4 KV memory-pool/backend wrapper work and GB10 serving proof are still pending.
- Ran a Linux/aarch64 verification pass on the GB10 host for `jethac/sglang@67c7967a1913960055e64c49c26c5f622c1f1ff1`.
  - result: `results/sglang_fp4_kv_sm121_pgx_verify_20260608T0205JST.md`
  - outcome: Linux/aarch64 branch fetch and detached worktree checkout passed; `python3 -m py_compile` passed for the touched SGLang files.
  - limitation: targeted `KV4Compatibility` pytest was not run because the host has no `python` shim and `python3` does not have `pytest` installed.
- Added SM-count awareness to diagnostics and benchmark harnesses.
  - new helper: `scripts/spark_hardware.py`
  - updated scripts: `spark_doctor.py`, `run_with_telemetry.py`, `openai_serving_benchmark.py`, `spark_smoke_suite.py`
  - new evidence: `results/spark_doctor_smcount_20260607T172142Z.md`
  - current GB10 comparison key: `NVIDIA_GB10:sm_121:sms_48`
  - audit doc: `docs/SM_COUNT_AWARENESS.md`
  - conclusion: current fork patches do not hardcode a 48-SM performance heuristic; future performance rows must still match `multi_processor_count`.
- Added and ran a standalone FlashInfer FA2 NVFP4 paged-KV correctness probe.
  - new script: `scripts/flashinfer_nvfp4_kv_probe.py`
  - result: `results/flashinfer_nvfp4_kv_probe_20260608T023901JST.json`
  - source: `jethac/flashinfer@e152cf4da4ab2a9d093b7d9d4b499198b0211c61`
  - env: `FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1`
  - hardware key: `NVIDIA_GB10:sm_121:sms_48`
  - outcome: NHD decode, NHD prefill, HND decode, and HND prefill all passed with cosine >= `0.99999946`.
  - limitation: this proves the patched FlashInfer FA2 kernel path and vLLM-style V-scale-factor de-swizzle, not clean vLLM/SGLang serving, KV capacity, quality, or throughput.
- Tightened llama.cpp wording.
  - Gemma 4 26B Q4_0 remains the blessed practical GGUF serving path.
  - The Q4_0 result does not prove native NVFP4/MXFP4 `sm_121a` tensor-core MMA dispatch; that needs a separate native-FP4 GGUF experiment.
- Tried a CPU-only Linux `aarch64` Docker verification route for the SGLang FP4 KV branch to avoid spending GPU time on Python-level gate tests.
  - result: `results/sglang_fp4_kv_sm121_cpu_docker_verify_20260608T0243JST.md`
  - target: `jethac/sglang@67c7967a1913960055e64c49c26c5f622c1f1ff1`
  - outcome: build failed before pytest while compiling `sglang-kernel-cpu`; the ARM64 `vaddq_f16` path hit a target-specific option mismatch.
  - conclusion: this does not disprove the SGLang FP4 KV gate patch, but the cheap CPU Docker verification route needs either a no-kernel pytest image or an ARM64 CPU-kernel build-flag fix.
- Fixed and validated the SGLang FP4 KV gate test setup.
  - fork commit: `jethac/sglang@eefe8aded`
  - result: `results/sglang_fp4_kv_sm121_pytest_20260608T0320JST.md`
  - outcome: targeted Linux `aarch64` `KV4Compatibility` pytest passed: `3 passed, 56 deselected`.
  - limitation: this is Python-level argument compatibility only; native FP4 KV pools, backend wrapper plumbing, quality, capacity, and serving remain pending.
- Added `scripts/cuda_build_target_audit.py` for build/JIT log target evidence before `.so` inspection.
  - first smoke artifact: `results/llamacpp_gemma4_26b_q4_0_build_target_audit_20260608T0325JST.json`
  - result: the existing llama.cpp server log contains accepted Spark target evidence through `CUDA : ARCHS = 1210`.
- Ran Gemma 4 26B-shaped FlashInfer FA2 NVFP4 KV probes.
  - sliding/local artifact: `results/flashinfer_nvfp4_kv_probe_gemma4_26b_sliding_1024_20260608T0340JST.json`
  - global/full artifact: `results/flashinfer_nvfp4_kv_probe_gemma4_26b_global_20260608T0335JST.json`
  - outcome: sliding/local shape `H_q=16`, `H_kv=8`, `D=256`, `page=16` passed NHD/HND decode and prefill at `kv_len=1024`, `qo_len=128`.
  - blocker: global/full shape `H_q=16`, `H_kv=2`, `D=512`, `page=16` failed all NHD/HND decode/prefill operations with FlashInfer FA2 invalid configuration from `prefill.cuh:3215`.
  - conclusion: Gemma 4 26B NVFP4 KV cannot be called ready for serving until the global-attention path is fixed or routed to a proven fallback.
- Promoted Qwen to a first-class speed/capacity benchmark lane.
  - issue: https://github.com/jethac/dgx-spark-hijinks/issues/20
  - new doc: `docs/QWEN_ON_DGX_SPARK.md`
  - rationale: Qwen is the cleaner path for NVFP4 weights, fp8-vs-NVFP4 KV capacity, and DFlash measurement; Gemma remains the harder model-family compatibility target.
  - AEON prior art: Qwen3.6 NVFP4+DFlash and Gemma 4 NVFP4-weight recipes are useful external GB10 evidence, but they do not prove our FA2 NVFP4-KV fork or SGLang `fp4_e2m1` KV.
- Pushed vLLM Qwen/DFlash SM12x stability branch.
  - fork commit: `jethac/vllm@0667185d5adaec32ff8cc8289a4d7716f6cdf966`
  - branch: `spark/hijinks-020-aeon-qwen-dflash-sm121a`
  - changes: guarded lazy fallback import for `_C_stable_libtorch`; speculative-decode CUDA graph capture-size alignment now applies to every non-`NONE` graph mode, including pure `PIECEWISE`.
  - verification artifact: `results/vllm_qwen_dflash_sm121a_patch_verify_20260608T0330JST.md`
  - limitation: local pytest collection is blocked by missing vLLM dev dependencies; GB10 Qwen3.6 NVFP4+DFlash serving reproduction remains pending.
- Captured SGLang Qwen2.5 1.5B fp8-vs-fp4 KV evidence.
  - summary: `results/sglang_qwen25_1_5b_fp8_vs_fp4kv_20260608T0332JST_summary.md`
  - fp8 row: NVIDIA SGLang 26.05 serves `Qwen/Qwen2.5-1.5B-Instruct` with FlashInfer attention, CUDA graphs enabled, hardware key `NVIDIA_GB10:sm_121:sms_48`, `3,113,713` KV tokens, and about `58-59 tok/s` decode.
  - stock fp4 FlashInfer row: fails at `KV4Compatibility`, rejecting FlashInfer for MHA FP4 KV.
  - stock fp4 Triton row: allocates `5,534,509` KV tokens, about `1.78x` fp8 capacity, then fails on missing `KVFP4QuantizeUtil`.
  - conclusion: fp8 is now a real SGLang Qwen comparator; stock fp4 KV is not a serving path yet, and the next after-row should use the `jethac/sglang` fork.
- Captured the matched SGLang Qwen BF16/auto comparator at `mem_fraction_static=0.40`.
  - artifacts: `results/sglang_qwen25_1_5b_bf16auto_040mem_20260608T0409JST_openai_benchmark.json` and server log.
  - result: BF16/auto KV allocated `1,557,709` tokens and decoded `58.89`, `58.59`, and `57.73 tok/s` across the standard short, medium, and long-prefill cases.
  - conclusion: fp8 roughly doubles KV pool tokens over BF16/auto at matched memory fraction without materially changing decode speed on this small Qwen row.
- Pushed the SGLang FP4 KV alias fix and ran patched overlay serving attempts.
  - fork commit: `jethac/sglang@98ad46961`.
  - change: `KVFP4QuantizeUtil` is now an alias of `BlockFP4KVQuantizeUtil`, matching the historical import used by the MHA KV memory-pool path.
  - FlashInfer attention overlay: clears the stock compatibility/import failures, allocates `5,539,718` FP4 KV tokens, targets `compute_121a,code=sm_121a`, then fails compiling FlashInfer FP4 decode at `vec_dtypes.cuh(117)`.
  - Triton attention overlay: normal graph capture stalls; disabling only standard CUDA graphs still enters piecewise graph capture and stalls; disabling both graph modes serves.
  - no-graphs Triton FP4 KV result: `5,541,103` KV tokens, smoke passes, but `short_decode` is only `0.276 tok/s` with repetitive output.
  - conclusion: the FP4 KV capacity path is real, but SGLang FP4 KV remains unblessed until the clean fork/dependency stack serves with graphs and acceptable quality.
- Added `scripts/record_openai_serving_row.py`.
  - purpose: capture smoke, benchmark, optional runtime/CUDA audits, and a manifest for one already-running OpenAI-compatible server.
  - verification: `python -m py_compile` passed and `--dry-run` emits relative artifact paths and portable command records.
- Captured llama.cpp Qwen2.5 1.5B Q4_K_M serving evidence.
  - model: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`, file `qwen2.5-1.5b-instruct-q4_k_m.gguf`.
  - binary: `/home/jethac/src/llama.cpp-b9536/build/bin/llama-server`, build `308f61c31 (9536)`.
  - artifact prefix: `results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST`.
  - server log: `NVIDIA GB10`, `CUDA : ARCHS = 1210`, `USE_GRAPHS = 1`, `BLACKWELL_NATIVE_FP4 = 1`.
  - compact OpenAI serving decode: `175.19`, `174.86`, and `166.66 tok/s` on short, medium, and long-prefill cases.
  - `llama-bench`: `pp512 12505.79 +/- 615.87 tok/s`, `tg128 178.10 +/- 0.95 tok/s`.
  - logprobs probe: still not lm-eval compatible; response exposes `logprobs.content` but not `tokens` and `token_logprobs`.
  - conclusion: practical Qwen GGUF serving is proven for llama.cpp on GB10; GGUF accuracy and native FP4/MXFP4 dispatch remain separate workstreams.
- Added the AEON vLLM reproduction runner and preflight.
  - new script: `scripts/run_aeon_vllm_reproduction.sh`
  - targets: `gemma26-dflash` and `qwen36-dflash`
  - preflight artifact: `results/aeon_vllm_reproduction_preflight_20260608T0430JST.md`
  - outcome: both AEON GHCR images resolve; Gemma/Qwen target and drafter HF repos are public and non-gated from the GB10 host.
  - limitation: no large model download or vLLM serving row has been started yet; this is readiness evidence for the next benchmark, not a performance result.
- Reproduced the AEON Gemma 4 26B NVFP4+DFlash vLLM row locally.
  - run ID: `aeon_gemma26_dflash_20260608T0436JST`
  - image: `ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2`
  - model: `AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4`
  - drafter: `z-lab/gemma-4-26B-A4B-it-DFlash`
  - artifact summary: `results/aeon_gemma26_dflash_20260608T0436JST_summary.md`
  - warmed compact row: `47.91`, `53.60`, and `98.38 tok/s` across short, medium, and long-prefill cases.
  - backend evidence: `FlashInferCutlassNvFp4LinearKernel`, `VLLM_CUTLASS` NvFp4 MoE, target `TRITON_ATTN`, drafter `FLASH_ATTN`, CUDA graphs, DFlash.
  - caveats: not a `jethac` fork speedup claim; PyTorch arch list reports `sm_120` but not explicit `sm_121`; server log warns about differing NVFP4 global scales across fused parallel layers; accuracy remains separate.
- Attempted the AEON Qwen3.6 NVFP4+DFlash vLLM reproduction.
  - run ID: `aeon_qwen36_dflash_20260608T0501JST`
  - model downloaded: `/home/jethac/models/aeon/qwen36-nvfp4`, about `22G`
  - drafter downloaded: `/home/jethac/models/aeon/qwen36-dflash`, about `905M`
  - blocker: `ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2` did not finish/register after the initial pull or a bounded `timeout 900` retry.
  - artifact: `results/aeon_qwen36_dflash_20260608T0501JST_summary.md`
  - interpretation: this is a container acquisition blocker, not yet a Qwen model-load, runtime, or kernel failure.
- Promoted the AEON Qwen vLLM runner default to the current `v2` image and recorded a stop point.
  - runner default: `ghcr.io/aeon-7/vllm-spark-omni-q36:v2`
  - artifact: `results/aeon_qwen36_dflash_v2_20260608T0555JST_stop_point.md`
  - blocker: the `v2` pull had been started and partially observed, but the GB10 host later stopped answering SSH and ping before final pull status could be inspected.
  - interpretation: Qwen speed remains a required benchmark lane alongside Gemma; this stop point is an acquisition/reachability failure, not a Qwen model/runtime/kernel result.
- Advanced the vLLM Qwen/DFlash branch to cover the remaining AEON source patches that apply to the current fork.
  - fork commit: `jethac/vllm@6804e1b81e6ea2ca53bb5021151bdad0f201b11d3`
  - artifact: `results/vllm_aeon_qwen_patch_port_20260608T0619JST.md`
  - changes: Qwen3.5/3.6 text registry entries, hybrid KV `block_size=None` safety, Mamba block-size fallback, and text-only M-RoPE fallback, on top of the existing lazy import and CUDA graph alignment fixes.
  - validation: Python compile and `git diff --check` passed; targeted pytest is blocked by missing local vLLM test dependency `tblib`.
  - interpretation: the fork is closer to the AEON Qwen runtime shape, but it is still not a Qwen speed result. The required next benchmark is still AEON Qwen36 NVFP4+DFlash serving, then a matched `jethac` fork row.
- Extended the llama.cpp native loglikelihood probe toward a GGUF lm-eval adapter.
  - script: `scripts/llamacpp_native_loglikelihood_probe.py`
  - artifact: `results/llamacpp_native_loglikelihood_probe_v2_selftest_20260608.json`
  - changes: explicit `--pair CONTEXT|||CONTINUATION` scoring, continuation-token logprob summation, greedy-match metadata, and an `lm_eval_loglikelihood_tuple` field.
  - validation: Python compile and self-test passed.
  - interpretation: this is adapter-shape readiness only. GGUF accuracy is still blocked until the native probe passes against a live llama-server and a tiny lm-eval task.
- Added a tiny llama.cpp native loglikelihood task harness.
  - script: `scripts/llamacpp_native_loglikelihood_task.py`
  - task file: `tasks/llamacpp_loglikelihood_smoke.jsonl`
  - dry-run artifact: `results/llamacpp_native_loglikelihood_task_dryrun_20260608.json`
  - validation: Python compile, task dry-run, no-server failure path, and `git diff --check` passed.
  - interpretation: this makes the next live GGUF accuracy proof one command after starting llama-server; it still does not prove paper-comparable GGUF accuracy until run against the server.
- Ran the llama.cpp native loglikelihood task against a live Qwen2.5 1.5B Q4_K_M llama-server.
  - summary: `results/llamacpp_native_loglikelihood_20260608T1331JST_summary.md`
  - server evidence: `NVIDIA GB10`, `CUDA : ARCHS = 1210`, `USE_GRAPHS = 1`.
  - result: likely continuations were scored, but the unlikely `zebra` continuation was missing from top-512 probabilities; task `ok=false`.
  - interpretation: the OpenAI schema blocker has a live native-endpoint follow-up, but the native top-N path still does not satisfy arbitrary continuation-token scoring for lm-eval.
- Added the public compatibility board.
  - doc: `docs/COMPATIBILITY_BOARD.md`
  - purpose: recurring runtime/status view covering vLLM, SGLang, llama.cpp, FlashInfer, LiteRT-LM, HF fallback, model lanes, live proof queue, and update cadence.
  - interpretation: this closes the documentation gap where issues existed but there was no single compatibility board; maintaining it remains part of every future row update.
- Added the wheel/container matrix.
  - doc: `docs/WHEEL_CONTAINER_MATRIX.md`
  - purpose: map proven, partial, blocked, debug-only, and side-runtime install/container paths to exact artifact evidence.
  - interpretation: the matrix moves solution area 2 from scattered notes to an explicit acceptance-evidence index, while still marking the clean Spark-blessed vLLM/SGLang/FlashInfer package set as missing.
- Tightened the benchmark protocol so Qwen speed/capacity is mandatory alongside Gemma for broad runtime claims.
  - Qwen is the clean first lane for SM121a throughput, speculative decode, NVFP4 weights, and fp8-vs-NVFP4 KV capacity.
  - Gemma remains required for the original workload and harder model-family compatibility path.
  - llama.cpp Qwen2.5 GGUF evidence is now recorded in the llama.cpp serving recipe instead of being left as a stale template-only section.
- Added a Qwen speed-lane runner.
  - script: `scripts/qwen_speed_lane.py`
  - sample rows: `tasks/qwen_speed_lane_sample.jsonl`
  - dry-run artifact: `results/qwen_speed_lane_dryrun_20260608_summary.json`
  - purpose: record already-running vLLM, SGLang, and llama.cpp Qwen servers through the shared `record_openai_serving_row.py` manifest path.
  - interpretation: this does not replace the missing vLLM Qwen36 live result; it makes the next live Qwen campaign repeatable once the host and image acquisition are healthy.
- Added the AEON prior-art port map.
  - doc: `docs/AEON_PRIOR_ART_PORT_MAP.md`
  - script: `scripts/aeon_prior_art_audit.py`
  - audit artifact: `results/aeon_prior_art_audit_20260608.json`
  - sources: local `CODEX_REPORT_AEON7.md`, AEON Qwen patch directory, AEON Gemma repo, AEON `vllm-dflash`, and current `jethac` forks.
  - conclusion: literal AEON patches are vLLM internals and are already represented in `jethac/vllm@6804e1b`; SGLang and llama.cpp need counterpart experiments, not blind vLLM patch copies.
  - SGLang nuance: current `jethac/sglang@98ad46961` already has DFlash-specific surfaces, so the next useful step is a serving proof after ordinary Qwen/Gemma rows are stable.
- Added a mechanical solution coverage audit.
  - script: `scripts/solution_coverage_audit.py`
  - audit artifact: `results/solution_coverage_audit_20260608.json`
  - status doc change: `docs/SOLUTIONS_STATUS.md` now preserves every numbered solution-plan row through `15` instead of folding benchmark design, observability, upstream coordination, forks, and recipes into unrelated row numbers.
  - issue tracker change: `docs/ISSUE_TRACKER.md` now records solution-plan IDs separately from GitHub issue numbers.
  - Qwen rule: Qwen speed/capacity is a required cross-cutting benchmark lane alongside Gemma, not a substitute for any numbered solution area and not optional for broad runtime claims.
- Added a serving-manifest claim-readiness audit.
  - script: `scripts/serving_manifest_audit.py`
  - audit artifact: `results/serving_manifest_audit_20260608.json`
  - converted the llama.cpp Qwen2.5 Q4_K_M row manifest from a manual note into strict JSON.
  - result: the llama.cpp Qwen row is now claim-ready for practical serving evidence; the AEON Gemma vLLM row remains not claim-ready in the strict audit because its build-target audit lacks accepted Spark target evidence.
  - dry-run Qwen lane manifests are explicitly classified as planning evidence only.
- Added a container target audit to separate family/PTX evidence from native Spark target proof.
  - script: `scripts/container_target_audit.py`
  - AEON Gemma artifact: `results/aeon_gemma26_dflash_20260608T0436JST_container_target_audit.json`
  - result: the AEON Gemma image/container has GB10 runtime evidence and SM120-family/PTX evidence through `TORCH_CUDA_ARCH_LIST=... 12.0+PTX` / `sm_120`, but no explicit native `sm_121` or `sm_121a` target evidence.
  - serving-manifest audit now records this as `family_or_ptx_count: 1` for Gemma while keeping strict Gemma claim readiness false.
- Added an NVFP4 checkpoint metadata audit before larger Qwen/Gemma speed work.
  - script: `scripts/nvfp4_checkpoint_audit.py`
  - sample fixture: `tasks/nvfp4_checkpoint_audit_sample`
  - sample artifact: `results/nvfp4_checkpoint_audit_sample_20260608.json`
  - purpose: classify compressed-tensors versus ModelOpt NVFP4 markers, flag quantized router/vision/visual tensors, and check Gemma EOS/control-token metadata without loading tensor data or using GPU time.
  - interpretation: this does not prove a live Qwen or Gemma speed row; it prevents future NVFP4-weight serving or GGUF conversion rows from silently using a bad checkpoint format.
- Added a counterpart evidence audit for AEON-derived non-vLLM work.
  - script: `scripts/counterpart_evidence_audit.py`
  - artifact: `results/counterpart_evidence_audit_20260608.json`
  - result at creation time: all seven counterpart proof rows were still missing, partial, or blocked. The later `jethac/vllm` Qwen3.6+DFlash row moved the vLLM requirement to claim-ready for serving evidence; the SGLang and llama.cpp counterpart rows remain missing or partial.
  - interpretation: AEON source-port coverage is useful but does not satisfy the SGLang/llama.cpp counterpart acceptance tests.
- Added live task contracts for the missing counterpart rows.
  - task file: `tasks/counterpart_evidence_tasks.jsonl`
  - validator: `scripts/counterpart_task_matrix.py`
  - artifact: `results/counterpart_task_matrix_20260608.json`
  - result: all seven missing or partial counterpart requirements have concrete command templates and expected claim artifacts.
  - interpretation: this is runbook readiness, not live proof. The counterpart evidence audit remains the source of truth for whether those rows have actually landed.
- Reconnected to the GB10 host through Tailscale and captured fresh doctor evidence.
  - reachable host: `thinkstationpgx-00b4.tail740c8d.ts.net`
  - reachable IP: `100.113.98.11`
  - stale/unreachable LAN IP from this client: `192.168.68.112`
  - artifacts: `results/spark_doctor_tailnet_reconnect_20260608T074035JST.json`, `results/spark_doctor_tailnet_reconnect_20260608T074035JST.md`, `results/counterpart_task_matrix_tailnet_reconnect_20260608T074035JST.json`
  - result: live doctor confirms Linux `aarch64`, `NVIDIA GB10`, compute capability `12.1`, driver `580.159.03`, and CUDA `13.0`.
  - caveat: system Python has no Torch, so this reconnect artifact does not record SM count; use a runtime/container doctor for SM-count-backed benchmark rows.
- Ran the first local AEON Qwen3.6 NVFP4+DFlash vLLM attempt after reconnect.
  - run id: `aeon_qwen36_dflash_tailnet_retry2_20260608T075346JST`
  - image/model state: `ghcr.io/aeon-7/vllm-spark-omni-q36:v2` was present; target and drafter weights were present under `/home/jethac/models/aeon`.
  - checkpoint audit: `results/aeon_qwen36_dflash_tailnet_retry2_20260608T075346JST_nvfp4_checkpoint_audit.json`, `ok=true`, compressed-tensors NVFP4, `124306` safetensors keys, `0` quantized sensitive keys.
  - backend evidence: server log resolved `Qwen3_5MoeForConditionalGeneration` and `DFlashDraftModel`, selected `FlashInferCutlassNvFp4LinearKernel`, `MARLIN` NvFp4 MoE, FlashAttention 2, CUDA graphs, and `585168` KV tokens.
  - failure: row manifest is `ok=false`; chat smoke produced `message.reasoning` but no normal content and no `spark-ok`; benchmark recorded completion-token counts but no valid output text, so this is not a speed row.
  - interpretation: image acquisition and model startup are no longer the vLLM Qwen blocker; the next blocker is response/content validation plus native-target proof.
- Fixed the AEON Qwen3.6 vLLM response/content row by disabling Qwen thinking through OpenAI `chat_template_kwargs`.
  - direct probe: `results/qwen_content_probe_20260608T0900JST_direct_chat_probes.json`
  - finding: baseline and prompt-level `/no_think` stayed in `message.reasoning`, while API-level `{"enable_thinking": false}` returned normal `message.content` for both `qwen36-fast` and `qwen36-deep`.
  - passing row: `results/aeon_qwen36_dflash_nothink_20260608T0834JST_row_manifest.json`, `ok=true`
  - compact decode: `50.37 tok/s` short, `55.84 tok/s` medium, `53.75 tok/s` long-prefill.
  - caveat: this is AEON's container/checkpoint, not a `jethac` fork speedup; build-target audit still lacks explicit native `sm_121`/`sm_121a` evidence and the server still warns about Marlin weight-only FP4.
- Built and smoke-started the matched `jethac/vllm` derived image for Qwen3.6.
  - derived image: `jethac-vllm-aeon-q36:6804e1b81`
  - base image: `ghcr.io/aeon-7/vllm-spark-omni-q36:v2`
  - fork commit: `jethac/vllm@6804e1b81e6ea2ca53bb5021151bdad0f201b11d3`
  - build artifact: `results/jethac_vllm_aeon_q36_6804e1b81_image_build_20260608T0845JST.log`
  - stop-point artifact: `results/jethac_qwen36_dflash_depstop_20260608T0850JST_summary.md`
  - result: image imports `vllm 0.1.dev1+g6804e1b81`, but serving exits before health because the AEON base environment lacks `compressed_tensors.compressors.pack_quantized`.
  - interpretation: fork parity is now blocked on dependency/API drift, not image build, Qwen weights, or `sm_121` kernels.
- Advanced the matched `jethac/vllm` Qwen3.6 row past the dependency stop point.
  - passing image: `jethac-vllm-aeon-q36:6804e1b81-ct017-humming-aeonfa2`
  - fork commit: `jethac/vllm@6804e1b81e6ea2ca53bb5021151bdad0f201b11d3`
  - summary artifact: `results/jethac_qwen36_dflash_aeonfa2_nothink_20260608T0908JST_summary.md`
  - image layers: `compressed-tensors==0.17.0`, `humming-kernels[cu13]==0.1.4` plus `pyelftools`, and AEON's original FA2 binary restored after a PyTorch ABI mismatch.
  - backend evidence: `Qwen3_5MoeForConditionalGeneration`, `DFlashDraftModel`, `FlashInferCutlassNvFp4LinearKernel`, `MARLIN` NvFp4 MoE, FlashAttention 2, CUDA graphs, and `1,251,446` KV tokens.
  - compact decode: `47.22 tok/s` short, `58.88 tok/s` medium, `61.62 tok/s` long-prefill.
  - interpretation: this is a passing fork-derived vLLM Qwen serving row, but not clean fork packaging and not native `sm_121a` target proof because it still depends on AEON's FA2 binary and only host-side audits were captured.
- Added vLLM clean-packaging hook and in-container target/JIT audit tooling.
  - fork commit: `jethac/vllm@db4b210c1`
  - new vLLM env knob: `VLLM_PRECOMPILED_SKIP_FLASH_ATTN=1`, which skips extracting bundled FA2/FA3 extensions from a precompiled wheel while preserving the rest of the precompiled extension set
  - campaign scripts: `scripts/run_vllm_incontainer_target_audit.sh` and `scripts/cuda_artifact_arch_audit.py`
  - prior in-container audit interpretation: the passing `jethac/vllm@6804e1b` Qwen image has GB10 runtime evidence but no inspected `sm_121`/`sm_121a` CUDA object evidence, so the next vLLM image must replace AEON FA2 with an ABI-matched clean build and rerun the no-think Qwen row.
- Advanced the vLLM clean FA2 image build to the next blocker.
  - fork commit: `jethac/vllm@a919d635d`
  - main repo script: `scripts/build_vllm_aeon_qwen_cleanfa2_image.sh`
  - artifact: `results/jethac_vllm_qwen_cleanfa2_build_20260608Tfixversion_summary.md`
  - raw log: `results/jethac_vllm_qwen_cleanfa2_build_20260608Tfixversion.log`
  - result: `VLLM_PRECOMPILED_SKIP_FLASH_ATTN=1` successfully skipped bundled FA2/FA3, and `VLLM_VERSION_OVERRIDE=0.1.dev1+ga919d635d` fixed the previous `setuptools-scm` failure from missing `.git` metadata in the Docker context.
  - native-target finding: top-level vLLM CMake accepted `12.1a` and printed `arch=compute_121a,code=sm_121a`, but the nested pinned vLLM FlashAttention project reduced its supported target to `12.0`, selected `FA2_ARCHS: 8.0+PTX`, and compiled `_vllm_fa2_C` with only `sm_80`/`compute_80`.
  - interpretation: the clean vLLM packaging path is past dependency/versioning, but not past native FA2. The builder now fails fast if nested FA2 does not select SM121/SM121a; next work is patching or forking the pinned vLLM FlashAttention source.
- Added and tested the vLLM-pinned FlashAttention FA2 SM121a patch path.
  - fork branch: `jethac/flash-attention@spark/hijinks-021-fa2-sm121a`
  - fork commit: `7d53245`
  - submodule: `third_party/vllm-flash-attention`
  - issue: [#21](https://github.com/jethac/dgx-spark-hijinks/issues/21)
  - build artifact: `results/jethac_vllm_qwen_cleanfa2_build_20260608Tpatchedfa2_summary.md`
  - result: patched nested vLLM FlashAttention CMake selected `CUDA supported target architectures: 12.1a`, `FA2_ARCHS: 12.1a`, and `_vllm_fa2_C` invoked `nvcc` with `arch=compute_121a,code=sm_121a`.
  - failure: the build then stopped because the copied FlashAttention tree lacked its nested `csrc/cutlass` submodule, causing missing `cute/tensor.hpp` and `cutlass/numeric_types.h`.
  - interpretation: the architecture-selection blocker is fixed; the current blocker is packaging the nested CUTLASS dependency into the clean build context. The builder now initializes `csrc/cutlass` before creating the Docker context.
- Completed the clean vLLM FA2 SM121a build and in-container target audit.
  - image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
  - main repo commit before build: `jethac/dgx-spark-hijinks@6b33492`
  - vLLM fork commit: `jethac/vllm@a919d635d`
  - FlashAttention fork commit: `jethac/flash-attention@7d53245`
  - build artifact: `results/jethac_vllm_qwen_cleanfa2_build_20260608Tpatchedfa2_cutlass_summary.md`
  - audit artifact: `results/jethac_vllm_qwen_cleanfa2_patchedfa2_cutlass_audit_20260608T2355JST_incontainer_target_audit.md`
  - result: `_vllm_fa2_C.abi3.so` built, installed, imported, and `cuobjdump` showed `sm_121a` cubins in the patched FA2 extension.
  - runtime evidence: `NVIDIA GB10`, compute capability `[12, 1]`, `48` SMs, Torch `2.12.0.dev20260408+cu130`, FlashInfer `0.6.9rc1`, vLLM `0.1.dev1+ga919d635d`.
  - remaining caveat: this is native-target proof for the vLLM FlashAttention extension only. Other vLLM objects still carry their existing mixed prebuilt architecture surface, including `sm_120`, `sm_100`, and `sm_90a`.
  - next gate: rerun the no-think Qwen3.6+DFlash serving row on this clean FA2 image.
- Ran the no-think Qwen3.6+DFlash serving row on the clean vLLM FA2 SM121a image.
  - image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
  - run id: `jethac_qwen36_dflash_cleanfa2_sm121a_nothink_record2_20260608T2359JST`
  - summary artifact: `results/jethac_qwen36_dflash_cleanfa2_sm121a_nothink_record2_20260608T2359JST_summary.md`
  - manifest: `results/jethac_qwen36_dflash_cleanfa2_sm121a_nothink_record2_20260608T2359JST_row_manifest.json`, `ok=true`.
  - smoke: normal OpenAI `message.content` returned `spark-ok` with `chat_template_kwargs={"enable_thinking": false}`.
  - compact decode: `61.07 tok/s` short, `56.97 tok/s` medium, `60.10 tok/s` long-prefill.
  - backend evidence: `Qwen3_5MoeForConditionalGeneration`, `DFlashDraftModel`, `FlashInferCutlassNvFp4LinearKernel`, `MARLIN` NvFp4 MoE, FlashAttention 2, FlashInfer FP4 GEMM autotune, CUDA graphs, and `1,241,920` KV tokens.
  - interpretation: this closes the clean vLLM Qwen packaging gap and pairs serving evidence with separate FA2 `sm_121a` cubin proof. It is not a speedup claim and it still does not prove native FP4 weight/MoE compute; the server warns that the selected weight path is Marlin weight-only FP4.
  - failed first recorder pass: `jethac_qwen36_dflash_cleanfa2_sm121a_nothink_20260608T2359JST` hit a PowerShell-to-SSH JSON quoting bug for `chat_template_kwargs`, producing a recorder `JSONDecodeError`. The server had started and served; classify that failed row as operator quoting, not runtime/model failure.
- Moved the clean SGLang FP4 KV row from corrupted graph serving to an autosafe no-graph capacity path.
  - fork branch: `jethac/sglang@spark/hijinks-018-fp4-e2m1-kv-sm121-serving`
  - live image: `nvcr.io/nvidia/sglang:26.05-py3` with editable source overlay and local FlashInfer JIT headers/source.
  - graph-enabled negative artifacts: `results/sglang_qwen_fp4kv_decode_dtype_chat_smoke_20260608.json`, `results/sglang_qwen_fp4kv_decode_dtype_raw_generate_20260608.json`, `results/sglang_qwen_fp4kv_decode_dtype_bad_output_20260608_server.log`.
  - layout probe: `results/sglang_nvfp4_kv_layout_probe_20260608.json`; both 4D and 3D scale-factor forms matched a faithful dequant reference at cosine `0.9999957`, so scale-rank was not the serving corruption root cause.
  - eager/no-graph positive artifacts: `results/sglang_qwen_fp4kv_eager_only_chat_smoke_20260608.json`, `results/sglang_qwen_fp4kv_eager_only_raw_generate_20260608.json`, `results/sglang_qwen_fp4kv_eager_only_server_20260608.log`.
  - auto-safe default artifacts: `results/sglang_qwen_fp4kv_autosafe_chat_smoke_20260608.json`, `results/sglang_qwen_fp4kv_autosafe_raw_generate_20260608.json`, `results/sglang_qwen_fp4kv_autosafe_server_20260608.log`.
  - result at this stage: the fork disables CUDA graph and piecewise graph capture for native FP4 KV unless `SGLANG_FP4_KV_ENABLE_CUDA_GRAPH=1` is set. The early autosafe smoke looked better than graph-enabled decode, but this was not enough evidence to bless quality.
  - interpretation: SGLang FP4 KV needed a matched fp8-vs-fp4 row with standardized quality checks before any serving claim.
  - related upstream note: TensorRT-LLM #11368 documents a separate GB10 FP4 GEMM problem where SM120 tile configs exceed GB10's 99 KiB shared-memory limit, explaining why SM12x dispatch/JIT targeting alone may not improve MoE-shaped FP4 GEMM performance.
- Recorded the matched SGLang Qwen autosafe fp8-vs-FP4 KV row.
  - summary: `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_summary.md`
  - fp8 comparator: `3,101,822` KV tokens, no-graph policy match, decode `56.73`, `56.81`, and `57.10 tok/s`, raw `2+2` returns `4`.
  - FP4 KV: `5,519,481` KV tokens, auto-safe no-graph policy, `1.779x` fp8 capacity, and `NVFP4 KV cache calibrated 28 layers from 4096 eager prefill tokens`.
  - negative result: standardized FP4 raw `2+2` and compact benchmark content fail quality, so this is a capacity proof rather than a blessed serving or speed row.

- Recorded the matched vLLM Qwen fp8-vs-NVFP4 KV capacity row.
  - summary: `results/vllm_qwen_nvfp4_kv_capacity_20260608T1455JST_summary.md`
  - image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv`
  - runtime ref: `ghcr.io/aeon-7/vllm-spark-omni-q36:v2 + jethac/vllm@a919d635d + jethac/flashinfer@e152cf4d`
  - fp8 comparator: `6,364,935` KV tokens, `24.28x` max concurrency at 262k context, decode `43.001`, `42.512`, and `42.684 tok/s`.
  - NVFP4 KV: `11,146,226` KV tokens, `42.52x` max concurrency at 262k context, decode `43.014`, `42.615`, and `42.898 tok/s`.
  - backend evidence: `kv_cache_dtype='nvfp4'`, `Using nvfp4 data type to store kv cache`, `Using AttentionBackendEnum.FLASHINFER backend`, and `Using FlashInfer FA2 backend for NVFP4 KV cache on SM12x with vLLM V-scale-factor deswizzle enabled.`
  - interpretation: this is the first end-to-end vLLM NVFP4-KV serving capacity proof on the Qwen lane (`1.751x` fp8 KV pool/concurrency with normal content). It is not a decode-speed win, not native FP4 weight/MoE proof, and not a Gemma proof. Future Gemma work follows `docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md` via source overlay and the standalone `D=512` FlashInfer diagnosis rather than more image-build iteration.

- Set standing vLLM-lane direction toward Gemma 4 (and earlier Gemmas) on Spark with NVFP4 KV.
  - doc: `docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md`
  - framing: Gemma's serving blocker and the NVFP4-KV blocker are the same blocker — heterogeneous/dual head dims (local `D=256`, global `D=512`) plus alternating SWA force `TRITON_ATTN`, so the native FA2 `sm_121a` path never engages and the FA2 NVFP4-KV path fails Gemma global `D=512` at `prefill.cuh:3215` while passing local `D=256`.
  - sequencing: no vLLM image builds in the dev loop; iterate via standalone FlashInfer kernel harness (the keystone `D=512` head-dim-guard-vs-SMEM-overflow diagnosis) and source overlay on the proven `cleanfa2-patchedfa2-cutlass` image. Image bake is the final gated deliverable (Objective E), not a dev step.
  - SM120 ride-along: build on hikarioyama's SM120 prior art and keep patches SM12x-family-shaped so the eventual PR serves the larger RTX PRO 6000 audience (vllm #31085). Emit both `sm_120a` and `sm_121a` (arch-specific; `120f` cannot emit native FP4 MMA per #3170). `a` cubins are not cross-capability portable, so `sm_120a` cannot run on GB10 — SM120 stays compiled-but-unclaimed, validated by hikari, not us.
  - verified (NVIDIA Blackwell Tuning Guide, 2026-06): RTX PRO 6000 = `sm_120`/CC 12.0, GB10 = `sm_121`/CC 12.1; both CC 12.x have 128 KB SMEM/SM and a 99 KB/thread-block limit, vs B200 CC 10.0 at 228 KB/SM. So TRT-LLM #11368's >99 KiB FP4 GEMM tile overflow is a 12.x-family constraint, not a GB10 quirk — a 99 KB-fitting tile fix is correct for both sm_120 and sm_121.
  - tracker: added an "Upstream Issues Referenced" section to `docs/ISSUE_TRACKER.md` (vllm #31085, vllm #31128, TRT-LLM #11368) and a native-FP4 target note; linked the direction doc from `README.md`.

- Set standing SGLang-lane direction: convert the proven NVFP4-KV capacity row into a blessed quality row.
  - doc: `docs/CODEX_DIRECTION_SGLANG_NVFP4_KV.md`
  - framing: SGLang already proves ~1.78× fp8 KV capacity on GB10, but output corrupts even in eager/no-graph mode. The FlashInfer FA2 NVFP4-KV kernel is correct standalone (`e152cf4d`, cosine ≥ 0.99999946) and the layout probe cleared scale-rank (dequant cosine 0.9999957), so the bug is in SGLang's integration of a correct kernel, not the kernel math.
  - prime suspect: the turn-1 convention mismatch — the SGLang patch routes SM12x through `nvfp4_kv_quantize` (encode/multiply convention), but the GB10-runnable FlashInfer FA2 consumer may expect the `fp4_quantize` decode/divide convention; mismatch = off-by-`s_enc²` garbage, which matches the symptom. First decisive test: swap SM12x to `fp4_quantize` + inverted global scale and check whether raw `2+2` returns `4` in eager mode.
  - objectives: (A) root-cause eager corruption via a SGLang-vs-standalone-FlashInfer numerical bridge, testing convention → V-scale layout (SGLang symmetric-linear, not vLLM B2 swizzle) → calibration → decode kernel; (B) confirm the force-compiled FP4 decode kernel is numerically correct, not just building; (C) fix CUDA-graph-capture corruption so capacity isn't stuck on the slow no-graph path; (D) land a blessed matched fp8-vs-FP4 row with quality passing; (E) SWA/Gemma later (Qwen/non-SWA first).
  - methodology: no image builds in the dev loop — iterate via editable source overlay on stock `nvcr.io/nvidia/sglang:26.05-py3` and use the standalone FlashInfer reference as ground truth.
  - tracker: linked the direction doc from `README.md`; SGLang NVFP4 KV remains issue #18.

- Set standing llama.cpp-lane direction: unblock GGUF lm-eval accuracy and prove-or-deny native FP4.
  - doc: `docs/CODEX_DIRECTION_LLAMACPP.md`
  - framing: llama.cpp is the fastest practical runtime on the box and already blessed for serving (~175 tok/s Qwen2.5 1.5B, ~77 tok/s Gemma 4 26B), but two campaign deliverables stay red: GGUF lm-eval accuracy (blocked since day one, 120 `loader_failed` rows) and native `sm_121a` FP4 dispatch (`BLACKWELL_NATIVE_FP4=1` is compiled in but never proven to engage on k-quant models).
  - keystone (A): get exact per-continuation-token logprobs from supplied/prompt-echo tokens, not top-N. The `n_probs=512` native path is a cleared dead end — it misses unlikely continuations (the `zebra` failure). Decisive first test: probe every `b9536` endpoint for the exact logprob of a supplied unlikely token, rank-independent; if none returns it, escalate to a build-pin or a `jethac/llama.cpp` endpoint fork. Fixing this gives the whole campaign a quantization-accuracy oracle it currently lacks.
  - (B): prove or deny native FP4 via cuobjdump/dispatch evidence (a k-quant model on an FP4-capable build proves nothing), and chase MXFP4/GPT-OSS — since llama.cpp avoids triton it may be the only working MXFP4-on-Spark path where triton #8335 blocks `sm_121a`. (C) larger Qwen3/3.6 GGUF row; (D) keep serving recipes current.
  - methodology: native host binaries, not containers — iterate against the running `llama-server`, rebuild/fork only if the stock server can't expose supplied-token logprobs. SM120/RTX PRO 6000 is simpler here: source-built per machine (`CMAKE_CUDA_ARCHITECTURES=121` vs `120`), no non-portable `a` cubins to ship.
  - guardrails: three claim classes never merged (serving blessed / accuracy blocked / native-FP4 unproven); top-N is not loglikelihood. Issues #8 (accuracy), #17 (serving).

- Ran the SGLang standalone NVFP4-KV convention bridge on GB10.
  - script: `scripts/sglang_nvfp4_kv_convention_probe.py`
  - artifact: `results/sglang_nvfp4_kv_convention_probe_20260608.json`
  - runtime: `nvcr.io/nvidia/sglang:26.05-py3`, FlashInfer `0.6.10+cf494fca.nv26.05.cu132.50619265`, Torch `2.12.0a0+5aff3928d8.nv26.05`, device `NVIDIA GB10`, capability `[12, 1]`.
  - shape: page-size-1 decode, `tokens=40`, `query_heads=64`, `kv_heads=4`, `head_dim=128`.
  - result: `fp4_quantize` with encode scale (`1 / decode_scale`) and FA2 reader decode scale passed (`attention_cosine_vs_source=0.9950249`, `attention_cosine_vs_dequant=0.9999955`).
  - result: `nvfp4_kv_quantize` with decode scale and FA2 reader decode scale also passed with the same cosine values.
  - negative: `nvfp4_kv_quantize` with encode scale plus FA2 reader decode scale failed completely (`attention_cosine_vs_source=0.0`).
  - interpretation: the raw FA2 reader is convention-matched for the two viable pairs above, and the known `nvfp4_kv_quantize` encode/decode crossing is invalid. Because the serving overlay had already tried the `fp4_quantize` convention and still produced corrupt text, the next SGLang suspect is integration state: calibration scale plumbing, memory-pool/backend scale application, V-scale layout, or the forced-compiled decode path, not the standalone FA2 reader math.

- Ran the llama.cpp supplied-token echo-logprobs probe against the pinned `b9536` server.
  - script update: `scripts/gguf_logprobs_probe.py` now tokenizes `--context` and `--continuation`, sends `context + continuation` with `echo=true`, and only passes if prompt `tokens` plus `token_logprobs` cover the supplied continuation span.
  - artifacts: `results/llamacpp_gguf_echo_logprobs_probe_20260608_max0.json`, `results/llamacpp_gguf_echo_logprobs_probe_20260608_max1.json`, `results/llamacpp_gguf_echo_logprobs_probe_20260608_summary.json`, `results/llamacpp_gguf_echo_logprobs_probe_20260608_server.log`.
  - target: Qwen2.5 1.5B Q4_K_M on `/home/jethac/src/llama.cpp-b9536/build/bin/llama-server`, prompt `The capital of Japan is zebra`.
  - tokenization: context tokens `[785, 6722, 315, 6323, 374]`; continuation tokens `[1147, 50213]`.
  - result: both `max_tokens=0` and `max_tokens=1` returned `choices[0].logprobs.content` for a generated token (`-striped`) and did not expose prompt `tokens` or `token_logprobs`; both rows have `ok=false`.
  - interpretation: the OpenAI echo path on pinned `b9536` cannot provide exact supplied-continuation logprobs for the `zebra` case. The llama.cpp accuracy lane needs either a newer server pin that exposes prompt-token logprobs or a `jethac/llama.cpp` endpoint fork.

- Re-ran the focused Gemma 4 26B global-attention FlashInfer FA2 NVFP4-KV blocker on the Qwen-proven vLLM image.
  - artifact: `results/flashinfer_nvfp4_kv_probe_gemma4_26b_global_nhd_debug_20260608.json`
  - image/source: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv`, FlashInfer `0.6.13` from `/opt/jethac-flashinfer`, git `e152cf4d`.
  - shape: Gemma global/full attention NHD, `batch_size=2`, `kv_len=128`, `qo_len=16`, `page_size=16`, `H_q=16`, `H_kv=2`, `D=512`, `dtype=bfloat16`, vLLM-style swizzled V scale factors.
  - result: both operations still fail at `include/flashinfer/attention/prefill.cuh:3215`; decode trait is `NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 NUM_MMA_KV=1 NUM_WARPS_Q=1 NUM_WARPS_KV=4`, prefill trait is `NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 NUM_MMA_KV=2 NUM_WARPS_Q=4 NUM_WARPS_KV=1`.
  - interpretation: Gemma NVFP4-KV remains blocked below vLLM routing. The next work is a FlashInfer FA2 `D=512` trait/tile fix or a mixed-KV fallback for Gemma global layers.
  - trait audit: `DISPATCH_HEAD_DIM` already has `case 512`; the failure is `KernelTraits::IsInvalid()` clause `NUM_MMA_Q * (8 * NUM_MMA_D_VO + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >= 256`. For `D=512`, `NUM_MMA_D_VO=32`, so the term is `264` for decode and `272` for prefill. This is a fragment/register-shape guard, not a missing head-dim table or primarily a 99 KiB SMEM overflow. Mixed KV for global layers is the pragmatic next vLLM/Gemma path.

- Ran the SGLang FP4 pool bridge on GB10.
  - script: `scripts/sglang_fp4_pool_bridge_probe.py`
  - artifacts: `results/sglang_fp4_pool_bridge_probe_20260608.json`, `results/sglang_fp4_pool_bridge_probe_prefill_20260608.json`
  - runtime: `nvcr.io/nvidia/sglang:26.05-py3` with `jethac/sglang@spark/hijinks-018-fp4-e2m1-kv-sm121-serving` overlaid through `PYTHONPATH`; FlashInfer `0.6.10+cf494fca.nv26.05.cu132.50619265`; device `NVIDIA GB10`, capability `[12, 1]`.
  - shape: `tokens=40`, `query_heads=64`, `kv_heads=4`, `head_dim=128`, page size 1, token slots `1..40` so slot 0 remains the SGLang padded slot.
  - decode result: `all_ok=true`; `attention_cosine_vs_dequant=0.9999946356`, `attention_cosine_vs_source=0.9958496094`, finite output, `key_dequant_cosine_vs_source=0.9955501556`, `value_dequant_cosine_vs_source=0.9954957962`.
  - widened result: decode and paged prefill both pass; prefill has `attention_cosine_vs_dequant=0.9999946356`, `attention_cosine_vs_source=0.9957648516`, finite output, and `passed=true`.
  - interpretation: `MHATokenToKVPoolFP4` writes packed K/V and FP8 scale buffers that FlashInfer FA2 can consume through the same getters used by serving for decode and paged prefill. The remaining SGLang FP4-KV corruption is downstream of the basic pool contract: backend wrapper metadata, graph/capture state, stale calibration state, or a model-serving path not covered by the synthetic bridge.

- Added and ran SGLang FP4-KV backend trace instrumentation.
  - fork commit: `jethac/sglang@d7d931f530160ba86a2d55b4636d64baaeda3bec`
  - knob: `SGLANG_FP4_KV_TRACE_BACKEND=1`
  - artifacts: `results/sglang_fp4_backend_trace_20260608T1536JST_summary.md`, `results/sglang_fp4_backend_trace_20260608T1536JST_server.log`, `results/sglang_fp4_backend_trace_20260608T1536JST_trace_excerpt.txt`, `results/sglang_fp4_backend_trace_20260608T1536JST_raw_2plus2.json`, `results/sglang_fp4_backend_trace_20260608T1536JST_chat_smoke.json`
  - command shape: NVIDIA SGLang 26.05 source overlay, `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`, `--kv-cache-dtype fp4_e2m1`, FlashInfer attention, page size 1, memory fraction 0.40, CUDA graph and piecewise graph disabled.
  - result: server reached readiness, allocated `5,516,867` FP4 KV tokens, calibrated 28 layers, and traced all 28 decode layers through native FP4 KV with packed `uint8` K/V, FP8 scale buffers, and finite per-layer global scales.
  - sanity: raw `2+2 is` returned ` 4, 2+2 is 4, 2+2 is`; chat smoke returned exactly `spark-ok`.
  - interpretation at the time: this was quality-positive debug evidence and showed the backend decode call contract matched the cleared pool bridge, but it was not yet a blessed SGLang FP4-KV row. It is superseded by the matched fp8-vs-FP4 trace row below.

- Ran the matched SGLang fp8-vs-FP4 KV row on the backend trace branch.
  - fork commit: `jethac/sglang@d7d931f530160ba86a2d55b4636d64baaeda3bec`
  - artifacts: `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_summary.md`, plus fp8/fp4 row manifests, OpenAI benchmarks, raw `2+2`, chat smoke, server logs, runtime probes, build-target audits, and trace excerpts.
  - command shape: NVIDIA SGLang 26.05 source overlay, `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`, `--attention-backend flashinfer`, page size 1, memory fraction 0.40, CUDA graph and piecewise graph disabled; FP4 row adds `--kv-cache-dtype fp4_e2m1` and `SGLANG_FP4_KV_TRACE_BACKEND=1`; fp8 comparator uses `--kv-cache-dtype fp8_e4m3`.
  - capacity: fp8 allocated `3,105,240` KV tokens; FP4 allocated `5,517,572` KV tokens, or `1.7769x` fp8.
  - trace: FP4 row calibrated 28 layers and logged all 28 decode plus all 28 `extend_merge_paged` layers through packed `uint8` K/V and FP8 scale buffers.
  - smoke: fp8 raw `2+2 is` returned ` 4. 2+2 is 4. 2+2 is`; FP4 returned ` 4, 2+2 is 4, 2+2 is`; both chat smokes returned `spark-ok`.
  - benchmark: fp8 produced normal compact benchmark text at about `57 tok/s`; FP4 short/medium/long benchmark content remained degraded even though raw/chat smoke passed.
  - interpretation: this retires the missing matched-comparator/request-trace task, but does not bless SGLang FP4 KV quality or speed. The next SGLang task is quality localization on the degraded benchmark prompts.

- Ran the SGLang FP4-KV logprob quality probe on the same `d7d931f` source-overlay path.
  - script: `scripts/openai_quality_probe.py`
  - artifacts: `results/sglang_qwen_fp4kv_d7d931f_logprob_quality_20260608T1609JST_summary.md`, `results/sglang_qwen_fp4kv_d7d931f_logprob_quality_20260608T1609JST_fp8_quality_probe.json`, `results/sglang_qwen_fp4kv_d7d931f_logprob_quality_20260608T1609JST_fp4_quality_probe.json`, and `results/sglang_qwen_fp4kv_d7d931f_logprob_quality_20260608T1609JST_compare.json`.
  - command shape: same NVIDIA SGLang 26.05 source overlay, same Qwen2.5 1.5B model, FlashInfer attention, page size 1, memory fraction 0.40, CUDA graph and piecewise graph disabled; probe requested generated-token logprobs for `short_decode` and `medium_decode`.
  - result: fp8 passed both probes. FP4 `short_decode` began with the same high-confidence prefix (`A local AI workstation`) then drifted into mixed Chinese/repetition; FP4 `medium_decode` diverged at token one (`the following code:` vs fp8 `**Engineering Note:`) and collapsed into repeated `import` text.
  - interpretation: the quality bug is now localized beyond "bad benchmark text": one standardized prompt is wrong from the first token, while another starts plausibly and corrupts later. Next useful SGLang work is a divergence-window trace around the failing `medium_decode` prompt, not another capacity row.

- Ran the SGLang native `/generate` logprob divergence probe for `medium_decode`.
  - script: `scripts/sglang_native_logprob_compare.py`
  - artifacts: `results/sglang_qwen_fp4kv_d7d931f_native_divergence_20260608T1626JST_summary.md`, `results/sglang_qwen_fp4kv_d7d931f_native_divergence_20260608T1626JST_native_logprob_compare.json`, and server/trace logs with the same prefix.
  - command shape: two simultaneous NVIDIA SGLang 26.05 source-overlay servers on ports 30012 and 30013; fp8 comparator versus FP4 KV candidate; same model, FlashInfer attention, page size 1, memory fraction 0.40, CUDA graph and piecewise graph disabled. The probe rendered the Qwen chat template explicitly, then called native `/generate` with `return_logprob=true`.
  - result: fp8 and FP4 used the same 56-token rendered prompt and matched through output tokens 0-3 (`**`, `Engineering`, ` Note`, `:`). First divergence was token index 4: fp8 chose ` Valid`, FP4 chose ` Validate`; both alternatives appeared in both top-k lists, but FP4 reversed their rank.
  - interpretation: under native `/generate`, the failure is an early decode distribution perturbation that compounds, not a total first-token collapse. The next SGLang question is why OpenAI Chat Completions looked worse than native rendered-template `/generate`.

- Ran the SGLang OpenAI-vs-native prompt reconciliation probe.
  - script: `scripts/sglang_openai_native_reconcile.py`
  - artifacts: `results/sglang_qwen_fp4kv_prompt_path_reconcile_20260608T173754JST_summary.md`, `results/sglang_qwen_fp4kv_prompt_path_reconcile_20260608T173754JST.json`, and matching fp8/FP4 server logs, trace excerpts, and container inspect files.
  - command shape: two simultaneous NVIDIA SGLang 26.05 source-overlay servers on ports
    30012 and 30013; fp8 comparator versus FP4 KV candidate; same Qwen2.5 1.5B model,
    FlashInfer attention, page size 1, memory fraction 0.40, CUDA graph and piecewise
    graph disabled. The probe compared OpenAI Chat Completions prompt IDs against local
    Qwen chat-template rendering and replayed native `/generate` from the same prompt IDs.
  - result: fp8 and FP4 OpenAI prompt IDs matched the local render exactly: 56 tokens,
    SHA-256 `5a5d4572e0e3d940a909b85dc4a00350094cbd1d55333c3d4f0a7974a91ee517`, no first
    prompt diff. FP4 OpenAI still diverged at token 4, while FP4 native `/generate` from
    the same prompt IDs diverged at token 0 (`**` -> `ark`).
  - interpretation: prompt serialization is retired as the cause. The remaining SGLang
    FP4-KV quality bug is endpoint/path-specific serving numerics or metadata; next inspect
    request metadata and pre-sampling logits/hidden state before touching kernel math again.
- Added the Gemma compatibility plan as a sequenced ladder across the whole family.
  - doc: `docs/GEMMA_COMPATIBILITY_PLAN.md`
  - rationale: "Gemma 4" is five models across four architectures (E2B/E4B dense-mobile+PLE+audio, 12B dense encoder-free multimodal, 26B-A4B MoE, 31B dense), plus Gemma 3 (dense, SWA, uniform head dim) and Gemma 3n (the superseded mobile line). "Fix Gemma" is a matrix, not a checkbox.
  - structure: one-new-complication-per-rung ladder. Rung -1 = per-variant config audit (settles where `D=512` lives). Main vLLM NVFP4-KV ladder: Rung 0 Qwen (done) -> Rung 1 Gemma 3 27B (isolate SWA / hybrid mixed-KV, also a shippable big-Gemma capacity win) -> Rung 2 Gemma 4 31B text-only (dense D=512 mixed-KV) -> Rung 3 Gemma 4 26B-A4B text-only (add MoE) -> Rung 4 Gemma 4 12B (encoder-free multimodal through decoder/KV). Mobile side track (llama.cpp/LiteRT): Gemma 3n -> Gemma 4 E2B/E4B (isolate PLE/audio/elastic).
  - discipline: prove-before-climb; assign each model to its natural runtime rather than filling all 15 cells; **measure attention geometry (per-layer head_dim, heads, KV heads, SWA layer map, KV bytes/token) from the running model every rung** — config is a hint, the running model is ground truth, and the mixed-KV layer classification must come from measured per-layer head_dim, not assumption.
  - provenance caveat: Gemma 4 lineup/architecture is operator-provided (post-cutoff) and must be confirmed by the rung -1 audit + per-rung measurement before building on it.

- Completed the Gemma Rung -1 config audit.
  - script: `scripts/gemma_rung_minus1_config_audit.py`
  - artifact: `results/gemma_rung_minus1_config_audit_20260608.json`
  - report: `docs/GEMMA_RUNG_MINUS1_CONFIG_AUDIT.md`
  - result: `google/gemma-3-27b-it` normalizes to uniform `D=128` (not the earlier
    plan's `D=256`), with 52 sliding layers and 10 full layers; no `global_head_dim=512`.
  - result: `D=512` appears in all audited Gemma 4 server configs. 12B has 40 sliding
    `D=256` + 8 full `D=512` layers and vision/audio config blocks; 31B has 50 sliding
    `D=256` + 10 full `D=512` layers and is dense; 26B-A4B has 25 sliding `D=256` + 5
    full `D=512` layers plus MoE (`128` experts, top-`8`).
  - decision: 31B isolates dense `D=512` before 26B-A4B adds MoE. The later ladder
    correction moves Gemma 4 12B to the final multimodal-KV rung. Next vLLM live rung
    remains Gemma 3 27B; next SGLang live work remains Qwen FP4-KV quality.

- Tightened the Gemma Rung -1 audit against cached QAT/server variants.
  - artifact: `results/gemma_rung_minus1_config_audit_strict_20260608.json`
  - script update: `scripts/gemma_rung_minus1_config_audit.py` now records config wrapper
    status, operator architecture hints, PLE-like keys, and config-derived BF16 raw KV
    bytes/token totals.
  - result: 12B QAT variants preserve the base 12B geometry: 40 sliding `D=256` + 8 full
    `D=512`; 31B W4A16 preserves 50 sliding `D=256` + 10 full `D=512`; 26B-A4B
    QAT-unquantized preserves 25 sliding `D=256` + 5 full `D=512` and MoE.
  - correction: config wrapper fields alone do not decide whether modality is quarantined
    in an unfired encoder or fused into the decoder/KV. The ladder now follows the
    operator-provided architecture: 31B and 26B-A4B are text-only encoder-quarantine rungs;
    12B is the final encoder-free multimodal-KV rung.

- Integrated the first llama.cpp NVFP4 GGUF runtime smoke from the parallel lane.
  - artifact: `results/llamacpp_nvfp4_runtime_gate_20260608T1748JST_summary.md`
  - source: `jethac/llama.cpp@19bba67c1`, built for `sm_121a`
  - result: cached AEON Qwen3.6 NVFP4 converted to NVFP4 GGUF, loaded in `llama-server`,
    and returned `The capital of Japan is Tokyo.`
  - profiler evidence: Nsight summary shows `GGML_TYPE_NVFP4` matmul and
    `quantize_mmq_nvfp4` kernels. This is runtime dispatch/smoke evidence, not an accuracy
    or speed benchmark.

- Completed the first llama.cpp native FP4 arch-build checkpoint in parallel with the
  Gemma audit.
  - submodule/fork: `third_party/llama.cpp` -> `jethac/llama.cpp`, branch
    `spark/native-fp4-sm121-20260608`
  - pinned commit: `19bba67c1f4db723c60a0d421aa0788bf4ddc699`
  - artifact: `results/llamacpp_native_fp4_arch_20260608T164917JST_summary.md`
  - result: `CMAKE_CUDA_ARCHITECTURES=121a` configures/builds and emits `sm_121a` cubins
    with `2592` `mxf4nvf4.block_scale.scale_vec::4X` PTX hits.
  - result: `CMAKE_CUDA_ARCHITECTURES=121` is accepted but rewritten by this llama.cpp pin
    to `121a`, so it is not an independent non-`a` build.
  - result: `CMAKE_CUDA_ARCHITECTURES=120f` fails at CMake configure-time under this CUDA
    13.0/CMake 3.28.3 toolchain.
  - interpretation: native block-scale FP4 code emission on `sm_121a` is proven for the
    pinned source build; runtime dispatch, correctness, and speed on an actual NVFP4 GGUF
    remain the next gate.

- Tightened the vLLM Gemma 3 27B Rung 1 live packet after host preflight.
  - script: `scripts/prep_vllm_gemma3_27b_rung1.sh`
  - doc: `docs/VLLM_GEMMA3_27B_RUNG1_PREP_20260608.md`
  - preflight artifact: `results/vllm_gemma3_27b_rung1_preflight_20260608.md`
  - result: the Spark-class Linux endpoint is reachable, idle, and has the target vLLM
    image locally, but the older `/home/jethac/src/vllm` and `/home/jethac/src/flashinfer`
    assumptions are false, the existing Linux repo checkout is stale/dirty, and
    `google/gemma-3-27b-it` was not found in the bounded HF cache search.
  - fix: generated packets now stream `docker logs -f` into `_server.log`, write container
    IDs, wait for `/v1/models` before recording rows, and remove row containers on exit.
    This prevents the next Gemma Rung 1 run from recording only a Docker container ID as
    the server log or racing readiness.
  - next gate: create/sync a clean Linux run checkout with initialized vLLM/FlashInfer
    submodules and gated Gemma 3 access, then run the fp8 comparator row before the NVFP4
    candidate.

- Prepared the clean Linux checkout for the vLLM Gemma 3 27B Rung 1 live row.
  - checkout: `/home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608`
  - superproject: `595dfb6dba863088707afadbad816a511b803f81`
  - source overlays: `jethac/vllm@8916796bc50926fd61e606718b194a71e2e31a24` and
    `jethac/flashinfer@e152cf4da4ab2a9d093b7d9d4b499198b0211c61`
  - generated packet:
    `docs/results/vllm_gemma3_27b_rung1_20260608TCHECKOUTJST_command_packet.sh`
  - artifact: `results/vllm_gemma3_27b_rung1_checkout_setup_20260608.md`
  - validation: generated packet passes `bash -n`.
  - remaining gate: `google/gemma-3-27b-it` still needs gated access/cache confirmation
    before the fp8 comparator row starts.

- Ran the bounded Hugging Face access probe for vLLM Gemma 3 27B.
  - script: `scripts/hf_model_access_probe.py`
  - artifact: `results/vllm_gemma3_27b_hf_access_probe_20260608T173133JST.json`
  - target: `google/gemma-3-27b-it`
  - container: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
  - result: `model_info` succeeds and reports `gated="manual"` at revision
    `005ad3404e59d6023443cb575daa05336842228a`, but a config/tokenizer-only
    `snapshot_download` fails with `GatedRepoError`.
  - environment: no `HF_TOKEN` was present inside the container; cache filesystem free
    space was about `2.50e12` bytes.
  - interpretation: the immediate blocker for the Gemma 3 fp8 comparator row is gated
    Hugging Face authentication/access, not disk headroom, target image availability, or
    source-checkout setup.

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
