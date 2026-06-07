# DGX Spark Solutions Status

Date: 2026-06-08

This file maps `docs/DGX_SPARK_SOLUTIONS.md` to current evidence. It is intentionally conservative: a row is `strong` only when artifacts prove the relevant acceptance test, not merely when a runtime once started.

## Status Map

| solution area | status | current evidence | missing proof |
|---|---|---|---|
| 1. `sm_121` target naming and build flags | partial | `spark_doctor` records GB10 compute capability `12.1` and `multi_processor_count=48`; llama.cpp logs/audits include `CUDA : ARCHS = 1210`; `scripts/container_target_audit.py` now records image/container family/PTX evidence separately from native target proof. | vLLM and SGLang still need explicit Spark-compatible build-target evidence in container logs or binary inspection, not just runtime device capability or `12.0+PTX` family evidence. |
| 2. Wheel and container matrix | partial | `docs/WHEEL_CONTAINER_MATRIX.md` now indexes proven, partial, blocked, and debug-only stack paths; AEON Gemma vLLM, NVIDIA SGLang Qwen BF16/fp8, llama.cpp source serving, LiteRT-LM side-runtime, and fork source probes are recorded with artifacts. | Clean Spark-blessed install/container matrix across vLLM, SGLang, FlashInfer, PyTorch-adjacent wheels, and CUDA 13 ARM64 remains missing. |
| 3. Kernel dispatch architecture awareness | partial | AEON Gemma logs show `TRITON_ATTN`, `FlashInferCutlassNvFp4LinearKernel`, `VLLM_CUTLASS` NvFp4 MoE, DFlash, and CUDA graphs. SGLang Qwen rows log attention/KV state. | Need broader no-silent-fallback proof across vLLM/SGLang and explicit failure messages for datacenter-only paths. |
| 4. Separate datacenter Blackwell from Spark Blackwell | partial | Diagnosis and recipe docs distinguish `sm_100`, `sm_120`, and `sm_121`; SM-count-aware hardware key exists. | More Spark-specific upstream/runtime docs are needed before the recipe set is user-boring. |
| 5. vLLM first-class Spark runtime | partial | AEON Gemma 4 26B NVFP4+DFlash serves locally with warmed compact decode around `48-54 tok/s` and long-prefill around `98 tok/s`; Gemma 12B unified serves through source/precompiled path; `jethac/vllm@6804e1b` now carries the AEON Qwen source fixes. The AEON Gemma container audit records GB10 runtime evidence plus `12.0+PTX`/`sm_120` family evidence. | Clean official/blessed vLLM path, accuracy check, Qwen3.6 NVFP4+DFlash reproduction, and explicit native `sm_121` or `sm_121a` build-target proof. |
| 6. Gemma 4 12B support | partial | `gemma4_unified` source/precompiled vLLM probe serves on GB10. | Clean release/nightly container, zero-shot lm-eval task, and performance-worthy path. |
| 7. NVFP4 end to end | mixed | AEON Gemma proves NVFP4 weight serving; FlashInfer FA2 NVFP4 KV standalone probe passes small and Gemma sliding/local shapes. | End-to-end vLLM/SGLang `nvfp4` KV serving, fp8-vs-NVFP4 quality/capacity/throughput rows, and Gemma global `D=512` path. |
| 7a. SGLang first-class Spark runtime | partial | SGLang Qwen2.5 1.5B BF16/auto and fp8 KV serve around `58-59 tok/s`; fp8 roughly doubles KV pool over BF16/auto. | Gemma serving path, clean graph-compatible FP4 KV, and output quality checks. |
| 7b. LiteRT-LM optional side runtime | partial | Linux aarch64 install, CPU generation, and CPU/GPU benchmark evidence exist. | GPU chat exits `-11`; no blessed GPU serving path. |
| 8. llama.cpp / lm-eval accuracy | missing | `gguf_logprobs_probe` identifies the schema mismatch; `scripts/llamacpp_native_loglikelihood_probe.py` prototypes lm-eval-shaped scoring; `scripts/llamacpp_native_loglikelihood_task.py` adds a tiny JSONL task harness with local dry-run evidence. | Native probe and task harness must prove arbitrary continuation-token logprobs against a live llama-server, then a tiny GGUF lm-eval/loglikelihood task must pass; current OpenAI-compatible API returns `logprobs.content`, not `tokens` plus `token_logprobs`. |
| 8a. llama.cpp practical serving path | strong | Gemma 4 26B Q4_0 and Qwen2.5 1.5B Q4_K_M serve through OpenAI API; `llama-bench`, build-target audit, model details, and hardware evidence are captured. `scripts/serving_manifest_audit.py` now marks the Qwen2.5 llama.cpp row claim-ready for serving evidence. | Native NVFP4/MXFP4 GGUF and paper-comparable GGUF accuracy remain separate. |
| 9. HF fallback containment | partial | Telemetry wrapper and failure annotator exist. | Historical `returncode=-9` rows need stronger OOM/resource evidence if HF fallback remains in comparisons. |
| 10. GB10 SM count and memory tuning | partial | Hardware comparison keys include compute capability and SM count; scripts collect `multi_processor_count`. | Performance tuning and regression thresholds across model families remain mostly unproven. |
| 11. Multi-Spark | missing | Single-unit assumption is documented. | No multi-Spark hardware or TP>1 validation. |
| 12. Improve benchmark design | partial | `scripts/spark_smoke_suite.py`, `scripts/openai_serving_benchmark.py`, telemetry wrapping, failure annotation, `scripts/qwen_speed_lane.py`, and `scripts/serving_manifest_audit.py` split smoke, serving, Qwen speed/capacity, fragile fallback, and claim-readiness evidence into narrower phases. | Live rows still need consistent phase completion, size-aware timeout policy, row-level failure explanations, and matched before/after manifests for every claimed runtime path. |
| 13. Observability | partial | `spark_doctor`, `cuda_so_audit`, `cuda_build_target_audit`, `container_target_audit`, runtime process probes, SM-count-aware hardware keys, server-log artifact capture, and serving-manifest audits exist. | Each blessed runtime still needs a no-silent-fallback artifact proving selected attention, quantization, KV, CUDA graph, and build/JIT targets before and after benchmarks; current AEON Gemma vLLM manifest has family/PTX container evidence but still lacks accepted native build-target evidence. |
| 14. Coordinate upstream ownership | partial | GitHub issues track the layer split; `docs/COMPATIBILITY_BOARD.md`, `docs/WHEEL_CONTAINER_MATRIX.md`, and `docs/AEON_PRIOR_ART_PORT_MAP.md` give maintainers a public status board, install matrix, and prior-art map. | Need recurring blessed-stack updates, public reproduction bundles, and upstream issue/PR taxonomy once matched GB10 before/after evidence exists. |
| 14a. Forks, submodules, worktrees, and subagents | partial | `jethac` FlashInfer, vLLM, and SGLang forks/submodules/worktrees exist; patch branches are documented; the vLLM Qwen branch now includes AEON-derived Qwen/DFlash runtime fixes; `docs/AEON_PRIOR_ART_PORT_MAP.md` separates direct vLLM ports from SGLang/llama.cpp counterpart work. | No upstream PRs until matched before/after GB10 story is proven; every future fork change still needs issue branch, worktree path, commit SHA, and reproduction command. |
| 15. Publish honest recipes | partial | Runtime recipes, compatibility board, wheel/container matrix, blessed-stack notes, and Qwen/Gemma docs now record what works, what is slow, what is broken, and what remains untested. | A clean-unit reproduction for the blessed vLLM/SGLang/llama.cpp stack is still missing, and the recipes must stay tied to exact commands, versions, artifacts, and go/no-go decisions. |

## Cross-Cutting Required Lanes

| lane | status | current evidence | missing proof |
|---|---|---|---|
| Qwen speed and capacity | partial | `docs/QWEN_ON_DGX_SPARK.md`, `docs/BENCHMARK_PROTOCOL.md`, `scripts/qwen_speed_lane.py`, and `tasks/qwen_speed_lane_sample.jsonl` make Qwen a mandatory runtime lane. SGLang Qwen2.5 BF16/auto and fp8 rows exist; llama.cpp Qwen2.5 Q4_K_M rows exist; vLLM Qwen3.6 source compatibility prep exists. The serving-manifest audit marks the llama.cpp Qwen2.5 Q4_K_M row claim-ready for practical serving evidence. | Live GB10 run against all target Qwen servers, especially vLLM Qwen3.6 NVFP4+DFlash once image acquisition is fixed. Broad runtime claims require both Qwen and Gemma rows, not one family generalized to the other. |
| Gemma compatibility and performance | partial | AEON Gemma 26B NVFP4+DFlash, vLLM Gemma 26B BF16, vLLM Gemma 12B source/precompiled, llama.cpp Gemma 26B Q4_0, and LiteRT-LM Gemma E2B side-runtime rows exist. | Clean official vLLM/SGLang paths, Gemma 12B release-container support, SGLang Gemma serving, and paper-comparable GGUF accuracy remain open. |

## Mechanical Coverage Audit

Run this after changing the solution plan, status table, issue tracker, or Qwen benchmark lane:

```bash
python3 scripts/solution_coverage_audit.py \
  --output results/solution_coverage_audit_20260608.json
```

Run this before using serving rows as runtime-claim evidence:

```bash
python3 scripts/serving_manifest_audit.py \
  --output results/serving_manifest_audit_20260608.json
```

## Highest-Leverage Next Proofs

Live GB10 required:

1. Reconnect to the host and finish AEON Qwen3.6 NVFP4+DFlash acquisition with `scripts/pull_container_with_evidence.sh`.
2. If the image registers, run `scripts/run_aeon_vllm_reproduction.sh qwen36-dflash ...` with `RECORD=1`.
3. Run `scripts/qwen_speed_lane.py --input tasks/qwen_speed_lane_sample.jsonl ...` against the live Qwen servers so vLLM, SGLang, and llama.cpp rows share one manifest shape.
4. Run a clean vLLM/SGLang fp8-vs-NVFP4 KV after-row with quality checks before blessing FP4 KV.

Offline or low-GPU work:

1. Validate `scripts/llamacpp_native_loglikelihood_probe.py` and `scripts/llamacpp_native_loglikelihood_task.py` against a live llama-server, then wire the same scoring shape into a tiny GGUF lm-eval/loglikelihood task.
2. Keep the acceptance table above current as each artifact lands.
3. Continue porting AEON/hikarioyama-compatible changes only when local evidence shows the corresponding blocker.
