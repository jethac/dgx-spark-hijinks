# DGX Spark Solutions Status

Date: 2026-06-08

This file maps `docs/DGX_SPARK_SOLUTIONS.md` to current evidence. It is intentionally conservative: a row is `strong` only when artifacts prove the relevant acceptance test, not merely when a runtime once started.

## Status Map

| solution area | status | current evidence | missing proof |
|---|---|---|---|
| 1. `sm_121` target naming and build flags | partial | `spark_doctor` records GB10 compute capability `12.1` and `multi_processor_count=48` in earlier Torch-backed artifacts; the 2026-06-08 Tailscale reconnect doctor confirms live `NVIDIA GB10` / compute capability `12.1` / CUDA `13.0`; llama.cpp logs/audits include `CUDA : ARCHS = 1210`; `scripts/container_target_audit.py` now records image/container family/PTX evidence separately from native target proof. | vLLM and SGLang still need explicit Spark-compatible build-target evidence in container logs or binary inspection, not just runtime device capability or `12.0+PTX` family evidence. |
| 2. Wheel and container matrix | partial | `docs/WHEEL_CONTAINER_MATRIX.md` now indexes proven, partial, blocked, and debug-only stack paths; AEON Gemma vLLM, NVIDIA SGLang Qwen BF16/fp8, llama.cpp source serving, LiteRT-LM side-runtime, and fork source probes are recorded with artifacts. | Clean Spark-blessed install/container matrix across vLLM, SGLang, FlashInfer, PyTorch-adjacent wheels, and CUDA 13 ARM64 remains missing. |
| 3. Kernel dispatch architecture awareness | partial | AEON Gemma logs show `TRITON_ATTN`, `FlashInferCutlassNvFp4LinearKernel`, `VLLM_CUTLASS` NvFp4 MoE, DFlash, and CUDA graphs. SGLang Qwen rows log attention/KV state. | Need broader no-silent-fallback proof across vLLM/SGLang and explicit failure messages for datacenter-only paths. |
| 4. Separate datacenter Blackwell from Spark Blackwell | partial | Diagnosis and recipe docs distinguish `sm_100`, `sm_120`, and `sm_121`; SM-count-aware hardware key exists. | More Spark-specific upstream/runtime docs are needed before the recipe set is user-boring. |
| 5. vLLM first-class Spark runtime | partial | AEON Gemma 4 26B NVFP4+DFlash serves locally with warmed compact decode around `48-54 tok/s` and long-prefill around `98 tok/s`; Gemma 12B unified serves through source/precompiled path; AEON Qwen3.6 NVFP4+DFlash passes OpenAI smoke and compact serving when `chat_template_kwargs={"enable_thinking": false}` is set; `jethac/vllm@6804e1b` now has a passing derived AEON Qwen row after dependency alignment, with compact decode `47.22`, `58.88`, and `61.62 tok/s`. The AEON Gemma container audit records GB10 runtime evidence plus `12.0+PTX`/`sm_120` family evidence. The derived `jethac/vllm` in-container audit records GB10 runtime evidence but no inspected `sm_121`/`sm_121a` CUDA object evidence. | Clean official/blessed vLLM path, accuracy check, clean fork wheel/container without AEON FA2 binary, and explicit native `sm_121` or `sm_121a` build-target proof. |
| 6. Gemma 4 12B support | partial | `gemma4_unified` source/precompiled vLLM probe serves on GB10. | Clean release/nightly container, zero-shot lm-eval task, and performance-worthy path. |
| 7. NVFP4 end to end | mixed | AEON Gemma proves NVFP4 weight serving; FlashInfer FA2 NVFP4 KV standalone probes pass small, Gemma sliding/local, and SGLang-style linear V-scale tuple-KV signatures; SGLang FP4 KV now serves Qwen correctly by auto-disabling CUDA graph capture after pre-capture calibration; `scripts/nvfp4_checkpoint_audit.py` guards NVFP4 checkpoint format, sensitive tensor quantization, and Gemma EOS metadata before Qwen/Gemma serving or GGUF conversion. TensorRT-LLM #11368 also shows FP4 CUTLASS GEMM on GB10 can fail below dispatch because SM120 tile configs exceed GB10's 99 KiB shared-memory limit. | End-to-end vLLM/SGLang `nvfp4` KV fp8-vs-NVFP4 capacity/throughput rows, graph-safe SGLang FP4 KV, Gemma global `D=512` path, live checkpoint audits for the actual Qwen/Gemma NVFP4 rows, and GB10-specific FP4 GEMM tile proof rather than family-only dispatch proof. |
| 7a. SGLang first-class Spark runtime | partial | SGLang Qwen2.5 1.5B BF16/auto and fp8 KV serve around `58-59 tok/s`; fp8 roughly doubles KV pool over BF16/auto. A standalone FlashInfer FA2 NVFP4 KV probe passes the SGLang-style linear V-scale tuple-KV signature on GB10. The clean FP4 KV source overlay now calibrates before capture; graph-enabled decode is corrupt, but the fork auto-disables graph capture for native FP4 KV and the default launch passes `spark-ok` plus raw `2+2 is 4`. | Gemma serving path, matched fp8-vs-fp4 KV comparator, capacity/throughput row with graph policy recorded, and a graph-safe FP4 KV fix if we want performance parity rather than correctness-only serving. |
| 7b. LiteRT-LM optional side runtime | partial | Linux aarch64 install, CPU generation, and CPU/GPU benchmark evidence exist. | GPU chat exits `-11`; no blessed GPU serving path. |
| 8. llama.cpp / lm-eval accuracy | missing | `gguf_logprobs_probe` identifies the schema mismatch; `scripts/llamacpp_native_loglikelihood_probe.py` prototypes lm-eval-shaped scoring; `scripts/llamacpp_native_loglikelihood_task.py` adds a tiny JSONL task harness with local dry-run evidence; `scripts/counterpart_evidence_audit.py` keeps the live task proof marked missing. | Native probe and task harness must prove arbitrary continuation-token logprobs against a live llama-server, then a tiny GGUF lm-eval/loglikelihood task must pass; current OpenAI-compatible API returns `logprobs.content`, not `tokens` plus `token_logprobs`. |
| 8a. llama.cpp practical serving path | strong | Gemma 4 26B Q4_0 and Qwen2.5 1.5B Q4_K_M serve through OpenAI API; `llama-bench`, build-target audit, model details, and hardware evidence are captured. `scripts/serving_manifest_audit.py` now marks the Qwen2.5 llama.cpp row claim-ready for serving evidence. | Native NVFP4/MXFP4 GGUF and paper-comparable GGUF accuracy remain separate. |
| 9. HF fallback containment | partial | Telemetry wrapper and failure annotator exist. | Historical `returncode=-9` rows need stronger OOM/resource evidence if HF fallback remains in comparisons. |
| 10. GB10 SM count and memory tuning | partial | Hardware comparison keys include compute capability and SM count; scripts collect `multi_processor_count`. | Performance tuning and regression thresholds across model families remain mostly unproven. |
| 11. Multi-Spark | missing | Single-unit assumption is documented. | No multi-Spark hardware or TP>1 validation. |
| 12. Improve benchmark design | partial | `scripts/spark_smoke_suite.py`, `scripts/openai_serving_benchmark.py`, telemetry wrapping, failure annotation, `scripts/qwen_speed_lane.py`, `scripts/serving_manifest_audit.py`, and `scripts/counterpart_task_matrix.py` split smoke, serving, Qwen speed/capacity, fragile fallback, claim-readiness, and missing-counterpart task contracts into narrower phases. | Live rows still need consistent phase completion, size-aware timeout policy, row-level failure explanations, and matched before/after manifests for every claimed runtime path. |
| 13. Observability | partial | `spark_doctor`, `cuda_so_audit`, `cuda_build_target_audit`, `container_target_audit`, runtime process probes, SM-count-aware hardware keys, server-log artifact capture, and serving-manifest audits exist. The Tailscale reconnect artifact proves the host is reachable at `100.113.98.11` and still reports GB10 / compute capability `12.1`; it also shows system Python has no Torch, so Torch-backed SM count needs an environment-specific doctor run. | Each blessed runtime still needs a no-silent-fallback artifact proving selected attention, quantization, KV, CUDA graph, and build/JIT targets before and after benchmarks; current AEON Gemma vLLM manifest has family/PTX container evidence but still lacks accepted native build-target evidence. |
| 14. Coordinate upstream ownership | partial | GitHub issues track the layer split; `docs/COMPATIBILITY_BOARD.md`, `docs/WHEEL_CONTAINER_MATRIX.md`, and `docs/AEON_PRIOR_ART_PORT_MAP.md` give maintainers a public status board, install matrix, and prior-art map. | Need recurring blessed-stack updates, public reproduction bundles, and upstream issue/PR taxonomy once matched GB10 before/after evidence exists. |
| 14a. Forks, submodules, worktrees, and subagents | partial | `jethac` FlashInfer, vLLM, and SGLang forks/submodules/worktrees exist; patch branches are documented; the vLLM Qwen branch now includes AEON-derived Qwen/DFlash runtime fixes; `docs/AEON_PRIOR_ART_PORT_MAP.md` separates direct vLLM ports from SGLang/llama.cpp counterpart work; `scripts/counterpart_evidence_audit.py` tracks whether those counterpart rows have live artifacts; `tasks/counterpart_evidence_tasks.jsonl` defines the seven missing live task contracts. | No upstream PRs until matched before/after GB10 story is proven; every future fork change still needs issue branch, worktree path, commit SHA, and reproduction command. |
| 15. Publish honest recipes | partial | Runtime recipes, compatibility board, wheel/container matrix, blessed-stack notes, and Qwen/Gemma docs now record what works, what is slow, what is broken, and what remains untested. | A clean-unit reproduction for the blessed vLLM/SGLang/llama.cpp stack is still missing, and the recipes must stay tied to exact commands, versions, artifacts, and go/no-go decisions. |

## Cross-Cutting Required Lanes

| lane | status | current evidence | missing proof |
|---|---|---|---|
| Qwen speed and capacity | partial | `docs/QWEN_ON_DGX_SPARK.md`, `docs/BENCHMARK_PROTOCOL.md`, `scripts/qwen_speed_lane.py`, and `tasks/qwen_speed_lane_sample.jsonl` make Qwen a mandatory runtime lane. SGLang Qwen2.5 BF16/auto and fp8 rows exist; llama.cpp Qwen2.5 Q4_K_M rows exist; AEON and derived `jethac/vllm` Qwen3.6 NVFP4+DFlash rows now pass compact serving with normal content after disabling thinking. The serving-manifest audit marks practical Qwen serving evidence claim-ready. | Larger llama.cpp Qwen3/Qwen3.6 GGUF row, SGLang DFlash/EAGLE row, clean vLLM fork packaging, and native `sm_121a`/FP4 target proof. Broad runtime claims require both Qwen and Gemma rows, not one family generalized to the other. |
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

Run this before saying AEON prior art has been covered outside vLLM:

```bash
python3 scripts/counterpart_evidence_audit.py \
  --output results/counterpart_evidence_audit_20260608.json
```

Run this before the next live GB10 session to confirm every missing counterpart row has a task contract:

```bash
python3 scripts/counterpart_task_matrix.py \
  --tasks tasks/counterpart_evidence_tasks.jsonl \
  --audit results/counterpart_evidence_audit_20260608.json \
  --output results/counterpart_task_matrix_20260608.json
```

## Highest-Leverage Next Proofs

Live GB10 required:

1. Replace the AEON FA2 binary dependency in the passing `jethac/vllm` Qwen image with a clean fork CUDA/FA2 build, then rerun the no-think Qwen row and the in-container `.so`/JIT target audit.
2. Keep the Qwen row native-target status separate until `sm_121`/`sm_121a` evidence is recorded or explicitly classified as family/PTX-only.
3. Run `scripts/qwen_speed_lane.py --input tasks/qwen_speed_lane_sample.jsonl ...` against the live Qwen servers so vLLM, SGLang, and llama.cpp rows share one manifest shape.
4. Turn the SGLang FP4 KV correctness-safe row into a benchmark row: matched fp8 comparator, capacity tokens, quality checks, and graph policy recorded. Current fork default disables graph capture for native FP4 KV because graph-enabled decode corrupts output.

Offline or low-GPU work:

1. Validate `scripts/llamacpp_native_loglikelihood_probe.py` and `scripts/llamacpp_native_loglikelihood_task.py` against a live llama-server, then wire the same scoring shape into a tiny GGUF lm-eval/loglikelihood task.
2. Keep the acceptance table above current as each artifact lands.
3. Keep `tasks/counterpart_evidence_tasks.jsonl` aligned with `results/counterpart_evidence_audit_20260608.json` as rows move from missing to live evidence.
4. Continue porting AEON/hikarioyama-compatible changes only when local evidence shows the corresponding blocker.
5. Add a focused FP4 GEMM note/test for the TensorRT-LLM #11368 class: GB10 may need separate tile configs even when SM12x dispatch and `121a` JIT targeting are correct.
