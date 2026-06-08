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
| 5. vLLM first-class Spark runtime | partial | AEON Gemma 4 26B NVFP4+DFlash serves locally with warmed compact decode around `48-54 tok/s` and long-prefill around `98 tok/s`; Gemma 12B unified serves through source/precompiled path; AEON Qwen3.6 NVFP4+DFlash passes OpenAI smoke and compact serving when `chat_template_kwargs={"enable_thinking": false}` is set; `jethac/vllm@6804e1b` has a passing derived AEON Qwen row at `47.22`, `58.88`, and `61.62 tok/s`; clean `jethac/vllm@a919d635d` plus `jethac/flash-attention@7d53245` serves Qwen3.6 NVFP4+DFlash at `61.07`, `56.97`, and `60.10 tok/s` with separate `sm_121a` FA2 cubin proof; the derived Qwen NVFP4-KV row proves FlashInfer FA2 NVFP4 KV selection with `1.751x` fp8 KV pool/concurrency and decode-speed parity; Gemma 3 Rung 1 preflight confirms the host/image are ready, and the clean run checkout plus syntax-checked packet are prepared. | Clean official/blessed vLLM path, accuracy check, native FP4 weight/MoE proof, Gemma 3 fp8/NVFP4 live rows after gated model cache/access, and Gemma 4 NVFP4-KV through source overlay after the `D=512` FlashInfer blocker is fixed or routed around. |
| 6. Gemma 4 12B support | partial | `gemma4_unified` source/precompiled vLLM probe serves on GB10. | Clean release/nightly container, zero-shot lm-eval task, and performance-worthy path. |
| 7. NVFP4 end to end | mixed | AEON Gemma proves NVFP4 weight serving; FlashInfer FA2 NVFP4 KV standalone probes pass small, Gemma sliding/local, and SGLang-style linear V-scale tuple-KV signatures; vLLM Qwen NVFP4-KV now serves through FlashInfer FA2 with `11,146,226` KV tokens versus fp8 `6,364,935` (`1.751x`) and normal content; SGLang FP4 KV now has a matched `d7d931f` row with `5,517,572` FP4 tokens versus `3,105,240` fp8 tokens (`1.7769x`), raw/chat smoke passing, and decode plus `extend_merge_paged` backend traces through packed `uint8` K/V and FP8 scale buffers, but standardized benchmark content still degrades; the SGLang convention bridge proves the raw FA2 reader is matched for `fp4_quantize` with encode scale and for `nvfp4_kv_quantize` with decode scale, while the crossed `nvfp4_kv_quantize` encode-scale case fails; the SGLang pool bridge proves `MHATokenToKVPoolFP4` output is consumable by FlashInfer FA2 decode and paged prefill at `attention_cosine_vs_dequant=0.9999946`; `scripts/nvfp4_checkpoint_audit.py` guards NVFP4 checkpoint format, sensitive tensor quantization, and Gemma EOS metadata before Qwen/Gemma serving or GGUF conversion. TensorRT-LLM #11368 also shows FP4 CUTLASS GEMM on GB10 can fail below dispatch because SM120 tile configs exceed GB10's 99 KiB shared-memory limit. | Gemma global `D=512` path, SGLang FP4 KV benchmark quality and graph safety after the pool/backend trace, live checkpoint audits for the actual Qwen/Gemma NVFP4 rows, and GB10-specific FP4 GEMM tile proof rather than family-only dispatch proof. |
| 7a. SGLang first-class Spark runtime | partial | SGLang Qwen2.5 1.5B BF16/auto and fp8 KV serve around `58-59 tok/s`; fp8 roughly doubles KV pool over BF16/auto. A standalone FlashInfer FA2 NVFP4 KV probe passes the SGLang-style linear V-scale tuple-KV signature on GB10, the convention bridge narrows corruption away from raw FA2 math for the viable quantizer/reader pairs, and the pool bridge clears `MHATokenToKVPoolFP4` layout/global-scale application for decode and paged prefill. The clean FP4 KV source overlay now calibrates before capture and auto-disables graph capture for native FP4 KV; the matched `d7d931f` row allocates `5,517,572` FP4 tokens versus `3,105,240` fp8 tokens, traces all 28 decode and `extend_merge_paged` layers, and passes raw/chat smoke. OpenAI and native logprob probes now localize the remaining quality bug: native rendered-template `/generate` matches fp8 through token 3 and diverges at token 4 by rank reversal, while OpenAI Chat Completions looks more severe. | OpenAI-vs-native prompt/path reconciliation, Gemma serving path, graph-safe FP4 KV, and an accepted build-target proof. |
| 7b. LiteRT-LM optional side runtime | partial | Linux aarch64 install, CPU generation, and CPU/GPU benchmark evidence exist. | GPU chat exits `-11`; no blessed GPU serving path. |
| 8. llama.cpp / lm-eval accuracy | blocked | `gguf_logprobs_probe` identifies the OpenAI schema mismatch; `scripts/llamacpp_native_loglikelihood_probe.py` and `scripts/llamacpp_native_loglikelihood_task.py` now have a live llama-server result. The live native task scored likely continuations but missed the unlikely `zebra` continuation at `n_probs=512`. The updated OpenAI echo probe also failed on pinned `b9536`: both `max_tokens=0` and `max_tokens=1` returned `choices[0].logprobs.content` for generated token `-striped`, not prompt `tokens`/`token_logprobs` for supplied continuation tokens `[1147, 50213]`. | Find a newer llama.cpp pin with prompt-token logprobs, prove full-vocabulary probabilities are practical, or add a `jethac/llama.cpp` native scoring endpoint before tiny GGUF lm-eval/loglikelihood can pass. |
| 8a. llama.cpp practical serving and native FP4 | strong-serving / partial-native-fp4 | Gemma 4 26B Q4_0 and Qwen2.5 1.5B Q4_K_M serve through OpenAI API; `llama-bench`, build-target audit, model details, and hardware evidence are captured. `scripts/serving_manifest_audit.py` now marks the Qwen2.5 llama.cpp row claim-ready for serving evidence. The `jethac/llama.cpp@spark/native-fp4-sm121-20260608` submodule at `19bba67c1` builds with `121a`, rewrites `121` to `121a`, and emits native `mxf4nvf4.block_scale` PTX for `sm_121a`; `120f` is rejected by this CMake/CUDA 13.0 toolchain. | Native NVFP4/MXFP4 GGUF runtime dispatch, correctness, and speed remain unproven; paper-comparable GGUF accuracy remains separate. |
| 9. HF fallback containment | partial | Telemetry wrapper and failure annotator exist. | Historical `returncode=-9` rows need stronger OOM/resource evidence if HF fallback remains in comparisons. |
| 10. GB10 SM count and memory tuning | partial | Hardware comparison keys include compute capability and SM count; scripts collect `multi_processor_count`. | Performance tuning and regression thresholds across model families remain mostly unproven. |
| 11. Multi-Spark | missing | Single-unit assumption is documented. | No multi-Spark hardware or TP>1 validation. |
| 12. Improve benchmark design | partial | `scripts/spark_smoke_suite.py`, `scripts/openai_serving_benchmark.py`, telemetry wrapping, failure annotation, `scripts/qwen_speed_lane.py`, `scripts/serving_manifest_audit.py`, `scripts/counterpart_task_matrix.py`, and the tightened Gemma 3 vLLM packet split smoke, serving, Qwen speed/capacity, fragile fallback, claim-readiness, missing-counterpart task contracts, readiness waits, and server-log capture into narrower phases. | Live rows still need consistent phase completion, size-aware timeout policy, row-level failure explanations, and matched before/after manifests for every claimed runtime path. |
| 13. Observability | partial | `spark_doctor`, `cuda_so_audit`, `cuda_build_target_audit`, `container_target_audit`, runtime process probes, SM-count-aware hardware keys, server-log artifact capture, and serving-manifest audits exist. The Tailscale reconnect artifact proves the host is reachable at `100.113.98.11` and still reports GB10 / compute capability `12.1`; it also shows system Python has no Torch, so Torch-backed SM count needs an environment-specific doctor run. | Each blessed runtime still needs a no-silent-fallback artifact proving selected attention, quantization, KV, CUDA graph, and build/JIT targets before and after benchmarks; current AEON Gemma vLLM manifest has family/PTX container evidence but still lacks accepted native build-target evidence. |
| 14. Coordinate upstream ownership | partial | GitHub issues track the layer split; `docs/COMPATIBILITY_BOARD.md`, `docs/WHEEL_CONTAINER_MATRIX.md`, and `docs/AEON_PRIOR_ART_PORT_MAP.md` give maintainers a public status board, install matrix, and prior-art map. | Need recurring blessed-stack updates, public reproduction bundles, and upstream issue/PR taxonomy once matched GB10 before/after evidence exists. |
| 14a. Forks, submodules, worktrees, and subagents | partial | `jethac` FlashInfer, vLLM, SGLang, and llama.cpp forks/submodules/worktrees exist; patch branches are documented; the vLLM Qwen branch now includes AEON-derived Qwen/DFlash runtime fixes; `jethac/llama.cpp@spark/native-fp4-sm121-20260608` is pinned for native FP4 arch testing; `docs/AEON_PRIOR_ART_PORT_MAP.md` separates direct vLLM ports from SGLang/llama.cpp counterpart work; `scripts/counterpart_evidence_audit.py` tracks whether those counterpart rows have live artifacts; `tasks/counterpart_evidence_tasks.jsonl` defines the eight live task contracts. | No upstream PRs until matched before/after GB10 story is proven; every future fork change still needs issue branch, worktree path, commit SHA, and reproduction command. |
| 15. Publish honest recipes | partial | Runtime recipes, compatibility board, wheel/container matrix, blessed-stack notes, and Qwen/Gemma docs now record what works, what is slow, what is broken, and what remains untested. | A clean-unit reproduction for the blessed vLLM/SGLang/llama.cpp stack is still missing, and the recipes must stay tied to exact commands, versions, artifacts, and go/no-go decisions. |

## Cross-Cutting Required Lanes

| lane | status | current evidence | missing proof |
|---|---|---|---|
| Qwen speed and capacity | partial | `docs/QWEN_ON_DGX_SPARK.md`, `docs/BENCHMARK_PROTOCOL.md`, `scripts/qwen_speed_lane.py`, and `tasks/qwen_speed_lane_sample.jsonl` make Qwen a mandatory runtime lane. SGLang Qwen2.5 BF16/auto and fp8 rows exist; SGLang FP4 KV records a matched `d7d931f` capacity/routing/smoke row at `1.7769x` fp8 capacity but fails standardized benchmark quality; llama.cpp Qwen2.5 Q4_K_M rows exist; AEON, derived, and clean-FA2 `jethac/vllm` Qwen3.6 NVFP4+DFlash rows pass compact serving with normal content after disabling thinking; vLLM Qwen NVFP4-KV records a matched fp8-vs-NVFP4 capacity row at `1.751x` KV pool/concurrency and decode parity. | Larger llama.cpp Qwen3/Qwen3.6 GGUF row, SGLang DFlash/EAGLE row, SGLang FP4-KV quality, and native FP4 weight/MoE proof. Broad runtime claims require both Qwen and Gemma rows, not one family generalized to the other. |
| Gemma compatibility and performance | partial | AEON Gemma 26B NVFP4+DFlash, vLLM Gemma 26B BF16, vLLM Gemma 12B source/precompiled, llama.cpp Gemma 26B Q4_0, and LiteRT-LM Gemma E2B side-runtime rows exist. | Clean official vLLM/SGLang paths, Gemma 12B release-container support, SGLang Gemma serving, and paper-comparable GGUF accuracy remain open. |
| Gemma ladder / Rung -1 config audit | done-config-only | `docs/GEMMA_RUNG_MINUS1_CONFIG_AUDIT.md` and `results/gemma_rung_minus1_config_audit_20260608.json` confirm Gemma 3 27B is uniform `D=128` with SWA and no `D=512`; Gemma 4 12B, 31B, and 26B-A4B all have full-attention `D=512`; 31B is the dense `D=512` isolation rung before 26B-A4B adds MoE. | Running-model geometry still must be measured per runtime/rung before any serving row is green. |

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

1. Follow `docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md`: use `/home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608` and `docs/results/vllm_gemma3_27b_rung1_20260608TCHECKOUTJST_command_packet.sh`; ensure gated `google/gemma-3-27b-it` access/cache, then run the Gemma 3 27B fp8 comparator row before the NVFP4 candidate. After Gemma 3 is green, return to the FlashInfer FA2 Gemma global `D=512` path or mixed-KV fallback.
2. Follow `docs/CODEX_DIRECTION_SGLANG_NVFP4_KV.md`: use the matched `d7d931f`, OpenAI logprob, and native `/generate` divergence rows as the baseline, then reconcile the OpenAI Chat Completions path with the native rendered-template path.
3. Keep native FP4 weight/MoE proof separate from FA2 `sm_121a` proof; the passing clean Qwen row still selects Marlin weight-only FP4.
4. Run `scripts/qwen_speed_lane.py --input tasks/qwen_speed_lane_sample.jsonl ...` against the live Qwen servers so vLLM, SGLang, and llama.cpp rows share one manifest shape.

Offline or low-GPU work:

1. Continue the llama.cpp lane from the live negative rows and the native-FP4 arch probe: test whether full-vocabulary scoring is practical, try a newer llama.cpp pin, or add a native endpoint that returns logprobs for supplied continuation tokens; separately run an actual NVFP4 GGUF on `jethac/llama.cpp@19bba67c1` to prove runtime dispatch/correctness.
2. Keep the acceptance table above current as each artifact lands.
3. Keep `tasks/counterpart_evidence_tasks.jsonl` aligned with `results/counterpart_evidence_audit_20260608.json` as rows move from missing to live evidence.
4. Continue porting AEON/hikarioyama-compatible changes only when local evidence shows the corresponding blocker.
5. Add a focused FP4 GEMM note/test for the TensorRT-LLM #11368 class: GB10 may need separate tile configs even when SM12x dispatch and `121a` JIT targeting are correct.
