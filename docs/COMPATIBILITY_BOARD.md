# DGX Spark Compatibility Board

Date: 2026-06-08 JST

Purpose: one public, recurring status view for the Spark-class GB10 local-AI stack. This board does not replace the detailed reports; it tells a reader what is usable today, what is only a proof, and what exact artifact is missing next.

Hardware scope: single Spark-class GB10 system, compute capability `12.1` / `sm_121`. Multi-Spark and TP>1 remain unvalidated.

Latest reachability evidence: `results/spark_doctor_tailnet_reconnect_20260608T074035JST.json` from `thinkstationpgx-00b4.tail740c8d.ts.net` / `100.113.98.11`. The old `192.168.68.112` LAN address was not reachable from the Windows client at that time.

## Status Legend

| status | meaning |
|---|---|
| `blessed` | usable for the named scope with local artifacts and clear caveats |
| `partial` | works for some rows, but not enough to recommend as a general Spark path |
| `debug-only` | useful evidence, not a usable serving path |
| `blocked` | next proof cannot proceed until the named external/runtime blocker is cleared |
| `side-runtime` | useful complement, not a main throughput path |

## Runtime Board

| runtime | status | best local evidence | not blessed yet | next proof | issue |
|---|---|---|---|---|---|
| vLLM | `partial` | AEON Gemma 4 26B A4B NVFP4+DFlash serves locally at about `48-54 tok/s` short/medium decode and `98 tok/s` long-prefill; Gemma 12B unified serves through a source/precompiled path; AEON Qwen3.6 NVFP4+DFlash starts and generates completion tokens but fails output validation | official/clean Spark container path, passing Qwen3.6 NVFP4+DFlash local row, matched `jethac/vllm` fork row, accuracy check, explicit `sm_121`/`sm_121a` build-target proof | fix Qwen3.6 response/content path, rerun `scripts/run_aeon_vllm_reproduction.sh qwen36-dflash ...` with `RECORD=1`, then run a matched fork row | [#6](https://github.com/jethac/dgx-spark-hijinks/issues/6), [#20](https://github.com/jethac/dgx-spark-hijinks/issues/20) |
| SGLang | `partial` | NVIDIA 26.05 serves Qwen2.5 1.5B BF16/auto and fp8 at about `58-59 tok/s`; fp8 doubles KV pool versus BF16/auto | Gemma serving, clean graph-compatible FP4 KV, output quality checks, explicit `sm_121` build-target proof | build clean `jethac/sglang` FP4-KV container and run fp8-vs-FP4 KV quality/capacity/throughput row | [#14](https://github.com/jethac/dgx-spark-hijinks/issues/14), [#18](https://github.com/jethac/dgx-spark-hijinks/issues/18) |
| llama.cpp | `blessed` for practical serving | Gemma 4 26B Q4_0 serves around `76 tok/s`; Qwen2.5 1.5B Q4_K_M serves around `167-175 tok/s`; logs include `CUDA : ARCHS = 1210` and CUDA graphs | paper-comparable GGUF lm-eval accuracy; native NVFP4/MXFP4 GGUF tensor-core proof; larger Qwen3/Qwen3.6 GGUF rows | start llama-server and run `scripts/llamacpp_native_loglikelihood_task.py --input tasks/llamacpp_loglikelihood_smoke.jsonl ...` | [#17](https://github.com/jethac/dgx-spark-hijinks/issues/17), [#8](https://github.com/jethac/dgx-spark-hijinks/issues/8) |
| FlashInfer | `partial` | SM121 `mm_fp4` dispatch patch enables `b12x`; FA2 NVFP4 KV standalone probes pass small and Gemma sliding/local shapes | serving speedup, Gemma global `D=512` path, end-to-end vLLM/SGLang NVFP4 KV, clean wheel/container proof | run clean forked FlashInfer with vLLM/SGLang after-row; keep `b12x` as enablement until serving rows prove speed | [#7](https://github.com/jethac/dgx-spark-hijinks/issues/7) |
| LiteRT-LM | `side-runtime` | CPU generation works; GPU benchmark runs for small Gemma row | GPU chat exits `-11`; not a throughput path | decide CPU/complement recommendation or isolate GPU chat crash | [#16](https://github.com/jethac/dgx-spark-hijinks/issues/16) |
| HF fallback | `partial` | telemetry and failure annotation scripts exist | several fallback rows die with `returncode=-9`; resource cause not always proven | add stronger OOM/resource evidence before using HF fallback in comparisons | [#9](https://github.com/jethac/dgx-spark-hijinks/issues/9) |

## Model And Feature Board

| lane | status | current call | next proof |
|---|---|---|---|
| Gemma 4 26B | `partial` | fast local vLLM NVFP4+DFlash row exists through AEON image; llama.cpp Q4_0 is practical and fast | accuracy checks, fork parity, and native/forked vLLM path |
| Gemma 4 12B | `partial` | source/precompiled vLLM probe serves at about `7.7 tok/s` | clean release/nightly container plus one zero-shot task |
| Qwen speed | `partial` | SGLang small Qwen rows and llama.cpp Qwen2.5 row exist; AEON vLLM Qwen36 starts and emits completion tokens but is not a valid speed row | passing AEON Qwen3.6 NVFP4+DFlash local serving row, then matched fork row |
| NVFP4 weights | `partial` | AEON Gemma proves compressed-tensors NVFP4 weight serving on GB10 | local Qwen36 row and clean fork parity |
| NVFP4 / FP4 KV | `debug-only` | standalone and overlay evidence prove capacity potential, but SGLang FP4 KV collapses to unusable speed or compile failures | clean graph-compatible serving with fp8 comparator and quality check |
| GGUF accuracy | `blocked` | OpenAI-compatible llama.cpp logprobs schema is insufficient; native task harness is ready | live llama-server native loglikelihood task, then tiny lm-eval/loglikelihood adapter |
| Multi-Spark | `blocked` | design-only | second unit or remote equivalent |

## Live Proof Queue

Run these when the GB10 host is reachable.

### vLLM Qwen36 NVFP4+DFlash

```bash
DOWNLOAD=0 DOCKER_PULL=0 RECORD=1 \
scripts/run_aeon_vllm_reproduction.sh \
  qwen36-dflash aeon_qwen36_dflash_YYYYMMDDTHHMMJST
```

Keep the row non-claim-ready until chat smoke returns normal validated content and the serving benchmark records usable output, not only completion-token counts.

### Qwen Speed Lane

After the target Qwen servers are already running:

```bash
python3 scripts/qwen_speed_lane.py \
  --input tasks/qwen_speed_lane_sample.jsonl \
  --campaign-id qwen_speed_lane_YYYYMMDDTHHMMJST \
  --continue-on-error
```

Before starting the seven AEON-derived counterpart rows, validate the live task matrix:

```bash
python3 scripts/counterpart_task_matrix.py \
  --tasks tasks/counterpart_evidence_tasks.jsonl \
  --audit results/counterpart_evidence_audit_20260608.json \
  --output results/counterpart_task_matrix_YYYYMMDDTHHMMJST.json
```

### llama.cpp Native Loglikelihood

After starting the target llama-server:

```bash
python3 scripts/llamacpp_native_loglikelihood_task.py \
  --url http://127.0.0.1:8080 \
  --n-probs 512 \
  --input tasks/llamacpp_loglikelihood_smoke.jsonl \
  --output results/llamacpp_native_loglikelihood_task_YYYYMMDDTHHMMJST.json
```

### SGLang FP4 KV After-Row

Use the clean `jethac/sglang` fork/container, not a site-package overlay, and record:

- BF16/auto or fp8 comparator
- FP4 KV row with the same model, prompts, memory fraction, and graph mode
- KV pool tokens
- selected attention backend
- CUDA graph status
- deterministic output sanity
- quality comparator before any speed/capacity claim

## Update Cadence

Update this board whenever any of the following changes:

- a new runtime row is captured
- a fork branch advances for vLLM, SGLang, FlashInfer, or llama.cpp
- a blocker changes from acquisition/setup to model/runtime/kernel failure
- a row becomes blessed or unblessed
- a new live-proof command supersedes one of the queue entries above

Each update must include:

- artifact path under `results/`
- exact runtime/container/commit
- hardware key with compute capability and SM count when live hardware is involved
- GitHub issue comment link when pushed publicly

Detailed acceptance evidence remains in `docs/SOLUTIONS_STATUS.md`, runtime-specific docs, and the `results/` artifacts.

The install/package-level companion is `docs/WHEEL_CONTAINER_MATRIX.md`.
