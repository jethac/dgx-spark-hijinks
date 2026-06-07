# Benchmarking Report

Date: 2026-06-07

This report summarizes the PGX Workstation benchmarking work performed so far and the state of the generated benchmark artifacts at the point monitoring was stopped.

## Scope

The work monitored and summarized the initial personal Gemma 4 benchmark run described by `BENCHMARK_PLAN.md` on `thinkstationpgx-00b4`, with results synchronized into:

- Remote generated report: `/home/jethac/gemma4-evals/20260606_BENCHMARKING.md`
- Local generated report: `B:\workshop\20260606_BENCHMARKING.md`
- This narrative report: `B:\workshop\BENCHMARKING_REPORT.md`

The personal benchmark run was not terminated. Monitoring was stopped per request; the last observed remote process was still running.

Later targeted Spark probes were run outside the original personal campaign to make the compatibility story more actionable. Those results are indexed in `docs/BASELINE_RESULTS.md` and summarized below.

## Campaign Setup

The benchmark plan was revised to run host-native on the PGX workstation rather than through Docker. The active toolchain and sources recorded in the generated report were:

- vLLM `0.22.1`
- stock llama.cpp `b9536`
- llama.cpp MTP checkout from PR `23398`
- lm-eval-harness via the local Python environment

The manifest contained:

- 152 accuracy rows
- 248 MTP rows

The personal benchmark run stages were:

1. smoke safetensors
2. full accuracy
3. safetensors throughput
4. GGUF throughput
5. MTP speed
6. final report generation

During execution, timeout handling was adjusted for larger models:

- Safetensors smoke probes were raised to 600 seconds normally and 2400 seconds for large probes.
- MTP speed runs were given 900 second default timeout and 2400 second large-model timeout.
- `run_mtp_speed.py` was updated to accept those timeout arguments and record `timeout_s`.
- `bash -n run_campaign.sh` and Python compilation of `run_mtp_speed.py` passed after the edits.

## Last Synced Artifact State

The latest local generated benchmark snapshot was:

- File: `B:\workshop\20260606_BENCHMARKING.md`
- Last write time: 2026-06-07 18:40:15
- Size: 19,517 bytes

At that snapshot:

| artifact | status |
|---|---:|
| smoke rows | 152 complete |
| smoke ok | 21 |
| smoke eval_failed | 11 |
| smoke loader_failed | 120 |
| full eval task records | 70 |
| full eval ok records | 65 |
| full eval failed records | 5 |
| throughput JSONL rows observed | 2 |
| MTP JSONL rows observed | 2 |

The generated report showed the throughput and MTP sections as partial/early data only. The full throughput and MTP stages had not yet run in the active personal benchmark process by the last synced snapshot.

## Last Live Observation

The last live status sample before stopping showed:

- Campaign PID: `310239`
- Stage: `full_accuracy`
- Active model row: `unsloth-26b-a4b-baseline-bf16`
- Active task: `hellaswag`, 10-shot
- Backend: vLLM
- Active command: `lm_eval --model vllm ... --tasks hellaswag --num_fewshot 10`
- GPU utilization: 96%
- Active vLLM engine present and compute-bound
- Full result rows still at 70, meaning the active HellaSwag row had not yet written its result

The remote benchmark was not stopped.

## Accuracy Progress

The full-accuracy stage had progressed through E2B, E4B, 12B, and part of the 26B-A4B rows. The following rows were complete across all selected tasks unless noted otherwise.

| row | backend | status | notable result |
|---|---|---|---|
| `google-e2b-baseline-bf16` | vLLM | complete | HellaSwag `acc_norm=0.350229`, ARC Challenge `0.331058` |
| `unsloth-e2b-baseline-bf16` | vLLM | complete | HellaSwag `acc_norm=0.350229`, ARC Challenge `0.329352` |
| `unsloth-e2b-qat-q4-0-unquantized-bf16` | HF | partial | zero-shot completed; ARC Challenge failed with `returncode=-9` |
| `google-e2b-qat-w4a16-w4a16` | vLLM | complete | HellaSwag `acc_norm=0.463653`, ARC Challenge `0.438567` |
| `unsloth-e2b-qat-w4a16-w4a16` | vLLM | complete | HellaSwag `acc_norm=0.463155`, ARC Challenge `0.438567` |
| `google-e4b-baseline-bf16` | vLLM | complete | HellaSwag `acc_norm=0.467935`, ARC Challenge `0.391638` |
| `unsloth-e4b-baseline-bf16` | vLLM | complete | HellaSwag `acc_norm=0.467935`, ARC Challenge `0.392491` |
| `unsloth-e4b-qat-q4-0-unquantized-bf16` | HF | failed | zero-shot group failed with `returncode=-9` |
| `google-e4b-qat-w4a16-w4a16` | vLLM | complete | HellaSwag `acc_norm=0.485162`, WinoGrande `0.532755` |
| `unsloth-e4b-qat-w4a16-w4a16` | vLLM | complete | HellaSwag `acc_norm=0.485162`, WinoGrande `0.539858` |
| `google-12b-baseline-bf16` | HF | partial | zero-shot completed; ARC Challenge failed with `returncode=-9` |
| `google-26b-a4b-baseline-bf16` | vLLM | complete | HellaSwag `acc_norm=0.494822`, ARC Challenge `0.414676` |
| `unsloth-26b-a4b-baseline-bf16` | vLLM | in progress | zero-shot, ARC Challenge, and WinoGrande completed; HellaSwag running |

Selected 26B-A4B results observed before stopping:

| row | BoolQ | PIQA | ARC Easy | ARC Challenge | WinoGrande | HellaSwag |
|---|---:|---:|---:|---:|---:|---:|
| `google-26b-a4b-baseline-bf16` | 0.690826 | 0.576170 | 0.358165 | 0.414676 | 0.546172 | 0.494822 |
| `unsloth-26b-a4b-baseline-bf16` | 0.688991 | 0.577258 | 0.359007 | 0.414676 | 0.548540 | running |

## Failure Modes

The main failure modes observed were:

- GGUF lm-eval compatibility: llama.cpp `b9536` `/v1/completions` logprobs did not match what lm-eval's GGUF adapter requires for loglikelihood scoring. These rows were treated as compatibility failures for paper-comparable accuracy.
- HF fallback memory/process failures: several HF-backed QAT/unquantized or larger rows exited with `returncode=-9`.
- Some vLLM load probes for unsupported 12B/QAT/mobile rows failed during smoke.

The campaign correctly recorded these as benchmark outcomes rather than silently skipping them.

## Runtime Notes

HellaSwag dominated wall time. Observed full HellaSwag durations included:

| row | elapsed seconds |
|---|---:|
| `google-e2b-baseline-bf16` | 5472 |
| `unsloth-e2b-baseline-bf16` | 5436 |
| `google-e4b-baseline-bf16` | 7890 |
| `unsloth-e4b-baseline-bf16` | 7860 |
| `google-e4b-qat-w4a16-w4a16` | 8401 |
| `unsloth-e4b-qat-w4a16-w4a16` | 8416 |
| `google-26b-a4b-baseline-bf16` | 9017 |

The active `unsloth-26b-a4b-baseline-bf16` HellaSwag row had not completed at the last observation.

## Throughput And MTP

Only early throughput/MTP rows were present in the synced snapshot:

| area | observed row |
|---|---|
| safetensors throughput | `google-e2b-baseline-bf16`, vLLM, prompt `506.98 tok/s`, generation `130.17 tok/s` |
| GGUF throughput | `google-e2b-qat-q4_0...gguf`, llama.cpp, prompt `3923.63 tok/s`, generation `122.11 tok/s` |
| MTP speed | `mtp-12b-ud-iq2_m-q8_0...`, prompt `1042.40 tok/s`, generation `36.40 tok/s` |

The active personal benchmark run had not reached the full throughput or MTP stages by the last synced report.

## Targeted Follow-Up Benchmarks

After the initial personal campaign snapshot, targeted compact probes were added to answer specific Spark/GB10 questions.

| target | result | interpretation |
|---|---|---|
| vLLM Gemma 4 E4B W4A16 | compact OpenAI harness around 50-52 tok/s decode | first before row for an already-running vLLM server |
| SGLang 26.05 Qwen BF16 | Qwen smoke passed; short/medium/long-prefill decode around 59-60 tok/s | SGLang works on GB10 for at least one supported BF16 model, but this is not Gemma or NVFP4 |
| SGLang Gemma 4 E2B | failed before health; default path crashed in Gemma4 audio tower, `--language-only` required encoder URLs | SGLang Gemma4 model glue blocker, not a proven `sm_121` kernel failure |
| FlashInfer SM121 `mm_fp4` source/JIT | patched auto-dispatch includes `b12x`; finite outputs on GB10 | dispatch enablement, not a speedup claim |
| FlashInfer model-shaped proxies | dense-decode proxies mixed; MoE-shaped proxies slower after the patch | the one-line `b12x` gate is not enough to make Spark faster |
| vLLM Gemma 4 26B A4B | serves in `vllm/vllm-openai:latest-cu130` at about 24 tok/s after `--max-num-batched-tokens 4096` | useful BF16/unquantized MoE serving baseline, not NVFP4 |
| vLLM AEON Gemma 4 26B A4B NVFP4+DFlash | serves in `ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2`; warmed compact row is 47.91, 53.60, and 98.38 tok/s across short/medium/long-prefill cases | first local vLLM Gemma 26B row materially above the BF16 baseline; gain comes from AEON NVFP4 checkpoint/container plus DFlash, not from our fork |
| vLLM Gemma 4 12B | source/precompiled vLLM commit `da1daf40` plus Transformers main serves at about 7.7 tok/s | proves `gemma4_unified` can run on GB10, but not a clean release container or performance win |
| llama.cpp Gemma 4 26B Q4_0 | OpenAI-compatible serving around 76 tok/s decode with `--reasoning off`; `llama-bench` tg128 around 77 tok/s | practical GGUF serving path is blessed; lm-eval GGUF accuracy remains blocked |
| LiteRT-LM Gemma 4 E2B | CPU chat works; GPU benchmark works; GPU chat prints output then exits `-11` | optional side-runtime evidence, not a main Spark performance path |
| SGLang Qwen2.5 1.5B BF16/auto vs fp8 | matched `mem_fraction_static=0.40` rows run at about 58-59 tok/s; fp8 KV pool is 3.11M tokens vs 1.56M BF16/auto | fp8 is a capacity win with decode-speed parity on this small Qwen row |
| SGLang Qwen2.5 1.5B patched FP4 KV | patched overlay serves only with both CUDA graph modes disabled; FP4 KV pool is 5.54M tokens but short decode is 0.276 tok/s with repetitive output | FP4 KV capacity path is real, but SGLang FP4 KV is not a usable speed path yet |
| llama.cpp Qwen2.5 1.5B Q4_K_M | OpenAI-compatible serving around 167-175 tok/s decode; `llama-bench` tg128 around 178 tok/s | practical Qwen GGUF serving is proven; lm-eval GGUF accuracy remains blocked by logprobs schema |
| vLLM AEON Qwen3.6 35B-A3B NVFP4+DFlash | target and drafter weights downloaded; image `ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2` did not register after initial pull plus `timeout 900` retry | blocked before serving; not a model-load or kernel result yet |

The current headline has changed: a local end-to-end vLLM Gemma 26B NVFP4 serving win is now banked through AEON's container and checkpoint. The open question is no longer whether a Spark-class GB10 can run a fast NVFP4 Gemma 26B vLLM path; it is how much of that path belongs upstream, how to reproduce it for Qwen, and which parts should be carried in `jethac` forks.

## SM120 Reference Work

Two hikarioyama reference repos are now tracked as prior art:

- `hikarioyama/vllm-nvfp4-kv-sm120` at `f6156ee3b22b24885a52c02bdafb34a9c201fe86`
- `hikarioyama/sglang-nvfp4-kv-sm120` at `9b2160f0fb8e11dbbb5171a57f06a02b0e9ba6e2`

They are relevant because they implement NVFP4 KV paths through vLLM/SGLang plus FlashInfer FA2 changes on SM120 RTX Blackwell systems. They are not GB10 `sm_121` validation. The repo policy is to build on them through clean `jethac` forks and worktrees, not vendor overlay trees into production images.

The vLLM reference changes the measurement priority. Its headline is not weight-GEMM speed: it reports roughly 1.78x fp8 KV pool and much higher maximum concurrency at matched utilization, while decode stays near fp8 parity. For Spark, the first useful proof should therefore measure KV pool tokens, maximum concurrency, hidden scratch allocations, quality, and long-context behavior. Decode tok/s is still recorded, but a flat decode result can be acceptable if capacity and quality improve.

## Open State

At the point monitoring stopped:

- Smoke was complete.
- Full accuracy was still running.
- The active row was `unsloth-26b-a4b-baseline-bf16` HellaSwag.
- Throughput stages were still pending in the active personal run.
- MTP stage was still pending in the active personal run.
- The final generated report was a current snapshot, not completion of that personal run.

## Files

- `B:\workshop\20260606_BENCHMARKING.md`: generated benchmark snapshot
- `B:\workshop\BENCHMARKING_REPORT.md`: this narrative status report
- Remote artifacts remain under `/home/jethac/gemma4-evals/results` and `/home/jethac/gemma4-evals/logs`.
