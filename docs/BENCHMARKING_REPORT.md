# Benchmarking Report

Date: 2026-06-07

This report summarizes the PGX Workstation benchmarking work performed so far and the state of the generated benchmark artifacts at the point monitoring was stopped.

## Scope

The work monitored and summarized the initial personal Gemma 4 benchmark run described by `BENCHMARK_PLAN.md` on `thinkstationpgx-00b4`, with results synchronized into:

- Remote generated report: `/home/jethac/gemma4-evals/20260606_BENCHMARKING.md`
- Local generated report: `B:\workshop\20260606_BENCHMARKING.md`
- This narrative report: `B:\workshop\BENCHMARKING_REPORT.md`

The personal benchmark run was not terminated. Monitoring was stopped per request; the last observed remote process was still running.

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
