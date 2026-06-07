# Gemma 4 On DGX Spark: Problems Encountered

Date: 2026-06-07

This report lists the problems encountered while running the Gemma 4 benchmark campaign on the DGX Spark / PGX workstation-class system observed as `NVIDIA GB10`. It covers fatal failures, nonfatal benchmark failures, compatibility issues, runtime and performance issues, and operational problems from setup and monitoring.

The campaign was still running when monitoring was stopped. The latest local generated benchmark snapshot used for this report was `B:\workshop\20260606_BENCHMARKING.md`, last synced at 2026-06-07 18:40.

## Executive Summary

The biggest practical problem was not model loading for the vLLM safetensors rows. Those ran reliably once the campaign settled. The biggest problems were:

- GGUF accuracy through lm-eval was blocked by a llama.cpp API/logprobs compatibility mismatch.
- HF fallback rows were fragile and repeatedly exited with `returncode=-9`, especially on QAT/unquantized and larger models.
- Full HellaSwag runs were extremely slow and dominated campaign runtime.
- Initial timeouts were too short for large model load probes and MTP benchmarking.
- The benchmark matrix was much larger than the machine could finish quickly: smoke completed, but full accuracy was still running and throughput/MTP stages were still pending when monitoring stopped.
- A later compact vLLM serving check showed Gemma 4 26B A4B can serve on GB10 at about 24 tok/s decode, but the observed path was BF16/unquantized Triton MoE, not FlashInfer NVFP4.

## Fatal Or Blocking Problems

### GGUF Accuracy Was Not Paper-Comparable

All GGUF accuracy rows through lm-eval and stock llama.cpp failed as paper-comparable accuracy rows.

The cause was an adapter/API mismatch:

- lm-eval's GGUF path expected echoed prompt/continuation `token_logprobs`.
- llama.cpp `b9536` `/v1/completions` returned current content-style logprobs for generated tokens only.

Impact:

- 120 smoke rows were recorded as `loader_failed`.
- GGUF accuracy could not be compared to lm-eval loglikelihood safetensors rows.
- GGUF was still usable for throughput through `llama-bench`, but not for the target accuracy path.

This is a benchmark compatibility blocker, not evidence that the GGUF model files themselves are unusable.

### HF Fallback Rows Hit Process Kills

Several HF-backed rows exited with `returncode=-9`. These were treated as real benchmark failures and recorded in the JSONL artifacts.

Observed full-eval failures included:

| row | task/group | backend | failure |
|---|---|---|---|
| `unsloth-e2b-qat-q4-0-unquantized-bf16` | ARC Challenge 25-shot | HF | `returncode=-9` |
| `unsloth-e4b-qat-q4-0-unquantized-bf16` | zero-shot group | HF | `returncode=-9` for BoolQ, PIQA, ARC Easy |
| `google-12b-baseline-bf16` | ARC Challenge 25-shot | HF | `returncode=-9` |

Impact:

- These rows did not complete all selected tasks.
- HF fallback is useful as a labeling path, but it is not a reliable substitute for vLLM on this matrix.
- The larger HF rows and QAT/unquantized rows appear to be memory/process-fragile on this system.

## Nonfatal Compatibility Problems

### SM120/SM121 / Blackwell-Specific Issues Were Implicit, Not Explicitly Logged

I searched the local reports and the remote benchmark logs for `sm120`, `sm_120`, `sm121`, `sm_121`, `compute_120`, `compute_121`, CUDA architecture, kernel, Triton, and related compiler/runtime terms. No explicit `sm120`- or `sm121`-targeted build or runtime error was found in the captured artifacts.

The system was observed as `NVIDIA GB10`, which NVIDIA's CUDA GPU table lists as compute capability 12.1, i.e. `sm_121`. By contrast, RTX PRO 6000 Blackwell and GeForce RTX 50-class cards are listed as compute capability 12.0, i.e. `sm_120`. So the exact DGX Spark target is `sm_121`, while many related community reports say `sm120` because they are discussing the broader RTX Blackwell desktop/workstation ecosystem.

After reviewing external SM120/SM121-focused repos, the better conclusion is more nuanced:

- The local logs did not show an explicit "unsupported sm120/sm121" or missing cubin error.
- But several failures line up with known Blackwell/vLLM ecosystem maturity issues.
- The campaign used conservative TP=1 vLLM execution, so it avoided some known no-P2P multi-GPU hangs.

External SM120 notes from `lna-lab/gemma4-12b-vllm-sm120` identify these relevant issues:

- Gemma 4 12B uses the newer `gemma4_unified` architecture.
- released vLLM up through `0.22.1` is reported not to load that architecture natively; the repo says vLLM nightly is needed for `Gemma4UnifiedForConditionalGeneration`.
- when vLLM falls back to the Transformers backend, it can crash in global-attention / projection paths because of the dual-head-dimension attention shape.
- on no-NVLink / no-P2P Blackwell systems, TP>1 can hang unless both `NCCL_P2P_DISABLE=1` and `--disable-custom-all-reduce` are used.

Those points explain why our 12B vLLM smoke rows are suspect as SM120/vLLM-version issues rather than plain model failures. We used vLLM `0.22.1`, and several 12B rows failed vLLM load probes before the campaign used HF fallback.

External SM120 notes from `hikarioyama/vllm-nvfp4-kv-sm120` identify a separate issue:

- vLLM's default NVFP4 KV path does not work on SM120 because SM120 lacks the relevant trtllm-gen NVFP4-KV cubins.
- their workaround routes NVFP4 KV through a patched FlashInfer FA2 path with explicit scale-factor strides.
- this is specifically about `--kv-cache-dtype nvfp4`, not ordinary fp8/bf16 KV paths.

That NVFP4 KV issue was not directly encountered in our benchmark because the campaign did not use the patched NVFP4 KV cache path. It is still relevant for future DGX Spark optimization work, but DGX Spark validation should specifically check `sm_121`, not only `sm_120`.

User-provided X/Twitter context from June 5-6, 2026 also points in the same direction: community users are explicitly asking NVIDIA for more attention on `sm120`/`sm121` support for RTX PRO 6000-class systems and DGX Spark, while an NVIDIA representative acknowledged the gap and said the team is trying to put more focus on it. This is not a benchmark artifact, but it supports the interpretation that the failures should be read against an immature RTX/SM120/SM121 inference software ecosystem rather than as isolated local setup mistakes.

In this run, the observed facts were:

- vLLM safetensors rows ran for E2B, E4B, and 26B-A4B with sustained GPU utilization.
- stock llama.cpp CUDA throughput worked for at least the early GGUF throughput row.
- llama.cpp MTP executed at least one speed row.
- the major GGUF accuracy problem was an lm-eval/llama.cpp logprobs schema mismatch, not a CUDA architecture failure.
- the repeated HF failures surfaced as `returncode=-9`, consistent with process kill/resource pressure rather than an explicit `sm120` unsupported-kernel error.

Current conclusion:

- confirmed from our logs: no explicit `sm120` or `sm121` compiler/kernel error.
- likely from external SM120 reports: some 12B vLLM failures are probably due to released vLLM `0.22.1` not yet handling `gemma4_unified` natively on this stack.
- avoided by our setup: no-P2P TP>1 hangs, because our accuracy runs used TP=1.
- not exercised: SM120 NVFP4 KV cache failure path, because the benchmark did not use `--kv-cache-dtype nvfp4`.

Recommended follow-up:

1. Re-test 12B Gemma 4 rows with a vLLM nightly that has native `Gemma4UnifiedForConditionalGeneration`.
2. If testing TP>1 on DGX Spark/no-NVLink configurations, set `NCCL_P2P_DISABLE=1` and use `--disable-custom-all-reduce`.
3. Do not enable `--kv-cache-dtype nvfp4` on SM120/SM121 unless using a validated patched FlashInfer/vLLM path or an upstream release that incorporates the equivalent fix.
4. Keep fp8 KV as the safer default for MTP/speculative decode until NVFP4 KV quality and backend wiring are validated for the target model.

### vLLM Load Probe Failures

Some safetensors rows failed vLLM smoke/load probes with `returncode=1`.

Examples from smoke:

| row | backend | status |
|---|---|---|
| `google-12b-qat-q4-0-unquantized-bf16` | vLLM | `eval_failed` |
| `google-12b-qat-w4a16-w4a16` | vLLM | `eval_failed` |
| `unsloth-12b-baseline-bf16` | vLLM | `eval_failed` |
| `unsloth-12b-qat-q4-0-unquantized-bf16` | vLLM | `eval_failed` |
| `unsloth-12b-qat-w4a16-w4a16` | vLLM | `eval_failed` |
| mobile/mobile-transformers E2B/E4B QAT rows | vLLM | `eval_failed` |

Impact:

- These rows did not advance into the full vLLM accuracy path.
- The campaign correctly classified them instead of silently falling back without labeling.

### Gemma 4 26B vLLM Needs A Larger Multimodal Batch Token Budget

A compact follow-up serving check on 2026-06-07 used `vllm/vllm-openai:latest-cu130` and `google/gemma-4-26B-A4B-it`.

The default launch failed before readiness with:

```text
ValueError: Chunked MM input disabled but max_tokens_per_mm_item (2496) is larger than max_num_batched_tokens (2048). Please increase max_num_batched_tokens.
```

The same model reached readiness after adding `--max-num-batched-tokens 4096`.

Impact:

- This is a recipe/configuration issue, not a model-capacity failure.
- Future Gemma 4 26B vLLM recipes on this image should include `--max-num-batched-tokens 4096` or another validated value above 2496.
- This failure should be documented distinctly from CUDA architecture and FlashInfer issues.

### QAT-Unquantized 26B Loads, But Not As A Quantized NVFP4 Path

The cached `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized` snapshot also reached readiness under vLLM with the same `--max-num-batched-tokens 4096` setting.

Observed server log facts:

- `quantization=None`
- `dtype=torch.bfloat16`
- attention backend: `TRITON_ATTN`
- MoE backend: `TRITON Unquantized MoE`

Impact:

- The QAT-unquantized snapshot is usable for a vLLM smoke/serving check.
- It should not be treated as proof of an end-to-end QAT, FP4, or NVFP4 serving path.
- Any future quantized benchmark must prove the actual backend and quantization path from logs, dispatch probes, or profiler evidence.

### Backend Comparability Is Limited

The matrix used multiple inference paths:

- vLLM for most safetensors rows
- HF fallback for rows vLLM could not handle
- stock llama.cpp for GGUF throughput
- llama.cpp MTP PR for MTP speed

Impact:

- Accuracy values are only paper-comparable for lm-eval loglikelihood rows.
- Throughput results from vLLM and llama.cpp should not be directly conflated.
- HF fallback rows should be read as a separate backend class.

## Runtime And Performance Problems

### HellaSwag Dominated Wall Time

Full HellaSwag 10-shot runs were the single largest time sink. Observed completed durations:

| row | elapsed seconds | approx minutes |
|---|---:|---:|
| `google-e2b-baseline-bf16` | 5472 | 91.2 |
| `unsloth-e2b-baseline-bf16` | 5436 | 90.6 |
| `google-e4b-baseline-bf16` | 7890 | 131.5 |
| `unsloth-e4b-baseline-bf16` | 7860 | 131.0 |
| `google-e4b-qat-w4a16-w4a16` | 8401 | 140.0 |
| `unsloth-e4b-qat-w4a16-w4a16` | 8416 | 140.3 |
| `google-26b-a4b-baseline-bf16` | 9017 | 150.3 |

Impact:

- Full accuracy could not finish quickly even though the system stayed healthy.
- Each new HellaSwag row required long passive monitoring.
- The campaign remained in `full_accuracy` for many hours.

### Larger Models Were Slow But Mostly Healthy Under vLLM

vLLM rows for E2B, E4B, and 26B-A4B generally ran healthily with sustained GPU utilization around 90-96% during active inference.

Performance concern:

- The larger rows were stable, but the long HellaSwag and ARC timings mean the full manifest is costly to run end to end.
- At the last stop point, full accuracy had not yet finished, and throughput/MTP stages were still pending.

### Instantaneous GPU Samples Could Be Misleading

Several task transitions showed momentary low GPU utilization, often 0-2%, while the subprocess or vLLM engine had just started. Follow-up samples generally showed the engine settling into 90%+ GPU utilization.

Impact:

- A single low `nvidia-smi` sample was not enough evidence of a stall.
- Health checks needed to consider process tree, elapsed time, engine process state, and repeated samples.

## Timeout And Operational Problems

### Initial Load Probe Timeout Was Too Short

The initial five-minute load/probe window was not enough for some large model cases.

Fix applied:

- Smoke probe timeout: 600 seconds
- Large smoke probe timeout: 2400 seconds
- MTP default timeout: 900 seconds
- MTP large timeout: 2400 seconds

Validation:

- `bash -n run_campaign.sh` passed.
- `run_mtp_speed.py` compiled successfully.

Impact:

- Future smoke/MTP runs are less likely to be killed prematurely.
- The active full-accuracy stage was not affected by this timeout change.

### Campaign Was Interrupted Earlier By Exit 137

The campaign log included an earlier smoke stage ending with `rc=137`, followed by resumed smoke and full-accuracy stages.

Impact:

- This indicates an earlier kill or resource pressure event.
- The later resumed smoke stage completed with `rc=0`.
- The active campaign recovered and continued.

### Monitoring Was Manual And Long-Running

Many status checks were implemented as local delayed commands that slept and then sampled the remote host.

Operational issue:

- After the user interrupted the turn, an old local delayed monitor session no longer accepted stdin.
- No new long-running monitor was started after the stop request.
- The remote campaign itself was not killed.

Impact:

- Reporting is based on the latest synced generated report and the last observed live state.
- There may be newer remote results after the stop point if the campaign continued.

## Result-State Problems

### Generated Report Was A Snapshot, Not Final Campaign Output

At the last synced snapshot:

| area | state |
|---|---|
| smoke | complete |
| full accuracy | still running |
| throughput | only early rows present |
| MTP | only early rows present |
| final report stage | not reached by active campaign |

Impact:

- `20260606_BENCHMARKING.md` is useful, but not a final end-of-campaign report.
- Full throughput and MTP conclusions cannot be drawn from the current snapshot.

### Pending Work Was Still Large

At the last synced generated report:

- Full eval records: 70
- Full eval ok records: 65
- Full eval failed records: 5
- Throughput observed rows: 2
- MTP observed rows: 2 in JSONL, with 1 row shown in the generated summary

Impact:

- Accuracy progress was substantial but incomplete.
- Throughput and MTP coverage were not representative of the full manifest.

## Minor Issues And Caveats

### MTP Acceptance Was Not Reported

The MTP row that appeared in the generated report had `acceptance=not_reported`.

Impact:

- Prompt and generation tok/s were recorded.
- Draft acceptance rate could not be analyzed from that row.

### Report Size And Preview Limits

The generated benchmark report intentionally previewed only a subset of long tables.

Impact:

- Full details require inspecting the raw JSONL files under `/home/jethac/gemma4-evals/results`.
- The local markdown is a summary snapshot, not a full dump of every row.

### Mixed Quantization And Publisher Rows Need Careful Grouping

The matrix includes Google and Unsloth publishers, baseline and QAT variants, safetensors and GGUF, BF16 and quantized forms.

Impact:

- Some results are comparable only within the same backend and task path.
- W4A16 rows often improved HellaSwag compared with baseline rows, but this should not be generalized across backends or incomplete rows.

## Practical Recommendations

1. Treat GGUF accuracy through the current llama.cpp/lm-eval path as blocked until the logprobs API mismatch is solved.
2. Keep HF fallback rows labeled and separate; do not merge them into vLLM comparisons.
3. Consider splitting HellaSwag into a separate long-running campaign, because it dominates wall time.
4. Keep the larger timeout settings for future smoke and MTP runs.
5. Before drawing final throughput or MTP conclusions, let the campaign reach those stages or run targeted throughput/MTP subsets directly.
6. For future monitoring, prefer point-in-time remote status checks over local long sleeps if the user may interrupt the session.

## Last Known Active Problem

When monitoring stopped, the active campaign was still in full accuracy:

- Active row: `unsloth-26b-a4b-baseline-bf16`
- Active task: HellaSwag 10-shot
- Backend: vLLM
- Health: GPU and engine healthy in the last observed startup sample
- Completion: not yet present in the last synced report

The remote process was intentionally left running.
