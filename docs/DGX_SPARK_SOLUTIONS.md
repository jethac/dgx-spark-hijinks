# DGX Spark Solutions Plan

Date: 2026-06-07

Goal: make DGX Spark boring for local AI.

Not heroic. Not "follow this Discord thread and install three nightlies." Boring.

That means a developer should be able to install a supported stack, run Gemma-class models through vLLM, SGLang, llama.cpp, or LiteRT-LM, know which kernels are being used, and get results that are both fast and explainable.

## 1. Name The Target Correctly

Weakness: people keep saying `sm120` when DGX Spark is `sm_121`.

Plan:

- Standardize docs and issue templates on: DGX Spark / GB10 = compute capability 12.1 = `sm_121`.
- Treat RTX PRO 6000 / RTX 50-series = `sm_120` as adjacent, not identical.
- In every build script, expose the target list explicitly instead of burying it in defaults.
- For CMake projects, use `CMAKE_CUDA_ARCHITECTURES="121-real"` for Spark-native builds.
- For Python/CUDA extension builds, document the equivalent `TORCH_CUDA_ARCH_LIST` and PTX fallback behavior per project.

Acceptance test:

- A fresh build log clearly shows `sm_121` or `121-real`.
- The resulting `.so` contains Spark-compatible code, verified with `cuobjdump`, `nvdisasm`, or project-native inspection.

## 2. Fix The Wheel And Container Matrix

Weakness: many packages assume x86_64, CUDA 12, or pre-Spark Blackwell targets.

Plan:

- Add CI build legs for Linux `aarch64` + CUDA 13 + Python versions used by DGX OS.
- Publish wheels for PyTorch-adjacent packages that currently force source builds on Spark.
- For vLLM, publish and document Spark-validated containers, not just generic Blackwell containers.
- Stop strict dependency pins that make the only working NGC PyTorch build unusable with vLLM.
- Add a small import/startup smoke test to every wheel release: import package, load CUDA extension, run one tiny kernel on `sm_121`.

Acceptance test:

- On a clean DGX Spark, `pip install` or the documented container path works without source compilation for the blessed stack.
- No runtime `libcudart.so.12` error on a CUDA 13 system.
- No warning that the GPU architecture is unsupported unless there is a documented and harmless compatibility path.

## 3. Make Kernel Dispatch Architecture-Aware

Weakness: libraries may have kernels for nearby architectures but fail to route Spark correctly.

Plan:

- Audit dispatch tables for `sm_121` handling across vLLM, SGLang, LiteRT-LM, FlashInfer, CUTLASS consumers, Triton custom kernels, TensorRT-LLM plugins, llama.cpp CUDA, and quantization libraries.
- Add explicit `sm_121` cases where code currently checks only `sm_80`, `sm_90`, `sm_100`, or `sm_120`.
- Ship PTX where appropriate so forward-compatible JIT can save users from missing SASS.
- Refuse silent fallback to old kernels when performance would be misleading; log the selected backend and architecture.
- Add one-line runtime diagnostics: selected attention backend, GEMM backend, quantization backend, KV cache backend, CUDA arch.

Acceptance test:

- Running a model prints or exposes the actual selected backend path.
- A Spark run never silently uses an Ampere-only or Hopper-only kernel path.
- Unsupported datacenter-only kernels fail with a useful error and recommended replacement.

## 4. Separate Datacenter Blackwell From Spark Blackwell

Weakness: `sm_100` recipes get copied to GB10 even when the hardware model is different.

Plan:

- Maintain separate recipe families:
  - B200/GB200/GB300 datacenter Blackwell
  - RTX Blackwell `sm_120`
  - DGX Spark GB10 `sm_121`
- Mark unsupported assumptions directly: TMEM, WGMMA, NVSwitch, NVLink, high-bandwidth HBM, high-SM-count tuning.
- In vLLM/SGLang docs, put Spark-specific flags in their own section instead of burying them under generic Blackwell.
- Add negative tests for flags known to hurt Spark, such as unnecessary eager mode or inappropriate distributed collectives.

Acceptance test:

- A user can find a Spark recipe without translating from H100/B200 instructions.
- The recipe states what was tested: model, quantization, container, driver, CUDA, backend, flags, and throughput.

## 5. Fix vLLM For Spark As A First-Class Runtime

Weakness: vLLM can work, but the working path is stack-sensitive.

Plan:

- Maintain an official Spark compatibility matrix by vLLM version/container tag.
- Keep CUDA graphs enabled by default; document `--enforce-eager` only as a workaround for specific bugs.
- Validate `sm_121` attention backends and make the default choice Spark-safe.
- Keep `--gpu-memory-utilization`, `--max-num-seqs`, and `--max-model-len` guidance tuned for unified 128 GB memory.
- Add startup checks that detect Spark and warn on known-bad flags.
- Track Gemma 4 architecture support explicitly, including `gemma4_unified`.

Acceptance test:

- A documented vLLM container serves a known Gemma-class model on Spark without manual dependency surgery.
- A smoke request works through the OpenAI-compatible API.
- The server reports CUDA graphs status and selected attention backend.

## 6. Fix Gemma 4 12B Support

Weakness: Gemma 4 12B uses newer architecture paths that released vLLM builds may not handle natively.

Plan:

- Retest Gemma 4 12B with the first vLLM release/nightly that has native `Gemma4UnifiedForConditionalGeneration`.
- Add Gemma 4 12B to the Spark CI smoke set.
- Test baseline BF16 and QAT variants separately; do not assume one proves the other.
- If Transformers fallback is needed, isolate it as a separate backend and collect memory/process telemetry.
- File minimal repros for any projection/global-attention shape failures.

Acceptance test:

- Gemma 4 12B starts under vLLM on Spark without HF fallback.
- One zero-shot lm-eval task and one generation request complete.
- Failure logs identify model architecture support separately from CUDA architecture support.

## 7. Make NVFP4 Real

Weakness: Spark's headline FP4 capability is only useful if NVFP4 is plumbed end to end.

Plan:

- Audit NVFP4 from checkpoint format through quantization metadata, loader, GEMM kernels, KV cache, serving flags, and output validation.
- Validate `--kv-cache-dtype nvfp4` on `sm_121` specifically before recommending it.
- Upstream or replace patched FlashInfer paths that route NVFP4 KV through working FA2 kernels.
- Track both vLLM and SGLang NVFP4 KV implementations. The SGLang SM120 implementation adds `fp4_e2m1` KV cache, FlashInfer FA2 patches, native FP4 memory pools, hybrid-SWA wiring, and per-layer global-scale auto-calibration before CUDA graph capture.
- Add correctness tests, not just speed tests: short deterministic prompts, logits sanity, and regression comparisons.
- Keep fp8 KV as the default until NVFP4 KV is proven on Spark for the target model family.

Acceptance test:

- NVFP4 model load, prefill, decode, and KV cache all use documented Spark-compatible paths.
- Output is numerically sane against a reference backend.
- Throughput improvement is measured against fp8/bf16 baselines, not assumed.

## 7a. Add SGLang As A First-Class Spark Runtime

Weakness: the first plan focused on vLLM and llama.cpp, but SGLang is another serious local serving runtime and already has community NVFP4 KV work for RTX Blackwell.

Plan:

- Add SGLang to the blessed-stack matrix alongside vLLM, LiteRT-LM, llama.cpp, and HF fallback.
- Track SGLang OpenAI-compatible serving smoke tests through `scripts/openai_chat_smoke.py`.
- Validate SGLang on the available single Spark before making any TP>1 claims.
- For NVFP4, study the `hikarioyama/sglang-nvfp4-kv-sm120` implementation: FlashInfer FA2 patches, `fp4_e2m1` KV, native FP4 pool, hybrid-SWA support, and global-scale auto-calibration.
- Treat the repo's small-model warning seriously: use fp8 KV for small models until NVFP4 quality is proven.
- Keep SGLang and vLLM results separate in reports unless backend, quantization, KV cache, and serving flags are identical enough to compare.

Acceptance test:

- SGLang imports or container startup is documented on Spark.
- A single-Spark SGLang server answers an OpenAI-compatible smoke request.
- The report records SGLang version/container, attention backend, KV cache dtype, quantization, CUDA graph mode, and memory use.
- NVFP4 SGLang is not blessed until output quality and speed are measured against fp8 KV on the same model.

## 7b. Evaluate LiteRT-LM For Gemma And Local Agents

Weakness: LiteRT-LM was not part of the first benchmark, but it may be relevant for Gemma, MTP, and local-agent prototyping.

Plan:

- Evaluate `google-ai-edge/LiteRT-LM` on Linux `aarch64` / DGX OS.
- Determine whether the Spark path uses CPU, CUDA GPU, LiteRT GPU, or another backend.
- Identify model format and conversion requirements.
- Smoke a Gemma model if supported.
- Compare generation throughput and ergonomics against llama.cpp, vLLM, and SGLang.

Acceptance test:

- LiteRT-LM has a documented build/install path on the Spark.
- One generation smoke completes or a concrete blocker is filed.
- The report records backend evidence, model format, throughput, and a go/no-go recommendation.

## 8. Fix llama.cpp / lm-eval Accuracy

Weakness: GGUF throughput works, but paper-comparable GGUF accuracy failed because lm-eval expected logprobs that the tested llama.cpp API did not provide.

Plan:

- Decide whether to fix the lm-eval adapter, pin a llama.cpp server version with compatible logprobs, or use a dedicated loglikelihood endpoint.
- Add an adapter test that checks echoed prompt/continuation token logprobs before running a full benchmark.
- Keep llama-bench throughput and lm-eval accuracy in separate result tables.
- Document which llama.cpp commit/API schema each benchmark used.

Acceptance test:

- A tiny GGUF lm-eval task produces valid loglikelihood scores.
- The adapter test fails fast if logprobs are missing or shaped differently.
- GGUF accuracy rows are no longer classified as loader failures for API-schema reasons.

## 8a. Bless llama.cpp As A Practical Serving Path

Weakness: llama.cpp may be the most reliable serving path today, but the first report only captured early throughput and the broken lm-eval accuracy adapter.

Plan:

- Pin a llama.cpp commit for Spark.
- Record CUDA build flags and architecture target.
- Run `llama-bench` and an OpenAI-compatible smoke test.
- Keep serving/throughput separate from paper-comparable accuracy.
- Decide where Ollama fits if its bundled llama.cpp path is more ergonomic.

Acceptance test:

- A single-Spark llama.cpp recipe serves a model from a clean checkout.
- The recipe records exact commit, build flags, `spark_doctor` evidence, throughput, and model details.
- The recipe clearly says whether lm-eval accuracy is supported for that server/API version.

## 9. Tame HF Fallback

Weakness: HF fallback rows were useful for labeling, but fragile; several runs died with `returncode=-9`.

Plan:

- Treat HF fallback as a separate backend class, never as a transparent replacement for vLLM.
- Add memory sampling around HF runs: RSS, GPU memory, unified memory pressure, swap, OOM killer logs.
- Lower batch/concurrency defaults for HF fallback on Spark.
- Prefer model-specific vLLM fixes over expanding HF fallback coverage.
- Add automatic failure annotation: killed, OOM suspected, unsupported architecture, shape error, timeout.

Acceptance test:

- A failed HF row records enough telemetry to say whether it was OOM/resource pressure or a real model error.
- Reports never mix HF fallback accuracy into vLLM comparisons without labeling it.

## 10. Tune For 48 SMs And Unified LPDDR5x

Weakness: "it runs" is not the same as "it is tuned for Spark."

Plan:

- Build a Spark-specific performance suite: prefill, decode, mixed prompt lengths, small-batch serving, long-context KV pressure.
- Tune tile sizes, occupancy, batch defaults, and graph capture sizes for GB10 instead of copying RTX PRO 6000 or B200 values. The first `spark_doctor` snapshot reported 48 CUDA multiprocessors on the available GB10 unit.
- Track memory bandwidth limits explicitly; 128 GB unified memory is the feature, but LPDDR5x bandwidth is the constraint.
- Add regression dashboards by model family and quantization type.

Acceptance test:

- Each blessed recipe has baseline tokens/sec, latency, memory, and backend metadata.
- A release cannot claim Spark support if it only passes functional tests and misses performance baselines by a large margin.

## 11. Handle Multi-Spark Correctly

Weakness: no-NVLink/no-P2P Blackwell systems can hang or crawl if distributed defaults assume datacenter topology.

Plan:

- For TP>1 or multi-node Spark, document required NCCL and vLLM flags.
- Validate `NCCL_P2P_DISABLE=1` and `--disable-custom-all-reduce` where appropriate.
- Prefer explicit Ray/NCCL recipes over generic distributed inference docs.
- Add startup topology checks that warn when P2P/NVLink assumptions are false.

Acceptance test:

- A two-Spark recipe starts, serves, and shuts down cleanly.
- A misconfigured distributed run fails early with a topology warning instead of hanging.

## 12. Improve Benchmark Design

Weakness: the initial personal benchmark matrix was too broad to finish quickly, and long HellaSwag rows dominated runtime.

Plan:

- Split the benchmark into phases:
  - smoke/load
  - short accuracy
  - long accuracy
  - throughput
  - MTP/spec decode
- Run HellaSwag as a separate long campaign.
- Use size-aware timeouts from the beginning.
- Save row-level progress after every task.
- Include backend, container, CUDA arch, selected kernels, and memory telemetry in every row.

Acceptance test:

- A failed row explains why it failed.
- A partial campaign still produces useful phase-specific conclusions.
- Long tasks no longer block all throughput/MTP discovery.

## 13. Add Observability For "Am I Using The Hardware?"

Weakness: GPU utilization alone is not proof of the right kernels.

Plan:

- Add a `spark-doctor` script that prints:
  - GPU name and compute capability
  - driver and CUDA runtime
  - PyTorch CUDA availability and arch list
  - vLLM version/container
  - FlashInfer/CUTLASS/Triton availability
  - selected attention and quantization backends
  - whether CUDA graphs are enabled
- Add optional `cuobjdump` inspection for installed CUDA extensions.
- Record this output in benchmark artifacts.

Acceptance test:

- Before a benchmark starts, the report proves whether the run is using `sm_121`-capable code.
- After a benchmark finishes, the report includes enough environment detail to reproduce the result.

## 14. Coordinate Upstream Ownership

Weakness: no single team owns "local AI works on Blackwell workstation/Spark systems."

Plan:

- Create a public Spark compatibility board across NVIDIA, vLLM, SGLang, LiteRT-LM, PyTorch, FlashInfer, llama.cpp, and key quantization projects.
- Track issues by layer: packaging, kernel dispatch, model architecture, runtime flags, performance, docs.
- Publish a monthly blessed stack: driver, CUDA, container, PyTorch, vLLM, FlashInfer, llama.cpp.
- Require each blessed stack to include a small public benchmark bundle.
- Make community repros easy to file: one command to collect environment, model, backend, and failure logs.

Acceptance test:

- Users can answer "what should I install today?" without reading five issue threads.
- Maintainers can reproduce Spark bugs without owning the hardware personally.

## 14a. Use Forks, Submodules, Worktrees, And Subagents

Weakness: patching upstream libraries through loose files or one dirty checkout will not scale.

Plan:

- Fork patched upstream libraries under `jethac`.
- Add each fork as a submodule under `third_party/`.
- Use one issue-named branch per patch stream.
- Use `git worktree` for parallel patch streams.
- Use subagents for independent runtime investigations and disjoint implementation areas.
- Keep each subagent's write scope separate to avoid merge conflicts.

Acceptance test:

- Any upstream code change is represented by a `jethac` fork submodule, issue branch, worktree path, commit SHA, reproduction command, and upstreaming plan.

## 15. Publish Honest Recipes

Weakness: users get trapped between marketing numbers and incomplete engineering notes.

Plan:

- Write recipes for concrete jobs:
  - Gemma local evaluation
  - local coding agent backend
  - NVFP4 MoE serving
  - GGUF llama.cpp serving
  - two-Spark distributed inference
- For each recipe, list what works, what is slow, what is broken, and what was not tested.
- Keep throughput numbers tied to exact commands and versions.
- Do not present paper-comparable accuracy unless the adapter path is validated.

Acceptance test:

- A developer can reproduce a recipe from a clean unit.
- The recipe says when to use Ollama/llama.cpp, when to use vLLM, and when not to bother yet.

## Immediate Attack Plan

If I were driving this this week:

1. Build `spark-doctor` and run it before every benchmark.
2. Pin one blessed vLLM container and one blessed llama.cpp build.
3. Retest Gemma 4 12B on a vLLM build with native `gemma4_unified`.
4. Add a tiny GGUF logprobs compatibility test before any lm-eval GGUF run.
5. Create a 30-minute Spark smoke suite that covers vLLM, SGLang, HF fallback, llama.cpp, NVFP4/fp8 KV, and one MTP row. LiteRT-LM is useful but opt-in because it is a side runtime, not the usual Spark performance path.
6. Publish the known-good stack and the known-bad flags.
7. File upstream issues with minimal repros, not benchmark dumps.

## The Definition Of Done

The ecosystem is fixed when this is boring:

```text
docker run ...
vllm serve google/gemma-4-12b ...
curl localhost:8000/v1/chat/completions ...
```

And the logs clearly say:

- running on GB10 / `sm_121`
- using Spark-validated kernels
- CUDA graphs enabled unless intentionally disabled
- quantization and KV cache paths selected correctly
- memory headroom sane

No mystery flags.

No accidental Ampere path.

No "works on H100, good luck."

Just the local AI box doing the job it was built to do.

## Source Inputs

- `DGX_SPARK_DIAGNOSIS.md`
- `GEMMA4_ON_DGX_SPARK.md`
- `BENCHMARKING_REPORT.md`
- NVIDIA CUDA GPU Compute Capability table: https://developer.nvidia.com/cuda/gpus
- NVIDIA DGX Spark Porting Guide: https://docs.nvidia.com/dgx/dgx-spark-porting-guide/dgx-spark-porting-guide.pdf
- vLLM DGX Spark blog: https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
- vLLM SM121 issue: https://github.com/vllm-project/vllm/issues/31128
- Gemma 4 12B vLLM SM120 notes: https://github.com/lna-lab/gemma4-12b-vllm-sm120
- vLLM NVFP4 KV SM120 notes: https://github.com/hikarioyama/vllm-nvfp4-kv-sm120
