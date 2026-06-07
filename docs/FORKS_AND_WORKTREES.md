# Forks, Submodules, And Worktrees

Policy: if this campaign needs changes to an upstream open-source library, the changes must live in a `jethac` fork and be added to this repo as a git submodule.

No loose long-lived patch directories.

Do not vendor hikarioyama's overlay trees into this repository or into production images. Treat them as SM120 prior art and attribution sources. Productionable changes must be re-derived as minimal patches on `jethac` forks of the true upstream projects, with upstream base commits, issue-named branches, local GB10 proof artifacts, and each upstream's contributing process preserved.

## Rules

- Fork upstream under `jethac`.
- Add the fork under `third_party/<repo-name>`.
- Create one branch per GitHub Issue.
- Use a separate git worktree for each active branch.
- Link every fork branch back to the hijinks Issue that requires it.
- Record upstream base commit, local branch, test command, and upstreaming plan.
- Follow each upstream repository's own contributing guide, style rules, tests, CI process, branch hygiene, and PR expectations. The `jethac` fork is only a staging area; patches should be shaped so upstream maintainers can review them normally.
- Add submodules only when carrying code. Reference clones can live outside the production flow or under `references/*-sm120-reference`.

## Naming

Submodules:

| upstream | fork | submodule path |
|---|---|---|
| `vllm-project/vllm` | `jethac/vllm` | `third_party/vllm` |
| `sgl-project/sglang` | `jethac/sglang` | `third_party/sglang` |
| `flashinfer-ai/flashinfer` | `jethac/flashinfer` | `third_party/flashinfer` |
| `ggml-org/llama.cpp` | `jethac/llama.cpp` | `third_party/llama.cpp` |
| `google-ai-edge/LiteRT-LM` | `jethac/LiteRT-LM` | `third_party/LiteRT-LM` |

Branches:

```text
spark/hijinks-<issue-number>-short-topic
```

Examples:

```text
spark/hijinks-001-sm121-build
spark/hijinks-003-gemma4-unified
spark/hijinks-004-backend-observability
spark/hijinks-014-sglang-runtime-sm121
spark/hijinks-018-sglang-nvfp4-kv-sm121
```

Worktrees:

```text
B:/workshop/worktrees/<repo-name>/<branch-slug>
```

Example:

```bash
git -C third_party/vllm worktree add B:/workshop/worktrees/vllm/spark-hijinks-003-gemma4-unified \
  -b spark/hijinks-003-gemma4-unified upstream/main
```

## Fork Command Pattern

```bash
gh repo fork vllm-project/vllm --org jethac --clone=false
git submodule add git@github.com:jethac/vllm.git third_party/vllm
git -C third_party/vllm remote add upstream https://github.com/vllm-project/vllm.git
git -C third_party/vllm fetch upstream
```

Use the same pattern for SGLang, FlashInfer, llama.cpp, and LiteRT-LM when a real code change is needed.

## Reference Repositories

Reference repos are allowed, but their names must not imply Spark validation.

For example, `hikarioyama/sglang-nvfp4-kv-sm120` should be treated as SM120 source context, not as a Spark fork. If cloned, name it something like:

```text
references/sglang-nvfp4-kv-sm120-reference
```

Active Spark work should use architecture-accurate names:

```text
sglang-runtime-sm121
sglang-nvfp4-kv-sm121
flashinfer-nvfp4-kv-sm121
vllm-gemma4-sm121
```

## NVFP4 KV Ownership Split

Build on hikarioyama's SM120 vLLM and SGLang NVFP4-KV work unless GB10 testing shows it is the wrong path, but keep ownership boundaries clean:

- FlashInfer owns FA2 kernel/page layout, explicit scale-factor strides, V scale-factor in-kernel deswizzle, JIT/codegen, SM12x/`121a` packaging, and kernel/unit harnesses.
- vLLM owns `--kv-cache-dtype nvfp4` routing, FlashInfer backend feature gates, scale-factor tensor and stride plumbing, serving tests, and docs. Do not carry FlashInfer kernels in vLLM PRs.
- SGLang owns `fp4_e2m1` KV dtype, KV memory pools, hybrid-SWA wiring, calibration before CUDA graph capture, server args, FlashInfer backend wrapper, and model-runner integration. Do not carry FlashInfer kernels in SGLang PRs.
- Keep the existing FlashInfer `mm_fp4` SM121 dispatch work separate from NVFP4 KV attention work; it is enablement evidence, not a serving-speed proof.

Suggested branches:

```text
jethac/flashinfer:spark/hijinks-007-fa2-nvfp4-kv-sm121
jethac/vllm:spark/hijinks-007-nvfp4-kv-sm121
jethac/sglang:spark/hijinks-018-fp4-e2m1-kv-sm121
```

## Spark NVFP4 KV PR Gate

A Spark NVFP4 KV PR is not ready until a clean GB10 `sm_121` wheel/container proves native NVFP4 KV selection, correctness against fp8/bf16, warmed performance/capacity deltas, and binary/JIT evidence for the selected kernels. Overlay-based success is useful debugging evidence only.

Required proof packet:

- clean build: wheel/container, no site-package overlays, no stale `flashinfer-jit-cache` or cubin mismatch
- build evidence: logs showing `sm_121`, `sm_121a`, or a documented valid SM12x family target such as `120f`
- binary/JIT evidence: `cuobjdump` or JIT-cache audit proving the claimed kernels are selected
- runtime evidence: real GB10 reports compute capability `(12, 1)` and logs show FlashInfer FA2 native NVFP4 KV, not fp8/bf16 fallback
- correctness: compare against fp8 or bf16 for prefill, decode, long context, GQA, page-size variants, CUDA graph replay, and peaked qK cases
- quality: deterministic prompt output plus PPL/retrieval-style checks on a model large enough for NVFP4 KV to be meaningful
- performance/capacity: warmed fp8-vs-NVFP4 runs with same model, prompts, memory fraction, CUDA graph mode, concurrency, TTFT, decode tok/s, and KV pool tokens
- scope labels: explicitly say if MLA/FlashMLA, Mamba/SSM, attention sinks, hybrid-SWA, MTP/spec decode, TP>1, or multi-Spark are untested

## Current State

Active submodules:

| submodule | upstream base | branch | worktree | status |
|---|---|---|---|---|
| `third_party/flashinfer` | `flashinfer-ai/flashinfer@a2870343` | `jethac/flashinfer@spark/hijinks-004-sm121-flashinfer` | `B:/workshop/worktrees/flashinfer/spark-hijinks-sm121-flashinfer` | patch branch pushed |
| `third_party/flashinfer` | `jethac/flashinfer@a42c8f07` | `jethac/flashinfer@spark/hijinks-007-fa2-nvfp4-kv-sm121` at `e152cf4d` | `B:/workshop/worktrees/flashinfer/spark-hijinks-007-fa2-nvfp4-kv-sm121` | FA2 explicit scale-factor stride/page patch pushed; inherits SM121 `mm_fp4` patch; GB10 build/runtime proof pending |
| `third_party/vllm` | `vllm-project/vllm@4dcd10e` | `jethac/vllm@spark/hijinks-007-nvfp4-kv-sm121` at `2c1405d` | `B:/workshop/worktrees/vllm/spark-hijinks-007-nvfp4-kv-sm121` | SM12x NVFP4 KV routes to FlashInfer FA2; GB10 build/runtime proof pending |
| `third_party/sglang` | `sgl-project/sglang@02be2e7` | `jethac/sglang@spark/hijinks-018-fp4-e2m1-kv-sm121` | `B:/workshop/worktrees/sglang/spark-hijinks-018-fp4-e2m1-kv-sm121` | fork, submodule, and branch pushed; code not ported yet |

FlashInfer patch:

- commit: `a42c8f07`
- branch URL: https://github.com/jethac/flashinfer/tree/spark/hijinks-004-sm121-flashinfer
- purpose: treat SM121 as SM12x for FP4 auto-dispatch and add `12.1a` to CUDA 12.9+/13 aarch64 JIT-cache build targets
- test coverage: added `test_mm_fp4_auto_prefers_b12x_for_sm121_nvfp4` under `tests/gemm/test_mm_fp4.py`
- local verification: Python syntax compile for touched Python files and the new test file
- upstream guidance checked: `CONTRIBUTING.md`
- relevant upstream process:
  - use editable install with `pip install --no-build-isolation -e . -v`
  - add unit tests in `tests/`
  - update docs when Python documentation changes
  - public CI may require a `ci-users` approval comment or `run-ci` label for outside contributors
  - Spark internal CI exists as `unit_test_spark` but is manual-trigger only
- local pytest limitation: targeted pytest collection fails in this Windows workspace because FlashInfer imports `tvm_ffi` through `tests/conftest.py`, and that dependency is not installed here.
- missing verification: FlashInfer runtime tests on GB10 and upstream CI

The new FlashInfer FA2 NVFP4 KV branch is deliberately based on the existing SM121 FlashInfer patch branch, not plain upstream. That preserves the earlier `mm_fp4` dispatch enablement while keeping KV cache changes separate for review and benchmarking.

FlashInfer FA2 NVFP4 KV patch:

- commit: `e152cf4d`
- branch URL: https://github.com/jethac/flashinfer/tree/spark/hijinks-007-fa2-nvfp4-kv-sm121
- ancestry: based on `a42c8f07`, so it includes the earlier SM121 `mm_fp4` auto-dispatch and `12.1a` JIT-cache build-target work
- purpose: port the FlashInfer side of the FA2 NVFP4 KV cache path by adding explicit scale-factor stride plumbing, independent K/V page strides, and optional vLLM B2 V scale-factor de-swizzle
- touched files: `csrc/batch_decode.cu`, `csrc/batch_prefill.cu`, `flashinfer/jit/attention/modules.py`, `flashinfer/jit/attention/utils.py`, `include/flashinfer/attention/persistent.cuh`, `include/flashinfer/attention/prefill.cuh`, `include/flashinfer/page.cuh`, `tests/jit/test_attention_utils.py`
- local verification: `python -m py_compile flashinfer/jit/attention/utils.py flashinfer/jit/attention/modules.py tests/jit/test_attention_utils.py` and `git diff --check` passed
- local pytest limitation: `python -m pytest tests/jit/test_attention_utils.py` fails to collect in this Windows workspace because `tvm_ffi` is not installed
- missing verification: clean FlashInfer GB10 build, harness proof, and vLLM/SGLang serving proof

vLLM SM12x NVFP4 KV routing patch:

- commit: `2c1405dd129d873d268b8baea78c5739cd384951`
- branch URL: https://github.com/jethac/vllm/tree/spark/hijinks-007-nvfp4-kv-sm121
- purpose: route SM12x `--kv-cache-dtype nvfp4` through FlashInfer FA2 while preserving the existing SM100 TRTLLM NVFP4 path
- touched files: `vllm/v1/attention/backends/flashinfer.py`, `tests/kernels/attention/test_flashinfer_nvfp4_sm12x_routing.py`, `docs/design/attention_backends.md`
- upstream guidance checked: `CONTRIBUTING.md`, `docs/contributing/README.md`, and `AGENTS.md`; commit was made with DCO sign-off
- local verification: Python syntax compile and staged `git diff --check` passed
- local pytest limitation: vLLM pytest collection is blocked in this Windows workspace because `tblib` is not installed
- local lint limitation: `ruff` is not installed in this Windows workspace
- missing verification: clean vLLM plus FlashInfer build on GB10 and a serving proof selecting FA2 native NVFP4 KV

Other forks should still be created only when the corresponding issue is ready to carry code.
