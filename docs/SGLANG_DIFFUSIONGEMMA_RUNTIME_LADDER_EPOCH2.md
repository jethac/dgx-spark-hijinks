# SGLang DiffusionGemma Runtime Ladder

Date: 2026-06-11 JST

Scope: epoch2 SGLang lane after rebasing onto upstream
`diffusion-gemma4-support`. The local DG-S0/DG-S2 foundation shell is no longer
the implementation path. It remains historical evidence for geometry and weight
mapping, but runtime work now starts from upstream `gemma4_diffusion.py` and
`Gemma4Renoise`.

## Base

SGLang branch:

- `jethac/sglang:spark/hijinks-024-diffusiongemma-upstream-rebase`
- current head: `3c381eaa6a`

What upstream provides:

- `DiffusionGemmaForBlockDiffusion`
- `Gemma4Renoise`
- multimodal image processor
- uniform-state dLLM scheduler path
- request guards for unsupported logprobs / penalties / structured output
- auto-policy for the cookbook path: Triton attention, eager mode, unchunked
  prefill

Our immediate fix on top:

- `0705924c1d` sets `is_uniform=True` on the DiffusionGemma
  `DllmConfig.from_server_args()` branch. Without it, the cookbook path can
  construct `DllmConfig` with an unbound `is_uniform` local.
- `651d55cd2e` adds a local `DiffusionGemmaConfig` fallback for environments
  whose installed Transformers build does not yet recognize
  `model_type=diffusion_gemma`. This is config loading only; the model class and
  `Gemma4Renoise` algorithm remain the upstream SGLang implementation.
- `06e4a98a56` adds a persistent-Ubicloud static source/runtime audit workflow
  for the DiffusionGemma surface. The green run is
  `results/sglang_dgemma_static_audit_persistent_20260612T1014JST/summary.md`.
  Scope is static audit only: no model weights, no serving, and no quality
  claim.
- `3c381eaa6a` adds a persistent-Ubicloud SGLang wheel build workflow. The
  green run is `results/sglang_wheel_persistent_20260612T1026JST/summary.md`.
  Scope is CPU package build only: no model weights, no runtime load, no
  serving, and no quality claim.

## Ladder

### DG-R0: Runtime Config Sanity

Prove the cookbook architecture enters the uniform-state dLLM path before
loading weights.

Gate:

- `DllmConfig.from_server_args()` returns `algorithm=Gemma4Renoise`
- `block_size=256`
- `max_running_requests=1`
- `is_uniform=True`

Status: green offline in the provisioned WSL env; see
`results/sglang_diffusiongemma_uniform_config_fix_20260611T2207JST.md`.

### DG-R1: Stock Upstream Runtime Smoke

Verify upstream's correctness-first path works on GB10 before adding our
performance stack.

Run shape:

- `sglang serve --model-path google/diffusiongemma-26B-A4B-it`
- `--dllm-algorithm Gemma4Renoise`
- `--trust-remote-code`
- accept upstream auto-policy: Triton attention, eager mode, unchunked prefill
- GB10 memory guardrails: one server, cgroup memory cap, no comparator
  concurrency

Gate:

- server reaches ready
- text-only chat request returns coherent output
- logs prove `Gemma4Renoise` and `DiffusionGemmaForBlockDiffusion`
- logs prove no FlashInfer/NVFP4 performance path is being claimed yet

Status: green on GB10 through the stock Triton/eager path; see
`results/sglang_dgemma_dgr1_stock_smoke_20260611T2340JST/summary.md`.

Caveat: the smoke response is coherent, but the server log reports many
checkpoint keys as uninitialized. The text-only audit is green in
`results/sglang_dgemma_dgr2_weight_warning_audit_20260611TmanualJST.md`;
multimodal/image quality remains blocked on a separate vision-load audit.

### DG-R2: Upstream Runtime Quality Baseline

Establish a small reproducible correctness baseline before performance changes.

Gate:

- audit the DG-R1 uninitialized-weight list and decide which entries are benign
  derived/cache/statistic tensors versus real load gaps
- deterministic short prompt set
- stable output under fixed seed
- one small supplied-answer or parseable QA check
- artifact records runtime policy and memory use

Status: RED for the deterministic text-only quality gate; see
`results/sglang_dgemma_dgr2_text_quality_20260612T0604JST/summary.md`.
The prerequisite weight-warning audit is green for text-only DG-R2; see
`results/sglang_dgemma_dgr2_weight_warning_audit_20260611TmanualJST.md`.

Follow-up diagnostics:

- `results/sglang_dgemma_dgr2_promptdiag_20260612T0745JST/summary.md`
  shows the original terse "answer only" prompts denoise to repeated `<eos>`
  tokens; less constrained OpenAI chat prompts answer correctly.
- `results/sglang_dgemma_cookbook_conformance_20260612T0831JST/summary.md`
  shows the public cookbook-style path, with no explicit
  `--dllm-algorithm-config`, is semantically runnable on broad chat prompts
  such as TCP/UDP, but is not byte-deterministic under the zero-bug gate and
  still has a short-prompt empty-output failure.
- `results/sglang_dgemma_dgr2_revised_text_quality_20260612T0847JST/summary.md`
  defines and passes the revised scoped text-only gate: direct OpenAI chat
  prompts for Tokyo, 2+2, and a DGX Spark use sentence are non-empty,
  byte-stable across two repeats, and semantically correct.

Do not include image prompts until the vision-path warning group has its own
audit. DG-R3 may now proceed only under the revised text-only DG-R2 scope above;
the original terse-prompt baseline remains RED and should stay cited as a known
prompt-pathology row rather than hidden.

### DG-R3: D=512 FlashInfer VO-Split Integration

Replace the stock Triton D=512 full-attention path only after the stock runtime
is green.

Integration points:

- `gemma4_diffusion.py` has two attention instances per layer:
  - causal encoder/context attention
  - bidirectional decoder/canvas attention
- SGLang wrapper construction must avoid the prior ctor-`jit_args` trap.
- Real runtime writer/read roundtrip is required; probes that fabricate cache
  contents are insufficient.

Gate:

- BF16/no-KV-quant serving still coherent
- D=512 global layers route through VO-split where expected
- no claim about NVFP4 yet

### DG-R4: Mixed-KV Safety Path

Enable the conservative SGLang-proven KV mode first: FP8-K + NVFP4-V.

Why first:

- It is the quality-green SGLang radix path from Qwen/Gemma 3.
- It avoids overclaiming full NVFP4 K behavior in a runtime with new
  bidirectional canvas semantics.

Gate:

- coherent output vs BF16/fp8 comparator
- capacity denominator audited as mixed-KV, not full NVFP4
- radix/prefix reuse behavior explicitly scoped

### DG-R5: Full NVFP4 K+V

Attempt the full capacity path after mixed-KV and VO-split are green.

Risk:

- DiffusionGemma repeatedly rereads prefix/canvas KV during denoising, so any
  FP4-K scale/feed issue will compound more visibly than in ordinary AR decode.

Gate:

- quality comparator vs mixed-KV/BF16
- capacity row separate from quality row
- no `--disable-radix-cache` blessed result

### DG-R6: Performance Campaign Row

Produce the public/blog-grade row.

Gate:

- before: stock upstream Triton/eager/unchunked path
- after: GB10 performance path with exact commits/images
- speed, capacity, and quality reported as separate claims
- all artifacts live under `results/`

## Rule

Do not merge the stock-runtime claim with the GB10-performance claim. Upstream
made DiffusionGemma runnable; our lane is to make it performant and honestly
measured on GB10.
