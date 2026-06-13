# SGLang DiffusionGemma Runtime Ladder

Date: 2026-06-11 JST

Scope: epoch2 SGLang lane after rebasing onto upstream
`diffusion-gemma4-support`. The local DG-S0/DG-S2 foundation shell is no longer
the implementation path. It remains historical evidence for geometry and weight
mapping, but runtime work now starts from upstream `gemma4_diffusion.py` and
`Gemma4Renoise`.

2026-06-12 closeout: the scoped DiffusionGemma ladder is green through DG-R7.
Remaining caveats are explicitly scoped inside the rung rows: the original terse
DG-R2 prompt baseline remains RED, DG-R4 mixed-KV is deferred with split-dtype
module-keying as the named blocker, DG-R7 is only a tiny stock-path image smoke,
and no CUDA-graph, long-context, or broad multimodal quality claim is made. The
current ship gate is the separate SGLang Gemma 4 autoregressive ladder.

## Base

SGLang branch:

- `jethac/sglang:spark/hijinks-024-diffusiongemma-upstream-rebase`
- closeout head for DG-R5/DG-R6/DG-R7 rows: `98bf8f129d`

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
- `f36ecf495b` gates the experimental DG-R3 FlashInfer path behind an explicit
  opt-in: `--attention-backend flashinfer` plus
  `SGLANG_FLASHINFER_VOSPLIT=1`. Stock `Gemma4Renoise` launches still force
  Triton, and CUDA graphs / chunked prefill remain disabled for DiffusionGemma.
  A persistent-Ubicloud wheel build for this runtime-code commit is green.
- `dec4c040a8` fixes the static-audit marker for that policy and was the
  DG-R3 policy checkpoint. The persistent-Ubicloud static audit is green; see
  `results/sglang_dgemma_dgr3_vosplit_policy_20260612T1050JST/summary.md`.
- `98bf8f129d` is the current branch head used by the later DG-R5/DG-R6/DG-R7
  closeout rows. It carries the intervening Gemma 4/MTP and NVFP4 safety fixes
  without changing the scoped DiffusionGemma claims below.

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
the follow-up static vision audit
`results/sglang_dgemma_vision_warning_static_audit_20260612T1554JST.md`
classifies the vision warning group as SGLang-created defaults rather than
missing checkpoint payload. Multimodal/image quality remains unclaimed until a
live image prompt or vision-forward gate runs; the later DG-R7 row supplies a
tiny stock-path color-recognition smoke only.

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

The vision-path warning group now has a static load audit in
`results/sglang_dgemma_vision_warning_static_audit_20260612T1554JST.md`: the
checkpoint has no matching payload for the warning-only no-scale norm,
layer-scalar, or clippable-bound defaults. The later DG-R7 live image smoke
exercises the real image processor and vision-forward path for two synthetic
color prompts only. Do not generalize it into broad image-quality,
FlashInfer-image, NVFP4-image, or benchmark claims. DG-R3 may now proceed only
under the revised text-only DG-R2 scope above; the original terse-prompt
baseline remains RED and should stay cited as a known prompt-pathology row
rather than hidden.

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

Status: green for BF16/no-KV-quant text-only serving through the explicit
FlashInfer VO-split opt-in. The stock runtime still uses Triton by default; the
experimental FlashInfer route requires both `--attention-backend flashinfer` and
`SGLANG_FLASHINFER_VOSPLIT=1`. Off-box source-policy validation is banked in
`results/sglang_dgemma_dgr3_vosplit_policy_20260612T1050JST/summary.md`. The
Spark serving gate is green in
`results/sglang_dgemma_dgr3_vosplit_smoke_20260612T112447JST/summary.md`: the
revised DG-R2 text gate passes, the opt-in warning is present, and D=512 global
layers route through VO-split trace labels with `head_dim_vo=256`.

The earlier run
`results/sglang_dgemma_dgr3_vosplit_smoke_20260612T111121JST/summary.md` stays
as a RED harness-diagnosis artifact, not a model-quality result: routing proof
was present but the parser expected the wrong trace shape, and the runner had
omitted the deterministic `Gemma4Renoise` config used by the revised DG-R2 gate.

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

Status: DEFERRED by the split-dtype de-scope decision in
`mail/0077_claude-to-codex_split-dtype-descoped-from-headline.md`. The
existing diagnostic row is RED at first live request; see
`results/sglang_dgemma_dgr4_mixedkv_smoke_20260612T114737JST/DIAGNOSIS.md`.
The row proves mixed-KV allocation (`kv_cache_dtype='fp4_e2m1'`,
`SGLANG_FP4_KV_MIXED_KV=1`, `mixed_kv=True`, FP4 hybrid subpools) and wrapper
construction for D=512 VO-split, then fails before the text-quality gate because
FlashInfer paged prefill is still planned with a single `kv_data_type=torch.uint8`
while SGLang supplies split K/V tensors (`K=torch.float8_e4m3fn`,
`V=torch.uint8`). This makes DiffusionGemma DG-R4 the named live consumer for
split-K/V paged-prefill module keying, but it is no longer a headline or
ship-blocking rung. The packet remains staged in
`docs/SGLANG_DIFFUSIONGEMMA_DGR4_MIXEDKV_PACKET_20260612.md` for future
split-dtype work. As of FlashInfer `3fa0775c`, the temporary
`k_data_type`/`v_data_type` plan kwargs are removed entirely; future DG-R4 work
needs real split-K/V module identity rather than replaying the old API shim.

### DG-R5: Full NVFP4 K+V

Attempt the full capacity path after mixed-KV and VO-split are green.

Risk:

- DiffusionGemma repeatedly rereads prefix/canvas KV during denoising, so any
  FP4-K scale/feed issue will compound more visibly than in ordinary AR decode.

Gate:

- quality comparator vs BF16/VO-split
- capacity row separate from quality row
- no `--disable-radix-cache` blessed result

Status: green for the scoped Spark text-only smoke; see
`results/sglang_dgemma_dgr5_fullnvfp4_smoke_20260612T145433JST/summary.md`.
The row uses `--kv-cache-dtype fp4_e2m1`, `SGLANG_FP4_KV_MIXED_KV=0`, the same
deterministic `Gemma4Renoise` config as DG-R2/DG-R3, and the explicit
FlashInfer VO-split opt-in. It proves text-only serving, full-NVFP4 K+V storage
(`mixed_kv=False`, FP4 K/V pools, FP4 module traces), and D=512 VO-split
routing (`head_dim=512`, `head_dim_vo=256`). It still does not claim capacity,
image quality, CUDA graph safety, or long-context quality. Packet:
`docs/SGLANG_DIFFUSIONGEMMA_DGR5_FULLNVFP4_PACKET_20260612.md`. Runner:
`scripts/run_sglang_dgemma_dgr5_fullnvfp4_smoke.sh`.

Matched BF16/auto-KV capacity denominator is also green; see
`results/sglang_dgemma_dgr5_capacity_pair_20260612T1517JST/summary.md`. At the
same current SGLang commit, image, page size, memory fraction, graph policy, and
VO-split launch shape, BF16/auto KV allocated `66560` full-layer / `53248` SWA
tokens, while full NVFP4 K+V allocated `237312` / `189696`, or about `3.56x`
KV token capacity versus BF16/auto KV for this model/launch envelope. This
capacity pair does not add image, CUDA graph, long-context quality, fp8
denominator, or throughput claims.

### DG-R6: Performance Campaign Row

Produce the public/blog-grade row.

Gate:

- before: stock upstream Triton/eager/unchunked path
- after: GB10 performance path with exact commits/images
- speed, capacity, and quality reported as separate claims
- all artifacts live under `results/`

Status: green for the scoped text-only performance pair; see
`results/sglang_dgemma_dgr6_perf_pair_20260612T152803JST/summary.md`.
Packet: `docs/SGLANG_DIFFUSIONGEMMA_DGR6_PERF_PACKET_20260612.md`. Runner:
`scripts/run_sglang_dgemma_dgr6_perf_pair.sh`.

The pair is intentionally a combined stack comparison, not an isolated
kernel benchmark: before is the stock SGLang DiffusionGemma policy path
(Triton attention, BF16/auto KV), and after is the GB10-tuned path
(FlashInfer VO-split, full NVFP4 K+V). Both rows passed the revised text-only
quality gate and OpenAI benchmark cases. The after row shows about `3.52x` KV
token capacity in the same performance launch envelope (`237568/67584`
full-layer tokens, `189952/54016` SWA tokens). Total completion-token
throughput is case-dependent: `1.016x` short, `0.902x` medium, `1.042x`
long-prefill, and `1.444x` natural-long-prefill. DiffusionGemma streaming emits
each measured completion as effectively one event, so `decode_tok_s` is not a
meaningful field for this row. DG-R5 remains the separate source for the
full-NVFP4 quality/capacity claims.

### DG-R7: Stock Multimodal Image Smoke

Exercise the image processor and vision-forward path after the static
vision-warning audit.

Gate:

- stock upstream policy path only: Triton attention, BF16/auto KV, eager/no
  graphs
- OpenAI chat `image_url` request with synthetic PNGs
- two repeats per image, stable non-empty answer
- semantic color check only

Status: GREEN for the revised scoped image smoke; see
`results/sglang_dgemma_dgr7_image_smoke_20260612T160944JST/summary.md`.
The red/blue image returns `Red and blue.` twice, and the green-square image
returns `The color is green.` twice. This proves the real multimodal request
path is live for a tiny deterministic color-recognition gate.

The earlier strict-prompt row
`results/sglang_dgemma_dgr7_image_smoke_20260612T160201JST/summary.md` stays
as a RED diagnostic: the red/blue image passed twice, but the terse
green-square prompt (`Answer with one color word`) returned empty twice. This
matches the DG-R2 terse-prompt pathology and is why the claim-grade image smoke
uses short descriptive prompts instead of one-word answer constraints.

Scope caveat: DG-R7 is not a broad image-quality benchmark and makes no
FlashInfer, NVFP4, capacity, throughput, image-generation, or long-context
claim.

## Rule

Do not merge the stock-runtime claim with the GB10-performance claim. Upstream
made DiffusionGemma runnable; our lane is to make it performant and honestly
measured on GB10.
