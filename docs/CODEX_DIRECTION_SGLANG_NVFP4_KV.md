# Direction: SGLang → NVFP4 KV on Spark (convert proven capacity into blessed quality)

> Standing direction for the SGLang lane. The current claim-ready SGLang row is mixed
> FP8-K + NVFP4-V: about `1.28x` fp8 KV capacity at a matched physical K+V byte budget,
> with radix-cache quality green under the prefix-cache graph guard. Full NVFP4 K+V
> remains the parked stretch route.

## 2026-06-11 update — Gemma 4 E4B rung 0 D=512 decode workaround staged

Artifact: `results/sglang_gemma4_e4b_rung0_20260611T141256JST/summary.md`.

SGLang Gemma 4 E4B text-only bring-up now reaches the intended next blocker under
`SGLANG_FLASHINFER_VOSPLIT=1`:

- source/provenance lines are present (`jethac/sglang@f3ebcf623`,
  `jethac/flashinfer@8d85fff9`, FlashInfer source paths under `/flashinfer-src`);
- serving dispatch logs prove SWA/local layers plan as `head_dim=256, head_dim_vo=256`;
- global prefill enters the two-pass VO-split route at `head_dim=512, head_dim_vo=256`;
- the remaining failure is decode: the D=512 global layer still enters the standard
  decode wrapper, which instantiates a symmetric `head_dim_qk=512;head_dim_vo=512`
  paged module and fails with `Unsupported max_mma_kv: 0`.

Interpretation: the prior SGLang wrapper-geometry bug is fixed, and the current red is
not evidence against the VO-split prefill route. It is the decode-side half already
called out in `docs/SGLANG_GEMMA4_RUNG_PREP.md`: global D=512 decode must route through a
VO-split-capable path (decode-as-prefill with `qo_len=1`, or an equivalent FlashInfer
decode API that carries `head_dim_vo`) before rung 0 can become a coherent serving row.
Do not edit `prefill.cuh` trait math in the SGLang lane; the `max_mma_kv` dispatcher work
is parked for the shared FlashInfer task.

Follow-up result: `results/sglang_gemma4_e4b_rung0_20260611T151226JST/summary.md`
reran with `jethac/sglang@9d78a007f` and proves the SGLang-side workaround reaches
runtime. Under `SGLANG_FLASHINFER_VOSPLIT=1`, D=512 global decode now plans through
the paged-prefill wrapper (`decode_as_prefill_vosplit*`) with `head_dim=512` and
`head_dim_vo=256`; D=256 sliding layers stay on normal decode. The remaining red is
FlashInfer dispatcher selection inside the VO-split paged-prefill path:
`Unsupported max_mma_kv: 0`. That is the r9 / `jethac/flashinfer@76af7982` target,
not a standard decode-wrapper routing failure.

Epoch-2 result: `results/sglang_gemma4_e4b_rung0_chat_20260611T180454JST/summary.md`
reran the same SGLang E4B text-only checkpoint with `jethac/flashinfer@76af7982`.
The dispatcher wall is closed and the OpenAI chat endpoint returns the coherent
answer `The capital of Japan is Tokyo.` while D=512 global prefill and decode both
route through `*_vosplit*` paged-prefill at `head_dim=512, head_dim_vo=256`.
The paired diagnostic `results/sglang_gemma4_e4b_chat_compare_20260611T175952JST/summary.md`
shows raw `/generate` still emits separator repetition for the same instruction.
Treat raw completion as a diagnostic only for this instruction-tuned checkpoint;
the rung-0 serving gate is chat-formatted quality plus geometry/provenance.

## 2026-06-10 update — deswizzle leak is falsified for the live SGLang failure

Artifacts:

- `results/flashinfer_nvfp4_page_deswizzle_matrix_20260610TmanualJST.md`
- `results/sglang_deswizzle_flag_check_20260610TmanualJST.md`
- `results/sglang_qwen_fp4kv_moduleproof_fast_20260610TmanualJST.md`

The standalone FlashInfer matrix proves an important negative-control mechanism:
SGLang-style linear V scale factors are correct with deswizzle off, but corrupt when
`FLASHINFER_PAGED_V_SF_DESWIZZLE=1` is applied. This is true for both page size 1 and
page size 16, and for both decode and paged prefill.

However, the live SGLang Qwen cached-prefix failure was rerun with module/env proof, and
the failing `extend_merge_paged` path had:

- `extra_cuda_flags=''`
- `deswizzle_macro_active=False`
- `BatchPrefillWithPagedKVCacheWrapper`
- FA2 backend, NHD layout, `torch.uint8` K/V carriers, `torch.float8_e4m3fn` K/V scales
- cached-prefix paged wrapper max KV length `55`

So do not continue chasing a vLLM deswizzle macro leak as the current root cause. It remains
a real corruption mechanism and a runbook guardrail, but it is not active in the reproduced
SGLang cached-prefix failure.

The active SGLang bug remains in the cached-prefix paged read / scale-feed convention handed
to FlashInfer, or in the FP4-K attention behavior reached by that path.

## 2026-06-10 update — vLLM proves FP4-K reuse; keep full-NVFP4 open

Artifact: `results/sglang_vs_vllm_fp4_k_reuse_diff_20260610T0255JST.md`.

The vLLM cross-check is now locked: full NVFP4 K+V with prefix caching ON produced a real
local cache hit (`vllm:prefix_cache_hits_total 3728.0`) and preserved the first token on the
Qwen smoke gate. This means FP4-K reuse is not categorically broken across runtimes.

Reframe the SGLang task accordingly:

- full NVFP4 K+V under SGLang radix remains red;
- mixed FP8-K / NVFP4-V remains the green fallback and capacity-insurance path;
- but the full-NVFP4 goal is still live. The next bug hunt should diff SGLang's FP4-K feed
  into FlashInfer attention against vLLM's working packed-cache path, especially
  `forward_extend_merge_paged`'s ragged suffix partial versus paged cached-prefix partial.

Do not spend more time on `_safe_merge_state` arithmetic or byte/page pairing unless new
evidence contradicts the existing traces. Both have already been cleared for the observed
row.

## 2026-06-10 update — mixed-KV fresh comparator is green

Artifacts:

- `results/sglang_mixed_kv_pool_probe_20260610T0036JST.md`
- `results/sglang_qwen_mixedkv_default_20260610T0042JST_summary.md`
- `results/sglang_qwen_fp8_vs_mixedkv_fresh_comparator_20260610TmanualJST.md`
- `results/sglang_qwen_mixedkv_quality3_20260610TmanualJST.md`
- `results/sglang_qwen_mixedkv_natural_longprefill_20260610TmanualJST.md`
- `results/sglang_qwen_mixedkv_graph_safety_20260610TmanualJST.md`

Fresh sequential comparator result, same model/runtime/page size/mem fraction:

| KV mode | allocatable tokens | K size | V size | short decode tok/s |
|---|---:|---:|---:|---:|
| fp8 K + fp8 V | 3,119,168 | 20.82 GB | 20.82 GB | 57.594 |
| fp8 K + NVFP4 V | 5,537,968 | 36.97 GB | 20.80 GB | 57.804 |

Observed allocator-token ratio: `1.775x`. Short-decode speed ratio: `1.004x`.

Important interpretation: the normalized bytes-per-token improvement is still the expected
mixed-path claim, about `1.28x`, because K remains fp8 while V is packed NVFP4 plus scale
factors. The `1.775x` number is the observed SGLang allocator token count under the same
launch settings, not a pure per-token storage-ratio claim.

The current repair direction is mixed KV, not more FP4-K scale tuning:

- K cache stays FP8 e4m3 to protect QK logits and prefix-only LSE.
- V cache stays packed NVFP4 plus FP8 scale factors to preserve a meaningful capacity win.
- Expected capacity claim is the mixed-path claim, about `16/(8+4.5) ~= 1.28x` over fp8,
  not the full `~1.78x` full-FP4 K+V claim.

The small GB10 page-size-1 probes now pass:

- FlashInfer split-K/V page-size-1 isolation with SGLang macro policy:
  - `fp8_k_bf16_v`: cosine `0.9999990463256836`
  - `bf16_k_nvfp4_v`: cosine `1.0`
  - `fp8_k_nvfp4_v`: cosine `1.0000001192092896`
- SGLang pool integration:
  - K buffer dtype `torch.float8_e4m3fn`
  - V buffer dtype `torch.uint8`
  - no K scale buffer
  - cosine versus pool reference `0.999995231628418`
  - max abs diff `0.0078125`

Important macro policy: do **not** compile SGLang's FlashInfer JIT modules with
`-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1`. That macro is vLLM-specific. Under that wrong
macro, the same page-size-1 NVFP4-V isolation drops to cosine around `0.88-0.90`; without
it, the isolation is exact. This explains the temporary red probe and should stay in the
runbook because accidental vLLM/SGLang macro sharing is easy.

Current status: mixed-KV is the practical SGLang route for Qwen2.5, but not fully blessed.
The default radix first-token gate is green and the fresh fp8 comparator shows decode
parity with larger allocatable KV capacity. The original repeated-sentence `long_prefill`
prompt is now demoted: fresh-server controls show fp8 and mixed-KV both trip repetition
flags on that prompt, so it is not a clean mixed-KV quality discriminator. A new
non-repetitive `natural_long_prefill` case passes for both fp8 and mixed-KV with speed
parity and no heuristic repetition flags.

Graph safety was initially tested with a narrow guarded path. With
`SGLANG_FP4_KV_ENABLE_CUDA_GRAPH=1`, the unguarded run captures both full and piecewise
CUDA graphs and serves short/medium requests with `cuda graph: True`; however, the same
`natural_long_prefill` case stops after four completion tokens (`1 .`,
`matched_stop=151645`) and the quality probe flags `too_short_for_requested_decode`.
The failure is isolated to graph mode plus radix reuse: a graph-enabled
`natural_long_prefill` run with `#cached-token: 0` passes, and the full graph-enabled
three-case sequence passes when rerun with `--disable-radix-cache` / `ChunkCache`
(`natural_long_prefill`: 128 completion tokens, no quality flags).

The first SGLang guard only covered the then-known unsafe shape: when
`SGLANG_FP4_KV_MIXED_KV=1` and an EXTEND batch has a nonzero cached prefix, piecewise CUDA
graph replay is disabled for that prefill. Decode graphs and no-prefix prefill graphs
remain enabled. The guarded default
radix row passes all three cases with graph capture enabled and radix cache on:
`short_decode` 64 tokens at `58.142 tok/s`, `medium_decode` 192 tokens at
`57.634 tok/s`, and `natural_long_prefill` 128 tokens at `57.723 tok/s`, all with no
quality flags. The server log proves cached-prefix prefills use `cuda graph: False` while
decode remains `cuda graph: True`. Artifact:
`results/sglang_qwen_mixedkv_graph_safety_20260610TmanualJST.md`.

Superseding policy: that guard is insufficient for blessable quality. The later
prefix-4096 control shows the graph-sensitive operation is also the cache-populating
prefill: graph-written prefix cache entries shift fp8 and mixed-KV PPL, while eager-written
prefix cache entries make the same row essentially equal. Until the graph-write path is
repaired, Qwen mixed-KV quality rows must disable graph capture for cache-populating
prefills as well as cached-prefix prefills. `--disable-radix-cache` remains diagnostic only.

The supplied-token PPL gate below was the interim guarded-graph result. Artifact:
`results/sglang_qwen_mixedkv_reuse_ppl_20260610TmanualJST.md`. A no-reuse 512-token
smoke row passed with fp8 and mixed-KV exactly equal, but that row had `cached_tokens=0`
and is only a harness smoke. The accepted rows warm a prefix and score the remaining
continuation through a real radix hit:

| ctx | reused prefix | scored tokens | PPL fp8 | PPL mixed | delta nats/token |
|---:|---:|---:|---:|---:|---:|
| 512 | 256 | 255 | 18.453757 | 18.431295 | -0.001218 |
| 2048 | 1024 | 1023 | 28.140193 | 28.392845 | 0.008938 |
| 8192 | 4096 | 4095 | 7.238053 | 8.056450 | 0.107121 |

Both fp8 and mixed-KV responses report the expected `cached_tokens`, with
`num_missing_tokens=0`, `num_mismatched_tokens=0`, and one explicitly skipped SGLang
logprob-span boundary placeholder. Server logs confirm the scored cached-prefix prefill
used `#cached-token: 256`, `#cached-token: 1024`, and `#cached-token: 4096`; for mixed-KV
these scored prefills ran under the guard (`cuda graph: False`) while the warmup/no-prefix
prefills still used graph replay. Capacity for the 512 launch was `3,116,067` fp8 tokens
versus `5,552,677` mixed-KV tokens (`1.782x` observed allocator ratio); the 2048 launch was
`3,117,451` fp8 tokens versus `5,560,832` mixed-KV tokens (`1.784x` observed allocator
ratio); the 8192 launch was `3,119,879` fp8 tokens versus `5,551,389` mixed-KV tokens
(`1.779x` observed allocator ratio).

Updated interpretation: mixed-KV remains the practical SGLang capacity route, but this
guarded-graph PPL table is no longer a clean quality baseline. The 8192 row's material PPL
cost (`+0.107121` nats/token) is now treated as a graph-written-prefix-cache failure
detector. The graph-safe baseline is the later all-eager prefix-4096 control, which shows
fp8 and mixed-KV effectively equal on the same corpus path. Full NVFP4 K+V remains a
separate red/open track.

That localization is now done. Artifact:
`results/sglang_qwen_mixedkv_8k_token_logprob_diagnostic_20260610TmanualJST.md`. The
detailed graph-enabled 8k rerun reproduces the loss at `+0.106689` nats/token and shows the first 1024
scored tokens after the 4096-token cached prefix account for about `55.9%` of the total
delta nats. The median token delta is near zero (`0.002143`), but the positive tail is
larger than the negative tail (`861.844` positive delta nats versus `-424.951` negative).
The 8k no-reuse control then isolates the issue to cached-prefix reuse: with
`reuse_prefix_len=0`, fp8 and mixed-KV are exactly equal at 8192 tokens (PPL `7.195014`
versus `7.195014`, `delta_nats_per_token=0.0`) while preserving the same mixed-KV
allocation (`5,542,470` tokens versus `3,116,223` fp8, `1.779x` observed). Server logs
confirm both scored requests used `#cached-token: 0`. Conclusion: do not blindly scale this
to 32k yet. The later graph-vs-eager control changes the repair target from "cached
NVFP4-V quality over long reused prefixes" to "graph-written prefix cache state"; a broad
mixed-KV long-prefill penalty remains falsified for the 8k no-reuse row.

The fixed-8k reuse-prefix sweep is now run. Artifact:
`results/sglang_qwen_mixedkv_reuse_prefix_sweep_ctx8192_20260610TmanualJST.md`. At fixed
`ctx=8192`, the PPL delta grows with reused-prefix length:

| reused prefix | delta nats/token |
|---:|---:|
| 0 | 0.000000 |
| 1024 | 0.003079 |
| 2048 | 0.026555 |
| 4096 | 0.106689 |
| 6144 | 0.114980 |

This initially looked like a cached NVFP4-V quality surface, but a later control supersedes
that interpretation. Artifact:
`results/sglang_qwen_mixedkv_prefix4096_graph_vs_eager_20260610TmanualJST.md`. On the same
4096-prefix / 8192-context corpus path, disabling all CUDA graph capture changes the row
from fp8 `7.238053` versus mixed `8.052973` (`+0.106689` nats/token) to fp8 `5.176347`
versus mixed `5.169650` (`-0.001295` nats/token). The scored cached-prefix request was
eager in both runs; the difference is that the graph-enabled sweep populated the prefix
cache with `cuda graph: True`, while the eager control populated it with `cuda graph:
False`.

Current interpretation: the active SGLang correctness bug is CUDA-graph cache-writing
state, not `_safe_merge_state`, radix page pairing, or an inherent NVFP4-V cached-prefix
quality limit. The practical policy is to disable graph capture for cache-populating
prefills as well as cached-prefix prefills until the graph-write path is repaired. Treat
the graph-enabled reuse-prefix sweep as a failure detector, not a blessable quality curve.

The prefix-cache graph-write guard is now implemented and validated. Artifact:
`results/sglang_qwen_mixedkv_prefix4096_prefixcacheguard_sweepcorpus_20260610TmanualJST.md`.
The guard lives in
`third_party/sglang/python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`:
with radix cache enabled, `PiecewiseCudaGraphRunner.can_run()` returns `False` for
EXTEND/prefill batches unless `SGLANG_ALLOW_PREFIX_CACHE_PREFILL_CUDA_GRAPH=1` is set for
experiments. This routes both cache-populating prefills and cached-prefix prefills through
eager while preserving regular decode CUDA graphs.

Validation on the same 8192-context / 4096-prefix corpus path, with graph capture enabled
globally, now gives fp8 PPL `5.176347` versus mixed-KV PPL `5.169650`
(`delta_nats_per_token=-0.001295`). Server logs prove both fp8 and mixed-KV used
`cuda graph: False` for the 4096-token cache write and the 4096-token cached read. The
mixed-KV allocation remains `5,552,890` tokens versus `3,116,450` fp8 tokens (`1.782x`).
This is the current SGLang Qwen mixed-KV quality baseline for prefix reuse on GB10.

The broader graph-safe fixed-8k sweep is now complete. Artifact:
`results/sglang_qwen_mixedkv_prefixcacheguard_reuse_prefix_sweep_ctx8192_20260610TmanualJST.md`.
It reruns reused prefixes `0, 1024, 2048, 4096, 6144` with graph capture enabled globally
but prefix-cache-writing and cached-prefix prefills forced eager by the guard:

| reused prefix | delta nats/token | allocator ratio |
|---:|---:|---:|
| 0 | 0.000000 | 1.776x |
| 1024 | -0.000267 | 1.779x |
| 2048 | 0.000436 | 1.778x |
| 4096 | -0.001295 | 1.769x |
| 6144 | -0.066981 | 1.779x |

Server logs prove the intended policy for every reused-prefix row: the cache-populating
prefill and the cached-prefix scoring prefill both report `cuda graph: False`, while mixed
rows still log `SGLang FP4 KV mixed mode enabled: K cache uses FP8 e4m3, V cache uses
packed NVFP4`. This promotes Qwen mixed-KV radix reuse from a single 4096-prefix control
to a graph-safe 8k prefix-ladder baseline. Full NVFP4 K+V remains separate and red/open.

The deep-prefix continuation is also complete. Artifact:
`results/sglang_qwen_mixedkv_prefixcacheguard_deep_prefix_sweep_ctx8192_20260610TmanualJST.md`.
It extends the same fixed-8k corpus path through reused prefixes `4096, 6144, 7168, 7680`.
All rows are `ok=true`; the deepest rows score only `1023` and `511` continuation tokens,
so their small positive deltas (`+0.008188` and `+0.010784` nats/token) are short-span
checks, not a stable quality trend. The run completes the requested prefix-depth curve for
the mixed-KV row.

Capacity denominator audit is now explicit. Artifact:
`results/sglang_qwen_mixedkv_capacity_denominator_audit_20260610TmanualJST.md`.
The `~1.78x` figure in the old SGLang mixed-KV rows was a pre-fix allocator-token ratio,
not the raw FP8-K + NVFP4-V storage ratio. The normalized mixed-KV storage gain is about
`16/(8+4.5) ~= 1.28x` versus fp8 at equal KV byte budget. The old `~1.78x` happened because
the mixed runs also allocated about `1.39x` more total log-reported K+V GB than fp8; the
allocator sized the pool from the logical full-FP4 dtype estimate while the experimental
pool physically stored FP8 K plus NVFP4 V.

The claim manifest for the current SGLang fallback row is
`results/sglang_qwen_mixedkv_claim_manifest_20260610TmanualJST.md`. Status:
claim-ready for Qwen mixed FP8-K + NVFP4-V on GB10 with radix cache ON and the prefix-cache
graph guard active. This is not a full NVFP4 K+V claim.

The pool-configurator denominator fix is now implemented and live-verified. Artifact:
`results/sglang_mixedkv_poolconfigfix_20260610TmanualJST.md`. The fixed mixed-KV row now
allocates `3,990,192` tokens versus fp8 `3,119,614` at the same `--mem-fraction-static
0.40`, with physical K+V bytes essentially equal (`41.62 GB` versus `41.66 GB`). The
current mixed-KV capacity claim is therefore the normalized `~1.28x` ratio. The older
`~1.78x` mixed-KV allocator ratio is a pre-fix overcommit artifact and should not be
quoted as current.

Gemma 3 27B Rung 1 has a first SGLang mixed-KV checkpoint:
`results/sglang_gemma3_27b_rung1_mixedkv_20260610TmanualJST.md`. Text-only Gemma 3 now
serves with experimental hybrid-SWA memory enabled, measured runtime geometry is uniform
`D=128` with 52 local and 10 global layers, and mixed FP8-K + NVFP4-V passes a short
sequential PPL comparator at `ctx=512`, `reuse_prefix_len=256`
(`delta_nats_per_token=+0.000690`). The SWA wrapper needed to select
`MHATokenToKVPoolFP4` for float4 KV; otherwise mixed-KV failed at allocation with
`fill_cuda` unsupported for `Float4_e2m1fn_x2`. This is a checkpoint, not a full Gemma
claim: long-context/deep-prefix Gemma 3, CUDA graphs, and full NVFP4 K+V remain pending.
The absolute PPL is high because the eval text is a deterministic repository markdown
corpus slice, not a cleaned benchmark corpus; the claim is the matched fp8-vs-mixed delta.

Gemma 4 E4B epoch-2 Rung 1 has a SGLang full-NVFP4 checkpoint:
`results/sglang_gemma4_e4b_rung1_fullnvfp4_20260611TmanualJST.md`. With
`jethac/sglang@96a9ff9ce` and `jethac/flashinfer@76af7982`, full NVFP4 K+V
(`--kv-cache-dtype fp4_e2m1`, `SGLANG_FP4_KV_MIXED_KV=0`) serves a short
OpenAI chat prompt coherently and passes a short native prompt-logprob PPL pair at
`ctx=512`, `reuse_prefix_len=256` against bf16/auto (`delta_nats_per_token=-0.190174`).
The D=512 globals route through `extend_paged_vosplit*` and
`decode_as_prefill_vosplit*`.

`96a9ff9ce` fixes the hybrid full-NVFP4 allocator denominator by accounting for
packed data plus scale buffers in `HybridSWAPoolConfigurator`. The fixed row reports
`1,274,008` full tokens versus bf16/auto `357,187` (`3.5668x`) and fp8 allocator
tokens `715,185` (`1.7814x`). Before the fix, full NVFP4 received only `716,992`
tokens and used only `16.15 GB` of KV memory, so it had quality but not capacity.

Do not over-claim the fp8 quality comparison. The fp8 comparator is still red:
SGLang allocates fp8 KV, warns that no fp8 scaling factors were provided, then times
out internally after 600 seconds and returns `Internal Server Error`. The current
claim is full-NVFP4 E4B short quality green versus bf16/auto plus allocator capacity
green versus fp8; fp8 quality remains a separate open issue.

Two upstream draft issues are banked for later filing:

- `results/upstream_draft_issue_sglang_prefix_cache_graph_write_20260610TmanualJST.md`
  covers the piecewise CUDA graph prefix-cache write corruption that affects fp8 as well
  as mixed-KV.
- `results/upstream_draft_issue_flashinfer_head512_selector_overpromise_20260610TmanualJST.md`
  covers the FlashInfer/vLLM `head_dim=512` selector-vs-kernel mismatch relevant to the
  future Gemma 4 rungs.

Stretch readiness is recorded in
`results/sglang_full_nvfp4_structural_route_readiness_20260610TmanualJST.md`. The full
NVFP4 K+V radix route should start as a separate issue-named implementation branch, not as
part of the mixed-KV claim packet.

## 2026-06-10 update — mixed-KV default radix first-token gate is green

Artifact: `results/sglang_qwen_mixedkv_default_20260610T0042JST_summary.md`.

The default Qwen radix-cache row was rerun with `SGLANG_FP4_KV_MIXED_KV=1`, radix cache
ON, page size 1, FlashInfer attention, CUDA graph disabled, and the existing source-stack
image. This directly targets the old failure where the second request reused a 55-token
FP4 cached prefix and flipped the first token from `**` to `ark`.

Result: green for the targeted first-token gate. Both request orders now return `**` on
the cached-prefix second request:

- OpenAI first then native second: native cached tokens `55`, first token `**`,
  logprob `-0.7601577043533325`.
- Native first then OpenAI second: OpenAI cached tokens `55`, first token `**`,
  logprob `-0.7601577043533325`.
- Fresh/no-reuse controls stay at `**`, logprob `-0.7235294580459595`.

The dense-cache summary audit passes with `ok: true`, `733` trace events, `648`
metric-bearing comparisons, and no findings.

Important caveat: this fixes the observed first-token radix failure; it does not prove
exact dense-vs-cached tensor equality. The first request-bound tensor difference remains
layer-0 attention output with cosine `0.4661444810372346`, `max_abs=0.2578125`, and
`rms=0.11784679304779001`. The difference no longer flips the tested first token, but
long-form quality still needs a real gate.

Capacity log: the mixed first-token gate reports `#tokens: 5573469`, `K size: 37.21 GB`,
`V size: 20.93 GB`, and `max_total_num_tokens=5573469`. The fresh sequential comparator
then reports `5,537,968` mixed-KV tokens versus `3,119,168` fp8 tokens, with short-decode
parity (`57.804` versus `57.594 tok/s`).

## 2026-06-09 update — K-side FP4 quantization is the current blocker

The dense-quant split is now run and changes the diagnosis:

- summary: `results/sglang_qwen_fp4kv_kvsplit_trace_20260609_summary.md`
- run id: `sglang_qwen_fp4kv_kvsplit_trace_c3dae30f_8b95253af_20260609T1335Z`
- source overlay: `jethac/sglang@8b95253af`, `jethac/flashinfer@c3dae30f`
- runtime image: `sglang-source-stack-c3dae30f-a8ad6a3ac`

The default radix row is still red (`**` dense/no-prefix versus `ark` on a 55-token
cached-prefix hit), and the first request-bound divergence is still layer-0 attention
output (`cosine=0.006467887232207366` on the sampled first head). But the split clears
the merge path as the primary suspect: previous prefix-reference tracing showed the
paged prefix contribution and base-2 LSE match a manual FP4 dequant reference, and the
final `_safe_merge_state` result matches the same online-softmax formula exactly.

The new K/V split shows the real sensitivity:

- actual dense FlashInfer output vs BF16 reference: `cosine=0.9999995231628418`
- FP4 K+V reference vs BF16: `cosine=0.7876883745193481`
- FP4 K-only reference vs BF16: `cosine=0.7893718481063843`
- FP4 V-only reference vs BF16: `cosine=0.996927797794342`

Interpretation: the row is K-side attention-logit sensitivity, not V reconstruction,
not page/scale pairing, and not merge math. Direct K reconstruction still has high raw
cosine (`~0.9967`) but large absolute error (`max_abs ~41-42`, RMS `~6`) on layer-0 K,
which is enough to move the softmax winner. The simple scale-convention suspicion was
also falsified: `scripts/sglang_fp4_quant_scale_probe.py` shows SGLang's current
scale convention and FlashInfer's helper convention both reconstruct a synthetic KV
tensor around `cosine ~0.9955`.

Follow-up direct scale diff, 2026-06-09: the captured failing run does not show a
dense-vs-cached global-scale mismatch. Layer 0 dense write/dequant and dense-quant
attention both record `k_global=0.1197916716337204` and
`v_global=0.0016276042442768812`; the failing `extend_merge_paged` cached-prefix call
receives the same `k_scale` and `v_scale`. FlashInfer's paged prefill/decode wrappers
use that handedness by multiplying `k_scale` into `sm_scale` and applying `v_scale` to
the output, matching the local dequant reference. Since the paged prefix `o2/s2` already
matches the local FP4-dequant attention reference, a stale value or reciprocal-scale
interpretation is falsified for this row. The key distinction is that the dense no-prefix
serving path uses BF16 K/V attention; the FP4-dequant reference inside that same dense
row is already bad on K-only.

K-scale policy probe, 2026-06-09: a scalar K scale is a real lever, but not a sufficient
fix. The inactive trace sweep in `jethac/sglang@dfd426442` tested K global-scale
multipliers `0.125,0.25,0.5,1,2,4,8` on the same layer-0 row. The best offline setting,
`0.125`, improved the FP4 K-only attention reference cosine from `0.7893718481063843`
to `0.9584803581237793`, and the FP4 K+V reference from `0.7876883745193481` to
`0.9561840295791626`, while direct K reconstruction dropped from about `0.9967` to
`0.8766`. The actual serving test in `jethac/sglang@e4f24bbd3` with
`SGLANG_FP4_KV_K_GLOBAL_SCALE_MULTIPLIER=0.125` did not pass: the radix-hit second
request changed from `ark` to `To`, not back to `**`, and the first layer-0 attention
divergence improved only from `0.006467887232207366` to `0.1657561728524288`. Treat
this as evidence that the current global K calibration is not attention-quality
optimal, but do not bless a scalar multiplier as a workaround.

Per-head K scale policy probe, 2026-06-09: `jethac/sglang@d6fa9d104` tested whether a
finer K policy can fit the current layout by quantizing each KV head with its own
amax-derived K global, folding the head/global ratio into the existing FP8 block-scale
buffer, and dequantizing under the single scalar K global that FlashInfer receives. The
best main-row setting is again multiplier `0.125`: FP4 K+V attention cosine improves
from `0.7876883745193481` to `0.954595148563385`, and K-only from
`0.7893718481063843` to `0.9574685096740723`, while direct K reconstruction drops to
`0.8836419582366943`. This is no better than the scalar `0.125` regime, and the scalar
runtime run already failed real radix-on serving. Do not implement per-head serving
policy as the next fix unless new evidence appears; it is currently a trace-only
falsifier for "head granularity is enough."

Dense-reference partial-state probe, 2026-06-09: `jethac/sglang@5b71bef3c` adds the
decisive reuse-vs-dense comparison. Result:

- cached Q versus dense no-prefix Q for the same logical token: `cosine=0.999993`
- suffix partial (`o1/s1`) versus dense suffix recompute: output `cosine=0.999994`
- merge math versus its own inputs: `cosine=0.99999988`
- cached-prefix partial (`o2/s2`) versus the FP4-dequant cached-page reference: output
  `cosine=0.999997`, LSE max abs `0.001953`
- cached-prefix partial versus the BF16 dense-prefix recompute: output
  `cosine=0.851723`, LSE max abs `484.75`
- full dense recompute versus cached merged output: `cosine=0.721851`

This corrects the previous framing. The no-prefix "dense" request serves over BF16 K/V
and only writes FP4 KV after attention; it does **not** prove that dense FP4 K is
quality-equivalent. The reuse path is internally consistent with the FP4 cache, but the
FP4 cached-prefix state is not close enough to the BF16 dense-prefix state. The remaining
SGLang bug is therefore not radix page pairing, stale scales, suffix attention, or
`_safe_merge_state`; it is the quality/semantics gap introduced when a later request
reuses the FP4-compressed prefix that the first request never had to attend through.

Next target: K-side policy/quality. Investigate calibrated K scale quality, per-head or
per-group K scales only if they are tied to a stronger quality objective than amax,
FP8/BF16 K with FP4 V, or model-specific gating for Qwen. Do not spend more time on
radix page pairing or `_safe_merge_state` for this row unless new evidence contradicts
the K-only split. A K-not-FP4 fallback must be reported with its capacity cost before it
can be considered a blessed serving result: naive FP8 K + NVFP4 V gives about
`16 / (8 + 4.5) = 1.28x` the fp8 KV pool, versus about `1.78x` for NVFP4 K+V.

Mixed-K/V ABI review, 2026-06-09:
`results/sglang_qwen_fp4kv_mixed_kv_abi_20260609_summary.md` records the stop-point
finding. The current FlashInfer FA2 paged-attention surface binds K and V to a single
`DTypeKV`: Python plan/run cache one `kv_data_type`, the JIT template emits one
`DTypeKV`, `paged_kv_t<DType, IdType>` has separate K/V pointers but one element type, and
prefill consumes `paged_kv_t<DTypeKV, IdType>`. Therefore FP8/BF16 K + NVFP4 V is not a
small SGLang memory-pool switch; it needs a FlashInfer mixed-KV attention API/kernel plus
SGLang integration. Keep the capacity estimate above, but treat mixed-K/V as a kernel/API
task until that surface exists.

## 2026-06-09 update — current-head source-stack retest still red

After vLLM Gemma 3 passed the short first-token gate with `jethac/flashinfer@c3dae30f`, we
rebuilt the SGLang lane as a reusable current-head source-stack image and reran the
Qwen FP4-KV cached-prefix/default row:

- artifact: `results/sglang_qwen_fp4kv_dense_cache_c3dae30f_e631a13fd_20260609T102017Z_summary.md`
- image: `sglang-source-stack-c3dae30f-e631a13fd`
- runner: `scripts/run_sglang_fp4_dense_cache_trace.sh`
- stack: editable `jethac/flashinfer@c3dae30f`, editable `jethac/sglang@e631a13fd`,
  source-built `sglang-kernel 0.4.3`
- case: `default`
- result: **red**. Current-head SGLang still emits the cached-prefix token flip:
  no-cache rows emit `**` (`-0.7235294580459595`), while 55-token radix-hit rows emit
  `ark` / `838` (`-0.5874708890914917`). Flush-between and namespace-isolated rows keep
  `cached_tokens=0` and stay clean.

Interpretation: the FlashInfer paged-prefill struct/plumbing fix that unblocked vLLM Gemma
does **not** by itself close SGLang Qwen FP4-KV radix reuse. The failure still follows
cached-prefix reuse, independent of endpoint order.

Trace status: post-hoc parser repair made this a clean request-bound localization artifact.
The comparator now treats warmup/health-check forwards as ignored provenance and validates
the `432` request-bound events strictly: `324` dense, `108` cached, `0` unknown, `648`
matched tensor comparisons, `0` metricless comparisons, and no schema findings. The first
localized request-bound divergence is layer-0 attention output equivalence: dense
full-prefill `o_rows` versus cached-prefix merged `merged_rows`
(`cosine=0.006467887232207366`, `max_abs=0.318359375`,
`rms=0.13599727805129772`) between dense `openai-first` and cached `native-second`.
So the artifact quality is green; the remaining red state is real FP4 cached-prefix
attention behavior, not missing trace request IDs or a later Qwen2/logits path.

Next diagnostic is implemented but not yet run: `jethac/sglang@a8ad6a3ac` adds inactive
`SGLANG_FP4_KV_TRACE_DENSE_QUANT_ATTENTION=1`. On the dense no-prefix path it recomputes
the sampled attention row with BF16 K/V and with SGLang's own FP4 quantize/dequantize pair
after cache global-scale calibration. This separates "FP4 KV quantization alone moves the
attention output" from "radix/paged reuse moves it beyond the FP4 quantization loss."

SM12x build note: the source build succeeds, but the build log contains repeated
performance warnings: `242` `.multicast::cluster` / `cp.async.bulk{.tensor}` advisories,
`109` references each to `compute_120a` and `compute_121a`, and `74` `setmaxnreg`
compatibility warnings. The warnings affect FP8 blockwise, int8/FP8 GEMM, W4A8 grouped
MoE, and FP8 blockwise MoE kernels. This is not the FP4-KV correctness bug, but it is a
separate SGLang SM12x performance-portability issue.

Previous attempt after the FlashInfer prefill fix:

- artifact: `results/sglang_qwen_fp4kv_after_fi0919_default2_20260609T1818JST_summary.md`
- runner: `scripts/run_sglang_fp4_dense_cache_trace.sh`
- case: `default`
- intended test: whether the same FlashInfer FP4 paged-prefill wrapper fix closes the
  Qwen FP4-KV cached-prefix failure.

Result: **inconclusive before serving**. The source-stack runner spent about 26 minutes
rebuilding `sglang-kernel` and was still at `82/127` build targets when stopped; no request
JSON or dense-cache comparison was produced. This does not falsify the FlashInfer cross-lane
hypothesis. It means the next SGLang step is packaging/build-loop work: prepare a reusable
source-stack image or narrow `sglang-kernel` build target first, then rerun the default
radix row from that prepared stack.

## ROOT CAUSE LOCALIZED — the radix/prefix cache (2026-06-08) — NEXT FOCUS
The first-token divergence test cornered it
(`results/sglang_qwen_fp4kv_radix_isolation_20260608T2038JST_summary.md`):
- fp8: OpenAI and native endpoints agree (`**`).
- FP4 default: diverge (OpenAI `**`, native `ark`/838).
- **FP4 with `--disable-radix-cache`: agree again (`**`/334).** Skip-warmup alone does NOT
  fix it; radix-off does.

So the SGLang FP4-KV bug is **radix/prefix-cache KV reuse** — a cached FP4 KV prefix is
mishandled on reuse. `--disable-radix-cache` is both the proof and a correctness workaround
(at the cost of prefix caching). **This remains a cross-lane pattern, but the simple
page/scale mismatch hypothesis is now weaker:** vLLM Gemma 3 27B also fails FP4 quality,
yet its high-limit trace shows sampled read-side packed data/scale bytes match write-side
bytes (`195 / 195`) and the failing short prompts do not skip SWA blocks; the follow-up
active-page dump shows the exact failing paged-prefill wrapper returns byte-like BF16
`out_after` whose first 16 values match the first 16 active packed V bytes; the CPU replay
against dequantized active pages then produces sane signed attention output with near-zero
cosine versus the real wrapper output. See
`docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md`. The SGLang `f76f80484` write/read trace
likewise clears the simplest stale/wrong-page scale-buffer version for sampled radix pages.
**SGLang's job:** prove whether the reused cached-prefix contribution is numerically
identical to a recomputed FP4 prefix contribution. A `--disable-radix-cache` row is
diagnostic/emergency workaround evidence only; it must not be blessed as an FP4-KV serving
result because prefix reuse is part of the serving behavior the capacity win must survive.

Instrumentation head: `jethac/sglang@ce1b6d15e` adds inactive-by-default
`SGLANG_FP4_KV_TRACE_RADIX=1` logs through `Req.init_next_round_input`,
`ForwardBatch.init_new`, and FlashInfer prefill/extend path selection. Use it to compare
the default FP4 native request against the radix-off request and prove whether the cached
prefix's packed KV bytes and FP8 scale buffers stay aligned.

Trace result (`results/sglang_qwen_fp4kv_radix_trace_20260608T213052JST_summary.md`):
the default FP4 native request fails (`**` vs `ark`/`838`) while reusing a 55-token prefix
(`prefix_indices_len=55`, `extend_prefix_lens_cpu=[55]`) and running
`forward_extend_merge_paged`; the radix-off row passes (`**` vs `**`) with
`prefix_indices_len=0`, `extend_prefix_lens_cpu=[0]`, and
`forward_extend_ragged_no_prefix`. This narrows the next fix to FP4 cached-prefix merge
page handling, not raw quantizer math, pool layout, prompt serialization, or graph capture.
The next hook must compare reused page IDs with K-data/K-scale/V-data/V-scale page IDs.

Page-pair trace result
(`results/sglang_qwen_fp4kv_page_pair_trace_20260608T214649JST_summary.md`):
`jethac/sglang@839cb7457` adds `SGLANG_FP4_KV_TRACE_PAGE_PAIR=1` and records the
FlashInfer paged plan plus FP4 data/scale view geometry. The default row still fails, but
the paged plan consumes the same 55 logical page IDs as the radix prefix (`4113..4167`),
and all 28 layers report matching first-dimension extents for K data, V data, K scale, and
V scale. So a gross page-list mismatch is not observed. The next hook moves inside
`extend_merge_paged`: sample actual FP4 data/scale bytes at the reused page IDs and log
`o1/s1/o2/s2` before `_safe_merge_state`.

Merge-state trace result
(`results/sglang_qwen_fp4kv_merge_trace_20260608T220823JST_summary.md`):
`jethac/sglang@991ac1e63` adds `SGLANG_FP4_KV_TRACE_MERGE_STATE=1` and records the
layer-0 cached-prefix merge. The default row still fails (`OpenAI **` vs native
`ark`/`838`) with `cached_tokens=55`; radix-off still passes (`**`/`334`) with
`cached_tokens=0` and no paged-prefix merge. The failing layer-0 trace sees readable,
nonzero packed K/V bytes and FP8 K/V scale bytes at pages `4113..4116`, and `o1/s1`,
`o2/s2`, and merged output are all finite. The merged sample matches the paged-prefix
sample, which is plausible for a 55-token cached prefix over a one-token ragged suffix and
is not alone proof of a merge bug. The next hook is now **write/read pairing**: trace
`MHATokenToKVPoolFP4.set_kv_buffer()` for the same physical page IDs and verify the bytes
and scale bytes written during cache fill match those read during radix reuse. If they
match, build a same-prompt no-prefix reference for the paged-prefix contribution and
inspect FlashInfer FA2 paged-prefix numerics / merge weighting.

Write/read trace result
(`results/sglang_qwen_fp4kv_write_read_trace_20260608T222204JST_summary.md`):
`jethac/sglang@f76f80484` adds `SGLANG_FP4_KV_TRACE_WRITE_READ=1`. The default row still
fails (`OpenAI **` vs native `ark`/`838`) with `cached_tokens=55`; radix-off still passes
(`**`/`334`) with `cached_tokens=0`. For layer 0 cached pages `4113..4116`, sampled K
data, V data, K scale, and V scale all match write input bytes = stored bytes = read bytes.
This clears the simple "scale buffer not copied / stale / wrong page" hypothesis for the
sampled pages. The next hook is now numerical: compare the cached FP4 paged-prefix
contribution against an equivalent no-prefix/full-ragged FP4 reference for the same
55-token prefix and query, then inspect FlashInfer FA2 paged-prefix dequant / LSE / merge
weighting if they diverge.

Prefix-reference trace result
(`results/sglang_qwen_fp4kv_prefix_ref_trace_20260608T2306JST_summary.md`):
`jethac/sglang@2a228949a` adds and fixes
`SGLANG_FP4_KV_TRACE_PREFIX_REF=1`. The default row still fails (`OpenAI **` vs native
`ark`/`838`) with `cached_tokens=55`; radix-off still passes (`**`/`334`) with
`cached_tokens=0`. For the sampled failing layer-0 request, cached paged-prefix `o2`
matches the dequantized torch reference (`cosine=0.999997`, `max_abs=0.0078125`), `s2`
matches after converting the reference LSE to FlashInfer's log2 convention
(`max_abs=0.001953125`), and manual `exp2` merge exactly matches `_safe_merge_state` at
BF16 precision (`max_abs=0.0`). This clears FP4 paged-prefix read/layout/dequant, LSE
units, and merge-state math for the sampled failing case. The next hook moves upward:
trace calibration/quantization-error impact at cache fill and request sequencing/state
across the OpenAI/native radix-hit pair.

Quant-error trace result
(`results/sglang_qwen_fp4kv_quant_error_trace_20260608T2325JST_summary.md`):
`jethac/sglang@d4fe78078` adds `SGLANG_FP4_KV_TRACE_QUANT_ERROR=1`. The default row still
fails (`OpenAI **` vs native `ark`/`838`) with `cached_tokens=55`; radix-off still passes
(`**`/`334`) with `cached_tokens=0`. Layer-0 dense-vs-dequant error is effectively
identical in the failing and passing rows: for the 56-token fill, K cosine is `0.996669`
and V cosine is `0.995715` in both rows, with the same global scales
(`k=0.1197916716337204`, `v=0.0016276042442768812`). This clears calibration/global-scale
selection and ordinary FP4 quant/dequant loss as the distinguishing factor. The next
hook is request-order/cache-state only: run the
`tasks/sglang_qwen_fp4kv_request_order_probe_20260608.md` packet to prove whether the
failure follows the second request / cached prefix, independent of endpoint.

Request-order trace result
(`results/sglang_qwen_fp4kv_request_order_20260608T2340JST_summary.md`):
`scripts/sglang_fp4_request_order_probe.py` proves the failure is endpoint-independent.
OpenAI-first/native-second fails on the native call (`cached_tokens=55`, `ark`/`838`);
native-first/OpenAI-second fails on the OpenAI call (`cached_tokens=55`, `ark`).
Flush-between and namespace isolation both keep `cached_tokens=0` and both endpoints emit
`**`. OpenAI `cache_salt + extra_key` and native `extra_key` are real radix namespaces; the
trace shows distinct namespace strings and no prefix hit. The remaining blocker is now
FP4 cached-prefix quality/reuse itself. Prior traces proved the bytes, scale bytes,
paged-prefix read, LSE convention, and merge are internally consistent for the sampled
failure, so the next fix/probe should compare full dense prefill versus FP4 cached-prefix
attention quality and test whether better global scales or a selective no-reuse policy is
needed.

Radix-reuse code-read checkpoint (2026-06-09): on a radix hit, SGLang reuses physical KV
slot IDs rather than copying K/V data or scale tensors into the new request.
`schedule_batch.py` gets cached physical indices from `tree_cache.match_prefix(...)`;
`common.py` writes those prefix indices into the request's `req_to_token` row and appends
only new suffix slots; `flashinfer_backend.py` `forward_extend_merge_paged` computes
suffix attention from fresh BF16 K/V and prefix attention from paged cached FP4 K/V.
`MHATokenToKVPoolFP4.set_kv_buffer()` writes packed K, packed V, K scale, and V scale to
the same physical slot, and `move_kv_cache()` copies all four buffers together. This is
consistent with the existing traces: stale/miscopied scale bytes are no longer the leading
theory. The next probe should compare dense full-prefill attention/logits against the FP4
cached-prefix read path, not another scale-copy trace.

Cached-prefix top-logprob post-analysis
(`results/sglang_qwen_fp4kv_cached_prefix_toplogprob_postanalysis_20260608T2346JST.md`):
the cached-prefix rows are not a small candidate-rank wobble. Baseline and reverse-order
failures both have `0 / 20` first-token top-logprob token overlap between the no-cache
first request and the 55-token cached-prefix second request; flush-between and namespace
isolation rows have `20 / 20` overlap. This keeps selective no-reuse in the diagnostic or
emergency-workaround bucket only. The accepted fix must preserve FP4 prefix reuse and
recover the top-logprob distribution.

No-code diagnostic switch checkpoint (2026-06-09): do not add a new switch before using
the existing ones.

- `SGLANG_FLASHINFER_USE_PAGED=1` forces extend away from
  `ragged suffix + paged cached prefix + _safe_merge_state` while keeping the radix prefix
  hit. Expected trace: `prefix_indices_len=55`, `extend_prefix_lens_cpu=[55]`,
  `use_ragged=False`, label `forward_extend_paged`, paged plan length `[56]`.
- `SGLANG_RADIX_FORCE_MISS=1` forces a radix miss without disabling the server's radix-cache
  feature path. Expected trace: `prefix_indices_len=0`, `extend_prefix_lens_cpu=[0]`, full
  prompt in `input_ids`, label `forward_extend_ragged_no_prefix` unless paged is also
  forced.

Next SGLang live matrix:

1. Default FP4: cached prefix hit, `forward_extend_merge_paged`, known bad.
2. `SGLANG_FLASHINFER_USE_PAGED=1`: cached prefix hit, full paged attention.
3. `SGLANG_RADIX_FORCE_MISS=1`: full recompute, no prefix hit.
4. Both env vars: full recompute through full paged attention.

Interpretation:

- If row 2 passes while row 1 fails, the bug is split ragged/paged merge interaction.
- If rows 1 and 2 fail while rows 3 and 4 pass, reused FP4 prefix state/contribution is
  the bug.
- If row 4 passes, it is the cleanest full-paged FP4 recompute comparator against cached
  full-paged reuse.

Matrix execution stop point
(`results/sglang_qwen_fp4kv_matrix_20260609tmatrix10jst.md`): the runner now gets past the
source-overlay build prerequisites and the FlashInfer version guard without downgrading the
runtime stack. It builds a throwaway image from `nvcr.io/nvidia/sglang:26.05-py3` with
rustup stable (`rustc 1.96.0`) and `protobuf-compiler`, removes stale FlashInfer packages
and cache state, upgrades CUTLASS DSL to `4.5.2`, installs editable
`jethac/flashinfer@4c3c0d99`, then installs editable
`jethac/sglang@d4fe78078`. The version probe reaches `flashinfer_python 0.6.13`,
`sglang-kernel 0.4.3`, and `sglang 0.5.12.post2.dev1018+gd4fe78078`. The server then
exits before health because PyPI `sglang-kernel 0.4.3` loads an `sm100/common_ops.abi3.so`
that is ABI-incompatible with the NVIDIA 26.05 Torch/CUDA stack on GB10
(`undefined symbol: _ZNK2at10TensorBase14const_data_ptrIiLi0EEEPKT_v`). The four matrix
rows therefore produced no request JSON and are not quality evidence. Next run must build
`sglang-kernel` from source against the same container Torch/CUDA stack, or source a
matching ARM64 CUDA 13.x wheel with SM121-compatible `common_ops`. Do not downgrade
SGLang, Torch, FlashInfer, or the container to make the guard pass.

Kernel source-build follow-up
(`results/sglang_kernel_source_build_20260609Tprobe3jst.md`): `jethac/sglang@d96869237`
adds default-on `sgl-kernel` CMake switches so the GB10 probe can build only the
precise-math `sm100` package path instead of compiling SM90/FA3/FlashMLA too. The narrow
source build against NVIDIA 26.05 Torch `2.12.0a0+5aff3928d8.nv26.05` and CUDA `13.2`
succeeds, emits `compute_121a` ptxas warnings but no hard failure, installs
`sglang-kernel 0.4.3`, and imports
`/usr/local/lib/python3.12/dist-packages/sgl_kernel/sm100/common_ops.abi3.so` on GB10. The
ABI blocker is therefore cleared. The next matrix run should prepare a source-stack image
once and run all four rows from it; rebuilding `sglang-kernel` inside each row is too slow.

Source-stack matrix result
(`results/sglang_qwen_fp4kv_matrix_20260609tprep1jst.md`): the reusable image
`sglang-source-stack-20260609tprep1jst` installs editable `jethac/flashinfer@4c3c0d99`,
editable `jethac/sglang@d96869237`, and the source-built `sglang-kernel 0.4.3` against
the NVIDIA 26.05 Torch/CUDA stack. The loaded extension is the rebuilt
`/usr/local/lib/python3.12/dist-packages/sgl_kernel/sm100/common_ops.abi3.so`, so the
runtime ABI blocker is cleared for real matrix rows. The four-row result keeps SGLang FP4
KV red but localizes it further:

- default cached-prefix reuse still fails: the second request reuses `55` cached tokens and
  returns `ark`/`838` while fresh rows return `**`/`334`;
- `SGLANG_FLASHINFER_USE_PAGED=1` with the radix hit changes the token to newline, but
  cached-prefix logprobs still differ from fresh rows, so full-paged attention alone is not
  an equivalence fix;
- `SGLANG_RADIX_FORCE_MISS=1` is clean: all rows have `cached_tokens=0`, token `**`, and
  identical logprob;
- force-miss plus full-paged is also internally clean, with `cached_tokens=0` and identical
  newline logprobs.

Decision: the failure follows FP4 cached-prefix reuse, not the PyPI wheel, FlashInfer
version guard, endpoint formatting, or ordinary full recompute. The next SGLang probe
should compare dense full-prefill attention/logits against the FP4 cached-prefix path.
`SGLANG_RADIX_FORCE_MISS=1`, namespace isolation, and `--disable-radix-cache` stay in the
diagnostic/emergency-workaround bucket; they are not the blessed capacity path.

## Why this, why now
The SGLang FP4 KV row already expands the KV pool ~1.78× over fp8 on GB10. The newest
`d7d931f` matched row improves the evidence: raw `2+2` and chat smoke pass, and backend
trace covers both decode and `extend_merge_paged`. But the standardized FP4 benchmark
text still degenerates, while fp8 produces normal text. The underlying FlashInfer FA2
NVFP4-KV kernel is proven correct standalone (`jethac/flashinfer@e152cf4d`, cosine ≥
0.99999946), and the layout probe ruled out scale-rank as the cause
(`results/sglang_nvfp4_kv_layout_probe_20260608.json`, dequant cosine 0.9999957).
**So the bug is in SGLang's integration or quality-sensitive serving path, not in the
basic kernel math or pool contract.** The job: find and fix that bug, then land a
blessed matched fp8-vs-FP4 serving row.

## Prime suspect — read this first
Turn 1 of this whole investigation was: "SM120/121 should use the `fp4_quantize`
fallback with **inverted global scale**, not `nvfp4_kv_quantize` — why?" The answer:
quantize and consumer are a matched pair on global-scale convention. `nvfp4_kv_quantize`
applies the encode scale by **multiply** (`s_enc = 6·448/amax`); the `fp4_quantize`
fallback applies the global scale by **divide** (`s_dec = 1/s_enc`). Crossing kernels
without reciprocating = off-by-`s_enc²` → NaN/garbage logits.

The current SGLang patch (`jethac/sglang@67c7967` → `eefe8aded`) **routes SM12x through
`nvfp4_kv_quantize`** (the multiply convention). The eager-mode corruption is exactly
the symptom you'd expect if its GB10-runnable FlashInfer FA2 consumer expects the
**divide** convention.

**CONFIRMED on hardware (2026-06-08).** The convention bridge
(`results/sglang_nvfp4_kv_convention_probe_20260608.json`) ran all four quantizer/reader
pairings on GB10. The FA2 reader is a **decode-convention reader**; matched pairs are
numerically correct (cosine 0.995 vs source, 0.99999 vs dequant), and the naive
`nvfp4_kv_quantize` + **encode** scale → decode reader is the literal **cosine-0.0**
off-by-`s_enc²` failure:

| quantizer | reader | attn cosine vs source | verdict |
|---|---|---:|---|
| `fp4_quantize` encode | decode | 0.995 | ✓ valid pair |
| `nvfp4_kv_quantize` **encode** | decode | **0.000** | ✗ the crossed garbage case |
| `nvfp4_kv_quantize` decode | decode | 0.995 | ✓ valid pair |
| `nvfp4_kv_quantize` decode | encode | 0.248 | ✗ mismatched |

So the kernel math is **exonerated** and the valid pairings are known: either
`fp4_quantize` + encode scale, or `nvfp4_kv_quantize` + **decode** (inverted) scale.
SGLang's remaining serving corruption means the **full path produces an unmatched pair**
— the bug is now in calibration / V-scale / backend integration (Objective A), not in
the convention at the raw-math level.

**vLLM is now your reference implementation.** vLLM consumes the *same* FlashInfer FA2
reader and, as of 2026-06-08, serves it **cleanly** with normal content and 1.751× fp8
capacity (`results/vllm_qwen_nvfp4_kv_capacity_20260608T1455JST_summary.md`, server log:
`Using FlashInfer FA2 backend for NVFP4 KV cache on SM12x with vLLM V-scale-factor
deswizzle enabled`). vLLM has therefore already produced a correct end-to-end matched
pair on GB10. Diff SGLang's quantize convention + V-scale layout + calibration against
what vLLM's clean row does; the divergence is the bug.

## GB10 memory safety (MANDATORY — read before any serving run)
GB10 has **unified memory**: the KV pool + model + a second comparator server all draw on
one shared ~119 GiB CPU+GPU pool. On 2026-06-09 a vLLM run exhausted it and deadlocked the
kernel (`docs/INCIDENT_20260609_OOM_DEADLOCK.md`); the same risk applies to SGLang's matched
fp8-vs-fp4 rows. Rules:
- Leave **≥15–20 GiB OS headroom**; keep `--mem-fraction-static` conservative (the FP4 KV
  pool is *large* by design — 1.78× fp8 — so a high fraction overshoots fast).
- **Never run the fp8 and fp4 comparator servers concurrently** at high mem-fraction — run
  them **sequentially** (stop server A before starting server B) or cap each so the sum
  < ~100 GiB. The matched-row pattern is the exact trigger from the incident.
- **Cgroup-limit the container** so a runaway is OOM-killed in-cgroup instead of wedging the
  kernel. Pre-run: `free -h` should show ~115 GiB available.

## Methodology / sequencing constraint
Do NOT build a container image in the dev loop. SGLang already iterates the right cheap
way and should keep doing so:
  1. **Editable source overlay on stock `nvcr.io/nvidia/sglang:26.05-py3`** with the
     `jethac/sglang` source + local FlashInfer JIT source (exactly the autosafe-row
     setup). Note overlay loses `.git`, so it is overlay evidence, not a pinned/wheel
     proof — fine for dev, label it honestly.
  2. **The proven standalone FlashInfer reference is ground truth.** Use
     `scripts/flashinfer_nvfp4_kv_probe.py` / `jethac/flashinfer@e152cf4d` and extend
     `scripts/sglang_nvfp4_kv_layout_probe.py` into a per-op numerical bridge to localize
     divergence. Most root-causing happens here, with zero serving.
A blessed serving row + clean container is the final deliverable, gated on quality
passing — not a dev step.

## What is already proven — do NOT redo these
- KV4Compatibility server-arg gates: `jethac/sglang@eefe8aded`, `3 passed / 56
  deselected` (`results/sglang_fp4_kv_sm121_pytest_20260608T0320JST.md`). Python-level
  arg compatibility only.
- `KVFP4QuantizeUtil` alias of `BlockFP4KVQuantizeUtil`: `jethac/sglang@98ad46961`.
- **Capacity**: matched autosafe row,
  `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_summary.md` — FP4 KV `5,519,481`
  tokens vs fp8 `3,101,822` = **1.779×**; calibration runs (`NVFP4 KV cache calibrated
  28 layers from 4096 eager prefill tokens`). This is capacity-proven; treat it as done.
- **Negative results that narrow the search**: scale-rank (4D vs 3D) is NOT the bug
  (layout probe dequant cosine 0.9999957); the FlashInfer FA2 NVFP4-KV kernel is correct
  standalone (e152cf4d). Do not re-investigate either — they're cleared.
- **Convention bridge is DONE** (`sglang_nvfp4_kv_convention_probe_20260608.json`, see
  table above): valid pairings identified, kernel math exonerated, bug localized to
  calibration / V-scale / backend integration. Do not re-run the raw-math bridge; the
  next work is finding where the *serving* path produces an unmatched pair.
- **Pool bridge is DONE** (`results/sglang_fp4_pool_bridge_probe_20260608.json`,
  `results/sglang_fp4_pool_bridge_probe_prefill_20260608.json`):
  `MHATokenToKVPoolFP4.set_kv_buffer()` writes packed K/V plus FP8 scale buffers that
  FlashInfer FA2 can consume directly. The widened probe used real pool getters for both
  `BatchDecodeWithPagedKVCacheWrapper` and `BatchPrefillWithPagedKVCacheWrapper`; both
  passed with `attention_cosine_vs_dequant=0.9999946`, while K/V dequantized back to the
  BF16 source at cosine about `0.9955`. This clears the basic pool layout and
  global-scale application surface for decode and paged prefill; the remaining serving
  corruption is later in backend wrapper/server sequencing, CUDA graph state, or a
  model-path difference not covered by the synthetic pool bridge.
- **Backend trace and matched comparator are now captured**
  (`results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_summary.md`):
  `jethac/sglang@d7d931f` adds opt-in `SGLANG_FP4_KV_TRACE_BACKEND=1` logging. A
  source-overlay Qwen run on the NVIDIA 26.05 image, with
  `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`, reached readiness, allocated `5,517,572`
  FP4 KV tokens versus `3,105,240` fp8 tokens (`1.7769x`), calibrated 28 layers, traced
  all 28 decode layers and all 28 `extend_merge_paged` layers through packed `uint8` K/V
  plus FP8 scale buffers, returned `spark-ok`, and produced sane raw `2+2` text. This
  is still not a blessed row because the FP4 standardized benchmark content remains
  degraded.
- **Logprob quality localization is DONE**
  (`results/sglang_qwen_fp4kv_d7d931f_logprob_quality_20260608T1609JST_summary.md`):
  `scripts/openai_quality_probe.py` compared the degraded FP4 prompts against the fp8
  comparator with generated-token logprobs. FP4 `short_decode` starts with the same
  high-confidence prefix as fp8 (`A local AI workstation`) then drifts into mixed
  Chinese/repetition; FP4 `medium_decode` diverges at token one (`the following code:`
  instead of `**Engineering Note:`) and collapses into repeated `import` text. This
  proves the quality bug is prompt/path-sensitive generation corruption, not just a
  missing backend trace or a capacity-only artifact gap.
- **Native `/generate` divergence-window probe is DONE**
  (`results/sglang_qwen_fp4kv_d7d931f_native_divergence_20260608T1626JST_summary.md`):
  rendering the same `medium_decode` chat prompt with the Qwen tokenizer and calling
  native `/generate` gives a sharper window. fp8 and FP4 match for the first four output
  tokens (`**`, `Engineering`, ` Note`, `:`), then diverge at token index 4: fp8 selects
  ` Valid` while FP4 selects ` Validate`. Both alternatives appear in both top-k lists,
  but FP4 reverses their rank. This is an early decode distribution shift that compounds,
  not a first-token catastrophe under native `/generate`.
- **OpenAI-vs-native prompt reconciliation is DONE**
  (`results/sglang_qwen_fp4kv_prompt_path_reconcile_20260608T173754JST_summary.md`):
  the OpenAI path and local Qwen chat-template render use identical 56-token prompt IDs
  for both fp8 and FP4 (`sha256=5a5d4572e0e3d940a909b85dc4a00350094cbd1d55333c3d4f0a7974a91ee517`).
  Prompt serialization is therefore not the cause. The endpoint split is real: FP4 OpenAI
  Chat Completions still starts plausibly and diverges at token 4, while native
  `/generate` from the same prompt IDs diverges at token 0 (`**` -> `ark`). The next bug
  surface is FP4 endpoint/request metadata or pre-sampling numerics, not chat-template
  tokenization.
- **Endpoint metadata localization packet is DONE**
  (`results/sglang_qwen_fp4kv_endpoint_metadata_20260608T1819JST_summary.md`):
  an offline pass over the prompt-reconciliation artifact records the same 56-token prompt
  hash for FP4 OpenAI and native paths, the OpenAI first token `**`, and the native first
  token `ark` (`838`). Existing backend traces cover decode and `extend_merge_paged`, but
  they are not request-tagged, so they cannot separate OpenAI request state from native
  `/generate` request state. The smallest next hook is now defined:
  `scripts/sglang_fp4_first_token_dump_patch.yaml` patches only `ModelRunner.sample()` to
  dump `next_token_logits` before/after `_preprocess_logits()` plus `ForwardBatch`
  `input_ids`, `positions`, `seq_lens`, and `rids`.

Read `docs/NVFP4_KV_PORTING_MAP.md` (SGLang Reference Map) and the autosafe summary
before starting.

## The SGLang problem, precisely
1. **Quality corruption in eager/no-graph serving** (keystone). The latest trace row
   passes raw/chat smoke, but the standardized benchmark content still degrades with
   CUDA graph and piecewise capture disabled. This is not merely a graph bug, and it is
   subtler than the earlier raw `2+2` failure.
2. **Worse corruption with CUDA graphs.** Graph-enabled decode corrupts output, so the
   fork currently auto-disables capture (`SGLANG_FP4_KV_ENABLE_CUDA_GRAPH=1` opt-in).
   This forces the slow no-graph path (an early variant measured 0.276 tok/s). Separable
   from #1 and likely a calibration-vs-capture state problem.
3. **FlashInfer FP4 decode kernel was force-compiled past errors** (`vec_dtypes.cuh`,
   group6 dtype mismatch, packed head-dim — `fi_fp4_decode_*` probes, commit `fb7f0a1`).
   Confirm the decode path is numerically correct, not merely compiling.

## Objectives, in order
**A. Fix the eager-mode quality corruption (keystone).**
The raw-math convention bridge is done (kernel exonerated; valid pairings known). The
bug is in the **serving path producing an unmatched pair**. Trace where SGLang's
end-to-end path diverges from a valid pairing, suspects in order:
   1. **Global-scale convention in the serving path** — the bridge proved
      `nvfp4_kv_quantize` + **encode** scale → decode reader is cosine-0 garbage. Confirm
      which scale SGLang actually feeds at serving time; switch to a valid pairing
      (`fp4_quantize` + encode, or `nvfp4_kv_quantize` + **decode/inverted**) and check
      raw `2+2`. **Cross-check against vLLM's clean row, which already gets this right.**
   2. **V-scale layout** — SGLang default is **symmetric-linear** V scale-factors;
      vLLM uses **B2 swizzle + in-kernel deswizzle**. SGLang must consume the FlashInfer
      kernel built **without** `FLASHINFER_PAGED_V_SF_DESWIZZLE` (vLLM's clean row builds
      it **with** the macro — so do not copy vLLM's flag, copy its *matched-ness*). A
      layout/macro mismatch corrupts V exactly like a convention mismatch.
   3. **Per-layer calibration application / prompt-path state** — the 28-layer /
      4096-token calibration must
      apply the same global scales at quantize and at in-kernel dequant. This is the most
      likely remaining culprit now that raw convention is understood. The prompt
      reconciliation probe proves OpenAI and native paths can use identical prompt IDs,
      yet FP4 OpenAI and native `/generate` diverge differently. Next compare FP4 request
      metadata, forward-mode/prefill state, and pre-sampling logits/hidden states between
      the two endpoints before changing the quantizer again.
   4. Only then the decode kernel itself (Objective B/C overlap).

**B. Confirm the FlashInfer FP4 decode kernel is numerically correct.**
The decode compile fixes (`vec_dtypes.cuh`, group6 dtype, packed head-dim) must be
validated against the standalone reference at the SGLang shapes, not just "it builds."
Decode is the daily-driver path; a subtly wrong decode kernel reproduces #1's symptom.

**C. Fix the CUDA-graph-capture corruption.**
Once eager is correct, graph corruption means calibration/global-scale state isn't
captured. Likely calibration-before-capture ordering or graphs capturing stale/
uncalibrated scales. Goal: serve FP4 KV with graphs on, so the capacity win isn't stuck
behind a slow no-graph path. Keep `SGLANG_FP4_KV_ENABLE_CUDA_GRAPH` as the gate until
proven.

**D. Land the blessed matched fp8-vs-FP4 serving row.**
Same model / prompts / `--mem-fraction-static` / `--page-size` / graph mode. Quality
must pass: raw `2+2` = `4`, coherent benchmark content, plus a real quality comparator
(PPL or retrieval sanity vs fp8/bf16). Record KV pool tokens, max concurrency, memory
telemetry, TTFT, warmed decode tok/s. Server log must prove native FlashInfer FP4 KV
selection — not fp8/bf16 fallback. Use Qwen2.5 1.5B (the established comparator) first.

**E. Gemma via SWA-aware mixed KV (Strategy B) — gated behind the Qwen quality fix.**
The shared Gemma blocker is a FlashInfer register/fragment-shape guard on `D=512` (see the
vLLM doc Objective B — `8*NUM_MMA_D_VO = 256 ≥ 256` before the KV term), so a true
full-FP4-KV global-attention kernel is a hard, separate FlashInfer track, not a quick fix.
The near-term Gemma path on **both** lanes is therefore **mixed KV**: NVFP4 on local
(`D=256`) layers, fp8/bf16 on global (`D=512`) layers — capturing most of Gemma's ~5:1
local:global capacity win while dodging the broken kernel.

SGLang's *mechanism* is its **hybrid-SWA subpool delegation** (`swa_memory_pool.py`,
`mem_cache/`): route local-attention layers to the FP4 subpool, global to fp8/bf16. This is
the SGLang counterpart to vLLM's per-layer `kv_cache_dtype_skip_layers` plumbing — the two
lanes implement the *same strategy through non-overlapping code* and cross-validate it.

**Prerequisite — do NOT start this until SGLang's Qwen FP4-KV quality is blessed
(Objectives A–D).** Gemma's SWA complexity will *mask* whether the long-sequence Qwen
degradation is actually fixed; building Gemma on an unblessed Qwen path is building on sand.
SGLang Gemma also has its own open blockers (issue #14). The shared surface is only the
FlashInfer guard — don't edit `prefill.cuh` trait math here; that lands once, in FlashInfer.
Coordinate attention geometry with the vLLM lane
(`docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md`).

Rung -1 config audit update (2026-06-08): `docs/GEMMA_RUNG_MINUS1_CONFIG_AUDIT.md` shows
Gemma 3 27B is the SWA-only server rung: uniform `D=128`, 52 sliding layers, 10 full layers,
and no `D=512`. Gemma 4 12B/31B/26B-A4B all carry full-attention `D=512`, with 26B-A4B
also adding MoE. Once Qwen FP4-KV quality is blessed, SGLang Gemma should mirror the vLLM
ladder by starting with Gemma 3 27B, not Gemma 4.

Ladder order update (2026-06-08): after Gemma 3 27B, the next SGLang Gemma rung is **Gemma
4 31B text-only**, not 12B. Operator-provided architecture says 31B and 26B-A4B are
encoder-based text+vision models, so text-only serving quarantines vision in the unfired
encoder and isolates attention/KV. Prove dense `D=512` mixed-KV on 31B, then add MoE on
26B-A4B text-only. Gemma 4 12B is last because its encoder-free multimodality is fused into
the decoder/KV path; it is the destination, not the stepping stone.

## Evidence gates (a row isn't a claim without these)
- Source-overlay/build evidence with a valid `sm_121a`/`compute_121a` FlashInfer target.
- `cuobjdump`/JIT-cache proof the running FP4 KV decode kernel matches the claimed path.
- Server log proving native FlashInfer FP4 KV selection (not fallback).
- Deterministic sanity (raw `2+2` = `4`) AND a quality comparator vs fp8/bf16/dequant.
- Capacity/concurrency vs an fp8 comparator at matched settings.
- CUDA-graph-replay coverage once graphs are re-enabled.
- Explicit scope labels: SWA/Gemma, page-size variants, TP>1, MTP/spec-decode untested.

## Guardrails
- **Keep capacity and quality claims separate.** The current mixed-KV capacity claim is
  `~1.28x` at matched physical K+V byte budget. Historical `~1.78x` mixed-KV rows were
  pre-fix allocator overcommit artifacts; full NVFP4 K+V `~1.78x` remains red/open.
- **Convention discipline (the turn-1 lesson).** Document the matched quantize↔consumer
  pair explicitly: which global-scale convention (multiply vs divide) and which V-scale
  layout (SGLang symmetric-linear, NOT vLLM B2 swizzle). Most of this lane's risk is a
  convention/layout mismatch between a correct quantizer and a correct consumer.
- **Lane ownership.** FlashInfer owns kernel/page/stride and the symmetric-linear V-scale
  behavior (even when a reference patch ships under the SGLang overlay). SGLang owns
  memory pool, KV dtype, calibration, FlashInfer wrapper plumbing, and server args. Do
  not put kernel math fixes in SGLang.
- **Validate on `sm_121a` only; SM120 ride-along.** Same policy as the vLLM doc: build on
  `hikarioyama/sglang-nvfp4-kv-sm120@9b2160f0` as prior art, keep patches SM12x-family-
  shaped, re-derive (do not vendor the overlay tree), emit `120a`+`121a` not `120f`,
  label SM120 compiled-but-unclaimed (hikari-validated, not us). The 99 KB/block SMEM
  ceiling is confirmed family-wide. See the "SM120 ride-along" section of
  `docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md`.
- **Maintain RTX PRO 6000 (SM120) compatibility in the quality fix, not just the build.**
  The convention/V-scale/calibration fix must land on the SM12x-family gate
  (`is_sm120_supported()` covers SM120 and GB10/SM121), derived from hikari's SM120
  reference — never an sm_121-only special case. Turn-1's `fp4_quantize`-fallback +
  inverted-scale guidance was always framed for the whole SM120/121 family; keep it that
  way so a correct GB10 result is also correct on the RTX PRO 6000 we can't test.
- **Hikari's working SM120 path is a debugging oracle for Objective A.** Their SGLang
  SM120 reference serves correctly on SM120, so it has already resolved the global-scale
  convention and V-scale layout for the family. If our GB10 output is corrupt and theirs
  is clean, diff our quantize/scale/calibration handling against hikari's — the
  divergence likely *is* the bug. This makes RTX PRO 6000 compatibility and the quality
  fix the same problem, not competing ones.
- Use issue-named worktrees (issue #18 for SGLang NVFP4 KV); reference #2555-style
  in-flight upstream work to avoid collisions on the backend selector.

## First concrete step (no image builds)
The matched `d7d931f` row, OpenAI logprob probe, native `/generate` divergence-window
probe, OpenAI-vs-native prompt reconciliation, and offline endpoint metadata localization
are done. Do not repeat them as-is. Prompt IDs match; prompt serialization is retired as
the cause, and untagged backend traces are insufficient for the endpoint split. The next
step is a live one-token FP4 run with `scripts/sglang_fp4_first_token_dump_patch.yaml`
enabled, then compare OpenAI versus native `next_token_logits` before and after
`ModelRunner._preprocess_logits()`. No serving image is required until quality passes.
