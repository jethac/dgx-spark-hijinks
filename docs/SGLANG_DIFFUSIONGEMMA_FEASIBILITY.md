# SGLang DiffusionGemma Feasibility

Date: 2026-06-11 JST

Scope: offline, code-anchored feasibility study for adding DiffusionGemma serving to
SGLang on DGX Spark / GB10. This is not an implementation and does not claim any live
serving result. The upstream serving reference today is vLLM's open `dgemma` branch and
the official `vllm/vllm-openai:gemma-aarch64-cu130` image; see
`docs/DG0_SERVING_STACK_RECON.md`.

Epoch-2 implementation checkpoint: `jethac/sglang` now has the DG-S0/DG-S2 foundation
scaffold: a local `DiffusionGemmaConfig` alias, `DiffusionGemmaForBlockDiffusion`
model shell, BF16 encoder/decoder-backbone weight remap into one Gemma 4 causal
backbone, self-conditioning parameter ownership, dLLM config recognition, and
`scripts/diffusion_gemma_config_audit.py` for metadata-only geometry manifests. This
does not claim BF16 parity or serving. Decoder denoise mode still raises
`NotImplementedError`; the next live gate is a Linux/Spark metadata manifest followed
by a BF16 weight-load manifest against the official vLLM image as oracle.

## Verdict

SGLang support is feasible, but it is a medium-large runtime feature, not a model-file
only port.

The reusable pieces are real:

- DiffusionGemma inherits the Gemma 4 mixed attention geometry exactly: sliding/local
  layers use D=256, global layers use D=512, with a 5:1 sliding/global layer pattern
  and a 1024-token sliding window.
- The SGLang Gemma 4 E4B route already proves the SGLang side of the D=512 dispatcher
  stack: D=512 globals use FlashInfer VO-split through paged prefill and decode-as-prefill,
  and full NVFP4 K+V is short-green at the expected allocator ratio.
- SGLang already has a diffusion-LLM scaffold under `sglang.srt.dllm`: a request phase
  machine, a scheduler mixin, a configurable algorithm plugin point, block-position
  handling, and existing SDAR / SDAR-MoE model examples that request full logits.

The missing pieces are also real:

- DiffusionGemma is not SDAR. It is a single Gemma 4 backbone run in two modes:
  encoder mode writes ordinary causal KV; decoder mode reads committed prefix KV, runs
  bidirectional attention inside a 256-token canvas, and does not write canvas KV.
- SGLang's current dLLM path is block-mask oriented. It fills a fixed masked block using
  `LowConfidence`, then streams the generated block. It does not model DiffusionGemma's
  block-autoregressive denoising loop, entropy-bound sampler, self-conditioning MLP,
  or encoder-commit pass.
- The hardest runtime surface is decoder-mode attention: bidirectional canvas attention
  plus read-only prefix KV, with no cache pollution from transient canvas tokens.

So the practical path is: port the architecture and scheduler first in BF16, compare
against the official vLLM image as the oracle, then enable the existing Gemma 4
FlashInfer / NVFP4 route once the mode semantics are correct.

## Inputs From DG-0 Recon

DiffusionGemma serving, per `docs/DG0_SERVING_STACK_RECON.md`:

- Architecture: `DiffusionGemmaForBlockDiffusion`.
- Base: Gemma 4 26B-A4B MoE text backbone with vision tower present but text-only serving
  can isolate the language/KV path.
- KV contract: YOCO-style two-mode backbone.
  - Encoder mode: causal attention, writes standard paged KV.
  - Decoder mode: bidirectional canvas attention, reads encoder KV, does not write KV.
- Canvas/sampler:
  - `canvas_length=256`
  - `max_denoising_steps=48`
  - entropy-bound sampler with `entropy_bound=0.1`
  - confidence/stability convergence gates
  - random renoise for non-converged positions
- Geometry:
  - sliding/local head dim 256
  - global head dim 512
  - 30 layers, full attention at layers 5/11/17/23/29
  - sliding window 1024
  - MoE: 128 experts, top-8
- Current Spark recipes force `TRITON_ATTN`; no NVFP4 KV path is currently proven there.

The GB10 thesis is therefore stronger than for AR Gemma 4: decoder denoising rereads
prefix KV many times per output block, so KV read bandwidth matters more than in ordinary
one-token decode.

## Existing SGLang Surfaces

### Diffusion Config

`third_party/sglang/python/sglang/srt/dllm/config.py` is the current dLLM entry point.
`DllmConfig.from_server_args()` activates only when `--dllm-algorithm` is set and maps
known architecture names to a fixed `block_size` and `mask_id`:

- `LLaDA2MoeModelLM`: block 32
- `SDARForCausalLM`: block 4
- `SDARMoeForCausalLM`: block 4

DiffusionGemma would currently raise `RuntimeError("Unknown diffusion LLM: ...")`.
The first config patch is to add `DiffusionGemmaForBlockDiffusion`, but that entry cannot
be only `{block_size: 256, mask_id: ...}`. It needs to carry canvas length, max denoising
steps, entropy-bound parameters, stability/convergence thresholds, random-renoise policy,
and whether the request is in encoder-commit or decoder-denoise mode.

### Request State

`third_party/sglang/python/sglang/srt/dllm/mixin/req.py` adds `dllm_phase`,
`dllm_block_offset`, and block-oriented `fill_ids` initialization. `_init_fill_ids_for_dllm()`
appends a full block of mask tokens, and `determine_dllm_phase()` decides prefill vs decode
by checking whether the current block contains the configured mask id.

For DiffusionGemma this is a starting point, not enough state. A request also needs:

- committed prefix length and current canvas start
- current canvas token ids, previous-step token ids, and optional random tokens for renoise
- per-position entropy/confidence/stability counters
- current denoising step, max step, and convergence status
- self-conditioning state or previous soft embeddings/logits, depending on the model class
- a hard separation between committed tokens that are eligible for prefix cache insertion
  and transient canvas tokens that must not be cached

### Scheduler

`third_party/sglang/python/sglang/srt/dllm/mixin/scheduler.py` provides the dLLM scheduling
loop. `SchedulerDllmMixin.get_new_batch_dllm()` moves requests through a `DllmManager`,
selects prefill or decode batches, builds a `ScheduleBatch`, and sets
`ForwardMode.DLLM_EXTEND`. `process_batch_result_dllm()` streams the returned token block
and releases cache when a request finishes.

This helps with request isolation and max-running-request gating, but DiffusionGemma needs
a different inner loop:

1. Encoder-prefill the prompt and write committed prefix KV.
2. Create a 256-token canvas.
3. Run decoder-denoise repeatedly over the canvas, reading prefix KV but not writing canvas KV.
4. Apply entropy-bound sampling and renoise non-converged positions.
5. When the block converges or hits the step cap, run encoder-commit for the block so its
   tokens become ordinary causal KV and can be reused by radix/prefix cache.
6. Stream or return the committed block, then repeat until `max_new_tokens` or stop criteria.

The current scheduler can host this, but `DllmReqPhase` likely needs explicit phases such
as `ENCODER_PREFILL`, `DECODER_DENOISE`, and `ENCODER_COMMIT` rather than only
incoming/staging prefill/decode.

### Algorithm Interface

`third_party/sglang/python/sglang/srt/dllm/algorithm/low_confidence.py` shows the current
algorithm contract. It runs full-logit forwards over a masked block, fills high-confidence
positions, and finally returns a tensor list of generated token ids.

DiffusionGemma needs a new algorithm implementation, not a retuned `LowConfidence`:

- use the entropy-bound sampler from the checkpoint/generation config
- maintain a temperature schedule across denoising steps
- check both entropy and argmax-stability convergence
- renoise non-converged positions with random vocab tokens
- consume self-conditioning state from the previous denoise step

The plugin mechanism in `sglang.srt.dllm.algorithm.__init__` can discover a new algorithm
class by module import, so the extension point is already usable.

### ForwardBatch And Attention Modes

`third_party/sglang/python/sglang/srt/model_executor/forward_batch_info.py` defines
`ForwardMode.DLLM_EXTEND`, treats it as an extend/prefill mode, and marks it as CUDA-graph
capable. It also overrides positions from `dllm_block_offset` when `batch.dllm_config` is
present.

`third_party/sglang/python/sglang/srt/layers/radix_attention.py` already has
`AttentionType.DECODER`, `DECODER_BIDIRECTIONAL`, and `ENCODER_ONLY`.
`third_party/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py` already has
custom-mask plumbing in paged prefill and a cross-attention update path.

That is useful, but DiffusionGemma needs a more specific contract:

- encoder mode should look like Gemma 4 causal decode/extend and save KV;
- decoder mode should set bidirectional/canvas masking, read the committed prefix pages,
  and pass `save_kv_cache=False` for canvas tokens;
- the backend must distinguish the prefix KV region from the dense canvas region without
  accidentally inserting canvas K/V into radix cache;
- any FlashInfer plan must still declare D=512 global layers through the VO-split path.

The existing `ENCODER_ONLY` SDAR examples are not sufficient because they do not model
read-only prefix KV plus transient bidirectional canvas K/V.

### Existing Diffusion Models

`third_party/sglang/python/sglang/srt/models/sdar.py` and `sdar_moe.py` are the best local
model references:

- both use `RadixAttention(..., attn_type=AttentionType.ENCODER_ONLY)`;
- both set `LogitsProcessor(..., return_full_logits=True)`, which the dLLM algorithm needs;
- `SDARMoeForCausalLM` demonstrates a dLLM MoE model class.

DiffusionGemma should not be derived directly from SDAR, but these files show the full-logit
and dLLM model wiring patterns.

The backbone reference is `third_party/sglang/python/sglang/srt/models/gemma4_causal.py`.
It already handles:

- heterogeneous head dims and KV heads by layer;
- Gemma 4 MoE routing and ModelOpt-style per-expert NVFP4/FP8 checkpoint layout;
- full-NVFP4 KV pool scales through `RadixAttention` layer objects;
- D=512 global layers through the campaign's FlashInfer VO-split routing.

## Required SGLang Work

### 1. Config And Registration

Add `DiffusionGemmaForBlockDiffusion` to the model registry by adding a new model file with
`EntryClass`. Extend `DllmConfig.from_server_args()` to recognize the architecture and parse
DiffusionGemma-specific generation config fields.

Do not hardcode only `block_size=256`. The server must record the actual model config:
canvas length, max denoising steps, sampler type, thresholds, head geometry, sliding/global
layer map, and max running requests. The running model remains the ground truth.

### 2. Weight Loading

Implement a DiffusionGemma model class that reuses the Gemma 4 language backbone but remaps
the checkpoint's encoder/decoder names into one SGLang module. Per DG-0 recon, the vLLM
implementation loads `model.encoder.language_model.*` into a single Gemma 4 backbone, skips
duplicated decoder backbone weights, and loads `model.decoder.self_conditioning.*` into a
decoder-only self-conditioning MLP.

The SGLang loader needs explicit remaps for at least:

- `model.encoder.language_model.*` -> Gemma 4 backbone params
- decoder duplicate backbone weights -> skipped or checked for equality
- `model.decoder.self_conditioning.*` -> new self-conditioning module
- vision tower weights -> skipped for text-only rungs unless multimodal is in scope

Start BF16. Only after BF16 matches the official vLLM image should the NVFP4 checkpoint be
loaded, because its GB10 path is unverified upstream.

### 3. Two-Mode Model Forward

The model class needs an explicit mode on `ForwardBatch` or request state:

- `encoder`: ordinary Gemma 4 causal forward, writes KV, uses existing SGLang Gemma 4
  FlashInfer/VO-split routing.
- `decoder`: same Gemma 4 layers and weights, plus self-conditioning, but attention is
  bidirectional over the canvas and read-only over prefix KV.

The current `Gemma4Attention.forward()` always calls `self.attn(..., save_kv_cache=...)`
with normal decoder semantics. DiffusionGemma needs a guarded path that sets the right
attention type/mask and disables KV writes in decoder mode.

### 4. Block-AR Scheduler

Replace the current mask-block lifecycle with a block-autoregressive lifecycle:

- prompt encoder prefill
- repeated decoder denoise over a 256-token canvas
- encoder commit of the converged canvas
- stream/return the committed block
- repeat

This likely belongs in `SchedulerDllmMixin` and `ReqDllmMixin`, but as DiffusionGemma-specific
phases rather than changing the SDAR path in place. The current dLLM `max_running_requests`
limit should be preserved; the official vLLM recipe uses a low cap because diffusion state
buffers are activation-heavy.

### 5. Canvas Attention

This is the largest unknown.

A correct first implementation can be slow if it is explicit and testable. Acceptable rung
order:

1. BF16 decoder canvas with no prefix reuse, using a dense/ragged custom mask.
2. BF16 decoder canvas reading committed prefix KV, no NVFP4.
3. FlashInfer paged-prefix read plus canvas custom mask.
4. D=512 VO-split and full NVFP4 KV only after the BF16 semantics match.

The custom-mask and cross-attention surfaces in `flashinfer_backend.py` are promising, but
the first proof must check the exact plan kwargs at runtime: query lengths, prefix lengths,
page size, window, `head_dim_qk`, `head_dim_vo`, `k_data_type`, `v_data_type`, and custom
mask presence. The campaign has repeatedly shown that shape-only probes miss runtime wrapper
state bugs.

### 6. Prefix Cache Semantics

Only encoder-committed tokens may be inserted into radix/prefix cache. Canvas tokens are
transient denoising state and must not become prefix-cache entries until the encoder commit
pass runs over the finalized block.

Green criteria for prefix cache:

- first request writes prompt + committed blocks as ordinary causal KV;
- repeated request proves prefix-cache hits on committed tokens;
- no cache hit is possible for an in-progress canvas;
- cached committed prefix KV produces the same canvas result as a fresh encoder prefix.

This is especially important for FP4 KV: the Qwen/SGLang radix work showed that "cache
reuse" is its own correctness surface.

### 7. FlashInfer / NVFP4 Enablement

Once BF16 is correct, the existing Gemma 4 work should transfer:

- D=512 global layers use VO-split with `head_dim_qk=512`, `head_dim_vo=256`.
- SGLang's linear V scale-factor layout should remain the native path; do not import
  vLLM's swizzled V-SF behavior.
- Full NVFP4 K+V should use the fixed hybrid denominator and the SGLang E4B full-NVFP4
  checkpoint as the allocator/routing baseline.

However, DiffusionGemma amplifies KV reads. A bad FP4 prefix-cache or canvas reader will
compound across 12-48 denoising passes. Require a writer-roundtrip gate and a vLLM-oracle
quality gate before quoting capacity.

## Proposed Rungs

### DG-S0: Static Recon And Config Load

Goal: SGLang recognizes the checkpoint architecture and prints measured geometry from the
running model, then exits before serving.

Gate:

- `DiffusionGemmaForBlockDiffusion` does not fall back to `TransformersForCausalLM`.
- Config reports canvas length, sampler, denoising step cap, mixed head dims, layer map,
  MoE parameters, and KV bytes/token.

### DG-S1: BF16 Model Load, No Diffusion Loop

Goal: load the BF16 checkpoint into a single Gemma 4 backbone plus self-conditioning module.

Gate:

- missing/unexpected weight report is explainable;
- encoder backbone weights load once;
- decoder duplicate weights are skipped or verified;
- self-conditioning weights load;
- text-only vision weights are quarantined.

### DG-S2: Encoder Commit Path

Goal: run ordinary causal encoder prefill/commit through SGLang Gemma 4 attention.

Gate:

- KV writer-roundtrip passes at D=256 sliding and D=512 global layers;
- D=512 globals route through VO-split;
- committed prefix cache hits are byte/layout correct.

### DG-S3: Decoder Canvas BF16, Single Request

Goal: run one 256-token canvas denoising step with bidirectional canvas attention and no
prefix-cache reuse.

Gate:

- canvas tokens do not write KV;
- full logits are returned for all canvas positions;
- output/logits compare against official vLLM image for a fixed prompt and seed.

### DG-S4: Block-AR Scheduler

Goal: implement repeated denoise, convergence, encoder commit, and block streaming for a
single request.

Gate:

- emits more than one 256-token block without cache pollution;
- stop handling works across block boundaries;
- output agrees with official vLLM within the expected sampler tolerance.

### DG-S5: Prefix Cache

Goal: repeated requests reuse committed prefix KV only.

Gate:

- metrics prove cache hits for committed prefix tokens;
- reused-prefix result matches fresh-prefix result;
- canvas tokens never appear in radix cache.

### DG-S6: FlashInfer And NVFP4

Goal: enable the GB10 path: FlashInfer, VO-split globals, full NVFP4 KV, then the NVFP4
checkpoint.

Gate:

- BF16 FlashInfer matches BF16 Triton/reference;
- full NVFP4 KV routes with correct scale tensors and no AOT/JIT cache ambiguity;
- quality and throughput compare against the official vLLM docker on the same prompt set;
- capacity ratio is reported separately from generation speed.

## Risks And Open Questions

- Upstream vLLM support is open-branch only as of this study; interfaces may still change.
- The current SGLang dLLM scheduler is SDAR-shaped and may need a parallel DiffusionGemma
  scheduler rather than incremental patches to `LowConfidence`.
- Decoder canvas attention is the main unknown: SGLang has mask and attention-type building
  blocks, but not the exact YOCO read-only-prefix plus bidirectional-canvas contract.
- SGLang's Gemma 4 checkpoint is dense E4B; DiffusionGemma is 26B-A4B MoE. The Gemma 4 MoE
  weight loader and NVFP4 expert paths need their own DG rung, not assumed from E4B.
- CUDA graphs are disabled for dLLM today in server-arg handling in several cases. Treat
  graph capture as a later optimization, not a correctness dependency.
- GB10 unified memory rules apply more strongly: diffusion state buffers add memory beyond
  weights and KV, so start with low `max_running_requests` and conservative memory caps.

## Recommended Next Move

Do not start by serving the NVFP4 checkpoint. Start with a code branch that implements only
DG-S0 and DG-S1:

1. Add a `diffusion_gemma.py` model shell with `EntryClass`.
2. Add `DiffusionGemmaForBlockDiffusion` config recognition.
3. Load BF16 weights into a Gemma 4 backbone plus a stub self-conditioning module.
4. Emit a measured geometry manifest and a weight-load manifest.

Only after the model-load manifest is clean should scheduler work begin. That keeps the
first SGLang DiffusionGemma artifact small, reviewable, and directly comparable to the
official vLLM branch.
