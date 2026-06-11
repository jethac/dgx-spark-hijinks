# DG-1 cache analysis - working notes (2026-06-11, in progress)

Source: upstream/dgemma branch, vllm/model_executor/models/diffusion_gemma.py
(1359 lines; fetched into the vllm worktree as upstream/dgemma).

## Confirmed so far
1. Single Gemma4 backbone, mode-switched: encoder = causal + KV WRITE
   (standard Gemma4 layers); decoder = bidirectional + KV READ-ONLY +
   self-conditioning. Mode set by DiffusionGemmaModelState.prepare_inputs().
2. **GLOBAL (full-attention) LAYERS HAVE NO v_proj - V = K** (pre-k_norm;
   "k_eq_v" backbone variant; checkpoint has no v_proj weights there).
   CONSEQUENCES TO VERIFY: (a) does the paged KV cache store K and V
   separately (duplicated) on global layers, or deduped? If duplicated,
   NVFP4 + dedup could compound capacity gains on exactly the D=512
   layers; (b) our VO-split reads V half-views - with V==K the read
   semantics still work but quantization error correlates between K and V
   (anomaly investigation relevance?); (c) k_norm applies to K-as-K but
   NOT V-as-K (V taken pre-norm) - layout/scale implications for the
   NVFP4 writer on global layers.
3. MoE router takes raw (non-prenormed) input (router_uses_prenormed_input
   = False).
4. Canvas machinery: denoise/commit via entropy-bound mask
   (eb_mask/renoise with random tokens, argmax canvas, valid_canvas_len
   handling near max_model_len) - file lines ~488-620.

## RESOLVED: the canvas "mask" is a per-request causal FLAG, not a mask
DiffusionGemmaModelState.build_attn_metadata passes `causal` as a per-request
GPU TENSOR (encoder/commit phase = True, denoise phase = False; mixed batches
supported; diffusion_gemma.py ~1015). Semantics check out: canvas tokens sit
after the prefix, so plan(causal=False) over (qo=canvas, kv=prefix+canvas)
yields full prefix visibility + intra-canvas bidirectionality - exactly the
needed pattern. CONSEQUENCE for queue item 3 (FlashInfer enablement): NOT
packed-custom-mask work. Needed instead: (a) per-causality wrapper grouping
in our FlashInfer builder (encoder-phase reqs -> causal wrapper, denoise-
phase -> non-causal wrapper - same split-and-group pattern as our
decode-as-prefill routing); (b) plan(causal=False) on the VO-split path,
probe-validated (trivial variant of existing probes; P520). The mm-prefix
custom-mask machinery is NOT required for DiffusionGemma.

## Still to read (continuation pointers)
- forward() dispatch (line ~326) + how mode reaches the attention layers;
- the bidirectional masking implementation in decoder mode (what the
  canvas mask actually is at the attention-backend level - feeds the
  canvas-mask FlashInfer work, queue item 3);
- DiffusionGemmaModelState.prepare_inputs() (which file? grep ModelState
  in the dgemma branch) - canvas slot/KV-slot mapping, prefix commit path;
- whether KV write uses standard reshape_and_cache (recon says yes -
  verify call site);
- build_attn_metadata import from gpu.attn_utils (line 50) - decoder-mode
  metadata construction.

## DG-0 numbers to interpret against this (e783da9)
25.6 tok/s sustained at 44-51 denoise steps/canvas (temp 0); flat tok/s
50->6000 prompt tokens; KV pool 1.63M tokens bf16; NVIDIA claims 150
(adaptive 12-16 steps presumably; sampler config = next DG-3 variable).
