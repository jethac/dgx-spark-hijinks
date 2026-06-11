# Window packet: DG-0 baseline + 31B bf16 anchor (one window, ~90 min)

Authored 2026-06-11 under the epoch-2 goal. Protocol/guardrails unchanged.

## Part 1 - 31B bf16 anchor row (task 17 completion; ~30 min, weights cached)
r9 image (jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r9 -
has the dispatcher fix BAKED; no overlay needed). Server: google/gemma-4-31B-it,
bf16 (NO kv dtype flag), env VLLM_FLASHINFER_VOSPLIT=1, --language-model-only,
max-model-len 8192, util 0.72. This config crashed pre-fix; must now serve.
Proof: FA2 VO split line, no max_mma_kv, EXT_PATH/image provenance.
Measure: ctx-8191 PPL (corpus abb63f0e), first-token, 3 decode reps, KV tokens.
COMPLETES the quality table: bf16-FlashInfer anchor vs nvfp4 4.281335 /
fp8 4.473945 / bf16-Triton 4.6532(suspect). Note the cross-runtime anomaly:
SGLang E4B full-NVFP4 also beats bf16 (-0.19 nats) - record, don't explain yet.

## Part 2 - DG-0 DiffusionGemma baseline (~50 min incl. 52GB pull)
Per docs/DG0_SERVING_STACK_RECON.md + task 24: official docker
vllm/vllm-openai:gemma-aarch64-cu130 (PR #45163 stack), BF16 checkpoint
google/diffusiongemma-26B-A4B-it, recipe flags (--diffusion-config
'{"canvas_length":256}', --max-num-seqs 4, recipe-pinned TRITON_ATTN),
--memory 100g, util <= 0.72. Measure: serving comes up; coherent completion;
tok/s at short + 2k + 8k-context prompts (the KV-read-amplification
curve's first points); GPU KV pool; per-layer/group dispatch geometry from
logs; canvas/denoise config echo. If the official image fails on GB10,
fall back to the eugr community recipe (recon doc) and record both.
NVFP4 checkpoint verification is NOT this window (state-buffer +
quantized-weights interplay deserves its own run).
