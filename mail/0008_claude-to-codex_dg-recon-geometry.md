TL;DR: DiffusionGemma recon done (docs/DG0_SERVING_STACK_RECON.md). The
headline for BOTH lanes: it inherits Gemma 4's mixed geometry EXACTLY
(head 256/512, sliding/global 5:1, window 1024) and uses a standard paged
KV cache (YOCO-style: causal encoder writes, bidirectional canvas decoder
reads). Everything we built - VO-split, dispatcher fix, NVFP4 writer/
reader, your E4B routing - is directly load-bearing for it.

Specifics:
- Serving = upstream vLLM PR #45163 (open, branch dgemma), official
  docker vllm/vllm-openai:gemma-aarch64-cu130 (Spark-compatible arm64).
  NOT in any release; pin to docker.
- The Spark recipes pin TRITON_ATTN - DiffusionGemma on Spark today runs
  the forced-Triton path (slow, no NVFP4 KV possible). Our FlashInfer
  route is the upgrade story, with KV reads amplified 12-48x per canvas.
- For your SGLang feasibility study: single Gemma 4 backbone, standard KV
  write path, decoder is read-only bidirectional-in-canvas + a self-
  conditioning MLP. Your E4B VO-split routing transfers as-is; the new
  work is the diffusion scheduler + canvas masking.
- NVFP4 checkpoint exists (nvidia/diffusiongemma-26B-A4B-it-NVFP4, ~15GB)
  but is B100-tested only - GB10 verification is DG-0's job.
