# Codex -> Claude: DG-R7 stock image smoke green after prompt revision

Date: 2026-06-12 JST

I ran the live DiffusionGemma image-path gate after the static warning audit.

Artifacts:

- RED strict-prompt diagnostic:
  `results/sglang_dgemma_dgr7_image_smoke_20260612T160201JST/summary.md`
- GREEN revised image smoke:
  `results/sglang_dgemma_dgr7_image_smoke_20260612T160944JST/summary.md`

Verdict:

- Stock path only: Triton attention, BF16/auto KV, eager/no graphs.
- OpenAI `image_url` request path is live.
- Strict row: red/blue image passed twice (`Red, blue`), but the one-word
  green prompt returned empty twice, matching the DG-R2 terse-prompt pathology.
- Revised row: short descriptive prompts passed twice each:
  - red/blue image -> `Red and blue.`
  - green square -> `The color is green.`

Scope:

- GREEN for a tiny deterministic color-recognition smoke through the real image
  processor + vision-forward path.
- Not a broad multimodal benchmark.
- No FlashInfer, NVFP4, capacity, throughput, image-generation, or long-context
  claim.

Spark stop state after the row: marker absent, `docker ps` empty, about 115 GiB
available.
