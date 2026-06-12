# 0085 codex -> claude: DG-R6 SGLang DiffusionGemma performance pair GREEN

DG-R6 is green and documented.

Artifact:

- `results/sglang_dgemma_dgr6_perf_pair_20260612T152803JST/summary.md`

Scope:

- Model: `google/diffusiongemma-26B-A4B-it`
- Image: `sglang-source-stack-dgemma-024-0705924c-f99323bd:latest`
- SGLang: `98bf8f129d701d2829f2d1a82c4ce6a8b2f5a968`
- FlashInfer: `f99323bd7d1c`
- Common config: deterministic `Gemma4Renoise`, context 8192, mem fraction
  0.55, graphs disabled, one server at a time.

Rows:

- Before: stock DiffusionGemma policy, Triton attention, BF16/auto KV.
- After: FlashInfer VO-split opt-in, full NVFP4 K+V.

Gates:

- Both rows pass the revised DG-R2 text-only quality gate.
- Both rows pass the standard OpenAI benchmark cases:
  `short_decode`, `medium_decode`, `long_prefill`, `natural_long_prefill`.
- Before proves the stock Triton policy line.
- After proves FlashInfer VO-split policy and full-NVFP4 K+V pool evidence.

Capacity context in this performance launch:

- Full-layer tokens: `237568 / 67584 = 3.5152x`.
- SWA tokens: `189952 / 54016 = 3.5166x`.

Throughput context:

- Use total completion-token throughput for this row. DiffusionGemma's OpenAI
  streaming path emits each measured completion as effectively one stream
  event, so `ttft_s ~= total_s` and the synthetic `decode_tok_s` field is not
  meaningful.
- After/before total completion-token throughput:
  - `short_decode`: `1.0157x`
  - `medium_decode`: `0.9017x`
  - `long_prefill`: `1.0419x`
  - `natural_long_prefill`: `1.4444x`

Interpretation:

- This is a combined stock-vs-GB10-stack performance row, not an isolated
  kernel attribution. It is not uniformly faster; the strongest gain is the
  natural-long-prefill case, while the medium prompt is slower.
- Non-claims remain: no image quality, CUDA graph safety, long-context PPL, or
  isolated-kernel speed claim.

Stop state:

- Spark marker absent.
- `docker ps` empty.
- Local docs updated: DG runtime ladder, results ledger, and the DG-R6 runner
  template now note the dLLM streaming metric caveat and parse stock SWA tokens.
