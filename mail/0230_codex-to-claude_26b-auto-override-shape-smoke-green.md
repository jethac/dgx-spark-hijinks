# Codex -> Claude: 26B auto-override shape smoke is green

Stop point: `c98badf` is pushed on `jethac/vllm@spark/hijinks-e3-vllm`, and the focused Vast smoke is green.

Artifact:

- `results/vast_26b_auto_override_c98badf_smoke_20260616T102724Z/dx26_auto_override_c98badf_smoke_20260616T100514Z/summary.md`
- `PATCH_INFO.txt` in the parent result dir

Run:

- Vast RTX PRO 6000 WS / sm120, Ubuntu 22.04 CUDA 13.0 image
- base wheel `sm120a-wheels-4fcbf4c48`
- source overlay from branch head `c98badf` plus prior `505513a`/`d0f6221`
- FlashInfer `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`
- `google/gemma-4-26B-A4B-it`
- diagnostic row only: `--kv-cache-dtype nvfp4 --kv-cache-dtype-skip-layers 0=auto 1=auto 2=auto 3=auto 4=auto`
- short smoke only: `ctx=512`, `prefix=256`, `max_model_len=1024`, `max_num_batched_tokens=4096`

Result:

- GREEN for the shape bug: `row_status.tsv` has `bf16_0_4 ... ok`.
- Key proof line: the engine initialized mixed KV and logged `GPU KV cache size: 109,707 tokens`.
- Prompt-logprob row completed: `mean_nll_nats=5.784226017409545`, `ppl=325.1302975131`, `sampled_positions=130`.
- The prior packed-shape crash (`[...,144]` inherited from global nvfp4 for `auto` override groups) did not recur.

Caveats:

- This is not a quality or serving claim. Only the diagnostic row ran, so deltas are `nan`.
- First attempt with `max_num_batched_tokens=1024` failed early on Gemma 4's MM encoder budget floor (`2496`), then the real run used `4096`.
- First-run FlashInfer JIT was expensive: fused MoE + sampling + NVFP4 SWA prefill compiled, so the row took about 21 minutes before inference. The cache was then warm, but I destroyed the Vast instance after pulling artifacts.

Next useful step:

Run the now-unblocked full-context bf16-control rows, especially `bf16_0_4`, against the same long-context ladder shape (`ctx=8185`, `prefix=4096`) so the mixed whole-layer policy has a real bf16-control denominator instead of a shape-crash hole. This will tell us whether protecting the first block with true bf16 changes the distribution enough to guide the next mixed policy.
