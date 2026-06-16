# 26B-A4B readout/hidden capture packet staged (offline, no live row)

Codex update, 2026-06-16 JST.

Vast is still clean but not usable from this side: `vastai show instances` returns no instances, and the last
allocation attempt still failed on account credit. I did not leave any cloud instance running.

Offline progress: staged the next readout discriminator for the vLLM 26B-A4B full-NVFP4 lane:

- `docs/vast_anchor/vllm_readout_capture_sitecustomize.py`
- `docs/vast_anchor/compare_readout_captures.py`
- `docs/vast_anchor/run_26b_readout_capture.sh`
- `docs/vast_anchor/launch_26b_readout_capture_live.sh`

Why this packet exists: the existing top-logprob packet uses public `prompt_logprobs` and can show the visible
distribution shift. It cannot tell whether the amplifier is already in final hidden states or only appears
after lm_head/readout. The new hook wraps `Gemma4ForCausalLM.compute_logits()`, skips one-row sample/decode
calls, and saves sampled final hidden rows plus raw top-k logits for each prompt-logprob chunk.

Comparator output is bucketed by local row inside each prompt chunk and reports:

- hidden cosine / rel-L2 vs bf16
- hidden RMS and mean-abs deltas
- logits top-1 match
- logits top-k Jaccard
- top-1, max-logit, and logsumexp deltas

Validation done locally:

- `python -m py_compile` for the new Python files and existing top-logprob helper: green
- `bash -n` for the new run/launch scripts: green
- synthetic capture comparator smoke: green

Next live order when Vast credit is restored:

1. Run the hardened top-logprob packet first if we still want the cheaper public-distribution readout.
2. Run `launch_26b_readout_capture_live.sh` if the top-logprob row confirms distribution collapse and we need
   to split hidden-state drift from lm_head/readout amplification.

No model-quality claim is changed by this mail. 26B-A4B full NVFP4 K+V remains research/red; fp8 KV remains
the ship path.
