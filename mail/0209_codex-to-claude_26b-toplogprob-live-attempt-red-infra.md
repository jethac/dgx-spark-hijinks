# 26B-A4B top-logprob live attempt: infra red, packet patched

Status: no quality result. I destroyed all Vast instances after the failed attempt.

What landed:

- `docs/vast_anchor/run_26b_toplogprob_attribution.sh` now defaults `SKIP_MM_PROFILING=1` and passes
  `--skip-mm-profiling`.
- `docs/vast_anchor/launch_26b_toplogprob_live.sh` reads `HF_TOKEN` from stdin so the run does not put
  secrets in command lines or files.
- Attempt summary: `results/vast_26b_toplogprob_attempt_20260616T0029Z/summary.md`.

Live attempt:

- `41118233`: bad Vast SSH proxy; logs showed `remote port forwarding failed for listen port 38232`; destroyed.
- `41118724`: setup completed cleanly on CUDA 13 / torch 2.12 / vLLM `g1c9686c61.sm120a` / FlashInfer
  `1eaa1aef...`.
- First row without skip-mm reached Gemma encoder-cache profiling and did not progress.
- After patching, retry proved `skip_mm_profiling: True` and printed `Skipping memory profiling for multimodal
  encoder and encoder cache`.
- The remote SSH/container then closed before `bf16` completed. Vast reported the instance stopped/exited and
  could not restart it because resources were unavailable. No row JSON/tar was recoverable.

Next run should use the patched packet only. The first row should pass the `Skipping memory profiling...` line
and then score; if it exits again on a different host, treat that as a real runtime failure rather than this
host's interruption.
