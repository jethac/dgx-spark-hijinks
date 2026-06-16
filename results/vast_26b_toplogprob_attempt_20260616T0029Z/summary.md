# Vast 26B-A4B top-logprob attribution attempt

Status: no quality row produced; live attempt stopped before `bf16` completed.

## Scope

- Model: `google/gemma-4-26B-A4B-it`
- Runtime: vLLM `0.1.dev1+g1c9686c61.sm120a`
- FlashInfer source overlay: `1eaa1aefc8d0a17bae5eb37eb9effff7a504fa0a`
- Torch/CUDA: `2.12.0+cu130` / CUDA `13.0`
- Device: RTX PRO 6000 Blackwell Workstation Edition, `sm_120`
- Vast instances:
  - `41118233`: destroyed after SSH proxy failed (`remote port forwarding failed for listen port 38232`).
  - `41118724`: Ubuntu 22.04 container, `sm_120`, 358 GB memory limit; destroyed after the container stopped and could not be restarted (`Required resources are currently unavailable`).

## What happened

1. Setup completed on `41118724`:
   - Torch import: `2.12.0+cu130`, CUDA `13.0`
   - FlashInfer import: `/root/flashinfer/flashinfer/__init__.py`
   - vLLM import: `0.1.dev1+g1c9686c61.sm120a`
2. First launch without `--skip-mm-profiling` reached:
   - `Encoder cache will be initialized with a budget of 4096 tokens, and profiled with 1 video items...`
   - No progress after roughly nine minutes past that line.
3. Patched the packet so `SKIP_MM_PROFILING=1` adds `--skip-mm-profiling`.
4. Relaunch reached the expected proof line:
   - non-default args included `skip_mm_profiling: True`
   - `Skipping memory profiling for multimodal encoder and encoder cache.`
5. The SSH session then closed from the remote side before the `bf16` row completed. Vast control plane reported the instance as `stopped/exited`; the start request remained queued because resources were unavailable. No row JSON or tar artifact was recoverable before destroy.

## Patch Landed

`docs/vast_anchor/run_26b_toplogprob_attribution.sh` now defaults `SKIP_MM_PROFILING=1` and passes
`--skip-mm-profiling` through to `vllm_toplogprob_attribution.py`.

`docs/vast_anchor/launch_26b_toplogprob_live.sh` was added to read `HF_TOKEN` from stdin, keeping secrets
out of command lines and files.

## Next Attempt

Retry on a more reliable Vast host or another sm120 box with the patched packet. The first row should show
`skip_mm_profiling: True` and `Skipping memory profiling for multimodal encoder and encoder cache` before
scoring. Do not rerun the old packet without this flag.
