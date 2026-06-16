# Vast 26B-A4B top-logprob retry blocked

Status: blocked before allocation by Vast account credit state. No model row was run.

## Context

The patched top-logprob packet from `f8e48fc` is ready and should be used for the next live attempt:

- `docs/vast_anchor/run_26b_toplogprob_attribution.sh`
- `docs/vast_anchor/vllm_toplogprob_attribution.py`
- `docs/vast_anchor/launch_26b_toplogprob_live.sh`

The prior live attempt proved that `--skip-mm-profiling` is necessary for this text-only Gemma 4 26B-A4B
discriminator.

## This Retry

Vast state before retry:

- Existing stopped/exited instance: `41119757`
- Image: `nvidia/cuda:13.0.1-devel-ubuntu24.04`
- Label: none
- This lane did not create it, so it was left untouched.

Replacement offer selected:

- Offer id: `37036746`
- GPU: RTX PRO 6000 WS, `sm_120`, 96 GB
- Host RAM: 257 GB
- Effective vCPUs: 128
- Geolocation: Utah, US
- Requested image: `nvidia/cuda:13.0.1-devel-ubuntu22.04`

Vast rejected the allocation:

```text
Your account lacks credit; see the billing page.
```

No Codex-created instance was allocated or left running in this retry.

## Next Action

Retry when the Vast account has credit or when the existing stopped instance is explicitly cleared for reuse or
destruction. Do not claim any model-quality result from this artifact.
