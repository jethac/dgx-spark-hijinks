# 26B-A4B top-logprob retry blocked before allocation

Status: no new model run. The packet is ready, but Vast refused a fresh instance before launch.

Current repo head before this note: `f8e48fc`.

What I tried:

- Checked Vast state: one stopped/exited instance exists, `41119757`, image
  `nvidia/cuda:13.0.1-devel-ubuntu24.04`, status `exited`, no label. I did not create it in the
  top-logprob attempt and did not destroy it.
- Filtered for a replacement sm120 96 GB PRO 6000 host, excluding the failed hosts from the prior attempt.
- Selected offer `37036746` (Utah, RTX PRO 6000 WS, 257 GB host RAM, 128 effective vCPUs) with the
  Ubuntu 22.04 CUDA 13 container image.
- Vast rejected `create instance` with: `Your account lacks credit; see the billing page.`

No Codex-created Vast instance was allocated this turn.

Next action once credit/account state is fixed:

```bash
scp docs/vast_anchor/{e3_setup_wget.sh,corpus_fetch.py,vllm_toplogprob_attribution.py,run_26b_toplogprob_attribution.sh,launch_26b_toplogprob_live.sh} root@HOST:/root/
```

Then run the patched packet only. The first row must show:

- non-default args include `skip_mm_profiling: True`
- `Skipping memory profiling for multimodal encoder and encoder cache.`

If a different host exits after that point too, treat it as a real runtime failure instead of the previous
host-specific interruption.
