# 26B-A4B top-logprob packet hardened for next Vast attempt

Status: offline patch only; no new model row.

I hardened the top-logprob packet after the prior Vast host killed the SSH session/container before the
`bf16` row completed.

Changes:

- `docs/vast_anchor/launch_26b_toplogprob_live.sh`
  - Defaults `DETACH=1`.
  - Reads `HF_TOKEN` from stdin, exports it in-memory, then launches the packet under `setsid`.
  - Prints `LAUNCHED pid=... out=... log=...` and exits, so a local SSH disconnect should not kill the row.
  - `DETACH=0` still gives the old foreground behavior when debugging.
- `docs/vast_anchor/run_26b_toplogprob_attribution.sh`
  - Adds `row_status.tsv`.
  - Stops before NVFP4 rows if `bf16` fails.
  - Tars a partial artifact on any exit via `FINAL_STATUS.txt`, so a timeout or row failure should still be
    recoverable as `${OUT}.tgz`.

Validation:

```bash
bash -n docs/vast_anchor/run_26b_toplogprob_attribution.sh
bash -n docs/vast_anchor/launch_26b_toplogprob_live.sh
```

Next live run pattern:

```bash
ssh -tt root@HOST 'cd /root && OUT=/root/dx26_toplogprob_attr_YYYYMMDDTHHMMZ ./launch_26b_toplogprob_live.sh'
# paste HF token once; command should return LAUNCHED immediately
ssh root@HOST 'tail -f /root/dx26_toplogprob_attr_YYYYMMDDTHHMMZ.log'
scp root@HOST:/root/dx26_toplogprob_attr_YYYYMMDDTHHMMZ.tgz ./results/
```

The first row must still show:

- `skip_mm_profiling: True`
- `Skipping memory profiling for multimodal encoder and encoder cache.`

Vast allocation remains blocked by account credit as of mail `0210`; this patch just makes the next attempt
more robust once a host is available.
