# Ubicloud GitHub Runner Smoke

Date: 2026-06-12 JST

Status: GREEN.

Run: https://github.com/jethac/dgx-spark-hijinks/actions/runs/27385841517
Commit: `96b709c371c4f7ad189ae45294c56591f9cf8820`

This verifies that the Ubicloud GitHub runner integration is active for
`jethac/dgx-spark-hijinks` on both runner families we need for cloud builds:

- `ubicloud-standard-8`: x64 job completed successfully.
- `ubicloud-standard-8-arm`: arm64 job completed successfully.

The workflow is `.github/workflows/ubicloud-smoke.yml`. It records basic runner
inventory (`uname`, `lscpu`, `nproc`, `free`, `df`) and can be re-run manually
with `workflow_dispatch`.

Next use: move CPU-heavy build workflows to Ubicloud labels so local CPU and the
Spark are reserved for GPU validation only.