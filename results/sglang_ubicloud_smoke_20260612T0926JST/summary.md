# SGLang Ubicloud GitHub Runner Smoke

Date: 2026-06-12 JST

Status: GREEN.

Run: https://github.com/jethac/sglang/actions/runs/27386264175
Commit: `1e4fe4793e4a09f46d26a3f0cdfb01612b4a7e4c`
Branch: `spark/hijinks-024-diffusiongemma-upstream-rebase`

This verifies that the Ubicloud GitHub runner integration is active for
`jethac/sglang` on both runner families needed by the campaign:

- `ubicloud-standard-8`: x64 job completed successfully.
- `ubicloud-standard-8-arm`: arm64 job completed successfully.

The workflow is `.github/workflows/hijinks-ubicloud-smoke.yml` in the SGLang
fork. It records basic runner inventory and can be re-run manually with
`workflow_dispatch`.

No local build or Spark CPU work was performed.