# SGLang Ubicloud 30-vCPU GitHub Runner Smoke

Date: 2026-06-12 JST

Status: GREEN.

Run: https://github.com/jethac/sglang/actions/runs/27386328408
Commit: `8544023d6c5e5ae907bf110cb496c163460d1bbd`
Branch: `spark/hijinks-024-diffusiongemma-upstream-rebase`

This verifies that the Ubicloud GitHub runner integration is active for
`jethac/sglang` on the same large runner classes intended for build work:

- `ubicloud-standard-30`: x64 job completed successfully.
- `ubicloud-standard-30-arm`: arm64 job completed successfully.

No local build or Spark CPU work was performed.