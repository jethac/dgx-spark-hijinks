# Ubicloud 30-vCPU GitHub Runner Smoke

Date: 2026-06-12 JST

Status: GREEN.

Run: https://github.com/jethac/dgx-spark-hijinks/actions/runs/27386349582
Commit: `f7f08e67f207a9d88ca4847afc35d432e4fe3273`

This verifies that the Ubicloud GitHub runner integration is active for
`jethac/dgx-spark-hijinks` on the same large runner classes intended for build
work:

- `ubicloud-standard-30`: x64 job completed successfully.
- `ubicloud-standard-30-arm`: arm64 job completed successfully.

No local build or Spark CPU work was performed.