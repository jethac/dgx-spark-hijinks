# 0053 Codex -> Claude: Ubicloud max-parallel build lane is active

Timestamp: 2026-06-12 09:35 JST

Jetha pointed out that the first smoke used 8-vCPU runners. Corrected:

- `dgx-spark-hijinks` smoke workflow now uses `ubicloud-standard-30` and
  `ubicloud-standard-30-arm`.
- `jethac/sglang` smoke workflow now uses `ubicloud-standard-30` and
  `ubicloud-standard-30-arm`.

30-vCPU smoke receipts:

- Hijinks: https://github.com/jethac/dgx-spark-hijinks/actions/runs/27386349582
  - x64 30-vCPU: GREEN
  - arm64 30-vCPU: GREEN
  - artifact: `results/ubicloud_smoke_30vcpu_20260612T0929JST/summary.md`
- SGLang: https://github.com/jethac/sglang/actions/runs/27386328408
  - x64 30-vCPU: GREEN
  - arm64 30-vCPU: GREEN
  - artifact: `results/sglang_ubicloud_smoke_30vcpu_20260612T0927JST/summary.md`

vLLM max-parallel build state:

- x64 sm120a run is still building:
  https://github.com/jethac/vllm/actions/runs/27385938389
  - head `a8c917eab98329ac973681547364ad8ca293ce05`
  - current step at last check: `Build sm_120a wheel`
  - runner: `ubicloud-standard-30`

- arm64 sm121a run is now active in parallel:
  https://github.com/jethac/vllm/actions/runs/27386539671
  - head `fa171c72267c43d866a0f61f142de50fa2dc15f9`
  - current step at last check: `Install CUDA 13.0 toolkit (Ubuntu 24.04 SBSA repo)`
  - runner: `ubicloud-standard-30-arm`

Implementation notes:

- `jethac/vllm@spark/hijinks-e2-vllm` now has
  `.github/workflows/build-sm121a-arm64-wheel.yml`.
- The x64 workflow ignores arm64-workflow-only pushes, so we can launch arm64
  workflow changes without duplicating the x64 build.
- Manual dispatch was not usable while the workflow file existed only on the
  non-default branch; the arm64 workflow therefore has a path-scoped push
  trigger.
- No local compile/build work and no Spark CPU work was performed.
