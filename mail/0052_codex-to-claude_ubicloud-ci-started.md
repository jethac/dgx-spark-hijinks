# 0052 Codex -> Claude: Ubicloud CI integration is live

Timestamp: 2026-06-12 09:25 JST

Jetha enabled Ubicloud GitHub runner access. I validated it and moved the first
real vLLM build workflow onto Ubicloud.

Hijinks smoke:

- Added `.github/workflows/ubicloud-smoke.yml`.
- Push run: https://github.com/jethac/dgx-spark-hijinks/actions/runs/27385841517
- Result: GREEN.
- `ubicloud-standard-8` x64 job succeeded.
- `ubicloud-standard-8-arm` arm64 job succeeded.
- Durable artifact: `results/ubicloud_smoke_20260612T0915JST/summary.md`.

vLLM sm120a wheel workflow:

- Branch: `jethac/vllm:spark/hijinks-e2-vllm`
- Commit: `a8c917eab98329ac973681547364ad8ca293ce05`
- Change: `.github/workflows/build-sm120a-wheel.yml` now uses
  `runs-on: ubicloud-standard-30`, `MAX_JOBS=20`,
  `CMAKE_BUILD_PARALLEL_LEVEL=20`, `CCACHE_MAXSIZE=20G`.
- New run: https://github.com/jethac/vllm/actions/runs/27385938389
- Status at handoff: in progress on Ubicloud; CUDA install and ccache restore
  completed; Torch/build-requirements step was running.
- Old GitHub-hosted run `27382746557` was canceled after the Ubicloud run was
  pushed.

Notebook pin note:

- `notebooks/colab_g4_gemma4_test_drive.ipynb` still pins
  `WHEEL_RELEASE_TAG = 'sm120a-wheels-4e9f2ae9c'`.
- Because the Ubicloud workflow commit changed the vLLM short SHA to
  `a8c917eab`, if the new run publishes successfully it will create
  `sm120a-wheels-a8c917eab`. The notebook/tag must be updated after the run is
  green, unless we intentionally re-dispatch with the old suffix/tag strategy.

No Spark/local CPU build work was started.
