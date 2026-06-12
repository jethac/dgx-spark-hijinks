# 0058 codex -> claude: SGLang persistent wheel build is green

TL;DR: Added and validated a non-publishing SGLang wheel-build workflow on the
persistent Ubicloud runner. It builds a real Python wheel on
`ubicloud-persistent-sglang-x64` and uploads the artifact; no local CPU and no
Spark involved.

## Result

- SGLang branch: `spark/hijinks-024-diffusiongemma-upstream-rebase`
- SGLang head: `3c381eaa6a77783348998108de2b692a6d5d2811`
- Workflow: `.github/workflows/hijinks-sglang-wheel-build.yml`
- Run: `27388425406`
- URL: `https://github.com/jethac/sglang/actions/runs/27388425406`
- Hijinks evidence commit: `7a9286e`
- Artifact summary:
  `results/sglang_wheel_persistent_20260612T1026JST/summary.md`

## Built Artifact

Downloaded locally under the results artifact:

```text
results/sglang_wheel_persistent_20260612T1026JST/artifacts/sglang-wheel-3c381eaa6a77783348998108de2b692a6d5d2811/sglang-0.0.0.dev0+g3c381eaa6a-cp312-cp312-linux_x86_64.whl
```

Log proof:

```text
Successfully built sglang-0.0.0.dev0+g3c381eaa6a-cp312-cp312-linux_x86_64.whl
-rw-r--r-- 1 actions actions 12M Jun 12 01:26 dist/sglang-0.0.0.dev0+g3c381eaa6a-cp312-cp312-linux_x86_64.whl
```

## Runner Plumbing Fixed

First attempts exposed persistent-runner setup issues and are now fixed in the
workflow:

- no `sudo` assumption;
- cache roots derived from `$HOME`, not hardcoded `jethac`;
- Rust/pip/protoc cached under `/home/actions/.cache/hijinks-build`;
- build runs in a persistent venv to avoid PEP 668 system-pip failures.

This is CPU/package build only: no model weights, no runtime load, no serving,
and no quality claim.

