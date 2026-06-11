# Colab G4 lane (reopened 2026-06-12): CI wheels -> Release -> notebook

Campaign: dgx-spark-hijinks epoch 2. Owner: Claude (bus name `colab-g4`).
Target runtime: **GCP/Colab G4 = NVIDIA RTX PRO 6000 Blackwell, CC 12.0 /
`sm_120`, 96 GB discrete** - the sibling arch of the Spark's `sm_121` and the
first box where the whole Gemma 4 ladder (E4B / 12B / 31B) fits at once.

## Architecture

```
push to jethac/vllm @ spark/hijinks-e2-vllm
        |
        v
GitHub Actions: .github/workflows/build-sm120a-wheel.yml   (NO GPU needed)
  apt cuda-toolkit-13-0 (nvcc 13.0) + torch 2.12.0+cu130
  use_existing_torch.py + pip wheel --no-build-isolation, TORCH_CUDA_ARCH_LIST=12.0a
  ccache restore/save brackets the build (6h-limit split design: re-run resumes)
  import-less sanity: wheel METADATA + cuobjdump sm_120a cubin grep
        |
        v
GitHub Release on jethac/vllm, tag sm120a-wheels-<shortsha>  (+ Actions artifact, 30d)
        |
        v
notebooks/colab_g4_gemma4_test_drive.ipynb  (animal-versioned; first cut KANGAROO)
  pip install the release wheel into a fresh venv -> serve Gemma 4 within minutes
```

## Wheel tag / URL convention

- Tag: `sm120a-wheels-<shortsha>` where `<shortsha>` = `git rev-parse --short=9`
  of the `spark/hijinks-e2-vllm` commit that built it (manual dispatch can add a
  suffix, e.g. `-retry1`).
- Release page: `https://github.com/jethac/vllm/releases/tag/sm120a-wheels-<shortsha>`
- Asset: one wheel, deterministic name
  `vllm-0.1.dev1+g<shortsha>.sm120a-cp38-abi3-linux_x86_64.whl`; the notebook does
  not hardcode it - it resolves the `.whl` asset via the GitHub API
  (`/repos/jethac/vllm/releases/tags/<tag>`), so only the tag is templated
  (`WHEEL_RELEASE_TAG` at the top of the env cell).
- Toolchain stamp (in the release notes, do not drift): nvcc 13.0
  (apt cuda-toolkit-13-0), torch 2.12.0+cu130, python 3.12,
  `TORCH_CUDA_ARCH_LIST=12.0a`. The notebook installs the same torch pin before
  the wheel so the ABI always matches.
- First tag: `sm120a-wheels-4e9f2ae9c` (from the workflow-introduction commit
  itself; runs 27382718191 push / 27382746557 dispatch).

## What is deliberately NOT done in Colab anymore

- **vLLM source builds.** The JACKAL-era F2 cell burned 1-2 h of paid G4 GPU
  time per fresh runtime compiling something that needs zero GPUs. Compile work
  moved to GitHub Actions wholesale. The notebook keeps ONE fallback source-build
  cell, off by default (`RUN_SOURCE_BUILD = False`), strictly for the window
  where the branch moved and CI hasn't published the new tag yet.
- **FlashInfer wheels - on purpose, the other direction.** FlashInfer stays a
  source tree (`jethac/flashinfer @ spark/hijinks-022-fa2-d512`) on PYTHONPATH
  with runtime JIT, because (a) this exact mode is known-good on Colab with the
  apt cuda-toolkit since notebook DINGO and on the P520, (b) JIT compiles only
  the kernels a session actually uses, keyed to the live GPU, while an AOT wheel
  would add CI hours for no install-time win, and (c) packaged
  flashinfer-python/jit-cache/cubin artifacts have been convicted as stale-kernel
  hazards before (WHEEL_CONTAINER_MATRIX r7 post-mortem); the notebook
  uninstalls any pip-pulled copies.
- **Per-package minimal CUDA installs.** Wholesale `cuda-toolkit-13-0` only
  (prefer-deterministic-over-minimal: the minimal set cost five user
  round-trips).

## G4 runtime selection notes

- Pick the **G4** runtime type (RTX PRO 6000 Blackwell). The notebook's guard
  cell parses `nvidia-smi --query-gpu=compute_cap` and hard-fails on anything
  that is not CC 12.x - A100/L4/T4 results would be meaningless (sm_120a cubins).
- CC 12.0 (G4) and 12.1 (Spark GB10) are validated separately - non-portable
  `a`-suffix cubins. This lane produces sm_120 evidence; the Spark stays the
  sm_121 capacity hero.
- 96 GB **discrete** VRAM: an OOM just kills the process; the Spark's
  unified-memory deadlock guardrails (INCIDENT_20260609) do not apply here.
  Menu adapts by VRAM: >=80 GiB all three Gemma 4 models, >=40 GiB E4B+12B,
  else E4B only (16 GB consumer Blackwell).

## Deliverables index

- CI: `jethac/vllm` `.github/workflows/build-sm120a-wheel.yml` (branch
  `spark/hijinks-ci-sm120a-wheel`, merged fast-forward into
  `spark/hijinks-e2-vllm` @ `4e9f2ae9c`).
- Notebook: `notebooks/colab_g4_gemma4_test_drive.ipynb` (KANGAROO; mirrored to
  the slash-free `colab` branch like its sibling probe-matrix notebook).
- Bisect rider: the notebook's Gemma 3 1B FLASH_ATTN-vs-FlashInfer C1 ctx-8191
  pair is the second sm_120 datapoint for
  `docs/BUG_FLASHINFER_GEMMA3_1B_SERVING_NUMERICS.md`.
- Mail: `mail/0047_claude-to-codex_colab-g4-lane-ci-wheels.md` (SGLang lane can
  adopt the same CI pattern).
