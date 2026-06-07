# Upstream Latest Release Audit

Date: 2026-06-07

Question: do the latest upstream vLLM and SGLang releases also exclude `b12x` from SM121 NVFP4 `mm_fp4` auto-dispatch?

Short answer: yes, by dependency inheritance, not because vLLM or SGLang each implement this particular heuristic themselves.

The `mm_fp4` auto-dispatch heuristic lives in FlashInfer. The latest vLLM and SGLang releases currently pin FlashInfer releases that still gate the `b12x` NVFP4 GEMM path on exact SM120 (`major == 12 and minor == 0`). Therefore those releases inherit the same SM121 exclusion unless they patch FlashInfer downstream or ship a different FlashInfer build than their release tag declares.

## Latest Release Tags Checked

Release metadata was checked with `gh release view` and the release tags were cloned under `B:/workshop/upstream-audit`.

| project | latest release tag | release commit checked | published | release URL |
|---|---|---:|---|---|
| vLLM | `v0.22.1` | `0decac0d96c4` | 2026-06-05 | https://github.com/vllm-project/vllm/releases/tag/v0.22.1 |
| SGLang | `v0.5.12.post1` | `5a15cde858ea` | 2026-05-26 | https://github.com/sgl-project/sglang/releases/tag/v0.5.12.post1 |
| FlashInfer stable | `v0.6.12` | `d768c14e7cf5` | 2026-05-29 | https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.12 |
| FlashInfer nightly | `nightly-v0.6.13-20260607` | `a28703432faa` | 2026-06-07 | https://github.com/flashinfer-ai/flashinfer/releases/tag/nightly-v0.6.13-20260607 |

Pinned FlashInfer tags checked because they are used by vLLM/SGLang releases:

| FlashInfer tag | commit checked | why checked |
|---|---:|---|
| `v0.6.11.post1` | `710e74e9dd23` | pinned by SGLang `v0.5.12.post1` |
| `v0.6.11.post2` | `064d9aa268fe` | pinned by vLLM `v0.22.1` |

## vLLM Latest Release

vLLM `v0.22.1` pins FlashInfer `0.6.11.post2`:

- `requirements/cuda.txt:12`: `flashinfer-python==0.6.11.post2`
- `requirements/cuda.txt:13`: `flashinfer-cubin==0.6.11.post2`
- `docker/Dockerfile:760`: `ARG FLASHINFER_VERSION=0.6.11.post2`
- `docker/Dockerfile:762`: installs `flashinfer-jit-cache==${FLASHINFER_VERSION}`
- `requirements/cuda.txt:24`: `nvidia-cutlass-dsl[cu13]==4.5.2`

FlashInfer `v0.6.11.post2` has the SM120-only `mm_fp4` auto-dispatch gate:

- `flashinfer/gemm/gemm_base.py:5718`: `_heuristic_func_mm_fp4`
- `flashinfer/gemm/gemm_base.py:5753`: `is_sm120 = major == 12 and minor == 0`
- `flashinfer/gemm/gemm_base.py:5756`: `if is_sm120 and use_nvfp4 and cuda_major >= 13:`

FlashInfer `v0.6.11.post2` release/JIT-cache build lists also omit `12.1a`:

- `.github/workflows/release.yml:185`: CUDA 12.9+/13 build target list includes `12.0f`, not `12.1a`
- `.github/workflows/nightly-release.yml:156`: same omission
- `docs/installation.rst:110`: example `FLASHINFER_CUDA_ARCH_LIST` includes `12.0f`, not `12.1a`

Conclusion for vLLM `v0.22.1`: the release pins a FlashInfer version whose `mm_fp4` auto-dispatch excludes SM121 from the `b12x` NVFP4 path.

## SGLang Latest Release

SGLang `v0.5.12.post1` pins FlashInfer `0.6.11.post1`:

- `python/pyproject.toml:30`: `flashinfer_python==0.6.11.post1`
- `python/pyproject.toml:31`: `flashinfer_cubin==0.6.11.post1`
- `docker/Dockerfile:22`: `ARG FLASHINFER_VERSION=0.6.11.post1`
- `docker/Dockerfile:345`: installs `flashinfer-jit-cache==${FLASHINFER_VERSION}`
- `python/pyproject.toml:40`: `nvidia-cutlass-dsl[cu13]==4.5.1`

FlashInfer `v0.6.11.post1` has the same SM120-only `mm_fp4` auto-dispatch gate:

- `flashinfer/gemm/gemm_base.py:5627`: `_heuristic_func_mm_fp4`
- `flashinfer/gemm/gemm_base.py:5662`: `is_sm120 = major == 12 and minor == 0`
- `flashinfer/gemm/gemm_base.py:5665`: `if is_sm120 and use_nvfp4 and cuda_major >= 13:`

FlashInfer `v0.6.11.post1` release/JIT-cache build lists also omit `12.1a`:

- `.github/workflows/release.yml:185`: CUDA 12.9+/13 build target list includes `12.0f`, not `12.1a`
- `.github/workflows/nightly-release.yml:156`: same omission
- `docs/installation.rst:110`: example `FLASHINFER_CUDA_ARCH_LIST` includes `12.0f`, not `12.1a`

Conclusion for SGLang `v0.5.12.post1`: the release pins a FlashInfer version whose `mm_fp4` auto-dispatch excludes SM121 from the `b12x` NVFP4 path.

## FlashInfer Latest Stable And Nightly

FlashInfer stable `v0.6.12` still has the SM120-only gate:

- `flashinfer/gemm/gemm_base.py:5694`: `_heuristic_func_mm_fp4`
- `flashinfer/gemm/gemm_base.py:5729`: `is_sm120 = major == 12 and minor == 0`
- `flashinfer/gemm/gemm_base.py:5732`: `if is_sm120 and use_nvfp4 and cuda_major >= 13:`

FlashInfer nightly `nightly-v0.6.13-20260607` still has the SM120-only gate:

- `flashinfer/gemm/gemm_base.py:5980`: `_heuristic_func_mm_fp4`
- `flashinfer/gemm/gemm_base.py:6015`: `is_sm120 = major == 12 and minor == 0`
- `flashinfer/gemm/gemm_base.py:6018`: `if is_sm120 and use_nvfp4 and cuda_major >= 13:`

Both tags also omit `12.1a` from the release/JIT-cache build target list and installation example:

- stable `v0.6.12`: `.github/workflows/release.yml:185`, `.github/workflows/nightly-release.yml:156`, `docs/installation.rst:110`
- nightly `nightly-v0.6.13-20260607`: `.github/workflows/release.yml:185`, `.github/workflows/nightly-release.yml:156`, `docs/installation.rst:110`

FlashInfer `v0.6.12` release notes mention SM120 W4A16 `b12x` kernels and SM120 FMHA AOT wheel work, but the checked source still excludes SM121 from this specific `mm_fp4` auto-dispatch fast path.

## Local Container Evidence

Separate from upstream release tags, the containers tested on the Spark also exclude `b12x` in real SM121 runtime dispatch:

- `nvcr.io/nvidia/sglang:26.05-py3`, FlashInfer `0.6.10+cf494fca.nv26.5.cu132.50619265`: real GB10 NVFP4 `mm_fp4` heuristic returned `["cudnn", "cutlass"]`.
- `vllm/vllm-openai:cu130-nightly-aarch64`, FlashInfer `0.6.8.post1`: real GB10 NVFP4 `mm_fp4` heuristic returned `["cudnn", "cutlass"]`.

Recorded artifact:

- `results/flashinfer_sm121_source_jit_20260607T1250Z.json`

## Patch Comparison

The campaign fork changes this exact behavior:

- fork: `jethac/flashinfer`
- branch: `spark/hijinks-004-sm121-flashinfer`
- commit: `a42c8f07`

Patched evidence:

- `flashinfer/gemm/gemm_base.py:6017`: `is_sm12x = major == 12`
- `flashinfer/gemm/gemm_base.py:6019`: SM12x + CUDA 13 + NVFP4 prefers `b12x`, then `cutlass`, then `cudnn`
- `tests/gemm/test_mm_fp4.py:146`: `test_mm_fp4_auto_prefers_b12x_for_sm121_nvfp4`
- `.github/workflows/release.yml:185` and `.github/workflows/nightly-release.yml:156`: aarch64 CUDA 12.9+/13 build lists include `12.1a`
- `docs/installation.rst:117`: Spark example includes `12.1a`

Runtime proof on the Spark:

- patched source returned `["b12x", "cutlass", "cudnn"]` on real GB10 / SM121.
- a tiny forced-`b12x` NVFP4 GEMM produced finite BF16 output with cosine similarity `0.9882067441940308` against BF16 `torch.mm`.
- source/JIT build path observed: `/root/.cache/flashinfer/0.6.13/121a/cached_ops/fp4_quantization_120f`

## Claim Boundaries

What is proven:

- latest vLLM release `v0.22.1` inherits the SM121 `b12x` auto-dispatch exclusion through its FlashInfer `0.6.11.post2` pin.
- latest SGLang release `v0.5.12.post1` inherits the SM121 `b12x` auto-dispatch exclusion through its FlashInfer `0.6.11.post1` pin.
- latest FlashInfer stable `v0.6.12` and latest nightly `nightly-v0.6.13-20260607` still contain the SM120-only `mm_fp4` auto-dispatch gate.
- tested vLLM and SGLang containers on the Spark show the same exclusion at runtime.

What is not proven:

- that every downstream binary published by vLLM or SGLang is built exactly from those pinned files.
- that every NVFP4 path in vLLM or SGLang is affected; this audit covers FlashInfer `mm_fp4` auto-dispatch and related JIT-cache targets.
- that the patched source improves end-to-end serving throughput; that still requires a clean before/after container or wheel set.

## Commands Used

```bash
gh release view --repo vllm-project/vllm --json tagName,name,publishedAt,isPrerelease,url,targetCommitish
gh release view --repo sgl-project/sglang --json tagName,name,publishedAt,isPrerelease,url,targetCommitish
gh release view --repo flashinfer-ai/flashinfer --json tagName,name,publishedAt,isPrerelease,url,targetCommitish
gh release list --repo flashinfer-ai/flashinfer --limit 10
```

Local clones:

```bash
git clone --depth 1 --branch v0.22.1 https://github.com/vllm-project/vllm.git B:/workshop/upstream-audit/vllm-v0.22.1
git clone --depth 1 --branch v0.5.12.post1 https://github.com/sgl-project/sglang.git B:/workshop/upstream-audit/sglang-v0.5.12.post1
git clone --depth 1 --branch v0.6.12 https://github.com/flashinfer-ai/flashinfer.git B:/workshop/upstream-audit/flashinfer-v0.6.12
git clone --depth 1 --branch nightly-v0.6.13-20260607 https://github.com/flashinfer-ai/flashinfer.git B:/workshop/upstream-audit/flashinfer-nightly-20260607
git clone --depth 1 --branch v0.6.11.post1 https://github.com/flashinfer-ai/flashinfer.git B:/workshop/upstream-audit/flashinfer-v0.6.11.post1
git clone --depth 1 --branch v0.6.11.post2 https://github.com/flashinfer-ai/flashinfer.git B:/workshop/upstream-audit/flashinfer-v0.6.11.post2
```
