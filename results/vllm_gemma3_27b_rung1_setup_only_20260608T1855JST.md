# vLLM Gemma 3 27B Rung 1 Setup-Only Check, 2026-06-08 18:55 JST

Status update: this setup check is retained only as evidence that the repaired
`25ab073ef` source and `4dcd10e` wheel base avoid the prior metadata 404. Its original
editable install allowed dependency resolution and downgraded core packages, including
Torch to `2.11.0+cu130` and FlashInfer to `0.6.12`. That downgrade is rejected for the live
Gemma rung. The accepted packet now installs vLLM with
`python3 -m pip install --no-build-isolation --no-deps -e .`, preserves the clean image's
Torch `2.12.0.dev20260408+cu130` and FlashInfer `0.6.9rc1`, and copies the ABI-matched
FA2 extension from `/opt/jethac-vllm` into the source overlay.

Purpose: prove the repaired Gemma 3 Rung 1 vLLM source/precompiled-wheel pair no longer
fails during editable install before spending a full model-load cycle.

Remote context:

- run checkout: `/home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608`
- vLLM checkout: `25ab073ef87f4443616fbaf00a2f6f09a9087c1f`
- image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
- vLLM precompiled wheel base: `4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa`
- install knobs:
  - `VLLM_USE_PRECOMPILED=1`
  - `VLLM_MAIN_CUDA_VERSION=13.0`
  - `VLLM_PRECOMPILED_SKIP_FLASH_ATTN=1`
  - `VLLM_VERSION_OVERRIDE=0.1.dev1+g25ab073ef`
  - `TORCH_CUDA_ARCH_LIST=12.1a`

Remote log:

```text
/home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608/results/vllm_gemma3_27b_rung1_20260608T1855JST_setup_only.log
```

Result:

- `pip install --no-build-isolation -e .` completed, but dependency resolution downgraded
  package versions and is no longer accepted for live rows.
- No `wheels.vllm.ai` metadata 404 occurred.
- Editable wheel built as `vllm-0.1.dev1+g25ab073ef-0.editable-cp312-cp312-linux_aarch64.whl`.
- Import probe passed:

```json
{
  "ok": true,
  "vllm": "0.1.dev1+g25ab073ef",
  "vllm_file": "/vllm-src/vllm/__init__.py",
  "flashinfer": "0.6.12",
  "flashinfer_file": "/usr/local/lib/python3.12/dist-packages/flashinfer/__init__.py",
  "torch": "2.11.0+cu130",
  "torch_cuda": "13.0",
  "device": "NVIDIA GB10",
  "capability": [12, 1],
  "geometry_hook": true,
  "wheel_base": "4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa",
  "source_commit": "25ab073ef87f4443616fbaf00a2f6f09a9087c1f"
}
```

Interpretation:

The previous vLLM Gemma 3 Rung 1 blocker, missing `cu130` wheel metadata for the
`8916796` overlay path, is cleared by moving the geometry hook onto the `a919d635d` lane
and using the `4dcd10e` precompiled wheel base.

The dependency state in this setup-only probe is not accepted. The later live packet fixes
that with `--no-deps`; the fp8 comparator row is recorded separately in
`results/vllm_gemma3_27b_rung1_fp8_20260608T1924JST.md`.

Remaining proof:

This is setup-only. It does not prove Gemma 3 model load, runtime geometry lines, KV
capacity, output quality, or FlashInfer source-overlay/backend selection. The fp8 comparator
row has since run and is recorded in
`results/vllm_gemma3_27b_rung1_fp8_20260608T1924JST.md`; the remaining Rung 1 proof is the
matching NVFP4 row with capacity and quality comparison against that fp8 baseline.
