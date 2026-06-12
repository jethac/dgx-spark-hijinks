# SGLang Gemma 4 AR Ladder Preflight

Run: `sglang_gemma4_ar_ladder_20260612T183849JST`
Date: 2026-06-12 JST

Status: RED before serving. No model weights were loaded and no quality,
capacity, or runtime claim is made.

## Intended Row

- Model: `google/gemma-4-12B-it`
- Pair: BF16/auto-KV comparator, then full NVFP4 K+V
- Image:
  `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack:epoch2-gemma4-tf511-12fca91`
- Digest recorded by the prior build:
  `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:bf24438b302c96e457b8a59f8a8dbaf109fab08013554be81e6957d4fb0f1a70`
- Spark guardrails: Docker `--memory 100g`, one server at a time,
  `mem_fraction_static=0.72`

## Result

The packet failed at `docker run` before the server could start:

```text
docker: no matching manifest for linux/arm64/v8 in the manifest list entries
```

The copied artifact therefore contains `bf16_status.txt=server_not_ready` and
no chat/PPL rows. `image_manifest_inspect.json` proves the pushed GHCR image has
only a `linux/amd64` runtime manifest plus an unknown-platform attestation
manifest. `base_image_manifest_inspect.json` proves the NVIDIA base image
`nvcr.io/nvidia/sglang:26.05-py3` does include `linux/arm64`, so the defect is
in our GitHub image workflow publishing the persistent x64 runner's native
platform only.

## Next Action

Rebuild through GitHub/Ubicloud with a `linux/arm64` target platform. Do not
build on Spark or local workstation CPU. After an arm64 image manifest is green,
rerun the same 12B row before advancing to 26B-A4B or 31B.
