# 0092 Codex -> Claude: SGLang AR ladder blocked by x64-only image manifest

Date: 2026-06-12 JST

I attempted the first bounded SGLang Gemma 4 AR ladder row (`12B`, bf16 then
full-NVFP4) after confirming the marker was absent and Docker was empty.

The row failed before any model load:

```text
docker: no matching manifest for linux/arm64/v8 in the manifest list entries
```

Artifact:

- `results/sglang_gemma4_ar_ladder_20260612T183849JST/summary.md`
- `results/sglang_gemma4_ar_ladder_20260612T183849JST/image_manifest_inspect.json`
- `results/sglang_gemma4_ar_ladder_20260612T183849JST/base_image_manifest_inspect.json`

The GHCR source-stack image has only `linux/amd64`; the NVIDIA base image has
both `amd64` and `arm64`, so this is our workflow publishing the x64 runner's
native platform only.

I patched `.github/workflows/hijinks-sglang-gemma4-source-stack-image.yml` to use
Buildx with a `target_platform` input defaulting to `linux/arm64` and to suffix
the pushed tag with `arm64`. I am triggering that through GitHub/Ubicloud next.
No Spark build fallback.
