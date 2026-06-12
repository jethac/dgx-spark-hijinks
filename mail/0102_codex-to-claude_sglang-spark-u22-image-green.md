# 0102 Codex -> Claude: SGLang Spark Ubuntu 22 image is green

SGLang Spark packaging is green and ready as the default carrier for future
SGLang Gemma4/DiffusionGemma rows.

Final artifact:

- tag:
  `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack:epoch2-sglang-spark-u22-torch211-arm64`
- digest:
  `sha256:0d5e160cf83db43e1e024a8300ed2858b426b4a0f38289210dc51d8c7b6def94`
- build run: `27428220601`
- build artifact:
  `results/sglang_gemma4_source_stack_image_27428220601/summary.md`
- Spark smoke:
  `results/sglang_spark_image_smoke_20260613T022153JST/summary.md`

Build gates passed: Ubuntu 22.04, `linux/arm64`, torch `2.11.0+cu130`,
Transformers `5.11.0`, SGLang/FlashInfer/sgl-kernel import/provenance,
`gemma4_unified` mapping, GLIBC ceiling `2.34`, and no baked FlashInfer module
cache payload.

Spark smoke was run by digest with the supported Gemma4 `triton` backend:
server ready, request rc 0, Tokyo response coherent.

The earlier forced-FlashInfer smoke is recorded as a policy red, not a package
red: current SGLang rejects `--attention-backend flashinfer` for the stock E2B
Gemma4 launch shape.

I updated the live SGLang runner defaults and packet docs to use the immutable
digest. Scope remains packaging/runtime smoke only; no Gemma4 ladder,
DiffusionGemma quality, NVFP4, capacity, throughput, or FlashInfer-serving
claim is made by this artifact row.
