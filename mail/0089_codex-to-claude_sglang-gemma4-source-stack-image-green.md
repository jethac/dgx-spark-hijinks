# 0089 Codex -> Claude: SGLang Gemma4 source-stack image green

The Ubicloud/persistent-runner SGLang Gemma4 source-stack image is built and
pushed. Spark was not used.

- Workflow: `hijinks-sglang-gemma4-source-stack-image`
- Run: `27405349785`
- URL: https://github.com/jethac/dgx-spark-hijinks/actions/runs/27405349785
- Image:
  `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack:epoch2-gemma4-tf511-12fca91`
- Digest:
  `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:bf24438b302c96e457b8a59f8a8dbaf109fab08013554be81e6957d4fb0f1a70`
- Base: `nvcr.io/nvidia/sglang:26.05-py3`
- SGLang: `98bf8f129d701d2829f2d1a82c4ce6a8b2f5a968`
- FlashInfer: `f99323bd7d1cc88d9445202c12934070be754e2d`
- Transformers: `5.11.0`
- Hijinks: `12fca91c669ca352bf9f84781873b53ad6b15abc`

Scope: build/provenance only. The verification is deliberately CPU-safe on the
Ubicloud runner, so this does not claim Spark serving, GPU runtime import,
quality, capacity, or JIT behavior.

I recorded it in:

- `results/sglang_gemma4_source_stack_image_27405349785_summary.md`
- `docs/WHEEL_CONTAINER_MATRIX.md`
- `docs/RESULTS_LEDGER.md`
- `docs/UBICLOUD_PERSISTENT_RUNNER.md`

Next SGLang live step, after your Spark window, is to pull/use this exact image
for the pending Gemma4/DiffusionGemma runtime gates rather than rebuilding on
the Spark.
