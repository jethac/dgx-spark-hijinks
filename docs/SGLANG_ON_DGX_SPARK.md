# SGLang On DGX Spark

Status: draft, not blessed.

Target: DGX Spark / ThinkStation PGX / GB10 = compute capability 12.1 = `sm_121`.

Scope: one Spark-class unit only. No TP>1 or multi-Spark claims yet.

## Why Track SGLang

SGLang is a serious serving runtime and should not be hidden under the vLLM plan. The `hikarioyama/sglang-nvfp4-kv-sm120` repo is especially relevant because it demonstrates the design shape for NVFP4 KV:

- `fp4_e2m1` KV cache
- FlashInfer FA2 kernel patches
- native FP4 KV memory pool
- hybrid-SWA wiring
- per-layer global-scale auto-calibration before CUDA graph capture
- fp4-vs-fp8 comparison discipline

That is SM120 RTX Blackwell evidence. It is not Spark validation. Our target is `sm_121`.

## Baseline First

Before NVFP4:

- install or run SGLang on the single GB10 unit
- capture `spark_doctor`
- start an OpenAI-compatible server
- run `scripts/openai_chat_smoke.py`
- establish BF16 or fp8 KV quality and speed

Only then test `fp4_e2m1`.

## NVFP4 Rule

Keep fp8 KV as the default recommendation until SGLang NVFP4 KV passes on Spark.

For NVFP4 validation, record:

- SGLang version/image/commit
- FlashInfer version or patch source
- model id/revision
- attention backend
- `--kv-cache-dtype fp4_e2m1`
- page size
- CUDA graph mode
- fresh JIT cache path
- deterministic prompt output
- fp4-vs-fp8 quality comparison
- prefill/decode speed
- memory/KV capacity difference
- whether patches are SM120-derived or SM121-specific

## Fork Rule

If SGLang needs source changes, fork `sgl-project/sglang` to `jethac/sglang`, add it as `third_party/sglang`, and do the patch in an issue-named worktree.

If FlashInfer needs source changes for the SGLang path, fork `flashinfer-ai/flashinfer` to `jethac/flashinfer`, add it as `third_party/flashinfer`, and use a separate worktree.

Use the supplied SM120 repo as reference context, not as a Spark-blessed submodule.
