# SGLang Qwen FP4-KV Retest After FlashInfer Prefill Fix, 2026-06-09 18:18 JST

Status: red/inconclusive before serving. This run did not test FP4-KV runtime quality.

Purpose: after the vLLM Gemma 3 first-token gate turned green with
`jethac/flashinfer@c3dae30f`, immediately test the cross-lane hypothesis that the same
FlashInfer paged-prefill argument/plumbing fix might close SGLang's radix/prefix-cache row.

Command shape:

- runner: `scripts/run_sglang_fp4_dense_cache_trace.sh`
- case: `default`
- model: `Qwen/Qwen2.5-1.5B-Instruct`
- KV: `--kv-cache-dtype fp4_e2m1`
- source stack: editable FlashInfer + editable SGLang + source-built `sglang-kernel`
- run id: `sglang_qwen_fp4kv_after_fi0919_default2_20260609T1818JST`

Artifacts:

- `results/sglang_qwen_fp4kv_after_fi0919_default2_20260609T1818JST_summary.json`
- `results/sglang_qwen_fp4kv_after_fi0919_default2_20260609T1818JST_dense_cache_trace_summary_audit.json`
- `results/sglang_qwen_fp4kv_after_fi0919_default2_20260609T1818JST_default_editable_install.log`

Outcome:

- The first attempt used the default readiness timeout and expired while `sglang-kernel` was
  still compiling.
- The second attempt extended readiness to `1800s`, but after about 26 minutes it was still
  building `sglang-kernel` (`82/127` targets) and had not launched SGLang. The run was stopped
  manually as a build-duration stop point.
- No request JSON or dense-cache comparison artifact was produced.
- The summary audit is correctly red because no serving row exists.

Notable build warnings:

- Multiple kernels compiled for `compute_120a` and `compute_121a`.
- `ptxas` repeatedly warned that `.multicast::cluster` on `cp.async.bulk{.tensor}` should use
  datacenter-family targets such as `sm_90a/sm_100a/...` rather than `sm_120a`/`sm_121a`, with
  expected performance loss on future architectures.
- Several `setmaxnreg` compatibility warnings were emitted.

Interpretation:

- This does not falsify the FlashInfer cross-lane hypothesis. It only shows that the current
  SGLang validation path is dominated by rebuilding too much `sglang-kernel` inside the live
  serving runner.
- The next SGLang step should be packaging/build-system work: produce a reusable source-stack
  image or narrower `sglang-kernel` build target first, then rerun the default radix row from
  that prepared image.
