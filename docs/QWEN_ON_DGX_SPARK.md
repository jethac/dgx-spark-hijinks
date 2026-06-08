# Qwen On DGX Spark

Status: active speed and capacity benchmark lane, issue #20.

Qwen is a first-class Spark target alongside Gemma, not a secondary check. Gemma exercises the hardest model-family path, but Qwen is the cleaner way to measure SM121a throughput, NVFP4 weights, speculative decode, and fp8-vs-NVFP4 KV capacity without Gemma 4's heterogeneous attention dimensions. Every claimed runtime path should eventually have at least one Qwen speed row and one Gemma row.

## Current Evidence

| runtime | row | status |
|---|---|---|
| vLLM | AEON-7 Qwen3.6 35B-A3B NVFP4 + DFlash / NVFP4-KV | local `v2` row passes OpenAI smoke and compact serving when `chat_template_kwargs={"enable_thinking": false}` is set; decode is about `50-56 tok/s` on the AEON image. A derived `jethac/vllm@6804e1b` row also passes at `47.22`, `58.88`, and `61.62 tok/s`, but depends on AEON's FA2 binary. The clean `jethac/vllm@a919d635d` + `jethac/flash-attention@7d53245` image now serves at `61.07`, `56.97`, and `60.10 tok/s` with separate `sm_121a` FA2 binary proof. A separate no-DFlash Qwen NVFP4-KV row records `1.751x` fp8 KV pool/concurrency through FlashInfer FA2 with decode parity; this is a capacity proof, not a speedup claim |
| SGLang | `Qwen/Qwen2.5-1.5B-Instruct` BF16/fp8/fp4-KV | local GB10 BF16/auto and fp8 rows pass at about 58-59 tok/s decode; the matched `d7d931f` fp8-vs-FP4 row proves `1.7769x` fp8 KV capacity and traces decode plus `extend_merge_paged`, with raw/chat smoke passing. The OpenAI logprob probe shows severe FP4 quality drift; the native `/generate` probe narrows one prompt to token-index-4 rank reversal; the prompt reconciliation row proves OpenAI prompt IDs match local Qwen chat-template IDs, so prompt serialization is not the cause. SGLang FP4 KV remains unblessed |
| llama.cpp | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` Q4_K_M | local GB10 row passes OpenAI smoke and compact serving at about 167-175 tok/s decode; lm-eval logprobs schema still blocked |

## vLLM Target

First reproduce the AEON Qwen path before changing source:

- image: `ghcr.io/aeon-7/vllm-spark-omni-q36:v2` by default; `v1.2` remains a historical compatibility target
- model: `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4`
- drafter: `z-lab/Qwen3.6-35B-A3B-DFlash`
- serving mode: compressed-tensors NVFP4 weights, DFlash speculative decode, `--attention-backend flash_attn`
- expected evidence: selected linear/MoE backends, CUDA graph mode, DFlash acceptance, TTFT, per-request decode, aggregate throughput, and zero-error soak result

Then test our forked stack:

- `jethac/vllm` with the SM12x NVFP4 KV FA2 routing and AEON-inspired DFlash stability patches
- `jethac/flashinfer` with SM121 `mm_fp4` dispatch and FA2 NVFP4-KV stride/page/deswizzle changes
- paired fp8-vs-NVFP4 KV runs on the same model, prompts, context length, memory fraction, graph mode, and concurrency
- `scripts/nvfp4_checkpoint_audit.py` output for any NVFP4-weight checkpoint used in the row, so format and sensitive-tensor handling are explicit

Current vLLM fork state:

- branch: `jethac/vllm@spark/hijinks-020-aeon-qwen-dflash-sm121a`
- passing row commit: `6804e1b81e6ea2ca53bb5021151bdad0f201b11d3`
- current fork head: `db4b210c1`
- source coverage: AEON lazy import fallback, CUDA graph alignment, Qwen3.5/3.6 text registry, hybrid KV `block_size=None` handling, Mamba block-size fallback, and text-only M-RoPE fallback
- source artifact: `results/vllm_aeon_qwen_patch_port_20260608T0619JST.md`
- passing fork-derived serving artifact: `results/jethac_qwen36_dflash_aeonfa2_nothink_20260608T0908JST_summary.md`
- image caveat: the passing container aligns `compressed-tensors==0.17.0`, adds `humming-kernels==0.1.4`, and restores AEON's FA2 binary after a PyTorch ABI mismatch. It is fork runtime parity, not clean fork packaging. The next image should use `VLLM_PRECOMPILED_SKIP_FLASH_ATTN=1` and supply an ABI-matched FA2 build instead.

Current local setup:

- `scripts/run_aeon_vllm_reproduction.sh qwen36-dflash RUN_ID` records the AEON Qwen reproduction row when `RECORD=1`.
- `scripts/pull_container_with_evidence.sh ghcr.io/aeon-7/vllm-spark-omni-q36:v2 RUN_ID` is the preferred image acquisition path when Docker pulls stall or fail to register.
- `scripts/qwen_speed_lane.py --input tasks/qwen_speed_lane_sample.jsonl --campaign-id RUN_ID` records already-running vLLM, SGLang, and llama.cpp Qwen servers with the shared row manifest wrapper.
- preflight artifact `results/aeon_vllm_reproduction_preflight_20260608T0430JST.md` confirms the GHCR image resolves and the Qwen target/drafter HF repos are public and non-gated from the GB10 host.
- current status: `results/aeon_qwen36_dflash_nothink_20260608T0834JST_row_manifest.json`, `results/jethac_qwen36_dflash_aeonfa2_nothink_20260608T0908JST_row_manifest.json`, and `results/jethac_qwen36_dflash_cleanfa2_sm121a_nothink_record2_20260608T2359JST_row_manifest.json` are `ok=true`; chat smoke returns normal `message.content` after disabling Qwen thinking with `chat_template_kwargs={"enable_thinking": false}`.
- clean native-target proof: `results/jethac_vllm_qwen_cleanfa2_patchedfa2_cutlass_audit_20260608T2355JST_incontainer_target_audit.md` proves the patched FA2 extension contains `sm_121a` cubins. The clean serving row selects FlashAttention 2, but its server log still does not contain accepted build-target strings.

Current local AEON Qwen evidence:

- `results/qwen_content_probe_20260608T0900JST_direct_chat_probes.json`: baseline and `/no_think` prompt rows stayed in `message.reasoning`; API-level `chat_template_kwargs={"enable_thinking": false}` produced `spark-ok` content for both `qwen36-fast` and `qwen36-deep`.
- `results/aeon_qwen36_dflash_nothink_20260608T0834JST_openai_benchmark.json`: compact serving passes with `50.37`, `55.84`, and `53.75 tok/s` decode for short, medium, and long-prefill cases.
- `results/jethac_qwen36_dflash_aeonfa2_nothink_20260608T0908JST_openai_benchmark.json`: derived `jethac/vllm` compact serving passes with `47.22`, `58.88`, and `61.62 tok/s` decode for short, medium, and long-prefill cases.
- `results/jethac_qwen36_dflash_aeonfa2_nothink_20260608T0908JST_server.log`: target `Qwen3_5MoeForConditionalGeneration`, drafter `DFlashDraftModel`, `FlashInferCutlassNvFp4LinearKernel`, `MARLIN` NvFp4 MoE, FlashAttention 2, CUDA graphs, `1,251,446` KV tokens, `4.77x` max concurrency at 262k context.
- `results/jethac_qwen36_dflash_cleanfa2_sm121a_nothink_record2_20260608T2359JST_summary.md`: clean FA2 image serves Qwen3.6 NVFP4+DFlash with `61.07`, `56.97`, and `60.10 tok/s` decode; server log shows `FlashInferCutlassNvFp4LinearKernel`, `MARLIN` NvFp4 MoE, FlashAttention 2, CUDA graphs, FlashInfer FP4 GEMM autotune, `1,241,920` KV tokens, and `4.74x` max concurrency at 262k context.
- `results/vllm_qwen_nvfp4_kv_capacity_20260608T1455JST_summary.md`: no-DFlash Qwen NVFP4-KV row through FlashInfer FA2 records fp8 `6,364,935` KV tokens versus NVFP4 `11,146,226` tokens, `1.751x` max-concurrency gain at 262k context, normal `message.content`, and decode-speed parity around `43 tok/s`.
- `results/aeon_qwen36_dflash_tailnet_retry2_20260608T075346JST_nvfp4_checkpoint_audit.json`: `ok=true`, compressed-tensors NVFP4, `124306` safetensors keys, `0` quantized sensitive keys.
- `results/aeon_qwen36_dflash_tailnet_retry2_20260608T075346JST_server.log`: target `Qwen3_5MoeForConditionalGeneration`, drafter `DFlashDraftModel`, `FlashInferCutlassNvFp4LinearKernel`, `MARLIN` NvFp4 MoE, FlashAttention 2, CUDA graphs, `585168` KV tokens, `4.73x` max concurrency at 262k context.
- `results/aeon_qwen36_dflash_nothink_20260608T0834JST_build_target_audit.json`: no accepted native `sm_121` or `sm_121a` target evidence in the server log.

Do not claim a fork speedup until server logs prove the selected kernel path and the before/after rows are matched. The current `jethac/vllm` row proves fork-derived serving parity through an AEON-based image, not an improvement.

## SGLang Target

Use Qwen for the first real SGLang NVFP4 KV validation:

1. Public BF16/fp8 Qwen baseline.
2. Same Qwen model with `--kv-cache-dtype fp4_e2m1 --attention-backend flashinfer --page-size 1`.
3. Deterministic output sanity plus a quality comparator.
4. KV pool tokens, maximum concurrency, TTFT, warmed decode, and selected backend logs.

Start with a standard-attention Qwen model before Qwen3.6 hybrid/MoE. Small models may be quality-negative controls for fp4 KV; a small-model incoherence result is not by itself a Spark kernel failure.

Current SGLang Qwen evidence:

- `results/sglang_qwen25_1_5b_fp8_vs_fp4kv_20260608T0332JST_summary.md`
- BF16/auto KV comparator: `Qwen/Qwen2.5-1.5B-Instruct`, FlashInfer attention, CUDA graphs enabled, `1,557,709` token KV pool, `57.7-58.9 tok/s` decode at `mem_fraction_static=0.40`.
- fp8 KV before row: `Qwen/Qwen2.5-1.5B-Instruct`, FlashInfer attention, CUDA graphs enabled, `3,113,713` token KV pool, about `58-59 tok/s` decode.
- stock `fp4_e2m1` with FlashInfer attention fails at the compatibility gate.
- stock `fp4_e2m1` with Triton attention reaches FP4 KV allocation, `5,534,509` tokens or about `1.78x` fp8 capacity, then fails on missing `KVFP4QuantizeUtil`.
- patched overlay using `jethac/sglang@98ad46961` gate/alias changes clears the stock SGLang blockers. FlashInfer attention reaches an `sm_121a` JIT compile and fails in FlashInfer FP4 decode. Triton attention serves only after disabling both standard and piecewise CUDA graphs, with `5,541,103` FP4 KV tokens, but the short decode row is only `0.276 tok/s` and output is repetitive.
- newer clean-source evidence on `jethac/sglang@spark/hijinks-018-fp4-e2m1-kv-sm121-serving` calibrates FP4 KV and disables CUDA graph and piecewise graph capture by default for native FP4 KV. The matched `d7d931f` row allocates `5,517,572` FP4 KV tokens versus `3,105,240` fp8 tokens (`1.7769x`), traces decode plus `extend_merge_paged`, and passes raw/chat smoke, but compact benchmark content still fails quality. The OpenAI logprob probe `results/sglang_qwen_fp4kv_d7d931f_logprob_quality_20260608T1609JST_summary.md` shows severe prompt-dependent drift; the native rendered-template probe `results/sglang_qwen_fp4kv_d7d931f_native_divergence_20260608T1626JST_summary.md` narrows `medium_decode` to a token-index-4 rank reversal; the prompt reconciliation row `results/sglang_qwen_fp4kv_prompt_path_reconcile_20260608T173754JST_summary.md` proves OpenAI and local rendered prompt IDs match; the endpoint metadata row `results/sglang_qwen_fp4kv_endpoint_metadata_20260608T1819JST_summary.md` confirms the current traces are not request-tagged and provides the first-token dump hook needed to split forward/KV state from logits preprocessing.

Interpretation: FP4 KV capacity and backend routing are real on this Qwen row, but benchmark quality is not fixed. The next after-row should dump request-tagged first-token logits before/after SGLang logits preprocessing before any throughput claim. Graph-enabled FP4 KV remains a separate corruption bug.

## llama.cpp Target

First Qwen GGUF row captured:

- model: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`, file `qwen2.5-1.5b-instruct-q4_k_m.gguf`
- binary: `/home/jethac/src/llama.cpp-b9536/build/bin/llama-server`, build `308f61c31 (9536)`
- alias: `qwen25-1.5b-q4_k_m-gguf`
- build evidence: server log reports `CUDA : ARCHS = 1210`, `USE_GRAPHS = 1`, and `BLACKWELL_NATIVE_FP4 = 1`
- compact serving decode: `175.19 tok/s` short, `174.86 tok/s` medium, `166.66 tok/s` long-prefill
- `llama-bench`: `pp512 12505.79 +/- 615.87 tok/s`, `tg128 178.10 +/- 0.95 tok/s`
- logprobs probe: still not lm-eval-compatible because the OpenAI response has `choices[0].logprobs.content`, not `tokens` plus `token_logprobs`

Artifacts use the prefix `results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST_*`.

Interpretation: llama.cpp practical Qwen GGUF serving is now proven on this GB10, and it is much faster than the small SGLang Qwen rows for single-stream decode. This row still does not prove native NVFP4/MXFP4 GGUF, Qwen3/Qwen3.6 behavior, or lm-eval accuracy.

Next llama.cpp Qwen proof:

- Run a Qwen3/Qwen3.6-class instruct GGUF row with the same `b9536` llama.cpp build, not another Qwen2.5 1.5B row.
- Start with a normal practical-serving quant such as Q4_K_M or Q4_0 to fill the larger-Qwen GGUF gap.
- Keep native FP4 separate: `BLACKWELL_NATIVE_FP4=1` in the build log is not proof unless the model artifact is NVFP4/MXFP4 GGUF and runtime/build audits show the native FP4 path.

## Required Artifacts

Every Qwen row should include:

- `spark_doctor` JSON and markdown
- runtime process probe
- NVFP4 checkpoint audit for NVFP4-weight rows
- CUDA build-target audit
- CUDA shared-object audit where applicable
- server log with backend selection
- OpenAI-compatible smoke and serving benchmark
- exact model revision and container/build commit
- hardware comparison key including compute capability and SM count

## Shared Runner

Use the JSONL-driven runner when the servers are already up:

```bash
python3 scripts/qwen_speed_lane.py \
  --input tasks/qwen_speed_lane_sample.jsonl \
  --campaign-id qwen_speed_lane_YYYYMMDDTHHMMJST \
  --continue-on-error
```

For a local command-shape check without touching live servers:

```bash
python3 scripts/qwen_speed_lane.py \
  --input tasks/qwen_speed_lane_sample.jsonl \
  --campaign-id qwen_speed_lane_dryrun \
  --dry-run \
  --continue-on-error
```

Each JSONL row is passed through `scripts/record_openai_serving_row.py`, so the output stays compatible with the existing benchmark protocol: chat smoke, compact OpenAI serving benchmark, optional runtime probe, optional CUDA shared-object audit, optional build-target audit from a server log, optional `llama-bench`, and optional GGUF logprobs probe.
