# Single-Spark vLLM Recipe

Status: draft, not blessed.

Use this recipe for one DGX Spark-class GB10 unit. Do not use this for multi-Spark runs.

## Rules

- Target GB10 as `sm_121`.
- Prefer a vLLM container or wheel explicitly validated for Spark.
- Keep CUDA graphs enabled unless a specific bug requires eager mode.
- Keep concurrency modest; Spark is a local/small-batch inference machine, not an H100 replacement.
- Record `spark_doctor` output before every run.
- Treat `hikarioyama/vllm-nvfp4-kv-sm120` as reference work for NVFP4 KV, not as a Spark result. Build on it unless GB10 testing shows a better path.

## Preflight

```bash
python3 scripts/spark_doctor.py --json > results/spark_doctor_before_vllm.json
python3 scripts/spark_doctor.py > results/spark_doctor_before_vllm.md
```

## Smoke

Start the vLLM server using the candidate stack, then run:

```bash
python3 scripts/openai_chat_smoke.py \
  --url http://127.0.0.1:8000 \
  --model MODEL_NAME \
  --output results/vllm_chat_smoke.json
```

## Result Requirements

A blessed result must record:

- exact container or wheel versions
- model id and revision
- CUDA driver/runtime
- PyTorch version
- vLLM version
- selected attention backend if visible
- build/JIT target audit path
- CUDA shared-object audit path
- CUDA graphs enabled/disabled
- prompt and generated token counts
- wall time and tokens/sec

## Gemma 4 12B Unified Probe

The current 12B compatibility proof is not a clean blessed stack. It used:

- `scripts/run_vllm_gemma4_12b_unified_probe.sh`
- base image `vllm/vllm-openai:latest-cu130`
- vLLM source commit `da1daf40bf18e5eaae04f26a80a537c8168a8bc2`
- `VLLM_USE_PRECOMPILED=1`
- Transformers main
- stale `flashinfer-jit-cache` removal

It served `google/gemma-4-12B-it`, but vLLM forced `TRITON_ATTN` and compact decode was about 7.7 tok/s. Use this as a packaging target, not a performance recommendation.

## Qwen Speed And Capacity Probe

Qwen is the preferred first target for measuring vLLM SM121a performance mechanics because it avoids Gemma 4's heterogeneous global/local attention complication.

AEON Qwen prior art to reproduce before claiming our own speedup:

- image: `ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2` or a documented newer successor
- model: `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4`
- drafter: `z-lab/Qwen3.6-35B-A3B-DFlash`
- key flags: `--quantization compressed-tensors`, `--attention-backend flash_attn`, DFlash `num_speculative_tokens=15`, long context, chunked prefill, prefix caching
- key evidence: selected linear and MoE backends, CUDA graph mode, DFlash acceptance, TTFT, decode tok/s, aggregate tok/s, and server stability

Fork after-row:

- use `jethac/vllm` plus `jethac/flashinfer`
- compare fp8 KV versus `--kv-cache-dtype nvfp4` on the same Qwen model, prompts, graph mode, memory fraction, and concurrency
- require logs proving FlashInfer FA2 NVFP4 KV for the NVFP4 row
- record KV pool tokens and maximum concurrency; decode-speed parity is useful if capacity improves

Do not transfer Gemma 4 conclusions onto Qwen or Qwen conclusions onto Gemma without model-specific serving rows.

Capture the row after the server is healthy:

```bash
python3 scripts/record_openai_serving_row.py \
  --backend vllm \
  --phase before \
  --run-id "$RUN_ID" \
  --url http://127.0.0.1:8000 \
  --model "$MODEL" \
  --container-image "$IMAGE" \
  --model-revision "$MODEL_REVISION" \
  --quantization compressed-tensors \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --attention-backend flash_attn \
  --server-log "results/${RUN_ID}_server.log" \
  --process-match vllm \
  --cuda-so-package vllm \
  --cuda-so-package flashinfer
```

## AEON Gemma 4 NVFP4 Weight Probe

AEON's Gemma 4 evidence is a vLLM NVFP4-weight path, not an FA2 NVFP4-KV result. Treat it as a separate reproduction target:

- image: `ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2`
- model: `AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4`
- drafter: `z-lab/gemma-4-26B-A4B-it-DFlash`
- env surface: `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass`, `VLLM_USE_FLASHINFER_MOE_FP4=0`, `VLLM_TEST_FORCE_FP8_MARLIN=0`, `VLLM_USE_FLASHINFER_SAMPLER=1`
- expected attention path: Gemma 4 may force Triton attention because local and global head dimensions differ

The local FlashInfer FA2 NVFP4-KV probe passed the Gemma 4 26B sliding/local shape but failed the global `D=512` shape. Do not use `--kv-cache-dtype nvfp4` for a blessed Gemma 4 vLLM row until that global path is fixed, routed to a proven fallback, or shown irrelevant to the selected model path.

## Experimental NVFP4 KV Fork Probe

Reference repo:

- `https://github.com/hikarioyama/vllm-nvfp4-kv-sm120`
- audited HEAD: `f6156ee3b22b24885a52c02bdafb34a9c201fe86`

Reference claims from SM120 RTX PRO 6000-class hardware:

- vLLM `--kv-cache-dtype nvfp4` routed through patched FlashInfer FA2.
- Step3.7-Flash 198B used TP=2, expert parallel, ModelOpt quantization, MTP K=1, `--max-model-len 131072`, and CUDA graph capture sizes `1,2,4,8,16`.
- Reported KV pool grew from about 1.66M fp8 tokens to about 2.96M-3.08M NVFP4 tokens, with decode in the same rough range.
- The expected primary win is KV capacity and concurrency, not a weight-GEMM speedup. Decode speed parity is enough if the KV pool expands.
- The reference B2 path matters because it avoids the hidden V scale-factor scratch allocation from the interim B1 design. Our telemetry must watch for unreported scratch or JIT-cache allocations that reduce usable KV pool.
- Scope limits: standard attention only; head dim 64/128/256/512; not MLA, Mamba/SSM, or attention sinks.

Spark proof requirements before blessing:

- first run the reference `harness/h_layout_b2.py` with the target model's `H_q/H_kv/D/page` and require the expected cosine/PASS result for the relevant layout
- one GB10 `sm_121` server starts with `--kv-cache-dtype nvfp4`
- logs or profiler evidence prove FlashInfer FA2 NVFP4 KV, not fp8/bf16 fallback
- build or JIT cache records `sm_121`, `sm_121a`, or a documented valid SM12x family target
- `scripts/cuda_build_target_audit.py` records the build/JIT target evidence before any `.so` inspection claim
- paired fp8 and NVFP4 runs use the same model, prompts, graph settings, and memory utilization
- KV pool tokens, maximum concurrency, quality, long-context behavior, and warmed decode speed are recorded
- Gemma 4's alternating local/global attention is checked against the reference repo's unsupported attention-sink caveat before treating any Gemma result as general
- if source changes are needed, port them through `jethac/vllm` and `jethac/flashinfer` worktrees instead of treating the SM120 overlay as a vendored production dependency
