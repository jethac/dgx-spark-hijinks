# DGX Spark Compatibility Board

Date: 2026-06-09 JST

Purpose: one public, recurring status view for the Spark-class GB10 local-AI stack. This board does not replace the detailed reports; it tells a reader what is usable today, what is only a proof, and what exact artifact is missing next.

Hardware scope: single Spark-class GB10 system, compute capability `12.1` / `sm_121`. Multi-Spark and TP>1 remain unvalidated.

Live execution runbook: `docs/LIVE_GB10_RUNBOOK.md`.
Machine-readable queue: `tasks/live_gb10_queue.jsonl`.
Queue audit: `results/live_task_queue_audit_20260609.json`.

Latest reachability evidence: `results/gb10_host_access_probe_tailnet_retry_20260609.md`.
The GB10 node is visible in the Tailscale control plane at `100.113.98.11`, but the
sanitized probe records `usable_for_live_work=false`: Tailscale ping times out, TCP/22
times out, SSH times out, and the peer reports relay traffic with `rx 0`. Live validation is
queued, but SSH is not currently a usable control path from this workspace.

## Latest Deltas

- `jethac/flashinfer@spark/hijinks-021-prefill-debug` commit `1230341d` adds inactive
  `FLASHINFER_PREFILL_DEBUG_ONCE=1` C++/JIT identity logging for the Gemma 3 NVFP4-KV
  FlashInfer paged-prefill failure. Run packet:
  `tasks/vllm_gemma3_flashinfer_prefill_debug_packet_20260609.md`; the packet now runs
  `scripts/flashinfer_prefill_debug_log_audit.py` to machine-check the paged FP4x2 identity,
  required scale tensors, and layout/page/head-consistent byte-carrier tensor views
  and tensor dump.
- `scripts/gemma_nvfp4_kv_quality_gate.py` now enforces a stricter Gemma NVFP4-KV quality
  gate. The current Gemma 3 NVFP4 candidate is machine-rejected in
  `results/gemma3_nvfp4_kv_quality_gate_current_red_20260609.json`, so capacity/routing
  evidence cannot accidentally promote the known-corrupt row.
- `results/gemma_compatibility_plan_audit_20260609.json` verifies that
  `docs/GEMMA_COMPATIBILITY_PLAN.md` still records the Gemma-specific PPL/quality rule, the
  outlier-risk caveat, the audited rung order, and capacity-not-speed framing.
- `scripts/install_sglang_source_stack.sh` now installs patched editable FlashInfer and
  source-builds `sgl-kernel` from `third_party/sglang`, so the SGLang matrix runner no
  longer depends on stale PyPI `flashinfer-cubin`, `flashinfer-jit-cache`, or
  `sglang-kernel` wheels.
- Fresh mechanical audits are recorded under `results/*_20260609.json`; counterpart task
  coverage remains complete for the six missing/partial non-vLLM rows.

## Status Legend

| status | meaning |
|---|---|
| `blessed` | usable for the named scope with local artifacts and clear caveats |
| `partial` | works for some rows, but not enough to recommend as a general Spark path |
| `debug-only` | useful evidence, not a usable serving path |
| `blocked` | next proof cannot proceed until the named external/runtime blocker is cleared |
| `side-runtime` | useful complement, not a main throughput path |

## Runtime Board

| runtime | status | best local evidence | not blessed yet | next proof | issue |
|---|---|---|---|---|---|
| vLLM | `partial` | AEON Gemma 4 26B A4B NVFP4+DFlash serves locally at about `48-54 tok/s` short/medium decode and `98 tok/s` long-prefill; Gemma 12B unified serves through a source/precompiled path; AEON Qwen3.6 NVFP4+DFlash passes compact serving at about `50-56 tok/s` when Qwen thinking is disabled through `chat_template_kwargs`; `jethac/vllm` derived AEON Qwen row passes at `47.22`, `58.88`, and `61.62 tok/s`; clean `jethac/vllm@a919d635d` + `jethac/flash-attention@7d53245` Qwen row passes at `61.07`, `56.97`, and `60.10 tok/s` with separate `sm_121a` FA2 cubin proof; Qwen NVFP4-KV through FlashInfer FA2 records `1.751x` fp8 KV pool/concurrency with decode parity; Gemma Rung -1 config audit and Gemma 3 live preflight are complete; `jethac/vllm@25ab073ef` adds the runtime geometry hook on top of the CUDA-13-proven `a919d635d` lane; HF access for gated Gemma 3 is cleared; Gemma 3 27B Rung 1 fp8 comparator serves with FlashInfer attention, measured `62`-layer SWA geometry, `882,851` KV tokens, `6.74x` concurrency, and unflagged benchmark output; the matching Gemma 3 NVFP4 row routes through FlashInfer FA2 and records `1.77x` KV capacity at decode parity, but output is corrupted; the refreshed first-token diagnostic proves the NVFP4 corruption is already present on the first generated token with `0.0` top-logprob overlap versus fp8; `scripts/gemma_nvfp4_kv_quality_gate.py` now rejects that known-bad row; `jethac/vllm@e2a8197a9` adds env-gated trace events for metadata, KV writes, NVFP4 split views, and SWA skip state; `tasks/vllm_gemma3_nvfp4_trace_packet_20260608.md` records the next exact run packet | official general Spark container, accuracy check, native FP4 weight/MoE proof, quality-correct Gemma NVFP4-KV, upstreamable packaging beyond the AEON Qwen recipe | run the Gemma 3 FlashInfer prefill debug packet, fix the paged-prefill failure, then pass `scripts/gemma_nvfp4_kv_quality_gate.py` before climbing to Gemma 4 31B text-only dense D=512 mixed-KV | [#6](https://github.com/jethac/dgx-spark-hijinks/issues/6), [#20](https://github.com/jethac/dgx-spark-hijinks/issues/20) |
| SGLang | `partial` | NVIDIA 26.05 serves Qwen2.5 1.5B BF16/auto and fp8 at about `58-59 tok/s`; fp8 doubles KV pool versus BF16/auto; clean `jethac/sglang` FP4-KV source overlay records `1.779x` fp8 KV capacity under an auto-safe no-graph policy; standalone convention bridge clears raw FA2 reader math for viable scale pairs; prompt reconciliation proves OpenAI prompt IDs match local Qwen rendering, so serialization is not the quality bug; endpoint metadata localization records the OpenAI/native FP4 first-token split; the paired FP4 dump shows same-hash 56-token OpenAI/native requests diverge already at prefill logits (`334` vs `838`), while the paired fp8 control returns `**` on both endpoints and both prefill candidates argmax `334`; radix isolation shows `--disable-radix-cache` fixes the FP4 first-token split and `--skip-server-warmup` alone does not; `jethac/sglang@ce1b6d15e` traces the default failure to 55 cached prefix tokens routed through `forward_extend_merge_paged`; `jethac/sglang@839cb7457` shows those same 55 page IDs and matching K/V data plus scale first dimensions; `jethac/sglang@991ac1e63` samples readable packed K/V bytes, FP8 scale bytes, and finite `o1/s1/o2/s2` plus merged output for layer 0; `jethac/sglang@f76f80484` shows sampled cached pages `4113..4116` match write input bytes = stored bytes = read bytes for K/V data and scale buffers; `jethac/sglang@2a228949a` shows cached paged-prefix `o2`, FlashInfer-base `s2`, and merge match a dequantized reference; `scripts/sglang_dense_cache_trace_compare.py` is now wired into the dense-cache runner to produce per-case dense-vs-cached trace comparison JSON | Gemma serving, FP4 KV output quality, graph-safe FP4 KV, explicit `sm_121` build-target proof | run the dense-cache trace packet and inspect the generated comparison artifact before changing cache policy | [#14](https://github.com/jethac/dgx-spark-hijinks/issues/14), [#18](https://github.com/jethac/dgx-spark-hijinks/issues/18) |
| llama.cpp | `blessed` for practical serving | Gemma 4 26B Q4_0 serves around `76 tok/s`; Qwen2.5 1.5B Q4_K_M serves around `167-175 tok/s`; logs include `CUDA : ARCHS = 1210` and CUDA graphs; `jethac/llama.cpp@19bba67c1` builds/emits native `mxf4nvf4.block_scale` PTX for `sm_121a`; converted AEON Qwen3.6 NVFP4 GGUF loads and returns a correct short smoke with Nsight `GGML_TYPE_NVFP4` matmul evidence | paper-comparable GGUF lm-eval accuracy; NVFP4 accuracy/speed versus BF16/Q8/Q4_K_M; full no-fallback native FP4 proof; larger repeatable Qwen3/Qwen3.6 rows | run matched PP/TG and correctness comparisons for the NVFP4 GGUF; separately try a newer pin or native endpoint after both top-512 and OpenAI echo probes failed to expose supplied-token logprobs | [#17](https://github.com/jethac/dgx-spark-hijinks/issues/17), [#8](https://github.com/jethac/dgx-spark-hijinks/issues/8) |
| FlashInfer | `partial` | SM121 `mm_fp4` dispatch patch enables `b12x`; FA2 NVFP4 KV standalone probes pass small and Gemma sliding/local shapes; `jethac/flashinfer@1230341d` stages one-shot generated-module/C++ identity logging for the Gemma 3 paged-prefill failure; `scripts/flashinfer_prefill_debug_log_audit.py` is ready to reject missing/non-FP4 paged module identity, missing scale tensors, and paged tensor geometry mismatches | serving speedup, Gemma global `D=512` path, end-to-end vLLM/SGLang NVFP4 KV, clean wheel/container proof, live run of the new prefill debug packet | run the Gemma 3 FlashInfer prefill debug packet first when host access returns; keep `b12x` as enablement until serving rows prove speed | [#7](https://github.com/jethac/dgx-spark-hijinks/issues/7) |
| LiteRT-LM | `side-runtime` | CPU generation works; GPU benchmark runs for small Gemma row | GPU chat exits `-11`; not a throughput path | decide CPU/complement recommendation or isolate GPU chat crash | [#16](https://github.com/jethac/dgx-spark-hijinks/issues/16) |
| HF fallback | `partial` | telemetry and failure annotation scripts exist | several fallback rows die with `returncode=-9`; resource cause not always proven | add stronger OOM/resource evidence before using HF fallback in comparisons | [#9](https://github.com/jethac/dgx-spark-hijinks/issues/9) |

## Model And Feature Board

| lane | status | current call | next proof |
|---|---|---|---|
| Gemma 4 26B | `partial` | fast local vLLM NVFP4+DFlash row exists through AEON image; llama.cpp Q4_0 is practical and fast | accuracy checks, fork parity, and native/forked vLLM path |
| Gemma 4 12B | `partial` | source/precompiled vLLM probe serves at about `7.7 tok/s`; ladder now treats it as the final encoder-free multimodal-KV rung, not the next/simple rung | clean release/nightly container plus one zero-shot task after text-only 31B/26B rungs are green |
| Gemma ladder | `config-audited` | Rung -1 artifact shows Gemma 3 27B is uniform `D=128`; Gemma 4 12B/31B/26B-A4B all carry full-attention `D=512`; operator architecture reorders the climb as Gemma 3 27B -> 31B text-only dense D=512 -> 26B-A4B text-only MoE -> 12B encoder-free multimodal-KV; plan audit confirms the Gemma-specific quality/PPL and capacity-not-speed gates | running-model geometry and encoder/modality confirmation per runtime/rung |
| Qwen speed | `partial` | SGLang small Qwen rows, SGLang FP4 KV capacity-only row plus OpenAI/native logprob and prompt-reconciliation quality-localization rows, llama.cpp Qwen2.5 row, passing AEON vLLM Qwen36 NVFP4+DFlash row, passing derived `jethac/vllm` Qwen36 row, passing clean-FA2 `jethac/vllm` Qwen36 row, matched vLLM Qwen fp8-vs-NVFP4 KV capacity row, and llama.cpp Qwen3.6 NVFP4 GGUF runtime smoke exist | larger llama.cpp Qwen3/Qwen3.6 GGUF benchmarks, llama.cpp NVFP4 GGUF accuracy/speed, SGLang FP4-KV quality, SGLang DFlash/EAGLE, native FP4 weight/MoE proof |
| NVFP4 weights | `partial` | AEON Gemma, AEON Qwen36, derived `jethac/vllm` Qwen36, and clean-FA2 `jethac/vllm` Qwen36 prove compressed-tensors NVFP4 weight serving on GB10 | native FP4 weight/MoE compute proof; current clean Qwen row still selects Marlin weight-only FP4 |
| NVFP4 / FP4 KV | `partial` | standalone probes prove the FA2 tuple-KV signature; vLLM Qwen NVFP4-KV records `1.751x` fp8 KV pool/concurrency with normal content and decode parity; SGLang FP4 KV records a matched `1.779x` fp8 capacity gain under auto-safe no-graph policy but fails quality | Gemma NVFP4-KV, SGLang quality-passing FP4 KV serving, graph-safe serving, and claimable throughput |
| GGUF accuracy | `blocked` | OpenAI-compatible llama.cpp logprobs schema is insufficient; live native task at `n_probs=512` scored likely continuations but missed the unlikely continuation. This is separate from the native-FP4 arch-build row, which only proves FP4 code emission. | direct supplied-token logprobs, practical full-vocabulary probabilities, or another native scoring path |
| Multi-Spark | `blocked` | design-only | second unit or remote equivalent |

## Live Proof Queue

Run these when the GB10 host is reachable. The ordered source of truth is now
`tasks/live_gb10_queue.jsonl`; validate it with:

```bash
python3 scripts/live_task_queue_audit.py \
  --queue tasks/live_gb10_queue.jsonl \
  --output results/live_task_queue_audit_YYYYMMDDTHHMMJST.json
```

The human runbook is `docs/LIVE_GB10_RUNBOOK.md`. The older subsections below remain as
context for specific rows, not as an override for queue priority.

### vLLM Qwen36 NVFP4+DFlash

```bash
DOWNLOAD=0 DOCKER_PULL=0 RECORD=1 \
scripts/run_aeon_vllm_reproduction.sh \
  qwen36-dflash aeon_qwen36_dflash_YYYYMMDDTHHMMJST
```

Keep the row non-claim-ready until chat smoke returns normal validated content and the serving benchmark records usable output, not only completion-token counts.
For Qwen thinking models, pass `CHAT_TEMPLATE_KWARGS_JSON='{"enable_thinking": false}'` when the goal is normal OpenAI `message.content` output.

### vLLM Gemma 3 27B Rung 1

Preflight artifact: `results/vllm_gemma3_27b_rung1_preflight_20260608.md`.

The packet generator now captures real Docker logs, waits for `/v1/models`, and removes
containers on exit. Clean checkout artifact:
`results/vllm_gemma3_27b_rung1_checkout_setup_20260608.md`. Generated packet:
`docs/results/vllm_gemma3_27b_rung1_20260608TCHECKOUTJST_command_packet.sh`.
HF access probe `results/vllm_gemma3_27b_hf_access_probe_20260608T173133JST.json`
confirms the model is manually gated: model metadata is visible, but even the small
config/tokenizer snapshot fails with `GatedRepoError` because no `HF_TOKEN` is present in
the container environment. Disk headroom is sufficient. Provide/verify HF auth before
starting the fp8 comparator. Follow-up artifact
`results/vllm_gemma3_27b_hf_access_probe_20260608T1832JST.md` clears the auth/cache
blocker: the container sees `HF_TOKEN` and downloads the config/tokenizer snapshot for
`google/gemma-3-27b-it`. The live blocker moved to vLLM packaging: the current
`3658ba712` overlay is configured to use precompiled wheel metadata from
`8916796bc50926fd61e606718b194a71e2e31a24`, but that metadata returns 404 for the `cu130`
paths, so the packet must be rebased to a source/precompiled-wheel pair with published
CUDA 13 metadata before fp8 serving starts. That rebase is now prepared as
`jethac/vllm@25ab073ef87f4443616fbaf00a2f6f09a9087c1f` with precompiled wheel base
`4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa`. The dependency-resolving setup-only artifact
`results/vllm_gemma3_27b_rung1_setup_only_20260608T1855JST.md` is retained only for the
metadata-404 fix; live rows now use `--no-deps` to avoid downgrading Torch/FlashInfer.
The fp8 comparator is captured in
`results/vllm_gemma3_27b_rung1_fp8_20260608T1924JST.md`: FlashInfer attention, uniform
`head_dim=128`, `52` local SWA layers, `10` full/global layers, `882,851` fp8 KV tokens,
`6.74x` concurrency, and unflagged benchmark output. The NVFP4 candidate is captured in
`results/vllm_gemma3_27b_rung1_nvfp4_20260608T1924JST.md`: FlashInfer FA2 routing and
`1.777x` capacity are proven, but output is corrupted, so Gemma NVFP4-KV is not green.

Current live packet: `tasks/vllm_gemma3_flashinfer_prefill_debug_packet_20260609.md`.
It uses `jethac/flashinfer@spark/hijinks-021-prefill-debug` commit `1230341d` and
`FLASHINFER_PREFILL_DEBUG_ONCE=1` to print generated dtype identity, FP4 requirement,
additional scale-factor tensors, runtime page/head/window fields, and TensorView
shape/stride/dtype for the first generated paged-prefill module.
`scripts/flashinfer_prefill_debug_log_audit.py` must pass on the server log before treating
the diagnostic as complete. Run this before adding kernel-side fragment dumps.

### Qwen Speed Lane

After the target Qwen servers are already running:

```bash
python3 scripts/qwen_speed_lane.py \
  --input tasks/qwen_speed_lane_sample.jsonl \
  --campaign-id qwen_speed_lane_YYYYMMDDTHHMMJST \
  --continue-on-error
```

Before starting the seven AEON-derived counterpart rows, validate the live task matrix:

```bash
python3 scripts/counterpart_task_matrix.py \
  --tasks tasks/counterpart_evidence_tasks.jsonl \
  --audit results/counterpart_evidence_audit_20260609.json \
  --output results/counterpart_task_matrix_YYYYMMDDTHHMMJST.json
```

### llama.cpp Native Loglikelihood

The first live task is recorded in `results/llamacpp_native_loglikelihood_20260608T1331JST_summary.md` and failed because the unlikely continuation was not present in top-512 probabilities. The follow-up OpenAI echo probe is recorded in `results/llamacpp_gguf_echo_logprobs_probe_20260608_summary.json` and also failed because pinned `b9536` exposes generated-token `logprobs.content`, not prompt `tokens`/`token_logprobs`. The next experiment should change the scoring implementation or server pin, not merely rerun the same command.

The current acceptance contract is `tasks/llamacpp_supplied_token_loglikelihood_contract_20260609.md`. For newer-stock echo testing, use `scripts/llamacpp_echo_logprobs_contract_runner.py`; it probes every row in `tasks/llamacpp_loglikelihood_smoke.jsonl` with both `max_tokens=0` and `max_tokens=1`, then bridges and audits the result. A green row must score each supplied continuation token by token id, including the unlikely `" zebra"` token, and then pass:

```bash
python3 scripts/llamacpp_loglikelihood_contract_audit.py \
  --artifact results/llamacpp_native_loglikelihood_task_YYYYMMDDTHHMMJST.json \
  --input tasks/llamacpp_loglikelihood_smoke.jsonl \
  --output results/llamacpp_native_loglikelihood_task_YYYYMMDDTHHMMJST_contract_audit.json
```

Top-N output is acceptable only if every supplied continuation token is present. The known top-512, pinned echo, and dry-run artifacts are red under this auditor, so do not use them as row-8 evidence.

Historical command shape:

```bash
python3 scripts/llamacpp_native_loglikelihood_task.py \
  --url http://127.0.0.1:8080 \
  --n-probs 512 \
  --input tasks/llamacpp_loglikelihood_smoke.jsonl \
  --output results/llamacpp_native_loglikelihood_task_YYYYMMDDTHHMMJST.json
```

### llama.cpp Native FP4 Runtime

The arch-build checkpoint is recorded in
`results/llamacpp_native_fp4_arch_20260608T164917JST_summary.md`: `121a` builds and emits
native block-scale FP4 PTX, `121` rewrites to `121a`, and `120f` is rejected by this
toolchain. The next row must run an actual NVFP4/MXFP4 GGUF on
`jethac/llama.cpp@19bba67c1f4db723c60a0d421aa0788bf4ddc699` and record runtime dispatch,
correctness versus BF16/Q8, and PP/TG speed. Do not cite the arch-build artifact as runtime
proof.

### SGLang FP4 KV After-Row

Use the clean `jethac/sglang` fork/container, not a site-package overlay, and record:

- BF16/auto or fp8 comparator
- FP4 KV row with the same model, prompts, memory fraction, and graph mode
- KV pool tokens
- selected attention backend
- CUDA graph status
- deterministic output sanity
- quality comparator before any speed/capacity claim

Current safe policy: the fork disables CUDA graph capture for native FP4 KV unless `SGLANG_FP4_KV_ENABLE_CUDA_GRAPH=1` is set. The matched `d7d931f` row proves capacity/backend routing but still corrupts standardized Qwen output. OpenAI and native logprob probes localize this to prompt/path-sensitive generation drift; prompt reconciliation proves OpenAI prompt IDs match local Qwen rendering, so the remaining bug is endpoint/path-specific FP4 KV numerics or metadata. The paired first-token dump row moves this earlier than `_preprocess_logits()`: same-hash OpenAI/native prompts already have different prefill argmaxes, and the fp8 control proves the endpoint sequence itself can agree. Radix isolation shows disabling radix cache fixes the FP4 first-token split; the `ce1b6d15e` trace shows the failing default native request reuses 55 cached prefix tokens and enters `forward_extend_merge_paged`, while radix-off recomputes the full prompt and passes. The request-order post-analysis shows cached-prefix rows have `0 / 20` first-token top-logprob overlap with the full-prefill distribution, while no-reuse rows have `20 / 20`. This is diagnostic, not a serving blessing. FP4 KV is not a blessed SGLang serving path yet, and no-reuse rows must not be counted as the capacity path.

Packaging note: use `scripts/prepare_sglang_source_stack_image.sh` or the per-case
installer `scripts/install_sglang_source_stack.sh`. The current runner source-builds the
patched stack instead of installing the PyPI `sglang-kernel` wheel.

Queued dense-cache trace command: `scripts/run_sglang_fp4_dense_cache_trace.sh` wraps the
same source-built stack plus `SGLANG_FP4_KV_TRACE_DENSE_CACHE=1` and
`scripts/sglang_fp4_request_order_probe.py`. Default `CASES=default` runs the known failing
cached-prefix row; use `CASES=full_paged` only as the follow-up comparator after the default
row localizes the first divergence. The comparator now requires at least one metric-bearing
dense/cached vector or top-k comparison and records `first_divergence`; structural trace-key
matches without comparable metrics are red.

## Update Cadence

Update this board whenever any of the following changes:

- a new runtime row is captured
- a fork branch advances for vLLM, SGLang, FlashInfer, or llama.cpp
- a blocker changes from acquisition/setup to model/runtime/kernel failure
- a row becomes blessed or unblessed
- a new live-proof command supersedes one of the queue entries above

Each update must include:

- artifact path under `results/`
- exact runtime/container/commit
- hardware key with compute capability and SM count when live hardware is involved
- GitHub issue comment link when pushed publicly

Detailed acceptance evidence remains in `docs/SOLUTIONS_STATUS.md`, runtime-specific docs, and the `results/` artifacts.

The install/package-level companion is `docs/WHEEL_CONTAINER_MATRIX.md`.
