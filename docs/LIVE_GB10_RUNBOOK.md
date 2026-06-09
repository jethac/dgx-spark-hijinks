# Live GB10 Runbook

Date: 2026-06-09 JST

Purpose: the single human-facing stop/start page for live work on the Spark-class GB10
host. The machine is intermittently unreachable, so live sessions should start from the
ordered queue instead of re-reading the long status docs.

Machine scope: one Spark-class GB10 system, compute capability `12.1` / `sm_121`.

Current control-plane state from this workspace:

- Tailnet IP: `100.113.98.11`
- Latest observed state: listed as active via relay, but `tailscale ping`, TCP/22, and
  SSH time out with `rx 0`.
- Latest probe artifact: `results/gb10_host_access_probe_20260609.md`.

## Preflight

Run from the Windows workspace before attempting live work:

```powershell
python scripts\gb10_host_access_probe.py `
  --host 100.113.98.11 `
  --ssh-user jethac `
  --output-json results\gb10_host_access_probe_RUN_ID.json `
  --output-md results\gb10_host_access_probe_RUN_ID.md
```

Proceed only when `usable_for_live_work` is `true`. If Tailscale ping, TCP/22, or SSH
fails, do not start a live run. Continue offline repo work instead.

When SSH is reachable, run the queue audit before picking work:

```bash
python3 scripts/live_task_queue_audit.py \
  --queue tasks/live_gb10_queue.jsonl \
  --output results/live_task_queue_audit_$(date +%Y%m%dT%H%MJST).json
```

The checked-in audit artifact for the current queue is
`results/live_task_queue_audit_20260609.json`.

## Execution Rule

Use `tasks/live_gb10_queue.jsonl` as the ordered source of truth.

Work top-down unless a row's blocker is still true. A skipped row needs a short artifact
or issue comment explaining the blocker, not silent reordering.

Every live run should record:

- run id;
- task id from `tasks/live_gb10_queue.jsonl`;
- exact fork commit/container/image;
- compute capability and `multi_processor_count`;
- server log or kernel log;
- acceptance artifacts named by the queue row;
- issue comment URL after pushing.

## Current Queue

The queue currently contains eight items:

1. `vllm_gemma3_flashinfer_prefill_debug`
   - packet: `tasks/vllm_gemma3_flashinfer_prefill_debug_packet_20260609.md`
   - issue: `#6/#7`
   - why: Gemma 3 NVFP4-KV corruption is below the Python-visible FlashInfer wrapper.
2. `sglang_qwen_fp4kv_dense_cache_trace`
   - packet: `tasks/sglang_qwen_fp4kv_dense_cache_trace_probe_20260609.md`
   - issue: `#18/#20`
   - why: SGLang Qwen FP4-KV is red only when cached FP4 prefix reuse is involved.
3. `vllm_qwen_clean_ppl_32k_128k`
   - packet: `tasks/vllm_qwen_nvfp4_kv_clean_ppl_sweep_20260609.md`
   - issue: `#7/#20`
   - why: the 8k clean-path PPL row is accepted; 32k/128k quality cost remains missing.
4. `llamacpp_native_fp4_correctness_speed`
   - packet: `tasks/llamacpp_nvfp4_correctness_speed_packet_20260608.md`
   - issue: `#17/#20`
   - why: sm_121a native FP4 build proof exists, but runtime correctness/speed does not.
5. `flashinfer_fp4_gemm_tile_smem`
   - packet: `tasks/flashinfer_fp4_gemm_tile_smem_probe_20260609.md`
   - issue: `#7/#13`
   - why: b12x dispatch is not tile/shared-memory viability.
6. `llamacpp_newer_echo_logprobs_probe`
   - task ref: `tasks/counterpart_evidence_tasks.jsonl#llamacpp_newer_echo_logprobs`
   - issue: `#8`
   - why: a newer llama.cpp pin may expose supplied-token logprobs, but must pass the
     contract audit.
7. `llamacpp_larger_qwen3_gguf`
   - task ref: `tasks/counterpart_evidence_tasks.jsonl#llamacpp_larger_qwen_gguf`
   - issue: `#17/#20`
   - why: the current llama.cpp Qwen speed row is only Qwen2.5 1.5B.
8. `qwen_speed_lane_shared_manifest`
   - packet: `tasks/qwen_speed_lane_sample.jsonl`
   - issue: `#20`
   - why: record already-running Qwen servers in one shared manifest shape.

## Do Not Run Yet

Do not run these as queue items unless their blockers change:

- SGLang Gemma serving rows: wait until SGLang Qwen FP4-KV quality is green.
- SGLang DFlash/EAGLE rows: wait until ordinary Qwen/SGLang quality is stable.
- vLLM Qwen NVFP4-KV capacity row from `counterpart_evidence_tasks.jsonl`: capacity is
  already recorded; the next Qwen KV task is quality at longer context.
- vLLM Qwen36 DFlash reproduction from `counterpart_evidence_tasks.jsonl`: AEON,
  derived, and clean-FA2 rows already exist; native FP4 weight/MoE proof is separate.

Completed/superseded diagnostics are tracked in `docs/COMPATIBILITY_BOARD.md` and should
not be repeated unless a new patch changes the hypothesis.

## After A Live Run

1. Update the relevant runtime doc and `docs/SOLUTIONS_STATUS.md`.
2. Regenerate audits:

```bash
python3 scripts/live_task_queue_audit.py \
  --queue tasks/live_gb10_queue.jsonl \
  --output results/live_task_queue_audit_YYYYMMDDTHHMMJST.json

python3 scripts/solution_coverage_audit.py \
  --output results/solution_coverage_audit_YYYYMMDDTHHMMJST.json
```

3. Commit code/docs/artifacts.
4. Push to `git@github.com:jethac/dgx-spark-hijinks.git`.
5. Comment on the issue named by the queue row.

## Claim Discipline

- Capacity is not speed.
- Dispatch is not tile viability.
- A build flag is not runtime dispatch.
- Top-N logprobs are not supplied-token loglikelihood unless every supplied token is
  present.
- No SGLang Gemma climb until Qwen FP4-KV quality is green.
- No upstream PRs until there is a matched before/after GB10 story.
