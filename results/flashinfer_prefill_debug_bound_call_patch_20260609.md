# FlashInfer Prefill Debug Bound-Call Patch

Date: 2026-06-09 JST

## Patch

- FlashInfer fork: `jethac/flashinfer`
- Branch: `spark/hijinks-021-prefill-debug`
- Commit: `96be2fa8` (`Bind prefill debug tensor logs to module calls`)
- Campaign packet: `tasks/vllm_gemma3_flashinfer_prefill_debug_packet_20260609.md`

## What Changed

The FlashInfer debug patch now emits `call_id`, `module_uri`, `module_key`, and `path` on
both the paged-prefill identity line and the tensor dump line. The campaign audit requires
those fields to match before accepting tensor geometry as evidence for a generated module.

## Local Validation

- `python -m py_compile flashinfer/jit/attention/modules.py` passed in the FlashInfer fork.
- `git diff --check` passed in the FlashInfer fork.
- `python -m py_compile scripts/flashinfer_prefill_debug_log_audit.py` passed in the
  campaign repo.
- A synthetic Gemma-shaped NHD FP4 paged-prefill log with matching
  `(path, call_id, module_uri, module_key)` passed the audit.
- The same synthetic log with a mismatched tensor `module_uri` failed with:
  `no paged identity line has a tensor dump with matching path/call_id/module_uri/module_key`.
- `python scripts/live_task_queue_audit.py --queue tasks/live_gb10_queue.jsonl --output results/live_task_queue_audit_20260609.json`
  passed after the tailnet recovery update.
- `python scripts/solution_coverage_audit.py --output results/solution_coverage_audit_20260609.json`
  passed.
- `git diff --check` passed in the campaign repo.

## Live Status

Host access has recovered; see `results/gb10_host_access_tailnet_recovered_20260609.md`.
The next live action is to run the Gemma 3 vLLM FlashInfer prefill debug packet on GB10.
