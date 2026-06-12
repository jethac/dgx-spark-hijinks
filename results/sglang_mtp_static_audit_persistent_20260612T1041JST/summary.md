# SGLang Gemma 4 MTP Static Audit - Persistent Runner

Date: 2026-06-12 JST

Scope: static source audit only. No model weights, no server, no GPU runtime,
and no serving-support claim.

## Result

GREEN.

- Workflow: `hijinks-gemma4-mtp-static-audit`
- Run: <https://github.com/jethac/sglang/actions/runs/27388967875>
- SGLang branch: `spark/hijinks-024-diffusiongemma-upstream-rebase`
- SGLang commit: `542525bf3c00a5d76c12ec9290f0ccb17e8da0a7`
- Runner: `ubicloud-persistent-sglang-x64`
- Runner host proof: `vmzrp4jj`
- CPU proof: `nproc = 16`
- CUDA toolkit proof: `nvcc release 13.0, V13.0.88`
- ccache proof: `/opt/build-cache/ccache`, max `100.0 GB`

## Gates Covered

- Gemma 4 assistant classes are present and registered:
  `Gemma4AssistantForCausalLM`, `Gemma4UnifiedAssistantForCausalLM`.
- Speculative hook markers are present for assistant detection, `NEXTN`/`EAGLE`
  promotion to `FROZEN_KV_MTP`, and `EAGLE3` rejection.
- `FrozenKVMTPWorker` markers show target memory pool use and target KV pool
  binding.
- `frozen_kv_target_view()` / `target_kv_pool_view()` markers show the draft
  attention backend swaps to the target KV pool.
- Target verify / draft extend routing markers are present in the FlashInfer and
  hybrid attention backends.
- Full-NVFP4 container markers and mixed-KV split dtype markers are present in
  the FlashInfer index updaters.

## Artifacts

- `run_27388967875.json`
- `run_27388967875.log`

## Interpretation

This closes only the no-weight static/import side of the SGLang Gemma 4 MTP
recon. Runtime support remains gated by the zero-bug serving ladder:
spec-on greedy output must match spec-off greedy output token-for-token at
temperature 0 before any speedup or acceptance-rate number is claimable.
