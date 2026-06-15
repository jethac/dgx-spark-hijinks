# Epoch-3 rebase status (vLLM v0.23.0 + FlashInfer main)

Triggered 2026-06-15: vLLM v0.23.0 dropped. Identity for all our commits = `Jetha Chan <jethachan@gmail.com>`.

## FlashInfer -> flashinfer-ai/flashinfer `main` (DONE, source side)
- Branch `jethac/flashinfer@spark/hijinks-e3-flashinfer`, worktree `B:/workshop/worktrees/flashinfer/spark-hijinks-e3-flashinfer`.
- Base: upstream/main `c15ac84c`. Cherry-picked all **20** Jetha commits from `spark/fp8-d512-clean-reject`
  (the canonical superset branch) — **clean, no real conflicts** (our nvfp4-KV kernel work is additive vs main).
- Verified: FA2_REJECT macro + nvfp4 prefill plumbing present, 0 markers. (User chose `main` over the
  0.6.12 tag — library moves fast; we build FlashInfer from source via PYTHONPATH so the pin is irrelevant.)

## vLLM -> v0.23.0 (DONE, source side)
- Branch `jethac/vllm@spark/hijinks-e3-vllm`, worktree `B:/workshop/worktrees/vllm/spark-hijinks-e3-vllm`.
- Base: v0.23.0 (`91df0fad4`). Strategy: **cherry-pick (hybrid)**.
- Phase 1: cherry-picked **34** of the 49-commit carry-set (union of Jetha commits across the parallel
  spark/hijinks-e2-* branches; duplicates/empties auto-skipped). Conflict policy: additive auto-resolved
  (`resolve_additive.py`); our-code files take-theirs (`take_theirs.py` — config.py, gemma4.py,
  model_states, our tests/utils); **flashinfer.py deferred**.
- Phase 2: ported `vllm/v1/attention/backends/flashinfer.py` as a single 3-way merge (v0.23.0 is near-
  identical to v0.22.1, so our 2194-line nvfp4/VO-split/mm-prefix addition merged with **1 conflict** —
  the SM12x-vs-sm100f nvfp4 routing, resolved to our superset).
- Verified: 0 conflict markers in vllm/csrc/tests; key touched files py_compile clean. 35 commits over v0.23.0.

## Remaining
1. **Build + test validation** on a vast sm_120 box: build vLLM e3 + FlashInfer e3 from source, re-run the
   12B/31B matched green ladder + 26B re-confirm. (Branches must be pushed to the forks first.)
2. Converge the FlashInfer pin with Codex's SGLang rebase (one ref: `spark/hijinks-e3-flashinfer`).
3. Resume the paused 26B read-capture on the fresh stack.

Tooling: `docs/vast_anchor/{resolve_additive,take_theirs}.py`.

## VALIDATED (2026-06-15) — both rebases serve green
Built the e3 wheel on the Ubicloud build machine (hijinks-build-x64, compile needs no GPU) and serve-tested
on a vast sm_120 box with FlashInfer-e3 (JIT). Two runtime API-drift issues surfaced (compile couldn't catch
them) and were fixed (commit e99078ddf):
1. `FlashInferBackend.supports_combination()` — make `device_capability` optional (v0.23.0's caller omits it).
2. `CommonAttentionMetadata.mm_req_doc_ranges` — re-add the field def (dropped in cherry-pick; flashinfer.py
   reference survived).

Matched green ladder on e3 (v0.23.0 + FlashInfer-main), BITWISE-IDENTICAL to pre-rebase:
| model | bf16 | nvfp4 (calibrated) | delta |
| --- | ---: | ---: | ---: |
| 12B | 8.2764 | 8.3003 (k=0.1,v=0.06) | +0.024 |
| 31B | 9.1789 | 9.1693 (k=v=0.05) | -0.010 |

The epoch-3 rebase is DONE and validated. Wheel rebuilding with the 2 fixes (warm ccache). FlashInfer ref
for SGLang to pin: `jethac/flashinfer@spark/hijinks-e3-flashinfer` (main + 20 nvfp4-KV commits).
