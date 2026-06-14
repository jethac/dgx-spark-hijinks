# SGLang 12B Recovery Sweep Diagnostic

Date: 2026-06-14 JST

Status: diagnostic red. No tested SGLang chunk/radix configuration recovered the
Gemma 4 12B ctx-8185 full-NVFP4 row toward the mail-0140 `~+0.19`
nats/token reference.

## Scope

- Request source: `mail/0143_claude-to-codex_characterize-sglang-recovery-curve-while-i-pin-the-kernel.md`
- Host: DGX Spark / GB10, sm_121
- Repo: clean Spark worktree at `dgx-spark-hijinks` `epoch2`
  `6fad98dedd976354887e64caa55d7069fd855266`
- Image:
  `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0bacd437f9917928a9bd7ba0dafbb37516f8e05b4b9727bbff796556c2cc7714`
- SGLang source ref in worktree: `f920e2d88af68031b745494f5435efb71ac93562`
- FlashInfer source ref in worktree: `f99323bd7d1cc88d9445202c12934070be754e2d`
- Model: `google/gemma-4-12B-it`
- Shape: ctx `8185`, reused prefix `4096`, scored suffix `4088` tokens,
  page size `1`
- KV dtype: full NVFP4 K+V (`fp4_e2m1`)
- Memory guardrail: one server at a time, Docker `--memory 100g`,
  `mem_fraction_static=0.72`
- Claim scope: single-arm diagnostic only, not a claim-grade matched ladder row

Baseline used for serving deltas:
`results/sglang_gemma4_12b_ar_matched_bf16_fullnvfp4_ctx8185_prefix4096_20260613T153712JST/STOP_SUMMARY.md`
reported bf16 NLL `4.571989822602`.

## Serving Recovery Curve

| row | radix | chunked prefill size | cached tokens | mean NLL | PPL | Delta vs bf16 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| old matched full-NVFP4 red | on | default | 4096 | 4.974959038640 | 144.742896 | +0.402969216038 |
| prior diagnostic | on | 2048 | 4096 | 4.926853552137 | 137.944793 | +0.354863729534 |
| new diagnostic | on | 1024 | 4096 | 5.050551528617 | 156.108539 | +0.478561706015 |
| new diagnostic | off | 1024 | 0 | 5.050551528617 | 156.108539 | +0.478561706015 |
| new diagnostic | on | 512 | 4096 | 5.055428890499 | 156.871797 | +0.483439067897 |
| new diagnostic | off | 512 | 0 | 5.055428890499 | 156.871797 | +0.483439067897 |

All four new rows reached readiness, returned `Tokyo`/`Tokyo` on the chat smoke,
and completed supplied-token PPL with `missing=0`, `mismatched=0`.

Interpretation:

- No tested SGLang config reaches the `~+0.19` target.
- Reducing `--chunked-prefill-size` below 2048 worsened the row rather than
  improving it.
- Radix cache on/off changes the cache-hit count (`4096` vs `0`) but not the
  NLL for the tested chunk sizes, so this diagnostic does not point at radix
  reuse as the driver.
- The SGLang ladder remains blocked pending the FlashInfer FA2/NVFP4
  accumulation fix or a different, explicitly scoped avoidance route.

## Exact HF SDPA Reference Check

Mail 0143 also asked to confirm the `~+0.19` floor on the SGLang stack using
`docs/vast_anchor/refsim_longctx.py`.

The first attempt failed because the packaged SGLang image does not include
`accelerate`, which `device_map="cuda"` requires:

- `results/sglang_12b_refsim_longctx_20260614T130509JST/refsim_stderr.log`

The rerun installed `accelerate` inside the ephemeral container and completed:

| environment | bf16 NLL | NVFP4-q/dq NLL | Delta |
| --- | ---: | ---: | ---: |
| prior vast.ai reference log in `docs/vast_anchor/pfx_results/refsim_longctx_ctx8185.log` | 8.3551 | 8.5483 | +0.1932 |
| Spark packaged SGLang image rerun | 8.3357 | 9.0306 | +0.6949 |

This does **not** confirm the `+0.19` floor on the Spark packaged SGLang stack.
Because the bf16 side is close to the prior log but the NVFP4 q/dq side is much
worse, treat this as an environment-sensitive exact-reference discrepancy until
discriminated. It should not be used to revise the serving interpretation by
itself.

## Artifacts

- Run status: `results/sglang_12b_recovery_sweep_20260614T124138JST_RUN_STATUS.tsv`
- c1024 radix on:
  `results/sglang_12b_recovery_c1024_radixon_20260614T124138JST/`
- c1024 radix off:
  `results/sglang_12b_recovery_c1024_radixoff_20260614T124138JST/`
- c512 radix on:
  `results/sglang_12b_recovery_c512_radixon_20260614T124138JST/`
- c512 radix off:
  `results/sglang_12b_recovery_c512_radixoff_20260614T124138JST/`
- Failed first refsim attempt:
  `results/sglang_12b_refsim_longctx_20260614T130509JST/`
- Completed refsim attempt:
  `results/sglang_12b_refsim_longctx_accel_20260614T130750JST/`

At stop point the Spark marker was absent and no live Docker containers remained.
