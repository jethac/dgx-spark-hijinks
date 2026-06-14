# SGLang Gemma 4 12B full-NVFP4 fixed literal scale diagnostic

Date: 2026-06-14 JST

Scope: diagnostic only, not claim-grade. This tests Claude mail 0153's vLLM global-scale hypothesis directly in SGLang by overriding the FP4 KV decode global scale to a fixed literal `0.1`.

## Runtime

- Host: DGX Spark / GB10 `thinkstationpgx-00b4`
- Repo: `dgx-spark-hijinks@2e9ab6556c7764209eeb520f1b0592ab3986d9a4`
- SGLang overlay: `jethac/sglang@794c8517a92010245fbe394294ad31743603825f`
- FlashInfer overlay: `jethac/flashinfer@3fa0775cafaf88da5e0fc3b987afa6bd75d9510c`
- Image: `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0bacd437f9917928a9bd7ba0dafbb37516f8e05b4b9727bbff796556c2cc7714`
- Model: `google/gemma-4-12B-it`
- Attention: FlashInfer VO-split, page size 1
- KV row: full NVFP4 K+V, `--kv-cache-dtype fp4_e2m1`
- Context: `8185`, reused prefix `4096`, scored continuation tokens `4088`
- Memory guardrail: single server, Docker `--memory 100g`, `--mem-fraction-static 0.72`

The in-container submodule git-rev probe did not resolve because the source overlay mounted a submodule `.git` file without the parent `.git/modules` tree. The Spark checkout was explicitly updated to the refs above before launch.

## Scale convention

SGLang's FP4 KV dequant convention is:

```text
x_bf16 ~= x_fp4 * block_scale * global_scale
```

The existing autocalibration computes:

```text
global_scale = amax / (E2M1_MAX * MAX_BLOCK_SCALE_FP8) * multiplier
             = amax / (6 * 448) * multiplier
             = amax / 2688 * multiplier
```

For the observed layer-0 K `amax=1.602`, the default SGLang global scale is about `0.000596`; the fixed literal `0.1` is about `168x` larger. This is why the earlier multiplier-only sweep did not actually test Claude's vLLM literal-scale regime.

## Results

| row | scale setting | NLL | PPL | delta vs bf16 |
| --- | --- | ---: | ---: | ---: |
| bf16 baseline | n/a | 4.571989822602 | 96.736406679507 | n/a |
| full-NVFP4 | default/source-overlay plateau | 4.974958672257 | 144.742241533287 | +0.402968849654 |
| full-NVFP4 | `SGLANG_FP4_KV_GLOBAL_SCALE_MULTIPLIER=2.0` | 4.915603363181 | 136.401584673245 | +0.343613540579 |
| full-NVFP4 | `SGLANG_FP4_KV_FIXED_GLOBAL_SCALE=0.1` | 4.637407471576 | 103.276253589820 | +0.065417648974 |

Chat smoke on the fixed-literal row returned `Tokyo` twice with radix reuse enabled. The row is not claim-grade because it used a source overlay and only ran `fullnvfp4`, not matched bf16/fp8/full-NVFP4 comparators across the ladder.

## Proof Lines

The server log confirms the fixed literal scale was active and handed to FlashInfer:

```text
NVFP4 KV auto-calibrated layer 0: k_amax=1.602 v_amax=15.88 k_gs=0.1 v_gs=0.1 ... k_fixed_global_scale=0.1 v_fixed_global_scale=0.1
FP4 KV FlashInfer module trace ... deswizzle_macro_active=False ... k_scale=0.10000000149011612 v_scale=0.10000000149011612
```

The global and VO-split layer traces both show the same fixed `k_scale`/`v_scale`, with SGLang's linear V scale-factor layout (`deswizzle_macro_active=False`).

## Artifacts

- `results/sglang_gemma4_12b_fullnvfp4_fixedgs010_ctx8185_prefix4096_20260614T231852JST/`
- Earlier multiplier sweep: `results/sglang_gemma4_12b_fullnvfp4_scale_diagnostic_20260614T2245JST.md`

## Interpretation

This supports the global-scale hypothesis. The bug is not fixed by changing SGLang's existing multiplier by a small factor; it is fixed substantially by moving to the same literal global-scale regime that vLLM uses.

The next implementation step is to replace the diagnostic env override with a real SGLang policy for FP4 KV global scale, then rerun the packaged-image matched ladder. The diagnostic result is strong enough to justify that code path, but it is not itself a serving claim.
