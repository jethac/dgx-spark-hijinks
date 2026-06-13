# SGLang Gemma 4 12B Matched bf16 vs Full-NVFP4

Status: RED for claim-grade quality. The matched delta is too large:
`+0.4029692160381897` nats/token.

## Scope

- Runtime: SGLang on DGX Spark, packaged Ubuntu 22.04 / arm64 / torch 2.11 image
- Image:
  `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0d5e160cf83db43e1e024a8300ed2858b426b4a0f38289210dc51d8c7b6def94`
- Commit: `5c2947e2c11ad55fcd1a7be7a34af9d2a2b2d2b8`
- Model: `google/gemma-4-12B-it`
- Row labels: `bf16 fullnvfp4`
- Shape: ctx `8185`, reused prefix `4096`, page size `1`, graphs disabled
- Corpus: one generated `ppl_corpus.md` shared by both arms in this run
- Memory guardrail: one server at a time, Docker `--memory 100g`

This row applies the matched-claim rule from
`docs/SHIP_GATE_SGLANG_GEMMA4_LADDER_PLAN.md`: same image, same commit, same
corpus bytes, same shape, same graph state, and only the KV dtype changes.

## Result

Both arms are transport/serving green:

- bf16 chat: `Tokyo` / `Tokyo`
- full-NVFP4 chat: `Tokyo` / `Tokyo`
- bf16 PPL: pass, `cached_tokens=4096`
- full-NVFP4 PPL: pass, `cached_tokens=4096`

The matched quality delta is red:

| ctx | PPL bf16 | PPL full-NVFP4 | delta PPL | NLL bf16 | NLL full-NVFP4 | delta nats/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8185 | 96.7364066795068 | 144.74289616094566 | 48.00648948143886 | 4.571989822602299 | 4.974959038640488 | 0.4029692160381897 |

This confirms the earlier cross-artifact warning: the large delta was not a
contaminated artifact mismatch. It reproduces in a single matched packaged-image
run.

## Interpretation

The SGLang Gemma 4 12B full-NVFP4 path is not claim-grade at this shape. The
serving/routing path works, but the quality delta is too large for a public
NVFP4-KV parity claim.

This row is separate from the fp8 comparator dispatcher red:

- fp8 is still blocked by the FlashInfer D=512/VO-split dispatcher issue recorded
  in `results/sglang_gemma4_12b_ar_claim_ctx8185_prefix4096_20260613T105511JST/STOP_SUMMARY.md`.
- full-NVFP4 does not hit that dispatcher failure, but it is quality-red against
  bf16 on the matched PPL gate.

## Host State

At stop point:

- marker: absent
- Docker: no running containers
- memory: ~115 GiB available

## Key Artifacts

- `google-gemma-4-12b-it/bf16_ppl.json`
- `google-gemma-4-12b-it/fullnvfp4_ppl.json`
- `google-gemma-4-12b-it/compare_bf16_vs_fullnvfp4.json`
- `google-gemma-4-12b-it/bf16_server.log`
- `google-gemma-4-12b-it/fullnvfp4_server.log`
- `manifest.json`
- `../sglang_gemma4_12b_ar_matched_bf16_fullnvfp4_ctx8185_prefix4096_20260613T153712JST_runner.log`
