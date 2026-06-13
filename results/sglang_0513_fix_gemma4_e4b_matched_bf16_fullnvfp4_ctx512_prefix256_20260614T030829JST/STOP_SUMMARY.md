# SGLang 0.5.13 Gemma 4 E4B Matched bf16 vs Full-NVFP4

Status: GREEN scoped E4B checkpoint on the rebuilt SGLang 0.5.13 image.

## Scope

- Runtime: SGLang on DGX Spark, packaged Ubuntu 22.04 / arm64 / torch 2.11 image
- Image:
  `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:97730002ac89ab95495c36fd7f189b3d1c648c7819fecb283ab07043d5be619e`
- Parent hijinks ref: `7cc1a7a6010e3f75e88b2d78e54c0d4d7c8aa52d`
- SGLang ref: `42ce5dad84ddf75da56282bc556d6df9f5c81303`
- FlashInfer ref: `f99323bd7d1cc88d9445202c12934070be754e2d`
- Model: `google/gemma-4-E4B-it`
- Rows: `bf16`, `fullnvfp4`
- Shape: ctx `512`, reused prefix `256`, page size `1`, graphs disabled
- Corpus: one generated `ppl_corpus.md` shared by both arms in this run
- Memory guardrail: one server at a time, Docker `--memory 100g`

This row is a small matched discriminator after the 12B quality-red result. It
does not prove broad SGLang NVFP4-KV support, but it shows the fixed 0.5.13
image can serve a Gemma 4 E4B full-NVFP4 radix-reuse PPL row cleanly.

## Result

Both arms are transport/serving green:

- bf16 chat: `TOKYO` / `TOKYO`
- full-NVFP4 chat: `TOKYO` / `TOKYO`
- bf16 PPL: pass, `cached_tokens=256`
- full-NVFP4 PPL: pass, `cached_tokens=256`

Matched quality delta:

| ctx | PPL bf16 | PPL full-NVFP4 | delta PPL | NLL bf16 | NLL full-NVFP4 | delta nats/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 174.79781100067538 | 143.8148668557961 | -30.98294414487927 | 5.163629940263592 | 4.968526825588333 | -0.1951031146752591 |

The negative delta is a single-corpus checkpoint, not a claim that NVFP4 is
generally better. It is useful evidence that the 12B `+0.402969` red is not a
universal "all SGLang full-NVFP4 Gemma rows fail" condition.

## Capacity

The row did not run a max-concurrency search. The allocator token counts show
the expected full-NVFP4 denominator improvement versus bf16 for this E4B shape:

| pool | bf16 tokens | full-NVFP4 tokens | ratio |
| --- | ---: | ---: | ---: |
| SWA | 644444 | 2279878 | 3.537744 |
| full | 805556 | 2849848 | 3.537740 |

This is a capacity-denominator observation from startup logs, not a throughput
or maximum-live-request benchmark.

## Runtime Evidence

bf16 pool:

```text
KV Cache is allocated. dtype: torch.bfloat16, #tokens: 644444, K size: 21.51 GB, V size: 21.51 GB
KV Cache is allocated. dtype: torch.bfloat16, #tokens: 805556, K size: 10.76 GB, V size: 10.76 GB
```

full-NVFP4 pool:

```text
KV Cache is allocated. dtype: torch.float4_e2m1fn_x2, #tokens: 2279878, K size: 21.40 GB, V size: 21.40 GB
KV Cache is allocated. dtype: torch.float4_e2m1fn_x2, #tokens: 2849848, K size: 10.70 GB, V size: 10.70 GB
```

VO-split route:

```text
SGLang FlashInfer VO split enabled: D=512 paged prefill and decode-as-prefill use two D_VO=256 passes.
SGLang FlashInfer wrapper geometries dispatch=WrapperDispatch.SLIDING_WINDOW geometries=[FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=256, head_dim_vo=256), FlashInferWrapperGeometry(num_qo_heads=8, num_kv_heads=2, head_dim=512, head_dim_vo=256)]
```

## Interpretation

The rebuilt `42ce5dad` SGLang 0.5.13 image has a scoped green E4B full-NVFP4
Gemma 4 row on Spark. This should be cited separately from the 12B full-NVFP4
quality-red row. The next broader claim still needs larger rungs and the fp8
comparator caveat remains open.

## Host State

At stop point:

- marker: absent
- Docker: no running containers
- memory: about 114 GiB available

## Key Artifacts

- `manifest.json`
- `google-gemma-4-e4b-it/bf16_ppl.json`
- `google-gemma-4-e4b-it/fullnvfp4_ppl.json`
- `google-gemma-4-e4b-it/compare_bf16_vs_fullnvfp4.json`
- `google-gemma-4-e4b-it/bf16_server.log`
- `google-gemma-4-e4b-it/fullnvfp4_server.log`
- `docker_ps_after.txt`
- `free_after.txt`
