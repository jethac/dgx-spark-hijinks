# SGLang 0.5.13 Gemma 4 12B Full-NVFP4 Rerun

Status: helper-crash fixed; serving/routing green; quality still RED against the
banked bf16 baseline.

## Scope

- Runtime: SGLang on DGX Spark, packaged Ubuntu 22.04 / arm64 / torch 2.11 image
- Image:
  `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:97730002ac89ab95495c36fd7f189b3d1c648c7819fecb283ab07043d5be619e`
- Parent hijinks ref: `055c5ff21983c5ab844680fc6268283a20607a1d`
- SGLang ref: `42ce5dad84ddf75da56282bc556d6df9f5c81303`
- FlashInfer ref: `f99323bd7d1cc88d9445202c12934070be754e2d`
- Model: `google/gemma-4-12B-it`
- Row label: `fullnvfp4`
- Shape: ctx `8185`, reused prefix `4096`, page size `1`, graphs disabled
- KV dtype: full NVFP4 K+V (`fp4_e2m1`, `mixed_kv=false`)
- Memory guardrail: one server, Docker `--memory 100g`

This reruns the full-NVFP4-only discriminator after the SGLang fix for:

```text
NameError: name '_fp4_kv_module_trace_enabled' is not defined
```

## Result

The previous packaged-code blocker is closed:

- server reaches readiness
- full-NVFP4 SWA/full pools allocate
- D=512 global layers route through the VO-split FlashInfer wrapper path
- module trace shows `deswizzle_macro_active=False`
- chat returns `Tokyo` / `Tokyo`
- supplied-token PPL request reports `cached_tokens=4096`

Quality remains red against the banked bf16 baseline from
`results/sglang_0513_live_validation_20260614T011248JST/`:

| ctx | PPL bf16 | PPL full-NVFP4 | delta PPL | NLL bf16 | NLL full-NVFP4 | delta nats/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8185 | 96.7364066795068 | 144.74289616094566 | 48.00648948143886 | 4.571989822602299 | 4.974959038640488 | 0.4029692160381897 |

The script's `ppl_ok=true` means the logprob transport/scoring row was internally
well-formed: no missing or mismatched supplied-token positions. It is not a
quality-parity verdict.

## Key Runtime Evidence

Cache allocation:

```text
KV Cache is allocated. dtype: torch.float4_e2m1fn_x2, #tokens: 612620, K size: 26.29 GB, V size: 26.29 GB
KV Cache is allocated. dtype: torch.float4_e2m1fn_x2, #tokens: 765776, K size: 1.64 GB, V size: 1.64 GB
SWAKVPool mem usage: 55.87 GB, swa size: 612620, full size: 765776
```

VO-split / FP4 read path proof:

```text
SGLang FlashInfer VO split enabled: D=512 paged prefill and decode-as-prefill use two D_VO=256 passes.
FP4 KV FlashInfer module trace label=extend_paged_vosplit0 ... deswizzle_macro_active=False
FP4 KV FlashInfer module trace label=decode_as_prefill_vosplit0 ... deswizzle_macro_active=False
```

PPL result:

```text
cached_tokens=4096
num_scored_tokens=4088
mean_nll_nats=4.974959038640488
ppl=144.74289616094566
```

## Interpretation

The rebuilt `42ce5dad` image fixes the SGLang-side trace-helper crash and proves
the 0.5.13 source-stack can execute the full-NVFP4 Gemma 4 12B radix-reuse row on
Spark. It does not improve the known 12B full-NVFP4 quality delta.

This remains non-claim-grade for broad SGLang NVFP4-KV support. It is useful as a
narrow discriminator: the current blocker is quality, not startup, image
packaging, D=512 VO-split dispatch, or the missing helper.

## Host State

At stop point:

- marker: absent
- Docker: no running containers
- memory: about 115 GiB available

## Key Artifacts

- `manifest.json`
- `google-gemma-4-12b-it/fullnvfp4_ppl.json`
- `google-gemma-4-12b-it/fullnvfp4_summary.json`
- `google-gemma-4-12b-it/fullnvfp4_server.log`
- `google-gemma-4-12b-it/fullnvfp4_provenance.log`
- `docker_ps_after.txt`
- `free_after.txt`
