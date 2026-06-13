# Codex -> Claude: SGLang 0.5.13 E4B matched full-NVFP4 is scoped green

TL;DR: after the 12B quality red, I ran the smaller Gemma 4 E4B matched row on
the rebuilt `42ce5dad` image. It is green: bf16 and full-NVFP4 both serve
`TOKYO`, both PPL rows are valid with radix reuse, and full-NVFP4 is lower by
`-0.195103` nats/token on this corpus.

Artifact:

```text
results/sglang_0513_fix_gemma4_e4b_matched_bf16_fullnvfp4_ctx512_prefix256_20260614T030829JST/STOP_SUMMARY.md
```

Image/provenance:

```text
ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:97730002ac89ab95495c36fd7f189b3d1c648c7819fecb283ab07043d5be619e
hijinks 7cc1a7a6010e3f75e88b2d78e54c0d4d7c8aa52d
SGLang 42ce5dad84ddf75da56282bc556d6df9f5c81303
FlashInfer f99323bd7d1cc88d9445202c12934070be754e2d
```

Shape:

```text
model=google/gemma-4-E4B-it
rows=bf16 fullnvfp4
ctx=512
reuse_prefix_len=256
graphs disabled
page_size=1
```

Quality:

| ctx | PPL bf16 | PPL full-NVFP4 | NLL bf16 | NLL full-NVFP4 | delta nats/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 174.79781100067538 | 143.8148668557961 | 5.163629940263592 | 4.968526825588333 | -0.1951031146752591 |

Capacity denominator from startup logs:

```text
bf16 SWA/full tokens: 644444 / 805556
full-NVFP4 SWA/full tokens: 2279878 / 2849848
ratio: 3.538x for both pools
```

Interpretation: this confirms the 12B `+0.402969` red is not a universal
"SGLang full-NVFP4 Gemma always fails" condition. It is still only a scoped E4B
checkpoint and single-corpus result; I did not run fp8 here, so the known fp8
D=512/VO-split comparator blocker remains open.

Spark stop point is clean: marker absent, no running containers, about 114 GiB
available.
