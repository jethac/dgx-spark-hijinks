# Codex -> Claude: SGLang 0.5.13 fixed image closes helper crash; 12B full-NVFP4 quality still red

TL;DR: rebuilt image `sha256:97730002...` gets past the missing-helper crash and
executes the Gemma 4 12B full-NVFP4 radix-reuse row, but the quality result is
still the same `+0.402969` nats/token red against bf16.

Artifacts:

```text
results/ghrun_27473134505/summary.md
results/sglang_0513_fix_gemma4_12b_fullnvfp4_only_ctx8185_prefix4096_20260614T024920JST/STOP_SUMMARY.md
```

Image/provenance:

```text
ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:97730002ac89ab95495c36fd7f189b3d1c648c7819fecb283ab07043d5be619e
SGLang 42ce5dad84ddf75da56282bc556d6df9f5c81303
FlashInfer f99323bd7d1cc88d9445202c12934070be754e2d
torch 2.11.0 / Ubuntu 22 / arm64
```

What is closed:

```text
NameError: name '_fp4_kv_module_trace_enabled' is not defined
```

The full-NVFP4 row now reaches readiness and runs:

```text
chat Tokyo/Tokyo
cached_tokens=4096
num_scored_tokens=4088
full-NVFP4 mean_nll=4.974959038640488
full-NVFP4 PPL=144.74289616094566
```

Runtime proof points:

```text
KV Cache is allocated. dtype: torch.float4_e2m1fn_x2, #tokens: 612620, K size: 26.29 GB, V size: 26.29 GB
KV Cache is allocated. dtype: torch.float4_e2m1fn_x2, #tokens: 765776, K size: 1.64 GB, V size: 1.64 GB
SGLang FlashInfer VO split enabled: D=512 paged prefill and decode-as-prefill use two D_VO=256 passes.
FP4 KV FlashInfer module trace ... deswizzle_macro_active=False
```

Quality verdict against the banked bf16 baseline from the same 0.5.13 packet:

| ctx | PPL bf16 | PPL full-NVFP4 | NLL bf16 | NLL full-NVFP4 | delta nats/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8185 | 96.7364066795068 | 144.74289616094566 | 4.571989822602299 | 4.974959038640488 | 0.4029692160381897 |

Interpretation: this is not a package/startup/VO-split/helper failure anymore.
It is the known 12B full-NVFP4 quality blocker. The script-level `ppl_ok=true`
only means supplied-token logprob transport was well-formed; it is not a parity
gate.

Spark stop point is clean: marker absent, no running containers, about 115 GiB
available.
