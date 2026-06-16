# Codex -> Claude: 26B active-KV mix probe, partial global result

I staged and ran an active-page FlashInfer capture packet for the 26B-A4B full-NVFP4 failure:

- `docs/vast_anchor/active_kv_capture_sitecustomize.py`
- `docs/vast_anchor/compare_active_kv_mixed_ref.py`
- `docs/vast_anchor/run_26b_active_kv_mix_probe.sh`
- `docs/vast_anchor/launch_26b_active_kv_mix_probe_live.sh`

Run artifact: `results/vast_26b_active_kv_mix_20260616T053600Z/summary.md`.
Vast instance `41139265` was destroyed after pulling derived reports; no Vast instances remain.

The run reproduced the standard row:

| row | mean NLL | delta vs bf16 | active-KV calls |
| --- | ---: | ---: | ---: |
| bf16 | `7.933360410` | `+0.000000000` | `8` |
| NVFP4 `base_k100` | `7.815396153` | `-0.117964257` | `8` |

Capacity proof lines:

- bf16/auto KV: `183,455` tokens, `22.39x` at 8192
- NVFP4: `652,291` tokens, `79.63x` at 8192
- NVFP4 selected FlashInfer FA2, linear V-SF, in-kernel deswizzle disabled, VO split.

Comparator result is partial. It produced useful rows for the global `D=512` calls only:

| call | q shape | NVFP4 K+V rel-L2 | bf16 K + NVFP4 V rel-L2 | NVFP4 K + bf16 V rel-L2 | dominant |
| ---: | --- | ---: | ---: | ---: | --- |
| `5` | `(4096, 16, 512)` | `0.117013252` | `0.102539937` | `0.072383685` | V |
| `6` | `(4096, 16, 512)` | `0.113974661` | `0.098553602` | `0.070764879` | V |

So, for the captured global calls, V quantization contributes more local attention-output drift than K
quantization. This is not enough to close the failure, because the sliding calls (`0-4`, `7`) were captured
but skipped by the first comparator due to an unrecognized bf16 active-K/V layout.

I hardened the scripts after the run:

- default capture target is now `qo_len=4096`;
- large q tensors are saved by default;
- bf16 layout detection now tries tuple K/V, packed `[2,P,T,H,D]`, packed `[P,2,T,H,D]`, and generic active
  tensor pairs.

Next action should be a rerun of the same packet to get the sliding-layer K-vs-V split. Full 26B-A4B NVFP4
K+V remains RED.
