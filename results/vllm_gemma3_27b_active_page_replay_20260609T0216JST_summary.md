# vLLM Gemma 3 27B Active-Page Replay, 2026-06-09

## Purpose

Replay the dumped Gemma 3 27B layer-5 FlashInfer paged-prefill calls against a CPU
dequantized attention reference. This follows
`results/vllm_gemma3_27b_active_page_dump_20260609T0216JST_summary.md`, which showed the
real wrapper returned byte-like BF16 output matching active packed V data bytes.

This is diagnostic-only, not a performance row.

## Method

Script: `scripts/vllm_active_page_replay.py`

For each `spark-active-page-prefill-dump/v1` payload:

1. Load `query`, `paged_kv_indices`, `paged_kv_last_page_len`, active K/V data pages, and
   active K/V FP8 scale pages.
2. Dequantize packed E2M1 K/V data using the FP8 scale buffers and global
   `k_scale` / `v_scale`.
3. Gather the paged sequence in wrapper order.
4. Run a CPU causal attention reference with Gemma 3 geometry (`32` Q heads, `16` KV
   heads, `D=128`) and GQA mapping.
5. Compare the reference with the dumped `out_after`.

Three variants were run:

- no V-scale deswizzle, no logits softcap;
- V-scale deswizzle, no logits softcap;
- V-scale deswizzle, logits softcap `50`.

The dump did not store `sm_scale` or `logits_soft_cap`, so the replay uses
`head_dim ** -0.5` for `sm_scale` and records the softcap assumption per artifact.

## Result

The two useful request dumps (`0002.pt`, `0003.pt`) are incompatible with a normal
dequantized attention result in every replay variant.

Best representative row: V-scale deswizzle enabled, logits softcap `50`.

| dump | tokens | reference mean | reference rms | `out_after` mean | `out_after` rms | cosine vs reference | mean abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0002.pt` | 18 | `-0.0286` | `1.9071` | `128.5285` | `147.4429` | `-0.00010` | `128.5625` |
| `0003.pt` | 23 | `-0.0309` | `1.9947` | `129.3900` | `148.0711` | `0.00241` | `129.4264` |

The replay reference head is signed and small:

```text
[0.0, -4.875, 0.40625, 0.0, 0.8125, -3.25, -0.40625, 0.0,
 0.40625, -0.40625, 1.625, 0.40625, -0.8125, -1.21875,
 -0.40625, -1.21875]
```

The real wrapper output head remains byte-like:

```text
[240.0, 1.0, 226.0, 137.0, 145.0, 20.0, 186.0, 185.0,
 33.0, 65.0, 47.0, 233.0, 91.0, 34.0, 145.0, 25.0]
```

The same conclusion holds without softcap and without V-scale deswizzle: the reference
mean remains near zero, RMS around `1.9..2.0`, and cosine against `out_after` stays near
zero.

## Conclusion

The Gemma 3 NVFP4-KV failure is now localized below vLLM page/scale pairing and below
generic dequantized attention math:

- exact active pages were captured from the failing wrapper call;
- CPU dequantization of those active pages produces sane signed K/V values;
- a causal attention reference produces sane signed output;
- the real FlashInfer paged prefill wrapper returns byte-range BF16 values instead;
- changing the V-scale deswizzle assumption or applying logits softcap does not explain
  the byte-like output.

The most likely next code audit is inside FlashInfer's paged-prefill NVFP4 path: verify
whether the FA2 prefill specialization is treating the packed uint8 V carrier as output
values, using the wrong V view/type, or binding the wrong template parameter for V element
conversion in the paged prefill wrapper.

## Artifacts

- `scripts/vllm_active_page_replay.py`
- `results/vllm_gemma3_27b_active_page_replay_20260609T0216JST_no_softcap.json`
- `results/vllm_gemma3_27b_active_page_replay_20260609T0216JST_no_softcap.md`
- `results/vllm_gemma3_27b_active_page_replay_20260609T0216JST_deswizzle_no_softcap.json`
- `results/vllm_gemma3_27b_active_page_replay_20260609T0216JST_deswizzle_no_softcap.md`
- `results/vllm_gemma3_27b_active_page_replay_20260609T0216JST_deswizzle_softcap50.json`
- `results/vllm_gemma3_27b_active_page_replay_20260609T0216JST_deswizzle_softcap50.md`
