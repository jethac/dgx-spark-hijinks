# G4-12B claim pair on r10 + first quantized cell (2026-06-12)

Same window as the order-control block (marker 06:11-08:14 JST). Runner:
`run_g412b_r10.sh`; gates: `run_r10_gates.sh` (this dir).

## The r10 image

`jethac-vllm-aeon-gemma4:9759e3b06-rebuiltc-76af7982-sm121a-r10`
(id `aed0da3f96b2`, full
`sha256:aed0da3f96b2de762916f036ea1906213ab4e4234f91d8fe9d4662c949b04248`)
= the verified final r9 (id-pinned `8c37bdbc4fdb...`) + **transformers
pinned 5.11.0** (the version the scorecard overlay proved serves
`gemma4_unified` green on vLLM 9759e3b06) + FlashInfer-cache re-scrub.
Builder: `scripts/build_vllm_gemma4_rebuiltc_r10_image.sh` (committed;
refuses to build if the r9 base id does not match the banked sha256; asserts
at build time transformers==5.11.0 imports and `gemma4_unified` is in the
transformers config mapping). Build log/summary:
`results/vllm_gemma4_rebuiltc_image_build_20260612T0620JST_r10{.log,_summary.md}`.

### Provenance gates (same set as r9) — ALL PASS

| gate | result |
|---|---|
| GPU import probe | vllm `0.1.dev1+g9759e3b06`, flashinfer `0.6.13`, torch `2.11.0+cu130`, humming `0.1.4`, **transformers `5.11.0` + `gemma4_unified` present**, GB10 CC 12.1 / 48 SMs, all 4 extension imports PASS (`r10_import_probe.txt`) |
| sm_121a cubins | `cuobjdump -lelf` on `_C.abi3.so`: sm_121a cubins present (`r10_cuobjdump_sm121.txt`) |
| linear-latch diag | verdict `writer wrote LINEAR V-SF`; linear cosine 0.9954940676689148 vs swizzled 0.9454609155654907 — identical to the r9 values (`r10_latch_diag.json`) |
| module-cache audit | `FLASHINFER_AOT_DIR` unset; no flashinfer payload in `/root/.cache` or `/tmp`; zero payload `.so` (`r10_module_cache_listing.txt`) |

Gate-harness note: the first import-probe invocation was a NO-STDIN false
green (heredoc without `docker run -i`; python read EOF, rc 0, empty
output). Caught by output inspection, re-run correctly, both attempts logged
in `status.txt`. Gates 2-4 were real on first run.

## The claim pair (google/gemma-4-12B-it, ctx 8191, util 0.72, C1 x2 bitwise)

| row | route proof | C1 (x2 bitwise) | KV tokens | decode tok/s | TTFT s | prefill tok/s | x4 tok/s | coherence |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Triton comparator (no knobs) | forced TRITON_ATTN x2, zero FI dispatch | 3.4373001938921166 | 449,488 | 7.54 | 0.995 | 2097 | 37.03 | COHERENT |
| FlashInfer (`VLLM_FLASHINFER_VOSPLIT=1`) | FLASHINFER x2 + "FA2 VO split (auto KV)", zero TRITON dispatch | 3.464887691589146 | 451,414 | 7.44 | 0.909 | 2295 | 36.72 | COHERENT |

- **R1: PASS.** delta FI−Tri = **+0.027587** nats (band +0.05).
- **Both C1 values bitwise-reproduce the 2026-06-12 scorecard dep-overlay
  pair** (r9 + in-container transformers upgrade) — the overlay was a
  faithful proxy; its numbers are now claim-grade on a baked image.
- Speed matches the overlay pair within noise (7.53/7.43 then, 7.54/7.44 now).
- `TRANSFORMERS_CHECK 5.11.0` logged by all three servers (no pip step, no
  overlay — baked).

## The quantized cell (first for this size)

| row | config | C1 (x2 bitwise) | KV tokens | capacity vs bf16 | delta vs bf16-Tri / bf16-FI | order provenance |
|---|---|---:|---:|---:|---:|---|
| nvfp4 | `--kv-cache-dtype nvfp4` + VOSPLIT + LINEAR_V_SF | 3.6834130552987028 | 1,587,074 | **3.53x** (vs 449,488) | +0.2461 / +0.2185 nats | **SCORE-FIRST, COLD** (C1 x2 before any smoke) |

Route proof: FLASHINFER-only + linear V-SF latch line + "FA2 VO split
(nvfp4 KV)". Smoke coherent (post-score).

Order/boot annotation (per today's Part A adjudication,
`results/claude_order_control_20260612/ORDER_CONTROL_SUMMARY.md`): at 31B,
nvfp4 is order- AND boot-stable (one bitwise profile across three boots and
both request orders), and the fp8-style per-boot bistability has not been
observed on any nvfp4 row. This 12B cell is nonetheless a SINGLE-BOOT value;
a second-boot repin is the cheap hardening step before this exact number
feeds a cross-window claim. Quality note, stated plainly: +0.246 nats is
inside the 0.5 RED band (not RED) but is the largest nvfp4-vs-bf16 C1 delta
in the family so far (g312b +0.074, g426b +0.129, 31B -0.332) — the
size-dependence of nvfp4 quality is real and one-directional claims
("nvfp4 ~free" or "nvfp4 better") remain wrong; 3.53x capacity at +0.25
nats is the honest 12B framing.

## VERDICT: the G4-12B open box CLOSES

The 2026-06-12 ~03:00 adjudication gated G4-12B retirement on "a
Transformers-bumped image (r10 spec) serves its paired cells green." r10 is
that image (spec met exactly: r9 recipe inherited id-pinned + transformers
5.11.0 + same provenance gates, all green) and both paired cells are green
end-to-end with bitwise C1 determinism, in-band R1 delta, coherent
transcripts, and clean R5 route proof on both sides. G4-12B joins the
retirement claim set; the size also now has its first quantized (nvfp4) KV
cell with explicit order provenance. Speed: parity (0.99x decode, TTFT -9%
on FI), consistent with every other size — the retirement case still rests
on capability + 31B quality + determinism, not speed.

## Artifacts

Per-row in `results/`: server logs, proof-line files, smoke JSONs, C1 a/b
PPL JSONs + stdout/stderr, bench JSONs (tri/fi), r10 gate artifacts
(`r10_*`). Status: `status.txt` (gates + rows).
