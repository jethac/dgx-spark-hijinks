# SGLang Gemma 4 E4B Packaged VO-Split Smoke

Status: GREEN for the packaged-image bf16/auto VO-split decode smoke.

## Scope

- Runtime: SGLang on DGX Spark, packaged Ubuntu 22.04 / arm64 / torch 2.11 image
- Image:
  `ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack@sha256:0d5e160cf83db43e1e024a8300ed2858b426b4a0f38289210dc51d8c7b6def94`
- Commit: `5c2947e2c11ad55fcd1a7be7a34af9d2a2b2d2b8`
- Model: `google/gemma-4-E4B-it`
- Row labels: `bf16`
- KV dtype: `auto`
- Shape: ctx `512`, reused prefix `256`, page size `1`, graphs disabled
- Purpose: rerun the staged Gemma 4 VO-split route on the packaged image and
  prove D=512 global decode no longer falls into the old symmetric
  `Unsupported max_mma_kv: 0` path.

## Result

- readiness: pass
- chat smoke: `TOKYO` / `TOKYO`
- supplied-token PPL: pass
- prefix reuse proof: score response reports `cached_tokens=256`
- PPL: `174.79781100067538`
- mean NLL: `5.163629940263592` nats/token
- scored tokens: `255`

## Route Proof

The server log proves the heterogeneous Gemma 4 geometry and the packaged
VO-split route:

- `SGLANG_GEMMA_KV_POOL_CONFIG full_layers=7 swa_layers=35 ...`
- global layers use D=512 / V=512:
  `layer=5 heads=8 kv_heads=2 head_dim=512 v_head_dim=512`
- global prefill uses `extend_paged_vosplit0/1` with
  `head_dim=512, head_dim_vo=256`
- global decode uses `decode_as_prefill_vosplit0/1` with
  `head_dim=512, head_dim_vo=256`
- no `Unsupported max_mma_kv`, `Invalid configuration`, or `NUM_MMA` failure is
  present in `bf16_server.log`

This is still a smoke/route row, not a matched NVFP4 quality row.

## Host State

At stop point:

- marker: absent
- Docker: no running containers
- memory: ~115 GiB available

## Key Artifacts

- `google-gemma-4-e4b-it/bf16_ppl.json`
- `google-gemma-4-e4b-it/bf16_summary.json`
- `google-gemma-4-e4b-it/bf16_server.log`
- `manifest.json`
- `../sglang_gemma4_e4b_packaged_vosplit_bf16_smoke_20260613T153208JST_runner.log`
