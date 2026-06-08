# SGLang FP4 KV Backend Trace

Date: 2026-06-08

This source-overlay trace run used `jethac/sglang@d7d931f530160ba86a2d55b4636d64baaeda3bec` on `nvcr.io/nvidia/sglang:26.05-py3` with:

```bash
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
SGLANG_FP4_KV_TRACE_BACKEND=1
PYTHONPATH=/workspace/sglang/python
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 30011 \
  --tp 1 \
  --dtype bfloat16 \
  --kv-cache-dtype fp4_e2m1 \
  --attention-backend flashinfer \
  --page-size 1 \
  --mem-fraction-static 0.40 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph
```

## Artifacts

- server log: `results/sglang_fp4_backend_trace_20260608T1536JST_server.log`
- trace excerpt: `results/sglang_fp4_backend_trace_20260608T1536JST_trace_excerpt.txt`
- raw sanity: `results/sglang_fp4_backend_trace_20260608T1536JST_raw_2plus2.json`
- chat smoke: `results/sglang_fp4_backend_trace_20260608T1536JST_chat_smoke.json`
- source commit: `results/sglang_fp4_backend_trace_20260608T1536JST_sglang_commit.txt`

## Result

- Startup reached readiness with FP4 KV allocated: `torch.float4_e2m1fn_x2`, `5,516,867` tokens, K and V each `20.72 GB`.
- The server calibrated NVFP4 KV for 28 layers from 4096 eager prefill tokens.
- The one-time backend trace recorded 28 decode-layer calls through native FP4 KV.
- Each traced layer passed packed K/V as `torch.uint8` tensors shaped `(5516868, 2, 64)`.
- K/V scale buffers were `torch.float8_e4m3fn` shaped `(5516868, 1, 2, 8)`.
- Per-layer `k_scale` and `v_scale` values were nonzero finite scalars.
- Raw `2+2 is` returned ` 4, 2+2 is 4, 2+2 is`.
- Chat smoke returned exactly `spark-ok`.

## Interpretation

This is a useful quality-positive debug row, not yet a blessed SGLang FP4-KV row. It proves that the current backend wrapper can feed FlashInfer FA2 native FP4-KV decode with the same packed-data/FP8-scale contract cleared by the pool bridge, and this run did not reproduce the earlier malformed `2+2` output.

The trace did not capture an `extend_*` line. The server completed calibration and likely consumed the first prefill path before the one-time trace hook saw a request-path extend call. The next SGLang proof should capture request prefill explicitly, then repeat a matched fp8-vs-FP4 row with the same branch and environment.
