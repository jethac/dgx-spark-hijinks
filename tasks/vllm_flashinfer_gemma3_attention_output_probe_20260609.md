# vLLM / FlashInfer Gemma 3 NVFP4 Attention-Output Probe, 2026-06-09

Status: next live GB10 diagnostic.

Purpose: test the failure localized by
`results/vllm_gemma3_27b_tensor_trace_20260609T0115JST_summary.md` without loading
Gemma 3 again. The live vLLM row shows NVFP4-KV `flashinfer_attn_output` tensors are
BF16-shaped but nearly nonnegative, with means around `124..126` and max values exactly
`255.0` on many layers. Sampled physical page/scale bytes already matched, so this task
probes the FlashInfer FA2 attention-output boundary directly.

## Geometry

Use the measured Gemma 3 27B Rung 1 geometry from the running vLLM server logs:

- `head_dim=128`
- `num_kv_heads=16`
- `num_qo_heads=32`
- `page_size=16`
- local/SWA layer window: `1024`
- global/full layers use the same head geometry

The standalone probe does not model Gemma layer norms, residuals, or logits. Its only
job is to catch whether FlashInfer FA2 NVFP4 output can become byte-like before vLLM
touches it.

## Script Change

`scripts/flashinfer_nvfp4_kv_probe.py` now records `actual_stats` and `expected_stats`,
including a `byte_like_nonnegative` flag, and accepts:

```bash
--k-global-scale FLOAT
--v-global-scale FLOAT
```

The previous probe always used `1.0` global scales. That cannot catch a misplaced
K/V scale application, so this packet must run both unit-scale and non-unit-scale rows.

## Run Packet

Use the CUDA-13 source-overlay pattern: mount this repo and the FlashInfer source
checkout, uninstall prebuilt FlashInfer packages inside the container, point Python at
`/flashinfer-src`, and set the vLLM-style V-scale deswizzle flag for swizzled rows.

Image used by the Gemma 3 vLLM diagnostic:

```bash
IMAGE=jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass
RUN_ROOT=/work
FLASHINFER_SRC=/flashinfer-src
OUT=results/vllm_flashinfer_gemma3_attention_output_probe_20260609TGB10
```

Container setup body:

```bash
python3 -m pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache || true
python3 -m pip install -q --upgrade \
  "nvidia-cutlass-dsl[cu13]>=4.5.0" \
  "apache-tvm-ffi>=0.1.6,!=0.1.8,!=0.1.8.post0,<0.2"
rm -rf /tmp/flashinfer-python-path
mkdir -p /tmp/flashinfer-python-path
ln -s /flashinfer-src/flashinfer /tmp/flashinfer-python-path/flashinfer
export PYTHONPATH="/tmp/flashinfer-python-path:${PYTHONPATH:-}"
```

Run these rows. `--signed-values` is required; without it the legacy synthetic generator
masks packed nibbles with `0x77`, producing a nonnegative reference that cannot test the
Gemma failure signature.

```bash
export FLASHINFER_EXTRA_CUDAFLAGS="-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1"

python3 scripts/flashinfer_nvfp4_kv_probe.py \
  --flashinfer-source-root /flashinfer-src \
  --output "${OUT}_swizzled_unit_scale.json" \
  --layouts NHD HND \
  --kv-container tuple \
  --v-scale-layout swizzled \
  --head-dim 128 \
  --num-kv-heads 16 \
  --num-qo-heads 32 \
  --page-size 16 \
  --kv-len 64 \
  --qo-len 16 \
  --k-global-scale 1.0 \
  --v-global-scale 1.0 \
  --signed-values

python3 scripts/flashinfer_nvfp4_kv_probe.py \
  --flashinfer-source-root /flashinfer-src \
  --output "${OUT}_swizzled_nonunit_vscale.json" \
  --layouts NHD HND \
  --kv-container tuple \
  --v-scale-layout swizzled \
  --head-dim 128 \
  --num-kv-heads 16 \
  --num-qo-heads 32 \
  --page-size 16 \
  --kv-len 64 \
  --qo-len 16 \
  --k-global-scale 0.03125 \
  --v-global-scale 0.03125 \
  --signed-values

python3 scripts/flashinfer_nvfp4_kv_probe.py \
  --flashinfer-source-root /flashinfer-src \
  --output "${OUT}_linear_control.json" \
  --layouts NHD HND \
  --kv-container tuple \
  --v-scale-layout linear \
  --head-dim 128 \
  --num-kv-heads 16 \
  --num-qo-heads 32 \
  --page-size 16 \
  --kv-len 64 \
  --qo-len 16 \
  --k-global-scale 0.03125 \
  --v-global-scale 0.03125 \
  --no-deswizzle-flag \
  --signed-values
```

## Acceptance

Green standalone result:

- every row has `all_ok=true`;
- actual output stats are signed/centered similarly to expected stats;
- no operation reports `actual_stats.byte_like_nonnegative=true` while the expected
  output is not byte-like.

Red standalone result:

- any row reproduces byte-like BF16 output (`min >= 0`, `max <= 255`, mean near `128`,
  or max exactly `255.0`), especially only in prefill or only with non-unit `v_scale`.

If standalone rows are green, the next vLLM source-overlay hook should trace the exact
FlashInfer wrapper arguments from the failing Gemma 3 request: `k_scale`, `v_scale`,
output dtype/shape/stride before the wrapper call, output dtype/shape/stride after the
wrapper call, and whether prefill or decode produced the first byte-like tensor.

## Live Result

Artifact: `results/vllm_flashinfer_gemma3_attention_output_probe_20260609T0134JST_summary.md`.

Rows:

- `signed_swizzled_nonunit_vscale`: `FLASHINFER_PAGED_V_SF_DESWIZZLE=1`, signed E2M1
  nibbles, `k_scale=v_scale=0.03125`.
- `signed_linear_control`: no deswizzle flag, signed E2M1 nibbles,
  `k_scale=v_scale=0.03125`.

Both rows passed for NHD/HND and decode/prefill:

- minimum cosine: `0.999997496604919`
- maximum absolute error: `0.0001220703125`
- `actual_stats.byte_like_nonnegative`: `0 / 4` operations in both rows
- actual output ranges are signed and centered near zero, e.g. NHD decode min
  `-0.0279541015625`, max `0.0255126953125`, mean `3.15e-06`.

Interpretation: generic FlashInfer FA2 NVFP4 attention output is not reproducing the
Gemma byte-like BF16 failure for signed Gemma-shaped `D=128` synthetic data, even with
vLLM-style V-scale deswizzle and non-unit K/V global scales. Next probe must use the
real vLLM/Gemma wrapper boundary: trace or dump the actual model `query`, split packed
K/V, K/V scale tensors, `k_scale`, `v_scale`, output buffer metadata, and wrapper result
for the first failing request.
