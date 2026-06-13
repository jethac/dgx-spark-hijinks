# SGLang 0.5.13 NVFP4 SF-Stride And Ladder Packet

Scope: Spark packet for the rebased SGLang `v0.5.13` source-stack image
(`jethac/sglang@spark/hijinks-025-sglang-0.5.13-rebase`, head
`74e0e4bb5f058b0e4acac10e769268bb2f9a0c85`). This packet answers whether the
rebased image fixes the Gemma 4 full-NVFP4 quality delta and records the actual
K/V scale-factor layout handed to FlashInfer.

## Preconditions

- Spark marker `/home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN` absent.
- `docker ps` empty.
- GB10 memory guardrails: one server at a time, `--memory 100g`, no concurrent
  comparators, `MEM_FRACTION_STATIC<=0.72`.
- Use only the packaged Ubuntu 22.04 / arm64 / torch 2.11 image from GitHub run
  `27466068365` once it succeeds. Do not use loose wheels or Spark-local builds.
- Do not set `FLASHINFER_PAGED_V_SF_DESWIZZLE=1` for SGLang. SGLang's native
  FP4 pool uses linear V scale factors.

## Image

Pending image tag from GitHub run `27466068365`:

```text
ghcr.io/jethac/dgx-spark-hijinks/sglang-gemma4-source-stack:epoch2-sglang-0513-74e0e4bb
```

After the run finishes, replace `IMAGE`/`IMAGE_DIGEST` below with the digest from
the workflow summary artifact and record it in `docs/WHEEL_CONTAINER_MATRIX.md`.

## Block A: Weight-Free SF Layout Probe

Run before any model load. This proves the page-size-1 SGLang linear-SF contract
at the Gemma 4 sliding-layer shape.

```bash
cd /home/jethac/spark_tmp/dgx-spark-hijinks-sglang-live
git fetch origin
git checkout epoch2
git pull --ff-only

RUN_ID=sglang_0513_nvfp4_sf_stride_probe_$(TZ=Asia/Tokyo date +%Y%m%dT%H%M%SJST)
mkdir -p "results/${RUN_ID}"

docker run --rm --gpus all --ipc=host --network=host \
  --memory 16g --memory-swap 16g \
  -w /hijinks \
  -v "$PWD:/hijinks" \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_CACHE_DIR=/tmp/flashinfer-cache-sglang-0513-layout \
  -e FLASHINFER_EXTRA_CUDAFLAGS="-gencode=arch=compute_121a,code=sm_121a" \
  "$IMAGE_DIGEST" \
  python3 scripts/sglang_nvfp4_kv_layout_probe.py \
    --tokens 64 \
    --query-heads 32 \
    --kv-heads 16 \
    --head-dim 256 \
    --output "results/${RUN_ID}/layout_probe_gemma4_swa_h256.json" \
  | tee "results/${RUN_ID}/layout_probe_stdout.log"
```

Gate:

- `all_ok: true`.
- Rank-4 SF shape is `[tokens, 1, kv_heads, head_dim/16]`.
- Rank-3 SF is rejected or fails the reference gate.
- No `FLASHINFER_PAGED_V_SF_DESWIZZLE` in the effective flags.

## Block B: 12B Matched Full-NVFP4 Row With SF Trace

Run the small matched row first. It is the discriminator for Claude's scale
granularity finding: pure block-16 NVFP4 format loss is near-lossless, while the
old served row behaved like a coarse/per-tensor V-SF read.

```bash
RUN_ID=sglang_0513_gemma4_12b_fullnvfp4_ctx8192_prefix4096_$(TZ=Asia/Tokyo date +%Y%m%dT%H%M%SJST) \
IMAGE="$IMAGE_DIGEST" \
IMAGE_DIGEST="$IMAGE_DIGEST" \
MODELS="google/gemma-4-12B-it" \
ROW_LABELS="bf16 fp8 fullnvfp4" \
CTX_LIST="8192" \
REUSE_PREFIX_LEN=4096 \
LOGPROB_START_LEN=4096 \
MEM_FRACTION_STATIC=0.72 \
GB10_DOCKER_MEMORY=100g \
GB10_DOCKER_MEMORY_SWAP=100g \
bash scripts/run_sglang_gemma4_ar_ladder_pair.sh
```

The runner sets:

- `SGLANG_FP4_KV_TRACE_MODULE=1`
- `SGLANG_FP4_KV_TRACE_BACKEND=1`
- `SGLANG_GEMMA4_TRACE_GEOMETRY=1`
- `FLASHINFER_PREFILL_DEBUG_ONCE=1`
- `FLASHINFER_EXTRA_CUDAFLAGS=-gencode=arch=compute_121a,code=sm_121a`

Gate:

- `bf16`, `fp8`, and `fullnvfp4` all serve and produce supplied-token PPL.
- Full NVFP4 first token is coherent and stable against bf16/fp8.
- Server logs contain the module/backend trace with K/V SF shape and stride.
- SGLang trace shows linear SF buffers and no active vLLM deswizzle macro.

Interpretation:

- If full-NVFP4 delta collapses toward `+0.01..+0.04` nats/token, the 0.5.13
  rebase plus SGLang linear SF path fixed the serving delta.
- If full-NVFP4 remains near the old large delta, preserve the server logs and
  treat the failure as an SF-layout/read-path bug. Compare the traced V-SF shape,
  stride, and FlashInfer plan against the block-16 oracle in
  `docs/NVFP4_FORMAT_VS_KERNEL_GEMMA4.md`.

## Block C: Ladder Continuation

Only after Block B is green or clearly scoped, continue the ladder:

```bash
RUN_ID=sglang_0513_gemma4_ar_ladder_$(TZ=Asia/Tokyo date +%Y%m%dT%H%M%SJST) \
IMAGE="$IMAGE_DIGEST" \
IMAGE_DIGEST="$IMAGE_DIGEST" \
MODELS="google/gemma-4-26B-A4B-it google/gemma-4-31B-it" \
ROW_LABELS="bf16 fp8 fullnvfp4" \
CTX_LIST="8192" \
REUSE_PREFIX_LEN=4096 \
LOGPROB_START_LEN=4096 \
MEM_FRACTION_STATIC=0.72 \
GB10_DOCKER_MEMORY=100g \
GB10_DOCKER_MEMORY_SWAP=100g \
bash scripts/run_sglang_gemma4_ar_ladder_pair.sh
```

Claude's prediction from `mail/0125`: 31B/E4B force linear V-SF through the
VO-split path, so they should be near-lossless if the head-256 swizzled path was
the whole large-delta mechanism. For SGLang, any green here must still cite its
own traced linear-SF evidence, not vLLM's swizzled/linear switch.

## Stop Point

At every stop point:

- capture `docker ps` and `free -h`;
- update `docs/RESULTS_LEDGER.md` and `docs/WHEEL_CONTAINER_MATRIX.md` with the
  exact image digest, commits, row scope, capacity ratio, and quality delta;
- write a `results/.../summary.md`;
- send Claude a numbered mail file under `mail/`;
- leave the marker absent unless explicitly yielding a Claude window.
