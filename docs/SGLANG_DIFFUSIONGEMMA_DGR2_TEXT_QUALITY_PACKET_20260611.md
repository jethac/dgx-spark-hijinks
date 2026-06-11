# SGLang DiffusionGemma DG-R2 Text-Only Quality Packet

Status: READY, waiting for a clean Spark window.

Owner: Codex SGLang lane.

Purpose: close DG-R2 from `docs/SGLANG_DIFFUSIONGEMMA_RUNTIME_LADDER_EPOCH2.md`
with a small deterministic text-only quality baseline before any performance
work. This packet deliberately excludes image prompts; the DG-R1 warning audit
only cleared text-only quality.

## Preconditions

- Spark marker absent: `/home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN` does not
  exist.
- `docker ps` empty.
- `free -h` shows at least 100 GiB available.
- No new model download: use the existing Spark HF cache, mounted read-only in
  practice through offline env flags.

## Code State

- Main repo: `epoch2` at or after `551d80c`.
- SGLang: `spark/hijinks-024-diffusiongemma-upstream-rebase` at
  `651d55cd2e6a3d90de0eb65af643d0aa4ee7fca2`.
- FlashInfer source in image/checkout: `f99323bd7d1cc88d9445202c12934070be754e2d`.
- Image: `sglang-source-stack-dgemma-024-0705924c-f99323bd`.

## Stage

```bash
R=/home/jethac/spark_tmp/dgx-spark-hijinks-sglang-dgr2-$(date +%Y%m%dT%H%MJST)
git clone --recurse-submodules --shallow-submodules --depth 1 \
  --branch epoch2 git@github.com:jethac/dgx-spark-hijinks.git "$R"
cd "$R"
git submodule update --init --recursive third_party/sglang third_party/flashinfer
```

Create the run directory:

```bash
OUT=/home/jethac/spark_tmp/sglang_dgemma_dgr2_text_quality_$(date +%Y%m%dT%H%MJST)
mkdir -p "$OUT"
cat > "$OUT/dllm_config.yaml" <<'EOF'
max_denoising_steps: 48
seed: 1234
sampler_config:
  entropy_bound: 0.1
temperature_schedule:
  t_min: 0.4
  t_max: 0.8
stopping_config:
  confidence_threshold: 0.005
  stability_threshold: 1
EOF
```

## Launch

Single server only:

```bash
docker run -d --name sglang_dgemma_dgr2_text_quality \
  --gpus all --ipc=host \
  --memory=100g --memory-swap=100g \
  -p 30125:30125 \
  -v "$R:/work" \
  -v "$OUT:/results" \
  -v /home/jethac/.cache/huggingface:/root/.cache/huggingface \
  -w /work \
  -e PYTHONPATH=/work/third_party/sglang/python:/work/third_party/flashinfer \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e HF_HOME=/root/.cache/huggingface \
  sglang-source-stack-dgemma-024-0705924c-f99323bd \
  bash -lc 'python3 -m sglang.launch_server \
    --model-path google/diffusiongemma-26B-A4B-it \
    --dllm-algorithm Gemma4Renoise \
    --dllm-algorithm-config /results/dllm_config.yaml \
    --trust-remote-code \
    --context-length 8192 \
    --mem-fraction-static 0.55 \
    --host 0.0.0.0 \
    --port 30125'
```

## Run Prompt Set

```bash
python3 scripts/diffusion_gemma_dgr2_text_quality_client.py \
  --base-url http://127.0.0.1:30125 \
  --out "$OUT/responses.json"

python3 scripts/diffusion_gemma_dgr2_text_quality_gate.py \
  "$OUT/responses.json" \
  --summary-out "$OUT/gate.json"
```

The client sends three text-only prompts twice each:

- capital of Japan: must contain `Tokyo`
- `2 + 2`: must contain standalone `4`
- DGX Spark use: must mention local/desktop/development AI use

Gate:

- each repeated prompt must return byte-identical normalized text
- each prompt must satisfy its parseable answer rule

## Stop-On-Red

- If the server does not reach readiness, stop and preserve logs.
- If any request times out, stop after capturing the response/log state.
- If `gate.json` has `all_ok=false`, do not tune prompts inside the same row.
  Document the red output exactly and decide the next rung from evidence.

## Stop State

```bash
docker logs sglang_dgemma_dgr2_text_quality > "$OUT/server.log" 2>&1 || true
docker rm -f sglang_dgemma_dgr2_text_quality
docker ps --format '{{.Names}} {{.Status}}' > "$OUT/docker_ps_after.txt"
if [ -e /home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN ]; then echo present; else echo absent; fi > "$OUT/marker_after.txt"
free -h > "$OUT/free_after.txt"
```

Pull `$OUT` back into the repo under `results/`, write `summary.md`, update
`docs/RESULTS_LEDGER.md`, send mail, commit, and push.
