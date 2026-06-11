# SGLang DiffusionGemma DG-R2 Prompt Diagnostic Packet

Status: READY, waiting for Docker-empty Spark window.

Owner: Codex SGLang lane.

Purpose: diagnose the DG-R2 text-only quality red without changing the DG-R2
gate. The existing claim remains RED until the original gate passes.

## Background

The DG-R2 quality gate in
`results/sglang_dgemma_dgr2_text_quality_20260612T0604JST/summary.md` showed:

- server loads and serves `google/diffusiongemma-26B-A4B-it`
- stock DiffusionGemma policy: Triton attention, eager, page size 256, BF16 KV
- `capital_japan` and `arithmetic_2_plus_2` return HTTP 200 with
  `finish_reason="length"` and nonzero `completion_tokens`, but empty
  `message.content`
- the DGX Spark descriptive prompt returns coherent text

Local code inspection suggests two likely surfaces:

- prompt sensitivity in the upstream `Gemma4Renoise` schedule
- fixed 256-token dLLM canvas interacting with normal OpenAI `max_tokens`
  clipping and special-token stripping

This packet probes those surfaces only. It does not tune the gate in-place.

## Preconditions

- `/home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN` absent.
- `docker ps` empty.
- `free -h` shows at least 100 GiB available.
- Use the existing B-backed/staged checkout:
  `/home/jethac/spark_tmp/dgx-spark-hijinks-sglang-dgr2-20260611T2344JST`.

If Docker is non-empty, do not launch. Mail coordination has already recorded
the marker-absent/Docker-busy state in
`mail/0042_codex-to-claude_marker-absent-docker-busy.md`.

## Launch

```bash
R=/home/jethac/spark_tmp/dgx-spark-hijinks-sglang-dgr2-20260611T2344JST
OUT=/home/jethac/spark_tmp/sglang_dgemma_dgr2_promptdiag_$(date +%Y%m%dT%H%MJST)
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

docker run -d --name sglang_dgemma_dgr2_promptdiag \
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

## Run

Use `/model_info` readiness, not `/health`; DG-R2 already proved `/health`
can stay 503 after readiness.

```bash
python3 "$R/scripts/diffusion_gemma_dgr2_prompt_diagnostic.py" \
  --base-url http://127.0.0.1:30125 \
  --out "$OUT/prompt_diagnostic.json"
```

The diagnostic captures, for each prompt variant:

- OpenAI chat, default special-token stripping
- OpenAI chat with `skip_special_tokens=false`
- native `/generate`, default stripping
- native `/generate` with `skip_special_tokens=false`

It includes short-answer prompts, less terse variants, the known-good DGX Spark
prompt, and full-canvas `max_tokens=256` variants for the failed factual
prompts.

## Stop

```bash
docker logs sglang_dgemma_dgr2_promptdiag > "$OUT/server.log" 2>&1 || true
docker inspect sglang_dgemma_dgr2_promptdiag > "$OUT/container_inspect.json" 2>&1 || true
docker rm -f sglang_dgemma_dgr2_promptdiag
docker ps --format '{{.Names}} {{.Status}}' > "$OUT/docker_ps_after.txt"
if [ -e /home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN ]; then echo present; else echo absent; fi > "$OUT/marker_after.txt"
free -h > "$OUT/free_after.txt"
```

Copy `$OUT` back to `results/`, write `summary.md`, update
`docs/RESULTS_LEDGER.md` only if the row adds a citable finding, send mail,
commit, and push.

## Interpretation

- If `skip_special_tokens=false` reveals text, the red is a detokenization /
  special-token surface.
- If full-canvas `max_tokens=256` reveals the expected answers while short
  `max_tokens` stays empty, the red is likely canvas-position/max-token
  interaction.
- If less terse prompts pass while terse prompts stay empty, the red is prompt
  sensitivity in the upstream runtime baseline.
- If chat and `/generate` disagree, the red is endpoint/template-specific.
- If all factual variants fail while the DGX descriptive prompt passes again,
  keep DG-R2 RED and scope upstream DiffusionGemma support as runnable but not
  quality-baseline green on this prompt class.
