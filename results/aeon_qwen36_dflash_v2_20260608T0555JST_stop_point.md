# AEON Qwen3.6 DFlash v2 Stop Point

Timestamp: 2026-06-08T05:55:34+09:00

Status: Qwen speed benchmarking is still required, but the GB10 host was unreachable before the current `v2` image-pull state could be inspected.

## Target

- image: `ghcr.io/aeon-7/vllm-spark-omni-q36:v2`
- model: `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4`
- drafter: `z-lab/Qwen3.6-35B-A3B-DFlash`
- intended runner: `scripts/run_aeon_vllm_reproduction.sh qwen36-dflash RUN_ID`

## Known Prior State

- The target weights were already cached at `/home/jethac/models/aeon/qwen36-nvfp4`, about `22G`.
- The DFlash drafter was already cached at `/home/jethac/models/aeon/qwen36-dflash`, about `905M`.
- `ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2` failed to register after two bounded pull attempts.
- A longer `v1.2` pull again reached late `Pull complete` lines without registering.
- A `v2` pull was then started with `docker pull --platform=linux/arm64 ghcr.io/aeon-7/vllm-spark-omni-q36:v2`.
- The last observed `v2` pull had progressed through multiple completed layers, including `9be51ddd2646: Pull complete`, but the image had not yet registered at the time of the last successful SSH session.

## Current Blocker

Follow-up inspection could not connect to the GB10 host:

```text
ssh: connect to host 192.168.68.112 port 22: Connection timed out
```

`Test-Connection 192.168.68.112 -Count 2 -Quiet` returned `False`.

No Qwen server was launched during this follow-up, and no model-load/runtime/kernel conclusion should be inferred from this stop point.

## Next Command

Once the host is reachable, inspect the pull and either record the failure or run the benchmark:

```bash
cd /home/jethac/dgx-spark-hijinks-aeon-qwen-run
docker image inspect ghcr.io/aeon-7/vllm-spark-omni-q36:v2
tail -120 results/aeon_qwen36_dflash_v2_20260608T0548JST_docker_pull.log
```

If the image is registered:

```bash
HF_CLI=/home/jethac/gemma4-evals/.venv/bin/hf \
RECORD_PYTHON=/home/jethac/gemma4-evals/.venv/bin/python \
MODELS_ROOT=/home/jethac/models/aeon \
RESULTS_DIR=results \
DOWNLOAD=0 \
DOCKER_PULL=0 \
RECORD=1 \
WAIT_TIMEOUT=2400 \
scripts/run_aeon_vllm_reproduction.sh qwen36-dflash aeon_qwen36_dflash_v2_YYYYMMDDTHHMMJST
```
