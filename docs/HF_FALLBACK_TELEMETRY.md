# HF Fallback Telemetry

Status: wrapper added and smoke-tested on GB10.

Tracked by:

- https://github.com/jethac/dgx-spark-hijinks/issues/9

HF fallback is a separate backend class. It is not a transparent substitute for vLLM/SGLang because it has different memory behavior and several imported benchmark rows exited with `returncode=-9`.

## Wrapper

Use `scripts/run_with_telemetry.py` around any fragile fallback command:

```bash
python3 scripts/run_with_telemetry.py \
  --run-id hf-gemma4-row-id \
  --backend hf \
  --model MODEL_ID \
  --timeout-s 2400 \
  --interval-s 5 \
  --output results/hf_gemma4_row_id_telemetry.json \
  -- python3 YOUR_HF_COMMAND.py --args ...
```

The report records:

- command, run id, backend, model, elapsed time, timeout, return code
- stdout/stderr tails
- process-tree RSS high-water sampling from `/proc`
- process-tree swap sampling from `/proc`
- `free -b` snapshots before, during, and after
- `nvidia-smi` snapshots when available
- kernel OOM evidence from `journalctl -k --since ... -g 'Out of memory|Killed process|oom-kill'`

`dmesg` may be unavailable to an unprivileged user on the Spark; the smoke artifact records that as a permission-denied diagnostic instead of failing the wrapper.

## Smoke Evidence

Artifact:

- `results/telemetry_spark_smoke_20260607T1405Z.json`

Command:

```bash
python3 run_with_telemetry.py \
  --run-id telemetry-spark-smoke-20260607T1405Z \
  --backend test \
  --timeout-s 20 \
  --interval-s 1 \
  --output telemetry_spark_smoke_20260607T1405Z.json \
  -- python3 -c "import time; x=bytearray(8*1024*1024); time.sleep(2); print(len(x))"
```

Result:

- return code: `0`
- failure class: `ok`
- elapsed: about 2.0 seconds
- peak process-tree RSS: about 17 MiB
- peak process-tree swap: `0`
- `nvidia-smi` observed `NVIDIA GB10`
- `journalctl` OOM query returned no entries

## Failure Annotation

`scripts/failure_annotator.py` consumes `spark-run-with-telemetry/v1` JSON reports. For failed telemetry runs it preserves:

- `failure_class`
- return code
- elapsed time
- timeout status
- peak RSS
- peak swap
- OOM evidence classification

This closes the reporting gap for future `returncode=-9` HF rows: a failed row can now say whether the kernel reported OOM/resource pressure, whether only a signal was observed, or whether stderr points to a model/runtime error.
