# Campaign Log

## 2026-06-07

- Created this public record repository.
- Imported the first Gemma 4 benchmark reports.
- Added the initial diagnosis and solution plan.
- Added first-pass tools:
  - `scripts/spark_doctor.py`
  - `scripts/gguf_logprobs_probe.py`
  - `scripts/openai_chat_smoke.py`
- Started tracking work through GitHub Issues.
- Ran `spark_doctor` on `thinkstationpgx-00b4` using `/home/jethac/gemma4-evals/.venv/bin/python`.
  - GPU: `NVIDIA GB10`
  - compute capability: `12.1` / `sm_121`
  - host: `aarch64`
  - driver: `580.159.03`
  - CUDA runtime reported by `nvidia-smi`: `13.0`
  - PyTorch: `2.11.0+cu130`
  - vLLM: `0.22.1`
  - FlashInfer: `0.6.11.post2`
  - PyTorch arch list: `sm_80`, `sm_90`, `sm_100`, `sm_110`, `sm_120`; no explicit `sm_121`
  - snapshot: `results/spark_doctor_20260607T110629Z.md`

## First Benchmark Campaign Summary

The first campaign was run on `thinkstationpgx-00b4` in `/home/jethac/gemma4-evals`.

At the last local sync:

- smoke rows: 152 complete
- smoke `ok`: 21
- smoke `eval_failed`: 11
- smoke `loader_failed`: 120
- full eval records: 70
- full eval `ok`: 65
- full eval failed: 5
- throughput rows observed: 2
- MTP rows observed: 2

The campaign was still in full accuracy when monitoring stopped. It was not killed.
