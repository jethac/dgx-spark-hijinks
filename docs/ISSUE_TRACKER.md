# Issue Tracker

This file maps the solution plan to GitHub Issues. Issue numbers are filled in after the issues are created.

| area | status | issue |
|---|---|---|
| `sm_121` target naming and build flags | planned | TBD |
| ARM64 + CUDA 13 wheel/container matrix | planned | TBD |
| kernel dispatch and backend observability | planned | TBD |
| vLLM Spark runtime path | planned | TBD |
| Gemma 4 12B vLLM support | planned | TBD |
| NVFP4 on Spark | planned | TBD |
| llama.cpp / lm-eval GGUF accuracy | planned | TBD |
| HF fallback telemetry and containment | planned | TBD |
| Spark-specific performance tuning | planned | TBD |
| single-machine benchmark redesign | planned | TBD |
| `spark-doctor` environment evidence | in progress | TBD |
| public recipes and blessed stack docs | planned | TBD |
| multi-Spark future work | blocked on hardware | TBD |

## Triage Rules

- Issues should contain a reproduction command or the exact missing evidence.
- Results must say which backend was tested: vLLM, HF fallback, llama.cpp, Ollama, or another path.
- DGX Spark means GB10 / compute capability 12.1 / `sm_121`.
- RTX PRO 6000 and RTX 50-series are `sm_120`; they are related but not the same validation target.
- Multi-Spark work remains design-only until we have more than one Spark-class unit.

