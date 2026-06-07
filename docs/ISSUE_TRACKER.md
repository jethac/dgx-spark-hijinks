# Issue Tracker

This file maps the solution plan to GitHub Issues. Issue numbers are filled in after the issues are created.

| area | status | issue |
|---|---|---|
| `sm_121` target naming and build flags | open | [#1](https://github.com/jethac/dgx-spark-hijinks/issues/1) |
| ARM64 + CUDA 13 wheel/container matrix | open | [#2](https://github.com/jethac/dgx-spark-hijinks/issues/2) |
| Gemma 4 12B vLLM support | open | [#3](https://github.com/jethac/dgx-spark-hijinks/issues/3) |
| kernel dispatch and backend observability | open | [#4](https://github.com/jethac/dgx-spark-hijinks/issues/4) |
| `spark-doctor` environment evidence | in progress | [#5](https://github.com/jethac/dgx-spark-hijinks/issues/5) |
| vLLM Spark runtime path | open | [#6](https://github.com/jethac/dgx-spark-hijinks/issues/6) |
| NVFP4 on Spark | open | [#7](https://github.com/jethac/dgx-spark-hijinks/issues/7) |
| llama.cpp / lm-eval GGUF accuracy | open | [#8](https://github.com/jethac/dgx-spark-hijinks/issues/8) |
| HF fallback telemetry and containment | open | [#9](https://github.com/jethac/dgx-spark-hijinks/issues/9) |
| single-machine benchmark redesign | open | [#10](https://github.com/jethac/dgx-spark-hijinks/issues/10) |
| public recipes and blessed stack docs | open | [#11](https://github.com/jethac/dgx-spark-hijinks/issues/11) |
| multi-Spark future work | blocked on hardware | [#12](https://github.com/jethac/dgx-spark-hijinks/issues/12) |
| Spark-specific performance tuning | open | [#13](https://github.com/jethac/dgx-spark-hijinks/issues/13) |
| SGLang Spark runtime path | open | [#14](https://github.com/jethac/dgx-spark-hijinks/issues/14) |
| upstream forks, submodules, and worktrees | open | [#15](https://github.com/jethac/dgx-spark-hijinks/issues/15) |
| LiteRT-LM on Spark | open | [#16](https://github.com/jethac/dgx-spark-hijinks/issues/16) |
| llama.cpp practical serving path | open | [#17](https://github.com/jethac/dgx-spark-hijinks/issues/17) |
| SGLang NVFP4 KV on Spark | open | [#18](https://github.com/jethac/dgx-spark-hijinks/issues/18) |

## Triage Rules

- Issues should contain a reproduction command or the exact missing evidence.
- Results must say which backend was tested: vLLM, SGLang, LiteRT-LM, HF fallback, llama.cpp, Ollama, or another path.
- DGX Spark means GB10 / compute capability 12.1 / `sm_121`.
- RTX PRO 6000 and RTX 50-series are `sm_120`; they are related but not the same validation target.
- Multi-Spark work remains design-only until we have more than one Spark-class unit.
- Upstream code changes require a `jethac` fork, submodule, and issue-named worktree.
