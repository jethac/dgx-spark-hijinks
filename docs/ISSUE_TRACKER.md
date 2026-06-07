# Issue Tracker

This file maps the solution plan to GitHub Issues. Issue numbers are filled in after the issues are created.

| area | status | issue |
|---|---|---|
| `sm_121` target naming and build flags | build/JIT target log audit added | [#1](https://github.com/jethac/dgx-spark-hijinks/issues/1) |
| ARM64 + CUDA 13 wheel/container matrix | open | [#2](https://github.com/jethac/dgx-spark-hijinks/issues/2) |
| Gemma 4 12B vLLM support | source/precompiled probe serves; clean release container pending | [#3](https://github.com/jethac/dgx-spark-hijinks/issues/3) |
| kernel dispatch and backend observability | build/JIT target audit plus existing runtime probes; backend-specific serving logs still pending | [#4](https://github.com/jethac/dgx-spark-hijinks/issues/4) |
| `spark-doctor` environment evidence | in progress | [#5](https://github.com/jethac/dgx-spark-hijinks/issues/5) |
| vLLM Spark runtime path | SM12x NVFP4 KV routing/deswizzle probe passed on GB10; standalone FlashInfer FA2 KV correctness passed; clean vLLM build and serving proof pending | [#6](https://github.com/jethac/dgx-spark-hijinks/issues/6) |
| NVFP4 on Spark | SM121 `b12x` GEMM dispatch enabled; FlashInfer FA2 KV stride/page and vLLM FA2 routing/deswizzle branches pushed; standalone GB10 FA2 NVFP4 KV passes small and Gemma 4 26B sliding shapes; Gemma 4 26B global `D=512` fails; SGLang integration and serving proof pending | [#7](https://github.com/jethac/dgx-spark-hijinks/issues/7) |
| llama.cpp / lm-eval GGUF accuracy | open | [#8](https://github.com/jethac/dgx-spark-hijinks/issues/8) |
| HF fallback telemetry and containment | telemetry wrapper added | [#9](https://github.com/jethac/dgx-spark-hijinks/issues/9) |
| single-machine benchmark redesign | smoke-suite orchestrator added; HF and MTP telemetry wrapped | [#10](https://github.com/jethac/dgx-spark-hijinks/issues/10) |
| public recipes and blessed stack docs | vLLM/SGLang recipes now require build/JIT target audit paths | [#11](https://github.com/jethac/dgx-spark-hijinks/issues/11) |
| multi-Spark future work | blocked on hardware | [#12](https://github.com/jethac/dgx-spark-hijinks/issues/12) |
| Spark-specific performance tuning | FlashInfer proxy null result recorded; SM-count-aware run keys added; serving-path tuning pending | [#13](https://github.com/jethac/dgx-spark-hijinks/issues/13) |
| SGLang Spark runtime path | BF16 Qwen passed; Gemma E2B failed; NVFP4 tracked in #18 | [#14](https://github.com/jethac/dgx-spark-hijinks/issues/14) |
| upstream forks, submodules, and worktrees | FlashInfer, vLLM, and SGLang forks/submodules/worktrees created; FlashInfer, vLLM, and SGLang KV patch branches pushed and submodules advanced | [#15](https://github.com/jethac/dgx-spark-hijinks/issues/15) |
| LiteRT-LM on Spark | optional side runtime; CPU path proven; GPU chat crash open | [#16](https://github.com/jethac/dgx-spark-hijinks/issues/16) |
| llama.cpp practical serving path | blessed for 26B Q4_0 serving | [#17](https://github.com/jethac/dgx-spark-hijinks/issues/17) |
| SGLang NVFP4 KV on Spark | `jethac/sglang@98ad46961` clears SM12x FP4 KV gate and alias blockers; patched overlay shows 1.78x fp8 pool potential and can serve only with graph paths disabled at 0.276 tok/s; clean graph-compatible serving still pending | [#18](https://github.com/jethac/dgx-spark-hijinks/issues/18) |
| before/after GB10 benchmark protocol | SM-count-aware hardware comparison keys added | [#19](https://github.com/jethac/dgx-spark-hijinks/issues/19) |
| Qwen speed and capacity benchmarks | SGLang Qwen BF16/auto and fp8 rows captured at matched memory fraction; patched fp4 KV capacity/debug row captured but not usable for speed; vLLM AEON reproduction and llama.cpp Qwen GGUF rows pending | [#20](https://github.com/jethac/dgx-spark-hijinks/issues/20) |

## Triage Rules

- Issues should contain a reproduction command or the exact missing evidence.
- Results must say which backend was tested: vLLM, SGLang, LiteRT-LM, HF fallback, llama.cpp, Ollama, or another path.
- DGX Spark means GB10 / compute capability 12.1 / `sm_121`.
- RTX PRO 6000 and RTX 50-series are `sm_120`; they are related but not the same validation target.
- Multi-Spark work remains design-only until we have more than one Spark-class unit.
- Upstream code changes require a `jethac` fork, submodule, and issue-named worktree.
