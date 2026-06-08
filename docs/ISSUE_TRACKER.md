# Issue Tracker

This file maps the solution plan to GitHub Issues. Issue numbers are filled in after the issues are created.

| plan id(s) | area | status | issue |
|---|---|---|---|
| `1` | `sm_121` target naming and build flags | build/JIT target log audit added | [#1](https://github.com/jethac/dgx-spark-hijinks/issues/1) |
| `2` | ARM64 + CUDA 13 wheel/container matrix | first matrix added; clean blessed stack still missing | [#2](https://github.com/jethac/dgx-spark-hijinks/issues/2) |
| `6` | Gemma 4 12B vLLM support | source/precompiled probe serves; clean release container pending | [#3](https://github.com/jethac/dgx-spark-hijinks/issues/3) |
| `3, 13` | kernel dispatch and backend observability | build/JIT target audit plus existing runtime probes; backend-specific serving logs still pending | [#4](https://github.com/jethac/dgx-spark-hijinks/issues/4) |
| `13` | `spark-doctor` environment evidence | in progress | [#5](https://github.com/jethac/dgx-spark-hijinks/issues/5) |
| `5` | vLLM Spark runtime path | AEON Gemma 4 26B NVFP4+DFlash serves locally at about 48-54 tok/s short/medium and 98 tok/s long-prefill; SM12x NVFP4 KV routing/deswizzle probe passed; AEON Qwen36 now passes with API-level thinking disabled; `jethac/vllm@6804e1b` derived AEON Qwen row also passes after dependency alignment and AEON FA2 binary restoration; clean `jethac/vllm@a919d635d` plus `jethac/flash-attention@7d53245` serves Qwen36 at 61.07/56.97/60.10 tok/s with separate `sm_121a` FA2 cubin proof; Qwen NVFP4-KV now records `1.751x` fp8 KV pool/concurrency with decode parity; remaining vLLM gaps are official packaging, Gemma NVFP4-KV, native FP4 weight/MoE proof, and accuracy | [#6](https://github.com/jethac/dgx-spark-hijinks/issues/6) |
| `3, 7` | NVFP4 on Spark | SM121 `b12x` GEMM dispatch enabled; FlashInfer FA2 KV stride/page and vLLM FA2 routing/deswizzle branches pushed; standalone GB10 FA2 NVFP4 KV passes small and Gemma 4 26B sliding shapes; vLLM Qwen NVFP4-KV serves with `1.751x` fp8 KV pool/concurrency and normal content; checkpoint audit added for NVFP4 format/sensitive tensor/EOS metadata; Gemma 4 26B global `D=512` fails; SGLang FP4 KV now has a capacity-proven no-graph serving path but still fails standardized output quality; SGLang convention bridge proves viable raw FA2 reader pairs and clears the invalid crossed `nvfp4_kv_quantize` encode-scale case; TensorRT-LLM #11368 documents a separate GB10 FP4 GEMM shared-memory tile gap | [#7](https://github.com/jethac/dgx-spark-hijinks/issues/7) |
| `8` | llama.cpp / lm-eval GGUF accuracy | native loglikelihood adapter prototype and tiny JSONL task harness ran against live llama-server; top-512 native probabilities scored likely continuations but missed the unlikely continuation; OpenAI echo on pinned `b9536` also returns generated-token `logprobs.content`, not prompt `tokens`/`token_logprobs`, so arbitrary continuation scoring remains blocked | [#8](https://github.com/jethac/dgx-spark-hijinks/issues/8) |
| `9` | HF fallback telemetry and containment | telemetry wrapper added | [#9](https://github.com/jethac/dgx-spark-hijinks/issues/9) |
| `12` | single-machine benchmark redesign | smoke-suite orchestrator, Qwen speed lane, HF telemetry, MTP/spec-decode wrapping, and counterpart task matrix exist | [#10](https://github.com/jethac/dgx-spark-hijinks/issues/10) |
| `4, 14, 15` | public recipes and blessed stack docs | vLLM/SGLang recipes require build/JIT target audit paths; compatibility board added | [#11](https://github.com/jethac/dgx-spark-hijinks/issues/11) |
| `11` | multi-Spark future work | blocked on hardware | [#12](https://github.com/jethac/dgx-spark-hijinks/issues/12) |
| `10` | Spark-specific performance tuning | FlashInfer proxy null result recorded; SM-count-aware run keys added; serving-path tuning pending | [#13](https://github.com/jethac/dgx-spark-hijinks/issues/13) |
| `7a` | SGLang Spark runtime path | BF16 Qwen passed; Gemma E2B failed; NVFP4 tracked in #18 | [#14](https://github.com/jethac/dgx-spark-hijinks/issues/14) |
| `14, 14a` | upstream forks, submodules, and worktrees | FlashInfer, vLLM, and SGLang forks/submodules/worktrees created; FlashInfer, vLLM, and SGLang KV patch branches pushed; vLLM Qwen branch advanced to AEON source patch coverage; AEON port map, counterpart evidence audit, and live task matrix added for SGLang/llama.cpp decisions | [#15](https://github.com/jethac/dgx-spark-hijinks/issues/15) |
| `7b` | LiteRT-LM on Spark | optional side runtime; CPU path proven; GPU chat crash open | [#16](https://github.com/jethac/dgx-spark-hijinks/issues/16) |
| `8a` | llama.cpp practical serving path | blessed for Gemma 4 26B Q4_0 and Qwen2.5 1.5B Q4_K_M serving | [#17](https://github.com/jethac/dgx-spark-hijinks/issues/17) |
| `7, 7a` | SGLang NVFP4 KV on Spark | `jethac/sglang` now clears early gate/alias blockers, calibrates before capture, and auto-disables CUDA graph capture for native FP4 KV because graph-enabled decode corrupts output; matched fp8-vs-FP4 autosafe row records `5,519,481` FP4 tokens versus `3,101,822` fp8 tokens, but standardized raw `2+2` and benchmark content fail quality; standalone convention bridge shows raw FA2 math is correct for `fp4_quantize` encode-scale and `nvfp4_kv_quantize` decode-scale pairs, so next debugging is calibration/V-scale/backend integration | [#18](https://github.com/jethac/dgx-spark-hijinks/issues/18) |
| `10, 12, 13` | before/after GB10 benchmark protocol | SM-count-aware hardware comparison keys and Qwen speed-lane runner added | [#19](https://github.com/jethac/dgx-spark-hijinks/issues/19) |
| `qwen-speed` | Qwen speed and capacity benchmarks | SGLang Qwen BF16/auto and fp8 rows captured; SGLang FP4 KV capacity row captures `1.779x` fp8 capacity but fails quality; llama.cpp Qwen2.5 1.5B Q4_K_M row captured at 167-175 tok/s; AEON Qwen36 NVFP4+DFlash passes vLLM smoke and compact serving at about 50-56 tok/s when `chat_template_kwargs={"enable_thinking": false}` is set; `jethac/vllm@6804e1b` derived AEON Qwen row passes at 47.22/58.88/61.62 tok/s but still lacks native FP4 weight/MoE proof; `jethac/vllm@a919d635d` clean FA2 row passes at 61.07/56.97/60.10 tok/s with separate `sm_121a` FA2 cubin proof; no-DFlash vLLM Qwen NVFP4-KV records `1.751x` fp8 KV pool/concurrency with decode parity; `scripts/qwen_speed_lane.py` records already-running Qwen servers | [#20](https://github.com/jethac/dgx-spark-hijinks/issues/20) |
| `vllm-fa2-sm121a` | vLLM FlashAttention FA2 native-target proof | `jethac/flash-attention@spark/hijinks-021-fa2-sm121a` adds SM121/SM121a to the vLLM-pinned FA2 CMake path and is wired as `third_party/vllm-flash-attention`; clean image build/import/cuobjdump proof still pending | [#21](https://github.com/jethac/dgx-spark-hijinks/issues/21) |

## Upstream Issues Referenced

External issues we track because they gate or motivate Spark/SM12x work. These are not
our issues; they record where the upstream fix or demand lives.

| upstream issue | relevance |
|---|---|
| [vllm-project/vllm #31085](https://github.com/vllm-project/vllm/issues/31085) | Add SM120 (RTX 6000/5000 Blackwell) native NVFP4 MoE kernels — concrete upstream demand for the SM12x-family-shaped NVFP4 work; the RTX PRO 6000 audience that makes our hikari-derived patches mergeable beyond Spark |
| [vllm-project/vllm #31128](https://github.com/vllm-project/vllm/issues/31128) | vLLM SM121 support tracking |
| [TensorRT-LLM #11368](https://github.com/NVIDIA/TensorRT-LLM/issues/11368) | SM120 CUTLASS FP4 GEMM tiles exceed the CC-12.x 99 KB/block shared-memory limit; confirmed family-wide (RTX PRO 6000 sm_120 and GB10 sm_121 share it), so a 99 KB-fitting tile fix serves both |

Native-FP4 target note: RTX PRO 6000 = `sm_120` (CC 12.0); GB10 = `sm_121` (CC 12.1).
Both need arch-specific (`a`) cubins for native block-scaled FP4 MMA — `120f`/`121f`
family targets cannot emit it. `a` cubins are not portable across compute capabilities,
so `sm_120a` cannot run on GB10; SM120 is a compiled-but-unclaimed target validated by
hikarioyama, not by us. See `docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md`.

## Triage Rules

- Issues should contain a reproduction command or the exact missing evidence.
- `docs/SOLUTIONS_STATUS.md` is the current acceptance-evidence index for `docs/DGX_SPARK_SOLUTIONS.md`.
- Results must say which backend was tested: vLLM, SGLang, LiteRT-LM, HF fallback, llama.cpp, Ollama, or another path.
- DGX Spark means GB10 / compute capability 12.1 / `sm_121`.
- RTX PRO 6000 and RTX 50-series are `sm_120`; they are related but not the same validation target.
- Multi-Spark work remains design-only until we have more than one Spark-class unit.
- Upstream code changes require a `jethac` fork, submodule, and issue-named worktree.
