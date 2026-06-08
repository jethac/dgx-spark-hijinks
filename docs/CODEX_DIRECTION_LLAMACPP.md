# Direction: llama.cpp on Spark (unblock GGUF accuracy; prove or deny native FP4)

> Standing direction for the llama.cpp lane. llama.cpp is the **practical daily-driver**
> on this box — by far the fastest runtime (~175 tok/s Qwen2.5 1.5B, ~77 tok/s Gemma 4
> 26B) and already blessed for serving. Two things are still red and both are real
> campaign deliverables: **GGUF lm-eval accuracy** (blocked since day one) and **native
> sm_121a FP4 dispatch** (now merged upstream as `GGML_TYPE_NVFP4` + Blackwell tensor-core
> MMQ, but unverified on our sm_121 box). This lane owns the campaign's *accuracy oracle*
> and — surprisingly — may be the simplest path to a first proof of native FP4 tensor-core
> use on this silicon in *any* runtime (one source-built binary, no container surgery).

## Why this, why now
"What Counts As Fixed" explicitly requires llama.cpp GGUF throughput and lm-eval accuracy
to be **separated and validated**. Throughput is done; accuracy has been blocked the
entire campaign (120 smoke rows `loader_failed` from a logprobs/schema mismatch). Fixing
it does more than close the llama.cpp lane — it gives the *whole campaign* a working
quantization-accuracy measurement tool, which we currently lack for any format.

## Methodology / sequencing constraint
llama.cpp is native host binaries, not containers — so the constraint is different from
the vLLM/SGLang lanes — and the difference cuts the *other* way. **Building llama.cpp
ourselves is cheap**: a single CMake C++ project that builds in minutes on GB10, not an
hour-long container. So for the native-FP4 work, building our own `jethac/llama.cpp` off a
recent master is the **primary tool, not a gated last resort** — the build itself (with
chosen arch flags) is how we answer the sm_121 dispatch question empirically rather than by
reading and guessing. `b9536`/`308f61c31` stays only as the known-good *serving baseline*
to diff against; keep the existing Python probe scripts pointed at it for comparison.

## What is already proven — do NOT redo these
- **Practical serving is blessed** (`docs/GGUF_LLAMA_CPP_STATUS.md`, issue #17):
  - Gemma 4 26B Q4_0 with `--reasoning off`: ~77 tok/s, `llama-bench` pp512 3021 / tg128
    77 (`results/llamacpp_gemma4_26b_q4_0_*`).
  - Qwen2.5 1.5B Q4_K_M: ~167–175 tok/s, `llama-bench` pp512 12505 / tg128 178
    (`results/llamacpp_qwen25_1_5b_q4_k_m_20260608T0420JST_*`).
  - Build evidence on both: `NVIDIA GB10`, `CUDA : ARCHS = 1210`, `USE_GRAPHS = 1`,
    `BLACKWELL_NATIVE_FP4 = 1`.
- **The accuracy blocker is characterized, and one dead end is cleared:** the OpenAI
  `/v1/completions` path returns `logprobs.content` (generated-token scoring), not
  `logprobs.tokens`/`token_logprobs`. The native `/completion` + `n_probs=512` adapter
  scores likely continuations but **misses unlikely tokens** (the `zebra` failure,
  `results/llamacpp_native_loglikelihood_20260608T1331JST_*`, `ok=false`). **top-N
  probabilities are fundamentally the wrong tool for lm-eval** — do not keep tuning
  `n_probs` upward hoping to catch the tail.

Read `docs/GGUF_LLAMA_CPP_STATUS.md` (Fix Options + Next Proof) and
`recipes/gguf_llamacpp_accuracy.md` before starting.

## The llama.cpp problem, precisely
1. **GGUF lm-eval accuracy is blocked.** lm-eval loglikelihood needs the **exact logprob
   of each supplied continuation token**, rank-independent — you feed context+continuation
   and read back each continuation token's own logprob. top-N sampling can't provide that
   for unlikely continuations. The fix needs **supplied-token / prompt-token (echo)
   logprobs**, not bigger top-N.
2. **Native sm_121a FP4 dispatch is unverified — but the code now exists upstream.** See
   "Upstream NVFP4 reality" below. NVFP4 is a real merged GGUF type with native Blackwell
   MMQ. `BLACKWELL_NATIVE_FP4=1`/`ARCHS=1210` mean our `b9536` build *compiled* that path —
   but every model we ran is Q4_0/Q4_K_M (integer k-quants), so it has **never been
   exercised** on our box, and upstream has open sm_120/121 compile + correctness issues.
   A k-quant model on an FP4-capable build proves nothing about native FP4 dispatch.
3. **Counterpart coverage gap.** Only Qwen2.5 1.5B exists; the speed/counterpart matrix
   still wants a larger Qwen3/Qwen3.6-class GGUF row on the same `b9536` build.

## Upstream NVFP4 reality (verified 2026-06)
llama.cpp moved faster than this lane's earlier notes assumed. As of spring 2026:
- **NVFP4 is a first-class GGUF type** (`GGML_TYPE_NVFP4 = 40`), merged into mainline with
  CUDA/SYCL/Vulkan kernels. MXFP4 (the gpt-oss format) has been in since Aug 2025.
- **Native Blackwell tensor-core FP4 MMQ landed in build b8967 (Apr 2026)**; our host runs
  the newer `b9536`, so the native path is compiled in (hence `BLACKWELL_NATIVE_FP4=1`).
  The kernel emits `mma.sync...mxf4nvf4.block_scale.scale_vec::4X` (m16n8k64) and repacks
  weights to an AoSoA tile layout at load under `BLACKWELL_MMA_AVAILABLE`.
- **The win profile matches our roofline thesis, independently:** PR #17906 reports the
  native FP4 path is ~25% faster on prompt-processing (PP) and ~10% *slower* otherwise —
  i.e. prefill-compute, not decode. "Older cards run FP4 but only see memory savings." This
  is the "capacity-everywhere, prefill-compute-only" conclusion from a different codebase.
- **The sm_120/121 arch wall is the same one our kernel lanes fight.** PR threads say
  "compile `120f`, it covers sm_121," but flashinfer #3170 says `120f` cannot emit native
  block-scaled FP4 MMA, and llama.cpp #19662 reports the block-scale MMA *failing to
  compile* on sm_120 under CUDA 13.1. Unresolved and toolkit-dependent. Upstream also flags
  NVFP4 correctness risk (#22042). Refs: PR #17906, PR #22196, issues #18250 (SM120 native
  NVFP4 MoE — twin of vllm #31085), #19662, discussion #22042.

## Objectives, in order
**A. Unblock GGUF lm-eval accuracy (keystone — and a campaign-wide tool).**
Get **exact per-continuation-token logprobs from supplied tokens**, not top-N:
   1. Determine whether any endpoint on the pinned `b9536` server returns the logprob of
      a *specific supplied* token regardless of rank — native `/completion` with prompt
      echo / `post_sampling_probs` over prompt tokens, or `/v1/completions` `echo=true`
      with prompt `token_logprobs`. The decisive test is recovering the `zebra` token's
      own logprob, not its presence in a top list.
   2. If an endpoint can: build the lm-eval loglikelihood adapter on it (sum of
      continuation-token logprobs + greedy-match boolean), pass the likely+unlikely pair
      test, then run one tiny lm-eval GGUF task end to end.
   3. If the stock server cannot expose supplied-token logprobs at any setting, escalate:
      pin a newer llama.cpp build that does, or open a `jethac/llama.cpp` fork +
      issue-named worktree adding a loglikelihood/prompt-logprob endpoint. That is the
      "different native scoring path" the status doc already flags as the fallback.
Deliverable: `scripts/gguf_logprobs_probe.py` (or the native probe) passes, plus one tiny
lm-eval task row — separating GGUF accuracy from throughput as the campaign requires.

**B. Determine whether native NVFP4 actually dispatches *correctly* on sm_121.**
The code exists (above); the question is whether it fires and is right on GB10. The build
*is* the experiment — `jethac/llama.cpp` builds in minutes on the box, so compile it
yourself with different arch flags and observe:
   1. **Resolve the arch gating by building it** (the keystone): read where
      `BLACKWELL_NATIVE_FP4` / `BLACKWELL_MMA_AVAILABLE` are defined and what arch guards the
      native FP4 MMQ path — then *settle it empirically*. Compile `jethac/llama.cpp` on GB10
      with `CMAKE_CUDA_ARCHITECTURES=121a`, then `121`, then `120f`, and record which
      compile (reproducing or dodging #19662's block-scale-MMA failure) and what `cuobjdump`
      shows for the FP4 MMA. That directly answers the #3170-vs-"`120f` covers sm_121"
      contradiction for our toolkit (CUDA 13.0) — no guessing from source. Same arch wall as
      the kernel lanes; cross-ref the "SM120 ride-along" section of
      `docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md`.
   2. **Confirm what actually emits**: `cuobjdump` your own build (and the installed
      `b9536`) for the `mxf4nvf4.block_scale` MMA, and trace whether an FP4 model dispatches
      it vs falls back. Record as fact, not inference.
   3. **The cheap experiment** (queue, don't block the read on it): convert/grab one NVFP4
      GGUF (e.g. Qwen3.6 NVFP4 via modelopt) and run it on `b9536`. Capture: (a) does the
      native FP4 MMA dispatch on sm_121, (b) is output correct vs a bf16/Q8 reference, (c)
      prefill (PP) speed vs Q4_K_M. That single row is our first proof-or-disproof of
      native FP4 tensor-core use on this silicon in any runtime.
   4. **MXFP4/GPT-OSS still differentiated:** triton #8335 blocks GPT-OSS on `sm_121a` for
      every triton-based stack; llama.cpp doesn't use triton, so it may be the only working
      GPT-OSS path on Spark. Worth a row once the NVFP4 dispatch question is answered.
   5. Optional capacity angle: llama.cpp KV-cache quant (`-ctk`/`-ctv q8_0`/`q4_0`) is a
      practical-runtime parallel to the NVFP4-KV work; a matched KV-dtype capacity row lets
      us compare the practical path against vLLM/SGLang.

## When to set up jethac/llama.cpp
We have two independent reasons to own the source and a build: the native-NVFP4-on-sm_121
investigation (Objective B) and a possible loglikelihood endpoint (Objective A.3). Do not
let that obscure the accuracy stop point: first run the newer-pin echo-span probe below.
Create the fork/submodule immediately after that probe fails, or when explicitly switching
to the native-FP4 lane.

1. Fork `ggml-org/llama.cpp` → `jethac/llama.cpp`; add as submodule `third_party/llama.cpp`.
2. Create the issue-named worktree branch off a **recent upstream master** ref (or the
   relevant native-FP4 PR, e.g. #22196) — pin + record the exact commit for reproducibility
   (master NVFP4 is in flux, so pin a chosen commit, don't float HEAD). Build it yourself
   on GB10 for the native-FP4 lane. Record what commit `b9536` was built from too, kept
   only as the serving baseline to diff against.
3. Build freely for source inspection and dispatch proof; just don't ship a kernel patch
   yet. Reading answers part of the arch question; building with different arch flags
   answers the rest (Objective B.1). Produce a findings artifact for Objective B.1/B.2,
   and for Objective A locate where
   `llama-perplexity`'s `--multiple-choice`/`--hellaswag` modes read per-token logits for
   *supplied* tokens (the loglikelihood primitive to lift into a server endpoint) and whether
   master's server already exposes prompt-token logprobs.
4. New issue for "native FP4 on sm_121" (distinct from #8 accuracy, #17 serving), or fold
   under #17 — operator's call.

**C. Add the larger Qwen3/Qwen3.6 GGUF serving row.** Same `b9536` CUDA build + row
recorder; closes the counterpart-matrix gap beyond Qwen2.5 1.5B. Low effort.

**D. Keep the practical serving recipes blessed and current.** Already validated; fold any
new model rows into `recipes/llamacpp_serving.md` and keep the `--reasoning off` /
thinking-disabled finding visible (it's the difference between empty and useful content).

## Evidence gates (a row isn't a claim without these)
- Build-target evidence (`CUDA : ARCHS = 1210`, build commit) recorded per row.
- For accuracy: the probe recovers an *unlikely* supplied token's own logprob; a tiny
  lm-eval task completes with paper-shaped loglikelihood tuples.
- For native FP4: cuobjdump/dispatch evidence that the FP4 tensor-core path actually
  executes on the model under test — not merely that the build supports it.
- Throughput rows keep backend, model, quant format, flags, and `llama-bench` numbers.
- Explicit claim-class labels (see guardrails) on every row.

## Guardrails
- **Three claim classes, never merged:** (1) practical serving — blessed; (2) lm-eval
  accuracy — blocked; (3) native-FP4 dispatch — unproven. `BLACKWELL_NATIVE_FP4=1` in a
  build never implies native FP4 is used at runtime.
- **top-N is not loglikelihood.** Do not ship an accuracy adapter that silently drops
  continuations outside the returned probability list — that under-counts exactly the
  hard cases lm-eval exists to measure.
- **A k-quant model proves nothing about native FP4.** Native FP4/MXFP4 claims require an
  actual NVFP4/MXFP4 GGUF plus runtime dispatch evidence.
- **SM120 / RTX PRO 6000 is simpler here than the kernel lanes — but not free.** llama.cpp
  is source-built per machine, so it sidesteps the wheel/cubin arch-portability problem:
  GB10 builds `CMAKE_CUDA_ARCHITECTURES=121` (`ARCHS=1210`), a RTX PRO 6000 builds `120`,
  no non-portable `a` cubins to ship. Caveat: source-built doesn't mean the FP4 path
  *compiles* on the consumer family — #19662 shows the block-scale MMA failing to build on
  sm_120 under CUDA 13.1. The arch wall just surfaces at build time instead of at
  cubin-packaging time. Document the right arch per card; don't assume a 1210 build is what
  a SM120 user runs.
- A `jethac/llama.cpp` fork + submodule + issue-named worktree is the home for **both** the
  deep-read and any later endpoint/kernel patch. Issues: #8 (accuracy), #17 (serving), plus
  a native-FP4-on-sm_121 issue.

## First concrete step
**Accuracy stop point:** the pinned `b9536` server already
failed both native top-N and OpenAI `echo=true` supplied-token probes. Before *writing* an
endpoint, run **one newer-pin echo-span probe** with `scripts/gguf_logprobs_probe.py`:
start a newer `llama-server` manually, then probe:

```bash
python3 scripts/gguf_logprobs_probe.py \
  --url http://127.0.0.1:18085 \
  --model qwen25-logprob-newer \
  --context "The capital of Japan is" \
  --continuation " zebra" \
  --max-tokens 0 \
  --output results/llamacpp_newer_echo_logprobs_max0.json

python3 scripts/gguf_logprobs_probe.py \
  --url http://127.0.0.1:18085 \
  --model qwen25-logprob-newer \
  --context "The capital of Japan is" \
  --continuation " zebra" \
  --max-tokens 1 \
  --output results/llamacpp_newer_echo_logprobs_max1.json
```

Pass condition: prompt `tokens` and `token_logprobs` cover continuation token ids
`[1147, 50213]`. If the newer pin still returns only generated-token
`choices[0].logprobs.content`, do not keep tuning top-N. Either run one bounded
full-vocab-practicality probe with `llamacpp_native_loglikelihood_probe.py`, or move
directly to `jethac/llama.cpp` with `third_party/llama.cpp` and an issue-named worktree
for a direct supplied-token loglikelihood endpoint.

Native FP4 can run in parallel as a separate lane after this docs stop point: set up
`jethac/llama.cpp` off a recent master ref, build on GB10, and use cuobjdump/runtime
dispatch evidence plus an actual NVFP4/MXFP4 GGUF to prove or deny the native FP4 path.
