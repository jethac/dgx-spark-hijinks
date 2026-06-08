# Direction: llama.cpp on Spark (unblock GGUF accuracy; prove or deny native FP4)

> Standing direction for the llama.cpp lane. llama.cpp is the **practical daily-driver**
> on this box — by far the fastest runtime (~175 tok/s Qwen2.5 1.5B, ~77 tok/s Gemma 4
> 26B) and already blessed for serving. Two things are still red and both are real
> campaign deliverables: **GGUF lm-eval accuracy** (blocked since day one) and **native
> sm_121a FP4 dispatch** (compiled in, never proven). This lane is mostly orthogonal to
> NVFP4 KV, but it owns the campaign's *accuracy oracle* and its best shot at MXFP4.

## Why this, why now
"What Counts As Fixed" explicitly requires llama.cpp GGUF throughput and lm-eval accuracy
to be **separated and validated**. Throughput is done; accuracy has been blocked the
entire campaign (120 smoke rows `loader_failed` from a logprobs/schema mismatch). Fixing
it does more than close the llama.cpp lane — it gives the *whole campaign* a working
quantization-accuracy measurement tool, which we currently lack for any format.

## Methodology / sequencing constraint
llama.cpp is native host binaries, not containers — so the constraint is different from
the vLLM/SGLang lanes. Do NOT rebuild llama.cpp in the dev loop unless an upstream
endpoint patch is genuinely required. Iterate against the already-validated running
`llama-server` (`b9536`, `308f61c31`) with the Python probe scripts. Rebuild (or fork)
only when Objective A proves the stock server cannot expose supplied-token logprobs.

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
2. **Native sm_121a FP4 dispatch is unproven.** `BLACKWELL_NATIVE_FP4=1` and `ARCHS=1210`
   are *compiled in*, but Q4_0/Q4_K_M are llama.cpp k-quants, not NVFP4/MXFP4 — so the
   tested models never exercise the native FP4 path. A k-quant model on an FP4-capable
   build proves nothing about native FP4 tensor-core dispatch.
3. **Counterpart coverage gap.** Only Qwen2.5 1.5B exists; the speed/counterpart matrix
   still wants a larger Qwen3/Qwen3.6-class GGUF row on the same `b9536` build.

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

**B. Prove or deny native sm_121a FP4 — and chase MXFP4/GPT-OSS as a differentiated win.**
   1. Use the build-target audit + a runtime dispatch probe (cuobjdump / JIT-cache /
      kernel trace) to determine what llama.cpp's FP4 path actually emits and whether it
      dispatches at all on the current models. Record it as fact, not inference.
   2. The high-value experiment: run an actual **MXFP4 GGUF (GPT-OSS-class)** on llama.cpp.
      triton #8335 blocks GPT-OSS on `sm_121a` (`.tile::gather4 .shared::cluster`
      unsupported) for every triton-based stack — but **llama.cpp does not use triton**,
      so it may be the *only* working MXFP4-on-Spark path. If it serves with native FP4
      dispatch proven, that is a real, differentiated community contribution.
   3. Optional capacity angle: llama.cpp KV-cache quantization (`-ctk`/`-ctv q8_0`/`q4_0`)
      gives a practical-runtime capacity story parallel to the NVFP4-KV work; a matched
      KV-dtype capacity row would let us compare the practical path against vLLM/SGLang.

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
- **SM120 / RTX PRO 6000 is simpler here than the kernel lanes.** llama.cpp is source-built
  per machine, so it sidesteps the wheel/cubin arch-portability problem entirely: GB10
  builds with `CMAKE_CUDA_ARCHITECTURES=121` (`ARCHS=1210`), a RTX PRO 6000 would build
  `120`. There are no non-portable `a` cubins to ship. Just document the right arch per
  card in the recipe; do not assume a 1210 build is what a SM120 user runs.
- Upstream code changes (an accuracy endpoint) require a `jethac/llama.cpp` fork,
  submodule, and issue-named worktree. Issues: #8 (accuracy), #17 (serving).

## First concrete step (no rebuilds yet)
The pinned `b9536` server has already failed both native top-N and OpenAI `echo=true`
supplied-token probes. There is no `third_party/llama.cpp` submodule or local llama.cpp
worktree yet; do not create one until the stock newer-pin probe below fails.

Run **one newer-pin echo-span probe** with the existing `scripts/gguf_logprobs_probe.py`:
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
