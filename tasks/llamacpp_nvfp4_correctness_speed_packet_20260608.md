# llama.cpp NVFP4 Correctness/Speed Run Packet, 2026-06-08

Purpose: advance the llama.cpp native NVFP4 lane after
`results/llamacpp_nvfp4_runtime_gate_20260608T1748JST_summary.md`.

The runtime dispatch gate is already green for the pinned
`jethac/llama.cpp@19bba67c1f4db723c60a0d421aa0788bf4ddc699`: a converted
NVFP4 GGUF loaded on GB10 and Nsight Systems saw `GGML_TYPE_NVFP4` matmul
kernels. The next stop point is correctness first, then speed. Do not claim
native FP4 speed quality until the correctness probe below has a recorded result.

This workspace did not have the Linux artifacts needed for a live run:

- `/home/jethac/src/llama.cpp-native-fp4-sm121-19bba67/build-native-fp4-121a-20260608T164933JST/bin/llama-server`
- `/home/jethac/spark_tmp/llamacpp_nvfp4_runtime_gate_20260608T1748JST/qwen36-nvfp4-nvfp4.gguf`
- `/home/jethac/models/aeon/qwen36-nvfp4`

Run the packet on the GB10 Linux host that has the prior runtime-gate artifacts.

## Required Inputs

Set these paths before running:

```bash
set -euo pipefail

LLAMA_DIR=/home/jethac/src/llama.cpp-native-fp4-sm121-19bba67
BUILD=$LLAMA_DIR/build-native-fp4-121a-20260608T164933JST
NVFP4_GGUF=/home/jethac/spark_tmp/llamacpp_nvfp4_runtime_gate_20260608T1748JST/qwen36-nvfp4-nvfp4.gguf

# Must be the same model family/checkpoint lineage as the NVFP4 GGUF.
# Prefer BF16. Q8_0 is acceptable if BF16 is not available.
REF_GGUF=${REF_GGUF:?set to a BF16 or Q8_0 Qwen3.6 reference GGUF}

RUN_ID=llamacpp_nvfp4_correctness_speed_$(date +%Y%m%dT%H%MJST)
RUN_DIR=/home/jethac/spark_tmp/$RUN_ID
REPO_RESULTS=/home/jethac/dgx-spark-hijinks/results/$RUN_ID
mkdir -p "$RUN_DIR" "$REPO_RESULTS"

for p in \
  "$BUILD/bin/llama-perplexity" \
  "$BUILD/bin/llama-cli" \
  "$BUILD/bin/llama-bench" \
  "$NVFP4_GGUF" \
  "$REF_GGUF"
do
  test -e "$p" || { echo "missing: $p" >&2; exit 2; }
done

{
  echo "run_id=$RUN_ID"
  echo "llama_dir=$LLAMA_DIR"
  git -C "$LLAMA_DIR" rev-parse HEAD
  "$BUILD/bin/llama-cli" --version || true
  nvidia-smi -L || true
  ls -lh "$NVFP4_GGUF" "$REF_GGUF"
} > "$REPO_RESULTS/run_info.txt" 2>&1
```

## Correctness Gate

Create a small fixed corpus. It is intentionally over 100 tokens so
`llama-perplexity --kl-divergence` prints summary statistics:

```bash
cat > "$RUN_DIR/correctness_corpus.txt" <<'EOF'
Tokyo is the capital of Japan. The city is known for rail stations, dense neighborhoods, and careful signage.
For a simple arithmetic check, two plus two equals four. A reliable assistant should not answer five.
Blackwell native FP4 inference should preserve ordinary factual completions, short reasoning, and stable next-token preferences.
The benchmark sentence repeats useful everyday facts: water freezes at zero degrees Celsius, the sun rises in the east, and Paris is the capital of France.
EOF
```

Save the reference log-probabilities, then compare the NVFP4 GGUF against them:

```bash
/usr/bin/time -f "elapsed=%E maxrss_kb=%M" \
  "$BUILD/bin/llama-perplexity" \
  -m "$REF_GGUF" \
  -f "$RUN_DIR/correctness_corpus.txt" \
  -c 512 \
  -b 512 \
  -ub 512 \
  -ngl 999 \
  --save-all-logits "$RUN_DIR/ref_logprobs.bin" \
  > "$REPO_RESULTS/ref_perplexity.stdout" \
  2> "$REPO_RESULTS/ref_perplexity.stderr"

/usr/bin/time -f "elapsed=%E maxrss_kb=%M" \
  "$BUILD/bin/llama-perplexity" \
  -m "$NVFP4_GGUF" \
  -f "$RUN_DIR/correctness_corpus.txt" \
  -c 512 \
  -b 512 \
  -ub 512 \
  -ngl 999 \
  --kl-divergence \
  --kl-divergence-base "$RUN_DIR/ref_logprobs.bin" \
  > "$REPO_RESULTS/nvfp4_kld.stdout" \
  2> "$REPO_RESULTS/nvfp4_kld.stderr"
```

Run two deterministic generation smokes against both models:

```bash
for model_name in ref nvfp4; do
  if [ "$model_name" = ref ]; then model="$REF_GGUF"; else model="$NVFP4_GGUF"; fi

  "$BUILD/bin/llama-cli" \
    -m "$model" \
    -p "Answer with one word. The capital of Japan is" \
    -n 4 \
    --temp 0 \
    --seed 1 \
    -c 512 \
    -ngl 999 \
    --no-display-prompt \
    > "$REPO_RESULTS/${model_name}_tokyo.txt" \
    2> "$REPO_RESULTS/${model_name}_tokyo.stderr"

  "$BUILD/bin/llama-cli" \
    -m "$model" \
    -p "Answer with one digit. 2 + 2 =" \
    -n 4 \
    --temp 0 \
    --seed 1 \
    -c 512 \
    -ngl 999 \
    --no-display-prompt \
    > "$REPO_RESULTS/${model_name}_2plus2.txt" \
    2> "$REPO_RESULTS/${model_name}_2plus2.stderr"
done
```

Correctness pass signal for this stop point:

- all commands exit `0`;
- `nvfp4_kld.stdout` contains finite `Mean    KLD` and `Same top p` values;
- no stdout/stderr contains `nan`, `inf`, `failed`, or `error loading model`;
- both `ref_tokyo.txt` and `nvfp4_tokyo.txt` contain `Tokyo`;
- both `ref_2plus2.txt` and `nvfp4_2plus2.txt` contain `4`;
- provisional quality is acceptable only if `Same top p >= 80%` on this corpus.

Fail signal:

- any load/crash/CUDA error;
- non-finite KLD output;
- `Same top p < 80%`;
- either NVFP4 deterministic smoke misses `Tokyo` or `4`.

If the pass signal fails, stop and summarize the failure. Do not run or cite speed as a
native FP4 win.

## Speed Gate

Run speed only after the correctness gate has a finite acceptable result. Keep the reference
row next to the NVFP4 row so the claim is matched.

```bash
for model_name in ref nvfp4; do
  if [ "$model_name" = ref ]; then model="$REF_GGUF"; else model="$NVFP4_GGUF"; fi

  "$BUILD/bin/llama-bench" \
    -m "$model" \
    -ngl 999 \
    -c 2048 \
    -p 512 \
    -n 128 \
    -r 5 \
    > "$REPO_RESULTS/${model_name}_bench_pp512_tg128.txt" \
    2> "$REPO_RESULTS/${model_name}_bench_pp512_tg128.stderr"

  "$BUILD/bin/llama-bench" \
    -m "$model" \
    -ngl 999 \
    -c 4096 \
    -p 2048 \
    -n 128 \
    -r 5 \
    > "$REPO_RESULTS/${model_name}_bench_pp2048_tg128.txt" \
    2> "$REPO_RESULTS/${model_name}_bench_pp2048_tg128.stderr"
done
```

Speed pass signal:

- all four bench commands exit `0`;
- each output reports backend `CUDA` and device `NVIDIA GB10`;
- NVFP4 `pp512` and `pp2048` rows are recorded separately from `tg128`;
- the summary states whether NVFP4 prefill is faster, tied, or slower than the reference.

## Summary Artifact

After running, add:

```bash
cat > "$REPO_RESULTS/summary.md" <<EOF
# llama.cpp NVFP4 Correctness/Speed, $(date +%Y-%m-%d)

Status: FILL_IN pass/fail.

Inputs:

- llama.cpp commit: $(git -C "$LLAMA_DIR" rev-parse HEAD)
- NVFP4 GGUF: $NVFP4_GGUF
- reference GGUF: $REF_GGUF

Correctness:

- Mean KLD: FILL_IN
- Same top p: FILL_IN
- Tokyo smoke: FILL_IN
- 2+2 smoke: FILL_IN

Speed:

- ref pp512/tg128: FILL_IN
- nvfp4 pp512/tg128: FILL_IN
- ref pp2048/tg128: FILL_IN
- nvfp4 pp2048/tg128: FILL_IN

Claim boundary:

- This is a small correctness/speed gate, not a full lm-eval row.
- Native FP4 dispatch evidence remains the prior Nsight runtime gate unless this run also
  captures a fresh profiler trace.
EOF
```

Then copy only compact artifacts into the repo. Keep large GGUF/logprob binaries under
`/home/jethac/spark_tmp`.
