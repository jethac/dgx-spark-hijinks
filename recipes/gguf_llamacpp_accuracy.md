# GGUF llama.cpp Accuracy Recipe

Status: blocked pending logprobs compatibility.

The initial personal benchmark run showed that llama.cpp throughput can work while lm-eval accuracy fails. Do not treat llama-bench throughput as proof that paper-comparable accuracy is available.

## Preflight

Start a llama.cpp OpenAI-compatible server for the target GGUF.

Then run:

```bash
python3 scripts/gguf_logprobs_probe.py \
  --url http://127.0.0.1:8080 \
  --output results/gguf_logprobs_probe.json
```

## Pass Criteria

The response must include arbitrary continuation-token logprobs in a shape that can score an echoed prompt/continuation pair. Generated-token top-k logprobs are not enough unless they can recover every requested continuation token's pre-sampling probability.

If this probe fails, do not run full lm-eval GGUF accuracy. Fix the adapter or pin a compatible llama.cpp server first.

## Current Result

The probe failed against stock llama.cpp `b9536` on 2026-06-07.

Artifact:

- `results/gguf_logprobs_probe_llamacpp_b9536_20260607T1145Z.json`

The server returned generated-token logprobs under `choices[0].logprobs.content`, but did not return `tokens` or `token_logprobs`. That is insufficient for the current lm-eval GGUF loglikelihood path.

The native `/tokenize` plus `/completion` top-N path has now been tested at `n_probs=512`
and failed the unlikely-continuation case: the likely continuations were scored, but the
`zebra` continuation was not present in the returned top-N probabilities. Do not keep
rerunning top-512 as an accuracy fix.

Next proof: test one newer llama.cpp server pin for supplied-token echo logprobs with
`scripts/gguf_logprobs_probe.py`. Pass only if prompt `tokens` and `token_logprobs` cover
the supplied continuation token ids, including the unlikely `zebra` case. If a newer pin
still returns only generated-token `choices[0].logprobs.content`, move to a bounded
full-vocabulary practicality probe or a `jethac/llama.cpp` endpoint fork.

Acceptance contract: `tasks/llamacpp_supplied_token_loglikelihood_contract_20260609.md`.
The required primitive is exact logprob for each supplied continuation token by token id,
plus summed logprob and greedy-match boolean. Top-N output is acceptable only when every
requested continuation token is present; the `" zebra"` smoke row must score.

Historical native top-N probe, now negative at `n_probs=512`:

```bash
python3 scripts/llamacpp_native_loglikelihood_probe.py \
  --url http://127.0.0.1:8080 \
  --n-probs 256 \
  --output results/llamacpp_native_loglikelihood_probe.json
```

The probe must pass both likely and unlikely continuations before any lm-eval adapter work is considered valid.

Once the probe passes, run the tiny JSONL task harness:

```bash
python3 scripts/llamacpp_native_loglikelihood_task.py \
  --url http://127.0.0.1:8080 \
  --n-probs 512 \
  --input tasks/llamacpp_loglikelihood_smoke.jsonl \
  --output results/llamacpp_native_loglikelihood_task.json
```

The harness scores each `context`/`continuation` pair through the same native `/tokenize` plus `/completion` path and emits `target_logprob_sum`, `all_tokens_greedy`, and an `lm_eval_loglikelihood_tuple` per row. The smoke task intentionally includes one likely continuation and one unlikely continuation. `results/llamacpp_native_loglikelihood_task_dryrun_20260608.json` proves the task file and command shape locally, but it is not live server proof.

Then audit the artifact against the supplied-token contract:

```bash
python3 scripts/llamacpp_loglikelihood_contract_audit.py \
  --artifact results/llamacpp_native_loglikelihood_task.json \
  --input tasks/llamacpp_loglikelihood_smoke.jsonl \
  --output results/llamacpp_native_loglikelihood_task_contract_audit.json
```

The audit must pass before using the GGUF row as lm-eval accuracy evidence.
