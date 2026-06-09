# llama.cpp Supplied-Token Loglikelihood Contract, 2026-06-09

Status: offline contract packet; live server/fork work still required.

Purpose: define the exact GGUF scoring primitive needed before llama.cpp can be used as
the campaign's paper-comparable lm-eval accuracy path. The current failure is API
semantics, not CUDA, model quality, or throughput.

## Negative Evidence To Preserve

- OpenAI-compatible llama.cpp `b9536` echo probe:
  `results/llamacpp_gguf_echo_logprobs_probe_20260608_summary.json`
  - `max_tokens=0` and `max_tokens=1` return generated-token
    `choices[0].logprobs.content`.
  - They do not return prompt `tokens` / `token_logprobs` covering the supplied
    continuation span.
- Native `/tokenize` + `/completion` top-N probe:
  `results/llamacpp_native_loglikelihood_20260608T1331JST_summary.md`
  - likely continuations score,
  - unlikely `" zebra"` is absent from top-512,
  - therefore top-N membership is not an lm-eval loglikelihood primitive.

Do not accept larger `n_probs` as the fix unless a bounded full-vocabulary practicality
probe proves every supplied continuation token is returned at acceptable cost. The pass
condition is target-token logprob by token id, not presence in a sampled top list.

## Required Primitive

Input:

```json
{
  "context": "The capital of Japan is",
  "continuation": " zebra"
}
```

Output:

```json
{
  "context_token_ids": [ /* tokenizer ids with normal BOS/chat behavior */ ],
  "continuation_token_ids": [1147, 50213],
  "continuation_token_logprobs": [
    {"token_id": 1147, "logprob": -12.34, "is_greedy": false},
    {"token_id": 50213, "logprob": -8.90, "is_greedy": false}
  ],
  "target_logprob_sum": -21.24,
  "all_tokens_greedy": false,
  "lm_eval_loglikelihood_tuple": [-21.24, false]
}
```

Required invariants:

- every supplied continuation token has an exact logprob, regardless of rank;
- unlikely continuations score instead of returning `missing`;
- tokenization is recorded for context and continuation separately;
- `target_logprob_sum` is the sum of continuation-token logprobs only;
- `all_tokens_greedy` is computed from each continuation token's greedy match;
- generated-token logprobs alone are not accepted;
- top-N response shapes are accepted only if every continuation token is present.

## Acceptance Smoke

Use `tasks/llamacpp_loglikelihood_smoke.jsonl`.

Required pass:

- `tokyo_likely`: scores and `expected_greedy=true`;
- `zebra_unlikely`: scores and `expected_greedy=false`;
- `qwen_name`: scores;
- task summary: `target_found == total`, `expected_greedy_mismatches == 0`,
  `ok == true`.

The `zebra_unlikely` row is the guard against silently shipping a top-N adapter.

## Allowed Implementation Routes

1. Newer stock llama.cpp pin:
   - run `scripts/gguf_logprobs_probe.py` against `/v1/completions`;
   - convert the max0/max1 probe artifacts with
     `scripts/llamacpp_echo_logprobs_to_contract.py`;
   - pass only if echoed prompt `tokens` and `token_logprobs` cover the continuation span
     and the converted artifact passes `scripts/llamacpp_loglikelihood_contract_audit.py`.
2. Bounded full-vocabulary practicality probe:
   - use native endpoints only if they can return the target token's logprob for every
     supplied token, including `" zebra"`;
   - record latency/payload cost before treating it as a real adapter path.
3. `jethac/llama.cpp` endpoint:
   - add an issue-named worktree if stock endpoints still cannot expose supplied-token
     logprobs;
   - lift the supplied-token scoring logic from llama.cpp's multiple-choice /
     perplexity path instead of relying on sampling top-N output.

## Deliverable

One live artifact must show the contract above for the smoke JSONL task before row 8 can
move out of blocked:

```bash
python3 scripts/llamacpp_native_loglikelihood_task.py \
  --url http://127.0.0.1:8080 \
  --input tasks/llamacpp_loglikelihood_smoke.jsonl \
  --output results/llamacpp_supplied_token_loglikelihood_YYYYMMDDTHHMMJST.json
```

If this still uses top-N, the artifact must remain red when `" zebra"` is missing. A green
artifact requires direct supplied-token logprobs or a proven full-vocabulary equivalent.

Audit command:

```bash
python3 scripts/llamacpp_loglikelihood_contract_audit.py \
  --artifact results/llamacpp_supplied_token_loglikelihood_YYYYMMDDTHHMMJST.json \
  --input tasks/llamacpp_loglikelihood_smoke.jsonl \
  --output results/llamacpp_supplied_token_loglikelihood_YYYYMMDDTHHMMJST_contract_audit.json
```

Known-red checks:

- `results/llamacpp_native_loglikelihood_20260608T1331JST_contract_audit.json`
  rejects the live top-512 artifact because token id `1147` from `" zebra"` was not scored.
- `results/llamacpp_native_loglikelihood_task_dryrun_contract_audit_20260609.json`
  rejects the dry-run artifact because it has no actual token logprobs.
- `results/llamacpp_gguf_echo_logprobs_probe_20260608_contract_audit.json`
  rejects the pinned `b9536` echo artifacts because they expose generated-token
  `logprobs.content`, not prompt `tokens` / `token_logprobs`.

Expected queue artifacts for the newer-stock echo probe:

- `results/${RUN_ID}_max0.json`
- `results/${RUN_ID}_max1.json`
- `results/${RUN_ID}_contract_artifact.json`
- `results/${RUN_ID}_contract_audit.json`

Bridge the newer-stock max0/max1 probes into the contract artifact:

```bash
python3 scripts/llamacpp_echo_logprobs_to_contract.py \
  --input tasks/llamacpp_loglikelihood_smoke.jsonl \
  --probe results/${RUN_ID}_max0.json \
  --probe results/${RUN_ID}_max1.json \
  --output results/${RUN_ID}_contract_artifact.json

python3 scripts/llamacpp_loglikelihood_contract_audit.py \
  --artifact results/${RUN_ID}_contract_artifact.json \
  --input tasks/llamacpp_loglikelihood_smoke.jsonl \
  --output results/${RUN_ID}_contract_audit.json
```
