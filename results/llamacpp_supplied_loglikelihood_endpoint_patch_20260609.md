# llama.cpp Supplied-Token Loglikelihood Endpoint Patch

Date: 2026-06-09

Status: fork endpoint implemented and compile-checked; GB10 runtime validation pending.

## Fork

- upstream/fork: `ggml-org/llama.cpp` -> `jethac/llama.cpp`
- branch: `spark/hijinks-008-supplied-loglikelihood`
- commit: `aa6a5961977139f23ae54dc8279fdac3d1494a77`
- worktree: `B:/workshop/worktrees/llama.cpp/spark-hijinks-008-supplied-loglikelihood`
- issue: https://github.com/jethac/dgx-spark-hijinks/issues/8

## Change

The fork adds a queued llama-server scoring endpoint:

- `POST /loglikelihood`
- `POST /v1/loglikelihood`

Request shape:

```json
{
  "context": "The capital of Japan is",
  "continuation": " zebra"
}
```

Response shape:

```json
{
  "context_token_ids": [],
  "continuation_token_ids": [],
  "continuation_token_logprobs": [],
  "target_logprob_sum": 0.0,
  "all_tokens_greedy": false,
  "lm_eval_loglikelihood_tuple": [0.0, false],
  "tokens_evaluated": 0
}
```

Implementation notes:

- Adds `SERVER_TASK_TYPE_LOGLIKELIHOOD`.
- Does not use the HTTP thread for model work.
- Reuses llama-server slot scheduling and queue/result plumbing.
- Scores supplied continuation tokens by reading logits for the previous token position.
- Computes target-token log-softmax directly from full logits rather than sorting top-N.
- Disables prompt cache for the scoring task to avoid skipped logits.
- Decodes at most one scored source token per batch so the endpoint does not require a broad global output-buffer change.
- Rejects multimodal payloads for this first endpoint, but text-only requests remain usable when the loaded model has multimodal support.

## Local Validation

Passed:

```text
git diff --check
```

Passed CPU-only WSL compile:

```bash
cmake -S /tmp/llama-loglikelihood-src \
  -B /tmp/llama-loglikelihood-build \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_CURL=OFF \
  -DGGML_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -G "Unix Makefiles"

cmake --build /tmp/llama-loglikelihood-build --target llama-server -j 4
```

Relevant build result:

```text
[ 97%] Building CXX object tools/server/CMakeFiles/server-context.dir/server-context.cpp.o
[ 98%] Linking CXX static library libserver-context.a
[100%] Building CXX object tools/server/CMakeFiles/llama-server-impl.dir/server.cpp.o
[100%] Linking CXX executable ../../bin/llama-server
[100%] Built target llama-server
```

Limitations:

- This was a CPU-only x86_64 compile check in WSL, not a CUDA/GB10 build.
- No live model endpoint smoke ran because the GB10 host was not reachable over Tailnet.
- The temporary WSL build directory was not available to a later WSL invocation, so no `--help` smoke is claimed.

## Remaining Gate

Run on the GB10 host when access is restored:

1. Build the fork on the Linux GB10 host.
2. Serve a small known GGUF model.
3. Run `tasks/llamacpp_loglikelihood_smoke.jsonl` through `/loglikelihood`.
4. Audit the artifact with `scripts/llamacpp_loglikelihood_contract_audit.py`.
5. Promote row 8 only if the unlikely `" zebra"` continuation has exact token logprobs and the audit passes.

Current host-access state: `results/gb10_host_access_probe_tailnet_retry_20260609.md`
records the Tailnet node visible but unusable for live work: Tailscale ping, TCP/22,
and SSH timed out with peer `rx_bytes=0`.
