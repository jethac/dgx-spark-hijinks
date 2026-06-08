# SGLang Qwen FP4-KV Endpoint Metadata Localization

Date: 2026-06-08 JST

Run id: `sglang_qwen_fp4kv_endpoint_metadata_20260608T1819JST`

Scope: SGLang lane only. No Gemma, no capacity rows, no image builds.

## What Ran

Docker was not reachable from this Windows workspace:

```powershell
docker ps --format "{{.ID}} {{.Image}} {{.Names}} {{.Ports}}"
```

failed with:

```text
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine
```

So no live server probe was started. Instead, the new metadata probe analyzed the latest
prompt-reconciliation artifact:

```powershell
python -m py_compile scripts/sglang_fp4_endpoint_metadata_probe.py
python scripts/sglang_fp4_endpoint_metadata_probe.py `
  --input-reconcile results/sglang_qwen_fp4kv_prompt_path_reconcile_20260608T173754JST.json `
  --fp4-server-log results/sglang_qwen_fp4kv_prompt_path_reconcile_20260608T173754JST_fp4_server.log `
  --run-id sglang_qwen_fp4kv_endpoint_metadata_20260608T1819JST `
  --output results/sglang_qwen_fp4kv_endpoint_metadata_20260608T1819JST.json
python -c "import yaml; from pathlib import Path; data=yaml.safe_load(Path('scripts/sglang_fp4_first_token_dump_patch.yaml').read_text()); assert data['patches'][0]['target'].endswith('ModelRunner.sample'); print('yaml_ok')"
```

## Artifacts

- `results/sglang_qwen_fp4kv_endpoint_metadata_20260608T1819JST.json`
- `scripts/sglang_fp4_endpoint_metadata_probe.py`
- `scripts/sglang_fp4_first_token_dump_patch.yaml`

## Result

The FP4 endpoint split is now localized at request-state / pre-sampling state, not prompt
IDs:

| field | OpenAI Chat Completions | native `/generate` |
|---|---|---|
| prompt SHA-256 | `5a5d4572e0e3d940a909b85dc4a00350094cbd1d55333c3d4f0a7974a91ee517` | same prompt IDs via native replay |
| prompt tokens | `56` | `56` |
| request decode cap | `max_tokens=192` | `max_new_tokens=192` |
| first generated token | `**` | `ark` (`838`) |
| first-token same? | no | no |

The FP4 server log had both 28-layer `decode` and 28-layer `extend_merge_paged` traces,
but those trace lines are not request-tagged. They prove the FP4 backend path was active;
they do not bind a trace group to OpenAI versus native.

## Smallest Next Hook

Use SGLang's existing source-patcher dumper to patch only
`sglang.srt.model_executor.model_runner.ModelRunner.sample`. This is the narrow point
where `LogitsProcessorOutput.next_token_logits` and `ForwardBatch` are both present before
the sampler chooses the first generated token.

Run-packet shape for the next Linux/container pass:

```bash
export DUMPER_ENABLE=1
export DUMPER_NON_INTRUSIVE_MODE=off
export DUMPER_DIR=/tmp/sglang_fp4_first_token_dump
export DUMPER_SOURCE_PATCHER_CONFIG=/workspace/scripts/sglang_fp4_first_token_dump_patch.yaml
export SGLANG_FP4_FIRST_TOKEN_DUMP=1
```

Then start the existing FP4 Qwen source-overlay server with the same no-graph flags as the
reconciliation row and send one-token metadata probes:

```bash
python scripts/sglang_fp4_endpoint_metadata_probe.py \
  --url http://127.0.0.1:30013 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --case medium_decode \
  --max-new-tokens 1 \
  --run-id sglang_qwen_fp4kv_endpoint_metadata_live_YYYYMMDDTHHMMJST \
  --output results/sglang_qwen_fp4kv_endpoint_metadata_live_YYYYMMDDTHHMMJST.json
```

Expected dump contents:

- `fp4_first_token__next_token_logits` before logits preprocessing
- `fp4_first_token__next_token_logits` after logits preprocessing
- `fp4_first_token__input_ids`
- `fp4_first_token__positions`
- `fp4_first_token__seq_lens`

## Exact Blocker / Next Issue

The remaining blocker is request-tagged first-token pre-sampling numerics. OpenAI and
native can share prompt IDs yet choose different FP4 first tokens, while the current
backend trace lacks `rid`/endpoint/forward-mode binding. Capture the above tensors for
OpenAI and native one-token requests, then compare whether divergence appears before or
after `ModelRunner._preprocess_logits()`.
