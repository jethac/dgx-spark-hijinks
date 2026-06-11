# SGLang DiffusionGemma DG-R2 Prompt Diagnostic

Date: 2026-06-12 JST

Status: DIAGNOSTIC COMPLETE. DG-R2 remains RED.

## Scope

This is a follow-up diagnostic for the DG-R2 text-only quality red in
`results/sglang_dgemma_dgr2_text_quality_20260612T0604JST/summary.md`.
It does not change the original DG-R2 gate.

- Model: `google/diffusiongemma-26B-A4B-it`
- Runtime image: `sglang-source-stack-dgemma-024-0705924c-f99323bd`
- KV dtype: `auto` / BF16
- Attention backend: Triton, forced by DiffusionGemma policy
- Page size: 256
- `mem_fraction_static`: 0.55
- CUDA graphs: disabled by the diffusion LLM path
- Diagnostic script: `scripts/diffusion_gemma_dgr2_prompt_diagnostic.py`

The run waited for a Docker-empty Spark window, claimed the marker, launched a
single server under `--memory=100g --memory-swap=100g`, and cleaned up afterward.

## Server Result

The server loaded and reached readiness:

```text
Load weight end. elapsed=367.45 s, type=DiffusionGemmaForBlockDiffusion, avail mem=62.35 GB, mem usage=49.18 GB.
KV Cache is allocated. dtype: torch.bfloat16, #tokens: 56320, K size: 5.40 GB, V size: 5.40 GB
KV Cache is allocated. dtype: torch.bfloat16, #tokens: 70656, K size: 0.68 GB, V size: 0.68 GB
The server is fired up and ready to roll!
```

`prompt_diagnostic_rc.txt` is `0`. The wrapper process returned nonzero only
because the ad hoc shell wrapper tried to `exit` with a CRLF-tainted `0`; cleanup
still completed and preserved the artifacts.

## Findings

The original empty outputs are explained by special-token stripping, but the
root cause is prompt sensitivity in the current upstream DiffusionGemma runtime:
the terse constrained chat prompts denoise to all-special-token canvases.

| Probe | Chat output | Chat with `skip_special_tokens=false` | Interpretation |
|---|---|---|---|
| `Answer with only the city name: What is the capital of Japan?` | empty | repeated `<eos>` | Original DG-R2 failure reproduced; visible empty text is all-special output. |
| `In one short sentence, what is the capital of Japan?` | `The capital of Japan is Tokyo.` | answer followed by repeated `<eos>` | Less constrained chat prompt passes. |
| `What is the capital of Japan?` | `The capital of Japan is **Tokyo**.` | answer + `<turn|>` + repeated `<eos>` | Direct chat prompt passes; full-canvas `max_tokens=256` gives the same answer. |
| `Answer with only the number: What is 2 + 2?` | empty | repeated `<eos>` | Original DG-R2 failure reproduced. |
| `In one short sentence, what is 2 + 2?` | `2 + 2 equals 4.` | answer followed by repeated `<eos>` | Less constrained chat prompt passes. |
| `What is 2 + 2?` | `2 + 2 = 4` | answer + `<turn|>` + repeated `<eos>` | Direct chat prompt passes; full-canvas `max_tokens=256` gives the same answer. |
| `Write the digit four.` | `4` | `4<turn|>` + repeated `<eos>` | Short direct chat can pass. |
| DGX Spark descriptive prompt | coherent DGX Spark sentence | sentence followed by repeated `<eos>` | Prior DG-R1/DG-R2 coherent prompt behavior reproduced. |

The native `/generate` endpoint is not a valid text-quality substitute for this
model in the current runtime. For the same probes, it returns mostly EOS-only
or unrelated/gibberish text, while the OpenAI chat endpoint returns coherent
answers for less constrained forms. The chat template is therefore part of the
current runnable path.

The `max_tokens=256` full-canvas variants do not rescue the problem. They behave
like their shorter direct-chat equivalents: direct prompts pass, while raw
`/generate` remains broken. This makes a simple "answer appears later in the
256-token canvas" explanation unlikely for the chat endpoint.

## Verdict

DG-R2 text-only remains RED because the original deterministic gate prompts
still fail. The diagnostic narrows the failure from "DiffusionGemma cannot
answer factual text prompts" to:

- server/load/text weights are sufficient for some coherent chat output;
- OpenAI chat formatting is required;
- terse "answer only" constraints can collapse to EOS-only canvases under the
  current `Gemma4Renoise` settings;
- the native `/generate` endpoint is not quality-green for this model.

Next useful work is to compare these SGLang `Gemma4Renoise` settings and chat
template behavior against the upstream cookbook/reference path before changing
the quality gate.

## Artifacts

- Raw diagnostic: `prompt_diagnostic.json`
- Server log: `server.log`
- Wait/claim log: `waiter.log`
- Run metadata: `run_meta.json`
- Stop state: `docker_ps_after.txt`, `marker_after.txt`, `free_after.txt`

## Stop State

```text
marker: absent
docker ps: empty
available memory: ~115 GiB
```
