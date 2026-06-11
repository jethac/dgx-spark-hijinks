# SGLang DiffusionGemma Cookbook-Style Conformance Row

Date: 2026-06-12 JST

Status: RED under the zero-bug deterministic gate; semantically confirms the
documented OpenAI-chat cookbook path is runnable.

## Scope

This row follows the public SGLang DiffusionGemma cookbook shape:

- `sglang serve --model-path google/diffusiongemma-26B-A4B-it`
- `--dllm-algorithm Gemma4Renoise`
- `--trust-remote-code`
- no explicit `--dllm-algorithm-config`
- OpenAI-compatible chat requests

The cookbook says the runtime settings for `Gemma4Renoise` are applied
automatically: Triton attention, eager mode, and unchunked prefill. It also
uses broad chat prompts with large `max_tokens`, for example "What are the key
differences between TCP and UDP?" with `max_tokens=1024`.

Reference: <https://docs.sglang.io/cookbook/autoregressive/Google/DiffusionGemma>

## Runtime

- Model: `google/diffusiongemma-26B-A4B-it`
- Runtime image: `sglang-source-stack-dgemma-024-0705924c-f99323bd`
- KV dtype: `auto` / BF16
- `dllm_algorithm_config`: `None`
- Attention backend: Triton, auto-forced by the DiffusionGemma policy
- Page size: 256
- CUDA graphs: disabled by the diffusion LLM path
- `mem_fraction_static`: 0.55

Server proof:

```text
Attention backend forced to triton for DiffusionGemma (head_dim 512 exceeds the flashinfer/fa3 cap).
dllm_algorithm='Gemma4Renoise', dllm_algorithm_config=None
Load weight end. elapsed=382.51 s, type=DiffusionGemmaForBlockDiffusion, avail mem=61.91 GB, mem usage=49.74 GB.
KV Cache is allocated. dtype: torch.bfloat16, #tokens: 54016, K size: 5.18 GB, V size: 5.18 GB
KV Cache is allocated. dtype: torch.bfloat16, #tokens: 67584, K size: 0.65 GB, V size: 0.65 GB
The server is fired up and ready to roll!
```

## Results

The exact cookbook prompt passed semantically twice, but not byte-identically:

| Prompt | Repeat 0 | Repeat 1 | Gate |
|---|---|---|---|
| `What are the key differences between TCP and UDP?` | coherent TCP/UDP comparison mentioning both protocols | coherent TCP/UDP comparison mentioning both protocols | RED only because text is not byte-identical |
| `What is the capital of Japan?` | `The capital of Japan is **Tokyo**.` | same | GREEN |
| `What is 2 + 2?` | `2 + 2 = 4` | same | GREEN |
| `Write the digit four.` | `4` | empty | RED |
| DGX Spark descriptive prompt | coherent local-AI/workstation sentence | coherent local-AI/workstation sentence with different wording | RED only because text is not byte-identical |

`cookbook_conformance_rc.txt` is `1` because the row requires byte-stability
and all expectation checks. This is the correct zero-bug result: broad
cookbook-style prompts are coherent, but the default no-YAML path is not
deterministic enough for a quality-green baseline, and a very short prompt can
still collapse to empty output.

## Interpretation

This resolves the immediate DG-R2 ambiguity:

- SGLang's documented DiffusionGemma OpenAI-chat path is runnable on GB10.
- The exact cookbook TCP/UDP prompt behaves like the docs imply: coherent text,
  not the EOS-only failure from the terse DG-R2 gate prompts.
- The stock default path without an explicit seed/config is not a zero-bug
  deterministic baseline; repeated broad answers can be semantically equivalent
  but not byte-identical.
- The earlier seeded DG-R2 prompt diagnostic remains the better reproducible
  failure artifact for prompt sensitivity.

Do not promote DG-R2 to green from this row. Treat the current state as:

- DG-R1 runnable smoke: GREEN.
- DG-R2 text-only quality baseline: RED.
- Cookbook conformance: semantically runnable but deterministic-gate RED.

## Artifacts

- Raw responses: `cookbook_conformance.json`
- Client stdout: `cookbook_conformance.stdout`
- Server log: `server.log`
- Run metadata: `run_meta.json`
- Stop state: `docker_ps_after.txt`, `marker_after.txt`, `free_after.txt`

## Stop State

```text
marker: absent
docker ps: empty
available memory: ~115 GiB
```
