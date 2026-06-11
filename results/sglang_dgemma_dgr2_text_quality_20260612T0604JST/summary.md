# SGLang DiffusionGemma DG-R2 Text-Only Quality Baseline

Date: 2026-06-12 JST

Status: RED.

## Scope

This is the DG-R2 text-only quality baseline from `docs/SGLANG_DIFFUSIONGEMMA_DGR2_TEXT_QUALITY_PACKET_20260611.md`.

- Model: `google/diffusiongemma-26B-A4B-it`
- Runtime image: `sglang-source-stack-dgemma-024-0705924c-f99323bd`
- Repo head: `0552a99b6620339c71bf966354e8ef08ee134aca`
- SGLang head: `651d55cd2e6a3d90de0eb65af643d0aa4ee7fca2`
- FlashInfer head: `f99323bd7d1cc88d9445202c12934070be754e2d`
- KV dtype: `auto` / BF16
- Attention backend: Triton, forced by DiffusionGemma policy because global `head_dim=512`
- Page size: 256
- `mem_fraction_static`: 0.55
- CUDA graphs: disabled by diffusion LLM path
- Mode: text-only; no image/multimodal claim

## Server Result

The server loaded from the local HF snapshot and reached serving readiness:

```text
Found local HF snapshot for google/diffusiongemma-26B-A4B-it ...
Load weight end. elapsed=308.96 s, type=DiffusionGemmaForBlockDiffusion, avail mem=62.40 GB, mem usage=49.17 GB.
KV Cache is allocated. dtype: torch.bfloat16, #tokens: 56320, K size: 5.40 GB, V size: 5.40 GB
KV Cache is allocated. dtype: torch.bfloat16, #tokens: 70656, K size: 0.68 GB, V size: 0.68 GB
The server is fired up and ready to roll!
```

The live warning list is the same class audited in `results/sglang_dgemma_dgr2_weight_warning_audit_20260611TmanualJST.md`; this row does not expand scope to image prompts.

Operational note: `/health` returned 503 even after SGLang logged readiness. For this row, `model_info` and successful `POST /v1/chat/completions` requests are the readiness evidence.

## Quality Gate

Gate: three prompts, two repeats each, byte-stable normalized output required, plus prompt-specific parse rule.

| Prompt | Stable | Answer rule | Result | Output |
|---|---:|---|---:|---|
| capital of Japan | yes | contains `Tokyo` | RED | empty string both repeats |
| 2 + 2 | yes | contains standalone `4` | RED | empty string both repeats |
| DGX Spark use | yes | mentions local/desktop/development AI use | GREEN | `The NVIDIA DGX Spark desktop is designed for high-performance AI development, data science, and machine learning prototyping in a compact, desktop-class form factor.` |

The failed prompts returned HTTP 200 with `finish_reason="length"` and nonzero `completion_tokens`, but `message.content=""`.

This is a deterministic quality red, not a load failure and not a crash.

## Artifacts

- Responses: `responses.json`
- Gate: `gate.json`
- Server log: `server.log`
- Run metadata: `run_meta.json`
- Model info: `model_info_before_client.json`
- Stop state: `docker_ps_after.txt`, `marker_after.txt`, `free_after.txt`

## Stop State

```text
marker: absent
docker ps: empty
```
