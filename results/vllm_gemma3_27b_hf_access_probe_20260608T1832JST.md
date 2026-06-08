# vLLM Gemma 3 27B HF Access Probe, 2026-06-08 18:32 JST

Purpose: verify that the gated `google/gemma-3-27b-it` model can be accessed from the
same container/cache shape used by the vLLM Gemma 3 27B Rung 1 packet, without recording
the token itself in the repository.

Host/user setup:

- A Hugging Face token file exists at `/home/jethac/.cache/huggingface/token`.
- `/home/jethac/.profile` exports `HF_TOKEN` from that token file when readable.
- SSH verification as `jethac` reports `HF_TOKEN_present=yes` and
  `token_file_readable=yes`.

Container probe:

- image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
- mounted cache: `/home/jethac/.cache/huggingface` -> `/hf-cache`
- model: `google/gemma-3-27b-it`
- model revision: `005ad3404e59d6023443cb575daa05336842228a`
- `huggingface_hub`: `1.9.2`
- token environment inside container: present

Result:

- `model_info` passed and reports `gated="manual"`, `private=false`, and `siblings=25`.
- Small snapshot download passed for:
  - `config.json`
  - `generation_config.json`
  - `special_tokens_map.json`
  - `tokenizer.json`
  - `tokenizer.model`
  - `tokenizer_config.json`
- Cache disk headroom at probe time: `2475908382720` free bytes on `/hf-cache`.

Interpretation:

The previous Gemma 3 Rung 1 blocker, missing gated Hugging Face auth, is cleared for the
`jethac` user and the container mount/env shape. The next failed run no longer died on HF
access; it died during the source-overlay editable install because the configured vLLM
precompiled wheel metadata commit was not published for the required `cu130` paths.

Follow-up blocker from the attempted packet run:

```text
Failed to fetch precompiled wheel metadata for variant cu130: HTTP Error 404: Not Found
Trying https://wheels.vllm.ai/8916796bc50926fd61e606718b194a71e2e31a24/cu130/vllm/metadata.json
Trying https://wheels.vllm.ai/8916796bc50926fd61e606718b194a71e2e31a24/vllm/metadata.json
metadata-generation-failed
```

Next action:

Do not rerun the current packet unchanged. Rebase the Gemma geometry overlay onto the
proven `a919d635d` lane, or otherwise select a vLLM source/precompiled-wheel pair with
published CUDA 13 metadata, then start with the fp8 comparator row before the NVFP4 row.
