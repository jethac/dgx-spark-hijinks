# SGLang DiffusionGemma DG-R1 Stock Runtime Smoke

Date: 2026-06-11 JST

Status: GREEN for DG-R1 stock-runtime smoke.

Scope: this proves that the upstream SGLang DiffusionGemma dLLM path can load
and answer coherently on GB10 through the correctness-first policy. It does not
claim FlashInfer, NVFP4, CUDA graph, or performance-path coverage.

## Code And Image

- Repo branch: `epoch2`
- SGLang branch: `spark/hijinks-024-diffusiongemma-upstream-rebase`
- SGLang commit: `651d55cd2e6a3d90de0eb65af643d0aa4ee7fca2`
- FlashInfer commit in source stack: `f99323bd7d1cc88d9445202c12934070be754e2d`
- Image: `sglang-source-stack-dgemma-024-0705924c-f99323bd`
- Image id: `sha256:b0f904496e768c12fee6da8aba6ba2fa84738b3d827b8c06dd6562f0add314d1`
- Host device: NVIDIA GB10, compute capability `12.1`

The run used a local `DiffusionGemmaConfig` fallback because the installed
Transformers build did not recognize `model_type=diffusion_gemma`. The fallback
only converts the checkpoint config into typed `text_config` and `vision_config`
objects; the model implementation and `Gemma4Renoise` algorithm are the upstream
SGLang implementation.

## Launch

```bash
python3 -m sglang.launch_server \
  --model-path google/diffusiongemma-26B-A4B-it \
  --dllm-algorithm Gemma4Renoise \
  --trust-remote-code \
  --context-length 8192 \
  --mem-fraction-static 0.55 \
  --host 0.0.0.0 \
  --port 30124
```

Container guardrails:

- `--memory=100g --memory-swap=100g`
- one server only
- no fp8/fp4 comparator concurrency
- `TRANSFORMERS_OFFLINE=1`
- `HF_HUB_OFFLINE=1`

## Runtime Evidence

`import_probe.json` records:

- `torch`: `2.12.0a0+5aff3928d8.nv26.05`
- `torch_cuda`: `13.2`
- `device`: `NVIDIA GB10`
- `capability`: `[12, 1]`
- `flashinfer`: `0.6.13`
- `sglang`: `0.0.0.dev1+g0705924c1`
- `config_class`: `DiffusionGemmaConfig`
- `text_config_class`: `DiffusionGemmaTextConfig`
- `head_dim`: `512`
- `swa_head_dim`: `256`
- `canvas_length`: `256`

`server.log` proves the stock policy:

- `Attention backend forced to triton for DiffusionGemma (head_dim 512 exceeds the flashinfer/fa3 cap).`
- `Setting page size to 256 for diffusion LLM inference`
- `chunked_prefill_size=-1`
- `disable_cuda_graph=True`
- `disable_piecewise_cuda_graph=True`
- `disable_overlap_schedule=True`
- `DiffusionGemmaForBlockDiffusion`
- `Gemma4Renoise`

Weight load and memory:

- load elapsed: `316.64 s`
- available memory before load: `111.47 GB`
- weight memory usage: `49.62 GB`
- available memory after load: `61.86 GB`
- BF16 SWA KV pool: `54272` tokens, K `5.20 GB`, V `5.20 GB`
- BF16 full KV pool: `67840` tokens, K `0.65 GB`, V `0.65 GB`
- `SWAKVPool mem usage: 11.70 GB`
- `max_total_num_tokens=67840`

The server reached readiness:

- `Uvicorn running on http://0.0.0.0:30124`
- `The server is fired up and ready to roll!`

Explicit smoke prompt:

```json
{
  "model": "google/diffusiongemma-26B-A4B-it",
  "messages": [
    {
      "role": "user",
      "content": "In one short sentence, say what the NVIDIA DGX Spark desktop AI computer is useful for."
    }
  ],
  "max_tokens": 64,
  "temperature": 0.0
}
```

Response:

> The NVIDIA DGX Spark is designed for high-performance local AI development,
> testing, and prototyping of machine learning models in a compact desktop form
> factor.

## Caveats

- This is a smoke row, not a quality baseline.
- The log reports many checkpoint keys as uninitialized, including router norm /
  root-size entries, V-norm entries, self-conditioning post-norm, and vision
  quantization/encoder entries. The output is coherent, but DG-R2 must audit
  these missing/uninitialized keys before any stronger quality claim.
- The first smoke prompt was ambiguous and produced an Apache Spark-oriented
  answer. The explicit DGX Spark prompt above is the citable response.

## Stop State

- Server container stopped.
- `docker ps` captured empty in `docker_ps_after.txt`.
- Claude marker was absent in `marker_after.txt`.
- `free_after.txt` shows `115Gi` available host memory.
