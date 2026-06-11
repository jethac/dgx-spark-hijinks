# DG-0 Serving Stack Recon: DiffusionGemma on vLLM / DGX Spark

Web-research recon, 2026-06-11 (model released 2026-06-10). No GPU runs, no code changes.
Goal: pin down how DiffusionGemma (Google DeepMind block-diffusion model on the Gemma 4
26B-A4B MoE base) is served with vLLM, to plan the DG-0 baseline run on DGX Spark
(GB10, sm_121, aarch64, 128 GB unified memory).

---

## 1. The serving stack

**Answer: (a) upstream vLLM — but on an official `vllm-project` branch (`dgemma`) that is
NOT yet merged to `main` and NOT in any tagged release. Distribution today is via official
multi-arch docker tags (`vllm/vllm-openai:gemma*`), plus a community patch stack for Spark.
It is not a fork, not a third-party plugin, and not transformers-backend/trust_remote_code
inside vLLM (native model class). A NIM container also exists but is not the only path.**

Evidence:

- vLLM blog, "DiffusionGemma: The First Diffusion LLM (dLLM) Natively Supported in vLLM"
  (2026-06-10): https://vllm-project.github.io/2026/06/10/diffusion-gemma (also mirrored at
  https://vllm.ai/blog/2026-06-10-diffusion-gemma). Integration is built on **model runner
  v2's `ModelState` abstraction** ("allows models to define their custom input preparation
  and provides hooks for managing per-request model-specific state"; hooks:
  `prepare_inputs()`, `prepare_attn()`, `custom_sampler()`, `add_request()`,
  `remove_request()`). Supported attention backends: "Triton Attention (`TRITON_ATTN`) and
  FlashAttention 4 (`FLASH_ATTN`)" with dynamic per-sequence causal switching.
- **PR: vllm-project/vllm #45163 "[Model] Add DiffusionGemma Support"** —
  https://github.com/vllm-project/vllm/pull/45163 — **state: OPEN** (created 2026-06-10,
  base `main`, head `vllm-project:dgemma`, i.e. a branch in the official org). PR body
  verbatim: *"See: https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it. Docker images
  are available under: `vllm-openai:gemma-cu130`"*.
- The `dgemma` branch vs `main` (as of 2026-06-11): 21 commits ahead / 31 behind. Adds
  `vllm/model_executor/models/diffusion_gemma.py`, `vllm/config/diffusion.py`,
  `vllm/transformers_utils/configs/diffusion_gemma.py`, and modifies scheduler
  (`v1/core/sched/scheduler.py`), model runner v2 (`v1/worker/gpu/model_runner.py`,
  `model_states/interface.py`), sampler, attention backends (flash_attn.py,
  triton_attn.py, triton_unified_attention.py), spec-decode rejection sampler, and
  NVFP4/FlashInfer MoE paths (`trtllm_nvfp4_moe.py`, `flashinfer_cutlass_moe.py`).
- Release timeline: latest vLLM releases are v0.22.1 (2026-06-05) and v0.22.0
  (2026-05-29) — **both predate the model; DiffusionGemma support is in no tagged
  release as of 2026-06-11.**
- Lineage: the blog credits **RFC #36155** ("[RFC]: dLLM support via plugin (spec-decode
  path reuse)", Red Hat authors Alon Kellner / Tomer Garber, closed) for the idea of
  reusing the speculative-decoding data path. The earlier dllm-plugin PR #42653
  ("feat(dllm): non-causal attention, scheduler hooks, and FlashInfer support for block
  diffusion models", for LLaDA2.0) was **closed unmerged**; the DiffusionGemma
  integration is a separate, official implementation on model runner v2.
- Docker tags on Docker Hub (`vllm/vllm-openai`), all pushed 2026-06-10:
  `gemma` (multi-arch **arm64+amd64**), `gemma-cu129`, `gemma-aarch64-cu129`,
  `gemma-aarch64-cu130`, `gemma-x86_64-cu130`, etc. **arm64 + cu130 images exist, which
  is what GB10/Spark needs.**
- Not trust_remote_code: the vLLM model class is native
  (`DiffusionGemmaForConditionalGeneration`, registered for the HF architecture string
  `DiffusionGemmaForBlockDiffusion`). On the HF side it's native in **transformers
  5.8.0.dev0** (config has no `auto_map`, repo contains no `modeling_*.py` — see §3).
  Note the eugr recipes and NVIDIA model card still pass `--trust-remote-code`
  (harmless/belt-and-suspenders; the google repo ships no remote code).
- Other engines: model card also documents SGLang (`python3 -m sglang.launch_server
  --model-path google/diffusiongemma-26B-A4B-it ...`), Transformers, MLX, Unsloth,
  llama.cpp (GGUF), and a NIM container (§4).
- `build.nvidia.com/spark/vllm` could not be fetched in this session (timed out twice,
  likely JS-rendered) — **open item**; see §4 for what NVIDIA's blogs say instead.

## 2. The checkpoints

| Variant | HF id | Notes |
|---|---|---|
| BF16 (official) | `google/diffusiongemma-26B-A4B-it` | 51.7 GB, 11 safetensors shards, Apache 2.0. https://huggingface.co/google/diffusiongemma-26B-A4B-it |
| NVFP4 (Model Optimizer) | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` | listed as "14B params", tensor types BF16 / F8_E4M3 / U8. https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4 |
| GGUF (community) | `unsloth/diffusiongemma-26B-A4B-it-GGUF` | llama.cpp path, not relevant for DG-0 |
| FP8/NVFP4 | Red Hat AI hub mirrors mentioned in search results (not verified) | open |

Google model card serving section (the card itself is minimal):

```bash
pip install vllm
vllm serve "google/diffusiongemma-26B-A4B-it"
```

No version pin and no trust_remote_code on the google card. The real serving reference is
the **official vLLM recipe** (https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it),
verbatim (single GPU, BF16):

```bash
docker pull vllm/vllm-openai:gemma   # required image with diffusion support built in

docker run -itd --name diffusiongemma \
    --ipc=host --network host --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:gemma \
        --model google/diffusiongemma-26B-A4B-it \
        --max-model-len 262144 \
        --max-num-seqs 4 \
        --gpu-memory-utilization 0.85 \
        --host 0.0.0.0 --port 8000
```

Full-featured variant adds: `--mm-processor-kwargs '{"max_soft_tokens": 1120}'`,
`--limit-mm-per-prompt '{"image": 7}'`, `--enable-auto-tool-choice`,
`--tool-call-parser gemma4`, `--reasoning-parser gemma4`.

Diffusion-specific flags documented by the recipe / blog:

- `--diffusion-config '{"canvas_length": 256}'` — block size
- `--hf-overrides '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}'`
- `--generation-config vllm` — "Overrides checkpoint's 256-token default"
- `--max-num-seqs 4` — "Prevents CUDA OOM from diffusion state buffers"
- `--gpu-memory-utilization 0.85` — "Reserves activation memory for denoising"

NVIDIA NVFP4 model card serving section, verbatim (card carries a disclaimer: *"vllm
serve commands below are tentative and subject to change until the supporting vLLM image
is publicly released"*):

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve nvidia/diffusiongemma-26B-A4B-IT-NVFP4 \
  --trust-remote-code \
  --max-num-seqs 4 \
  --attention-backend TRITON_ATTN \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --override-generation-config '{"max_new_tokens": null}' \
  --default-chat-template-kwargs '{"enable_thinking":true}'
```

NVFP4 card details: *"obtained by quantizing the weights and activations of
Gemma-26B-A4B-IT to NVFP4"* (16 -> 4 bits/param); supported on Blackwell and Hopper
(H100); **tested on Blackwell B100 — not GB10/Spark**. Accuracy table (thinking on):
GPQA-D 69.4 -> 68.6, AIME25 68.33 -> 67.33, GSM8K 94.54 -> 94.01, MMLU-Pro 81.0 -> 80.7.
(Caveat: the card's quantization sentence names "Gemma-26B-A4B-IT", so it is slightly
ambiguous whether the table is the diffusion variant's own baseline — flag, don't assume.)

## 3. Internals (DG-1 prep): canvas / prefix KV cache

**Primary sources:** vLLM blog (above); vLLM `dgemma` branch model code
`vllm/model_executor/models/diffusion_gemma.py` (1,359 lines, public); HF `config.json`
and `generation_config.json`; Google model card at
https://ai.google.dev/gemma/docs/diffusiongemma/model_card. **No dedicated arXiv tech
report found as of 2026-06-11** — the closest academic ancestor is Block Diffusion
(BD3-LM), arXiv:2503.09573, plus the (unpublished) Gemini Diffusion work. Open question:
whether DeepMind publishes a report later.

Two-mode architecture — module docstring from `diffusion_gemma.py` verbatim:

> "Single Gemma4 backbone run in two modes (like YOCO):
> - encoder mode: causal attention, writes KV cache
> - decoder mode: bidirectional attention, reads encoder KV, doesn't write
>
> Same weights, same layers. The only decoder-unique component is a self-conditioning MLP."

So, answering the specific DG-1 questions:

- **Standard KV cache for the prefix: YES.** Encoder mode is ordinary causal Gemma 4
  attention and writes paged KV exactly as the autoregressive model would. The blog states
  the encoder "runs twice per block: once to prefill the prompt, and once to 'commit' a
  finished block", and that because committed KV is byte-identical to AR layout,
  **"vLLM's automatic prefix caching works out of the box."**
- **Bidirectional attention within the canvas: YES**, decoder/denoise mode only; it reads
  the prefix KV but never writes. Canvas length fixed at 256
  (`config.json: "canvas_length": 256`). Once a canvas converges it is committed
  (encoder pass) and a fresh canvas starts — block-autoregressive.
- **Sampler/convergence** (`generation_config.json` verbatim): `max_denoising_steps: 48`,
  `confidence_threshold: 0.005`, `stability_threshold: 1`,
  `sampler_config: {"_cls_name": "EntropyBoundSamplerConfig", "entropy_bound": 0.1}`,
  temperature schedule `t_max: 0.8 -> t_min: 0.4` (linear decay over remaining steps —
  visible in the vLLM kernel code), `max_new_tokens: 256`. Convergence per blog: argmax
  canvas stable for the configured steps AND mean entropy below threshold, or step cap.
  Non-converged positions are "renoised" with random vocab tokens (visible in code:
  `denoise_canvas = torch.where(eb_mask, new_tokens, random_tokens)`).
- **Head dims — inherits Gemma 4's split: YES.** `config.json` `text_config`:
  `head_dim: 256` (sliding layers), `global_head_dim: 512` (full-attention layers),
  `num_attention_heads: 16`, `num_key_value_heads: 8` (sliding),
  `num_global_key_value_heads: 2` (global). This is exactly the Gemma 4 26B-A4B mixed-KV
  geometry this worktree cares about.
- **Sliding+global layer pattern: YES**, `layer_types` = 5x `sliding_attention` then 1x
  `full_attention`, repeated over 30 layers (full attention at layers 5, 11, 17, 23, 29).
  `sliding_window: 1024`. RoPE: full-attention layers use `rope_theta 1e6` with
  `partial_rotary_factor 0.25` ("proportional" type); sliding layers `rope_theta 1e4`.
  `final_logit_softcapping: 30.0`, `max_position_embeddings: 262144`, vocab 262,144,
  hidden 2816, MoE `num_experts: 128`, `top_k_experts: 8`, `moe_intermediate_size: 704`,
  tied embeddings. Vision tower: gemma4_vision, 27 layers, 280 soft tokens/image.
- **Modeling code location:** HF repo has NO `modeling_*.py` and no `auto_map` —
  reference implementation is native in **transformers 5.8.0.dev0**
  (`architectures: ["DiffusionGemmaForBlockDiffusion"]`, `model_type: diffusion_gemma`).
  vLLM implementation reuses `Gemma4Model` and loads `model.encoder.language_model.*`
  and `model.decoder.*` checkpoint weights into ONE backbone (decoder duplicates
  skipped), plus `model.decoder.self_conditioning.*` -> a gated MLP that feeds previous
  denoising-step soft embeddings back in.
- The scheduler side reuses the spec-decode data path: `DiffusionConfig.canvas_length`
  "also determines the number of speculative tokens scheduled per step"
  (`vllm/config/diffusion.py` on `dgemma`).

## 4. DGX Spark specifics

- **NVIDIA blogs** (no GB10-specific vLLM playbook found in NVIDIA/dgx-spark-playbooks
  repo yet — only the generic `nvidia/vllm` playbook):
  - https://developer.nvidia.com/blog/run-diffusiongemma-on-nvidia-for-developer-ready-high-throughput-text-generation/
    — "up to 150 tokens/sec on NVIDIA DGX Spark"; NIM:
    `nvcr.io/nim/google/diffusiongemma-26b-a4b-it:latest` (docker run with
    `NGC_API_KEY`, port 8000). No vLLM version pin given.
  - https://blogs.nvidia.com/blog/rtx-ai-garage-local-gemma-diffusion/ — 1,000 tok/s
    H100, 150 tok/s Spark, 2,000 tok/s DGX Station; "vLLM playbooks are available for
    DGX Spark, RTX PRO and DGX Station".
  - `build.nvidia.com/spark/vllm` itself: **could not fetch (timeouts); open item** —
    verify on a browser whether it now lists a DiffusionGemma-specific recipe/image tag.
- **Forums thread** "Community Docker adds support for DiffusionGemma" (DGX Spark/GB10),
  https://forums.developer.nvidia.com/t/community-docker-adds-support-for-diffusiongemma/372831:
  eugr added four **solo-only** recipes to https://github.com/eugr/spark-vllm-docker:
  `diffusion-gemma-bf16-thinking`, `diffusion-gemma-bf16`,
  `diffusion-gemma-nvfp4-thinking`, `diffusion-gemma-nvfp4`. Usage:
  `./hf-download.sh google/diffusiongemma-26B-A4B-it && ./run-recipe.sh
  diffusion-gemma-bf16-thinking --solo` (also `sparkrun run @eugr/<recipe>`).
  Reported on GB10: **no OOM / attention-backend issues**; generation much faster than
  AR but **prefill notably slower**; throughput variable, **102-179.2 tok/s** in vLLM
  logs. Benchmarking caveat: model "emits results in 256 token blocks by default, even
  if response is shorter" — generate >256 tokens and use `--exact-tg`.
- **eugr recipe command verbatim** (`recipes/diffusion-gemma-bf16-thinking.yaml`;
  container `vllm-node`, nightly-prebuilt vLLM wheels, `mods/diffusiongemma` applies
  backport patches **only if** the installed vLLM lacks upstream support — it greps for
  `class DiffusionGemmaForConditionalGeneration`, `vllm/config/diffusion.py`, etc.):

  ```bash
  vllm serve google/diffusiongemma-26B-A4B-it \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --load-format fastsafetensors \
    --attention-backend TRITON_ATTN \
    --max-num-seqs 10 \
    --enable-prefix-caching \
    --diffusion-config '{"canvas_length":256}' \
    --override-generation-config '{"max_new_tokens": null}' \
    --reasoning-parser gemma4 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --moe-backend triton
  ```

  The NVFP4 recipe serves `nvidia/diffusiongemma-26B-A4B-it-NVFP4` with the same shape
  (no `--moe-backend triton`, adds `--chat-template fixed_chat_template.jinja` from the
  mod, no `--max-num-seqs` cap). The mod also carries gemma4 streaming-reasoning and
  content-channel-sanitizer patches and a `dynamic_causal` flash-attn compatibility patch.
- **Memory:** BF16 weights 51.7 GB + KV at 262K max-model-len fits the Spark's 128 GB
  unified memory at `--gpu-memory-utilization 0.8` (recipes run it). NVFP4 ~15 GB
  weights. Note divergence: official recipe caps `--max-num-seqs 4` (H100, diffusion
  state-buffer OOM) vs eugr's `10` on Spark (more headroom, lower concurrency goals).
- **Official images on Spark:** `vllm/vllm-openai:gemma` and `gemma-aarch64-cu130` are
  published for linux/arm64 (pushed 2026-06-10), so the official image is in principle
  runnable on GB10 directly — but only the eugr community stack has GB10 *reports* so far.

## DG-0 run plan recommendation

Run DG-0 on the eugr community docker (`eugr/spark-vllm-docker`, recipe
`diffusion-gemma-bf16` with `--solo`; thinking off for benchmark determinism, or the
`-thinking` variant if we want parity with NVIDIA's numbers) serving the **BF16
checkpoint `google/diffusiongemma-26B-A4B-it`** — BF16 first because it is the reference
implementation target (vLLM blog claims accuracy-match vs HF reference) and NVFP4 on
GB10 is untested upstream (NVIDIA tested B100 only). Keep the recipe flags as-is
(`--attention-backend TRITON_ATTN`, `--diffusion-config '{"canvas_length":256}'`,
`--enable-prefix-caching`, `--override-generation-config '{"max_new_tokens": null}'`,
`--moe-backend triton`), but consider dropping `--max-num-seqs` to 4 to match the
official recipe for comparability. As a cross-check, also try the official
`vllm/vllm-openai:gemma-aarch64-cu130` image directly (arm64 image exists; if it boots
on GB10 it removes the community-patch variable). Benchmark with >256-token generations
and `--exact-tg`-style accounting (256-token block emission), and record both prefill
and generation tok/s separately (TTFT is ~10x worse than AR by design); target
sanity number is NVIDIA's ~150 tok/s generation on Spark (community logs: 102-179).

**Blockers / open questions:**

1. **Not in any vLLM release**: support lives on the `vllm-project:dgemma` branch
   (PR #45163, still open, 21 commits ahead / 31 behind main as of 2026-06-11). Any
   plan to rebase our own vLLM tree must cherry-pick those commits (they touch
   scheduler, model runner v2, sampler, and both attention backends) or wait for the
   merge + next release (>v0.22.1). Pin DG-0 to the docker image, not pip.
2. **NVFP4 on GB10 unverified**: NVIDIA card says "tentative... until the supporting
   vLLM image is publicly released" and tested only B100/Hopper; eugr ships nvfp4
   recipes but the forum thread reports no NVFP4-on-Spark numbers. Treat the NVFP4 DG-0
   leg as exploratory.
3. **FLASH_ATTN (FA4) backend on sm_121 unknown** — every Spark recipe and the NVIDIA
   card use `TRITON_ATTN`; assume Triton-only on GB10 until proven otherwise.
4. **build.nvidia.com/spark/vllm playbook unfetched** (JS/timeout) — confirm manually
   whether NVIDIA published a Spark-specific DiffusionGemma vLLM playbook or image tag.
5. **No arXiv tech report** for DiffusionGemma yet; internals above are sourced from the
   vLLM blog + public `dgemma` branch code + HF configs. The deepest public reference
   for DG-1 is `vllm/model_executor/models/diffusion_gemma.py` on that branch.
6. Minor: NVFP4 card's eval-table baseline wording names "Gemma-26B-A4B-IT" (AR model)
   — ambiguous; re-verify which baseline that table uses before quoting it.

## Sources

- vLLM blog: https://vllm-project.github.io/2026/06/10/diffusion-gemma (mirror: https://vllm.ai/blog/2026-06-10-diffusion-gemma)
- vLLM recipe: https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it
- PR #45163: https://github.com/vllm-project/vllm/pull/45163 (branch `dgemma`)
- RFC #36155: https://github.com/vllm-project/vllm/issues/36155 ; PR #42653 (closed, LLaDA2.0 dllm-plugin)
- HF BF16: https://huggingface.co/google/diffusiongemma-26B-A4B-it (config.json, generation_config.json fetched raw)
- HF NVFP4: https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4
- Google model card: https://ai.google.dev/gemma/docs/diffusiongemma/model_card ; dev guide: https://developers.googleblog.com/diffusiongemma-the-developer-guide/
- NVIDIA tech blog: https://developer.nvidia.com/blog/run-diffusiongemma-on-nvidia-for-developer-ready-high-throughput-text-generation/ ; RTX AI Garage: https://blogs.nvidia.com/blog/rtx-ai-garage-local-gemma-diffusion/
- Forums: https://forums.developer.nvidia.com/t/community-docker-adds-support-for-diffusiongemma/372831
- Community docker: https://github.com/eugr/spark-vllm-docker (recipes/diffusion-gemma-*.yaml, mods/diffusiongemma/run.sh fetched raw)
- Docker Hub tags: https://hub.docker.com/r/vllm/vllm-openai/tags (gemma*, arm64+amd64, 2026-06-10)
- Block Diffusion (BD3-LM): https://arxiv.org/abs/2503.09573
