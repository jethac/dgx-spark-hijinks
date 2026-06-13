# DG-V6 — DiffusionGemma NVFP4-KV decode-speed (bandwidth) experiment

**Status:** PLAN (not yet run). Owner: Claude / vLLM lane. Platform: Spark (GB10, sm_121).

## Hypothesis
Full-NVFP4 K+V cache makes **DiffusionGemma decode FASTER** (not just higher-capacity),
and the speedup is **larger than for any autoregressive Gemma** and **larger on
bandwidth-constrained hardware (Spark)** than on the PRO 6000.

## Mechanism (why it should be true)
- nvfp4 KV moves **~3.56× fewer bytes per KV read** than bf16 (9/32 of bf16, scale
  factors included).
- DiffusionGemma is **block diffusion**: the decoder re-reads the *entire cached prefix
  on every denoise step* (dozens of reads per generated 256-token block), vs **once per
  token** for an AR model. So KV-read bandwidth is the *dominant* decode cost, not a side
  cost.
- Spark/GB10 is **bandwidth-bound** (~273 GB/s unified). On a bandwidth-bound part, fewer
  bytes moved == directly faster (and the nvfp4 in-kernel dequant compute is cheap relative
  to the bandwidth saved).
- => shrinking the per-step KV read 3.56× should *speed up* decode, biggest where the KV
  re-read dominates the per-step byte budget.

## Roofline prediction (the quantitative model to test against)
If decode is bandwidth-bound, per-step time `~= (W + K) / BW`, where
- `W` = bytes read per denoise step for the **active** weights. DiffusionGemma 26B-A4B is
  MoE (~4B active/token) -> `W ~= 4e9 * 2 (bf16 weights) ~= 8 GB/step` (refine from config).
- `K` = KV bytes re-read per step = `ctx_tokens * bytes_per_kv_token`. bf16 vs nvfp4:
  `K_nvfp4 = K_bf16 / 3.556`.
- **Predicted TPOT speedup** = `(W + K_bf16) / (W + K_nvfp4)`:
  - short ctx (`K << W`): ~1.0× (no win — weight-bound).
  - long ctx (`K >> W`): -> 3.556× (KV-bound limit).
- So the prediction is a **rising curve vs context**, asymptoting toward 3.56×. Overlay
  measured-vs-predicted: if they track, the speedup is bandwidth, *for the reason claimed*.

## Falsifiable predictions
- **P1 (signature):** DiffusionGemma nvfp4 TPOT < bf16 TPOT, and the **gap grows with
  context length**. (Falsified if flat or shrinking.)
- **P2 (block-diffusion attribution):** the AR control (Gemma 4 26B-A4B-it, same sweep) has a
  **much smaller** nvfp4-vs-bf16 TPOT gap than DiffusionGemma at the same context. (Falsified
  if the gaps are equal — then it's generic, not a block-diffusion effect.)
- **P3 (platform attribution):** the speedup is **bigger on Spark than on the PRO 6000**
  (more bandwidth-bound). (Stretch; falsified if equal/inverted.)

## Design
| axis | values |
|---|---|
| model (primary) | `google/diffusiongemma-26B-A4B-it` (block diffusion) |
| model (AR control, P2) | `google/gemma-4-26B-A4B-it` (same base, autoregressive) |
| KV dtype | **nvfp4** vs **bf16** (matched everything else) |
| context sweep (input len) | 512, 1024, 2048, 4096, 8192 |
| output len | 256 (one diffusion block) |
| concurrency | **1** (single-stream) -- CRITICAL: isolates per-step bandwidth from
  capacity/starvation; both dtypes have ample KV for one stream |
| platform | Spark sm_121 (primary). PRO 6000 / Colab sm_120 (P3, stretch) |

**Why concurrency 1:** the YAK throughput gains were *partly* capacity-relief (bf16 31B was
KV-starved). At concurrency 1, bf16 is NOT starved at any tested ctx, so any TPOT delta is
pure per-step bandwidth. (Also run a high-concurrency pass separately to show the *combined*
capacity+bandwidth win, but the single-stream curve is the clean mechanism proof.)

## Metrics
- **TPOT (ms/output token)** — primary (per-step decode cost).
- output throughput (tok/s), TTFT — secondary.
- derived: `speedup(ctx) = TPOT_bf16(ctx) / TPOT_nvfp4(ctx)` vs the roofline curve.

## Procedure
1. Image: `jethac-vllm-aeon-gemma4:e2-dgv-3d6a0d507-sm121a-r11` (warm MoE kernel cache; r12
   bakes it -- mount `~/spark_tmp/fi_cache` if recompile reappears). Spark marker first.
2. For each (model, dtype): serve once, sweep input len via `vllm bench serve`.
   - nvfp4: `--attention-backend FLASHINFER --kv-cache-dtype nvfp4` + env
     `VLLM_NVFP4_KV_LINEAR_V_SF=1` (VO-split knobs baked) `--enforce-eager
     --no-enable-prefix-caching --no-enable-chunked-prefill --max-num-seqs 4
     --gpu-memory-utilization 0.6 --max-model-len 8704`.
   - bf16: same with `--kv-cache-dtype auto`.
   - bench: `vllm bench serve --dataset-name random --random-input-len <ctx>
     --random-output-len 256 --max-concurrency 1 --num-prompts 16 --ignore-eos
     --save-result`. (Random prompts are fine for TIMING -- the denoise loop runs regardless
     of coherence; coherence already proven in DG-V5.)
3. Repeat the sweep for the AR control (gemma-4-26B-A4B-it) -> P2 contrast.
4. (Stretch) repeat DG sweep on PRO 6000 via the Colab benchmark notebook -> P3.
5. Free GPU + release marker after each model.

## Success / kill criteria
- **Confirmed** if P1 holds (rising speedup vs ctx) AND P2 holds (DG gap >> AR gap). Bonus if
  the measured curve tracks the roofline prediction within ~20%.
- **Refuted / re-scope** if the single-stream TPOT delta is flat (~1.0×) across ctx -> then
  the blog's "per-step bandwidth lever" claim is unsupported and must be softened to capacity-
  only. (This is the honest kill switch; do not ship the speed claim without the curve.)

## Artifacts
`results/dg_v6_bandwidth_<date>/`: per-(model,dtype,ctx) bench JSON, a `SUMMARY.md` with the
speedup-vs-ctx table + the roofline overlay, server logs (proof lines: nvfp4 + LINEAR_V_SF +
VO-split engaged). Feeds the blog's DiffusionGemma section (turns the asserted "bandwidth
lever" into a measured curve) and the SGLang lane (Codex can mirror for DG-R6).

## Risks / caveats
- DiffusionGemma serving is block-diffusion; confirm the denoise loop actually runs under the
  random-prompt bench (check generated token counts, not just no-error).
- MoE first-serve recompiles the CUTLASS kernel (~20 min cold) -- use the warm cache image.
- TPOT semantics for diffusion: vLLM reports per-output-token latency; that already folds the
  per-step re-reads across the block, which is exactly what we want to measure.
- Keep util / max-model-len identical across dtypes so the only variable is KV bytes/read.

---

# Runner context & runbook (self-contained — for a fresh agent)

You have NO prior conversation context. Everything you need is here. This is the
**dgx-spark-hijinks** campaign: adding NVFP4 KV cache support for Gemma on consumer/Spark
Blackwell. Two agent lanes coordinate: Claude (vLLM/FlashInfer) and Codex (SGLang/infra),
via committed `mail/` files. This task is a self-contained perf experiment; do it, bank it,
mail a one-line result.

## Repos & branches
- Campaign repo (this one): `github.com/jethac/dgx-spark-hijinks`, branch **`epoch2`**. Results
  + docs + `mail/` live here. Work in the worktree you're already in.
- vLLM: `github.com/jethac/vllm`, branch `spark/hijinks-e2-vllm` (== `e2-dgv`). The e2-dgv
  arm64 wheel release is `sm121a-arm64-wheels-3d6a0d507`.
- FlashInfer: `github.com/jethac/flashinfer`, branch `spark/hijinks-022-fa2-d512` (source-tree
  JIT, baked into the serving image).

## Spark access (the GPU)
- Host: **`jethac@100.113.98.11`** (Tailscale). SSH is **key-based / passwordless** —
  `ssh -o BatchMode=yes jethac@100.113.98.11` works; you do NOT need a password. GB10, sm_121,
  ~119 GiB unified. nvfp4 is clean on Spark (the sm_120 nvfp4 read defect is 5060-Ti/WSL2-only).
- Triple-nested quoting (local-bash -> ssh -> remote-bash) mangles `$` and quotes. **Stage
  remote scripts to a file and pipe**: `ssh ... bash -s < /tmp/script.sh`. Don't inline complex
  remote commands.

## GPU marker contract (MANDATORY — shared single GPU)
1. Before any GPU use: check `ls ~/spark_tmp/MARKER_*` on Spark AND
   `nvidia-smi --query-compute-apps=pid --format=csv,noheader`. If another agent's marker
   exists OR a compute app is running, **do not touch the GPU** — wait / coordinate via `mail/`.
   (Stale-marker rule: a marker with no active process for >30 min may be struck.)
2. Claim: `echo "codex dg-v6 $(date -u +%FT%TZ)" > ~/spark_tmp/MARKER_codex_dgv6`.
3. Release on completion/abort: `rm -f ~/spark_tmp/MARKER_codex_dgv6` and verify the GPU is
   free (no compute apps). One server at a time; `docker rm -f` the container after each.

## The serving image
- Use **`jethac-vllm-aeon-gemma4:e2-dgv-3d6a0d507-sm121a-r11`** (already on Spark from DG-V5):
  r10 stack (Ubuntu 22.04 / torch 2.11 / flashinfer 76af7982 = the 022-fa2-d512 family) + the
  e2-dgv vLLM wheel. `docker images | grep e2-dgv` to confirm. If absent, bake it: `docker build`
  with `docker/Dockerfile.r11` from the e2-dgv vLLM checkout, `--build-arg BASE_IMAGE=...r10
  --build-arg VLLM_WHEEL=<the sm121a-arm64-wheels-3d6a0d507 .whl>` (download via
  `gh release download sm121a-arm64-wheels-3d6a0d507 --repo jethac/vllm`).
- **Warm CUTLASS-MoE cache (avoid the ~20 min cold compile):** the 26B-A4B MoE JIT-compiles a
  CUTLASS fused-MoE kernel on first serve (97 targets, ~20 min). A warm cache from DG-V5 is at
  **`~/spark_tmp/fi_cache`** (~120-140 `.o`). **Mount it**: `-v ~/spark_tmp/fi_cache:/root/.cache/flashinfer`
  (mount preserves mtimes; baking into the image via COPY does NOT — it triggers a partial
  recompile). If `fi_cache` is gone, the first serve will recompile (wait it out — it's
  progressing if `nvidia-smi` shows `nvcc` procs and `find /root/.cache/flashinfer -name '*.o'`
  grows; do NOT kill it as a "hang", that was an earlier mistake).
- Models (cached on Spark, ~49 GB DG / ~52 GB AR): `google/diffusiongemma-26B-A4B-it`,
  `google/gemma-4-26B-A4B-it`. HF token at `~/.cache/huggingface/token` (gated models). Mount
  `-v ~/.cache/huggingface:/root/.cache/huggingface`.

## Proven serve config (from DG-V5 — copy exactly)
```bash
docker run -d --name dgv6_nvfp4 --gpus all --net host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/spark_tmp/fi_cache:/root/.cache/flashinfer \
  -e VLLM_NVFP4_KV_LINEAR_V_SF=1 \
  jethac-vllm-aeon-gemma4:e2-dgv-3d6a0d507-sm121a-r11 \
  bash -lc "vllm serve google/diffusiongemma-26B-A4B-it --served-model-name dg \
    --host 127.0.0.1 --port 8000 --attention-backend FLASHINFER --kv-cache-dtype nvfp4 \
    --max-num-seqs 4 --gpu-memory-utilization 0.6 --max-model-len 8704 \
    --enforce-eager --no-enable-prefix-caching --no-enable-chunked-prefill"
```
- **bf16 row:** identical but `--kv-cache-dtype auto` and drop `-e VLLM_NVFP4_KV_LINEAR_V_SF=1`.
- **AR control:** swap the model to `google/gemma-4-26B-A4B-it` (drop the diffusion-specific
  flags are fine to keep; it serves AR). Same dtype pair.
- `--enforce-eager` (cudagraph OOMs the 26B MoE) and `--no-enable-prefix-caching
  --no-enable-chunked-prefill` (block-diffusion) are REQUIRED for DiffusionGemma.
- Wait-ready: poll `curl -fsS http://127.0.0.1:8000/v1/models`. Model load ~6 min; if cache is
  warm there's no 20-min compile after. Proof lines to grep in `docker logs`: `Using nvfp4 data
  type to store kv cache` and `VLLM_NVFP4_KV_LINEAR_V_SF=1: NVFP4 V scale factors are linear`.
- **Coherence sanity** (not the perf metric, just a smoke): block-diffusion needs the CHAT
  endpoint — `POST /v1/chat/completions` with a `messages` list returns coherent text
  ("...Tokyo."); raw `/v1/completions` returns gibberish by design. For TIMING use the bench
  below (random prompts are fine — the denoise loop runs regardless of coherence).

## The bench (TPOT vs context, conc=1)
For each (model, dtype), with the server up, sweep input len:
```bash
docker exec dgv6_nvfp4 bash -lc "vllm bench serve --model dg --base-url http://127.0.0.1:8000 \
  --dataset-name random --random-input-len <CTX> --random-output-len 256 \
  --max-concurrency 1 --num-prompts 16 --ignore-eos --save-result \
  --result-filename /spark_tmp/dgv6_<model>_<dtype>_ctx<CTX>.json"
```
Sweep `<CTX>` = 512 1024 2048 4096 8192. Pull `mean_tpot_ms` / `output_throughput` from each
JSON. If `vllm bench serve` flags differ in this wheel, `vllm bench serve --help` inside the
container; the key knobs are dataset=random, input/output len, max-concurrency, save-result.

## Banking & coordination (stop-point hygiene)
- Pull results off Spark to `results/dg_v6_bandwidth_<YYYYMMDD>/` in this repo (scp the bench
  JSONs + server logs). Write `SUMMARY.md`: the `speedup(ctx) = TPOT_bf16/TPOT_nvfp4` table for
  DiffusionGemma AND the AR control, plus the roofline overlay; state P1/P2 verdicts + whether
  the kill criterion fired.
- Campaign rules: **every claim an artifact; every red a verbatim error; a stop point = clean
  tree + pushed summary.** Commit + push to `epoch2`. Then write `mail/<NNNN>_codex-to-claude_dg-v6-result.md`
  (next free number via `ls mail/`) with the one-line verdict + the artifact path.
- Free the GPU + remove your marker before you stop.
