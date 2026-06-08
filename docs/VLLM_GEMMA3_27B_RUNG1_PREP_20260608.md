# vLLM Gemma 3 27B Rung 1 Prep Checkpoint

Date: 2026-06-08 JST

Scope: prep only for `google/gemma-3-27b-it` on the proven vLLM NVFP4-KV stack. No
container was started and no GPU work was consumed.

## Target Row

- model: `google/gemma-3-27b-it`
- served model name: `gemma3-27b-it`
- runtime image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
- vLLM overlay: `jethac/vllm@25ab073ef87f4443616fbaf00a2f6f09a9087c1f`
- FlashInfer overlay: `jethac/flashinfer@e152cf4da4ab2a9d093b7d9d4b499198b0211c61`
- precompiled wheel base: `4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa`
- max model length: `131072`
- GPU memory utilization: `0.85`
- attention backend: `flashinfer`
- comparator: `--kv-cache-dtype fp8`
- candidate: `--kv-cache-dtype nvfp4`
- Rung -1 correction to honor: Gemma 3 27B is dense with uniform `head_dim=128`, not
  `256`; it has hybrid SWA/full attention but no `D=512` problem.

## Prep Script

Use:

```bash
mkdir -p docs/results

STAMP=20260608T0000JST \
  bash scripts/prep_vllm_gemma3_27b_rung1.sh \
  third_party/vllm \
  third_party/flashinfer \
  /home/jethac/.cache/huggingface \
  results \
  > docs/results/vllm_gemma3_27b_rung1_20260608T0000JST_command_packet.sh
```

The script only prints the command packet. It does not call Docker, initialize CUDA, or
start serving.

## Exact Artifacts

With `STAMP=20260608T0000JST`, the live run must produce:

- `docs/results/vllm_gemma3_27b_rung1_20260608T0000JST_command_packet.sh`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_fp8_flashinfer_container_id.txt`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_fp8_flashinfer_docker_logs_pid.txt`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_fp8_flashinfer_server.log`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_fp8_flashinfer_import_probe.txt`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_fp8_flashinfer_editable_install.log`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_fp8_flashinfer_row_manifest.json`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_fp8_flashinfer_runtime_probe.json`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_fp8_flashinfer_openai_benchmark.json`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_fp8_flashinfer_chat_smoke.json`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_fp8_flashinfer_build_target_audit.json`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_fp8_flashinfer_quality.json`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_nvfp4_kv_flashinfer_container_id.txt`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_nvfp4_kv_flashinfer_docker_logs_pid.txt`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_nvfp4_kv_flashinfer_server.log`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_nvfp4_kv_flashinfer_import_probe.txt`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_nvfp4_kv_flashinfer_editable_install.log`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_nvfp4_kv_flashinfer_row_manifest.json`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_nvfp4_kv_flashinfer_runtime_probe.json`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_nvfp4_kv_flashinfer_openai_benchmark.json`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_nvfp4_kv_flashinfer_chat_smoke.json`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_nvfp4_kv_flashinfer_build_target_audit.json`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_nvfp4_kv_flashinfer_quality.json`
- `results/vllm_gemma3_27b_rung1_20260608T0000JST_quality_compare.json`

## Required Env Flags

- `VLLM_USE_V1=1`
- `VLLM_LOGGING_LEVEL=DEBUG`
- `VLLM_SPARK_KV_GEOMETRY_LOG=1`
- `SPARK_FLASHINFER_SOURCE_ROOT=/flashinfer-src`
- `FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1`
- `TORCH_CUDA_ARCH_LIST=12.1a`
- `CUDA_MODULE_LOADING=LAZY`
- `HF_TOKEN` passed through for gated model access

Do not pass `--kv-cache-dtype-skip-layers` for the green row. This rung is meant to prove
uniform-`D=128` Gemma SWA with all decoder attention layers on the selected KV dtype.

## Live Packet Behavior

The generated packet now starts each Docker container detached, writes the container ID,
streams `docker logs -f` to the expected `_server.log`, waits for
`http://127.0.0.1:8000/v1/models`, and only then records the OpenAI serving row. The
server log is therefore the actual vLLM log stream, not the Docker client output. A shell
`EXIT` trap removes both row containers if readiness or recording fails.

Live preflight from 2026-06-08 found the Spark-class host reachable and idle, with the
target image present and no Docker containers running. It also found that the older
`/home/jethac/src/vllm` and `/home/jethac/src/flashinfer` paths are absent; use initialized
repo submodules or a clean run checkout instead. `google/gemma-3-27b-it` was not present
in the existing Hugging Face cache, so the first live row needs gated HF access and enough
download space/time.

Prepared checkout: `/home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608`.
The generated packet is committed at
`docs/results/vllm_gemma3_27b_rung1_20260608TCHECKOUTJST_command_packet.sh` and passes
`bash -n`. In that checkout, `third_party/vllm` and `third_party/flashinfer` are
intentionally checked out at the Gemma 3 overlay commits named above, so `git submodule
status` shows leading `+` markers for those two submodules.

The vLLM overlay commit includes the env-gated runtime geometry hook. The remote run
checkout was updated and verified:

```text
+3658ba7123c3eb2211f18a882af1b993112fadb1 third_party/vllm (v0.13.0rc1-5248-g3658ba712)
298:                "SPARK_GEMMA_KV_GEOMETRY layer=%s heads=%s kv_heads=%s "
593:            "SPARK_GEMMA_KV_SPEC layer=%s spec=%s block_size=%s "
```

HF access probe:
`results/vllm_gemma3_27b_hf_access_probe_20260608T173133JST.json`. The model metadata is
visible and reports manual gating, but a config/tokenizer-only snapshot fails with
`GatedRepoError` because no `HF_TOKEN` is present in the container environment. Disk
headroom is sufficient. The fp8 comparator row should not be started until HF auth/access
is available.

Remote auth recheck:
`results/vllm_gemma3_27b_rung1_auth_recheck_20260608T181438JST.md`. SSH key auth works,
the host is reachable and idle, no Docker containers were running, but `HF_TOKEN` is still
absent, no root or `jethac` Hugging Face token file exists, and `google/gemma-3-27b-it` is
still not cached under `/home/jethac/.cache/huggingface/hub`. The live packet is now
geometry-ready, but starting it before HF auth/cache clears would only test Hugging Face
authentication failure.

HF access cleared:
`results/vllm_gemma3_27b_hf_access_probe_20260608T1832JST.md`. A user-scoped token file
and profile loader now make `HF_TOKEN` available for `jethac`, and the container probe
passes `model_info` plus the config/tokenizer snapshot for
`google/gemma-3-27b-it@005ad3404e59d6023443cb575daa05336842228a`. The next packet
attempt did not fail on Hugging Face access; it failed during editable vLLM install because
`wheels.vllm.ai` has no `cu130` metadata for
`8916796bc50926fd61e606718b194a71e2e31a24`. Do not rerun the current packet unchanged.
Repair the vLLM source/precompiled-wheel pair first, likely by moving the geometry hook
onto the proven `a919d635d` lane or another commit with published CUDA 13 metadata.

Source/wheel repair prepared:
`jethac/vllm@25ab073ef87f4443616fbaf00a2f6f09a9087c1f` cherry-picks the Gemma geometry hook
onto the proven `a919d635d` lane. The regenerated packet now uses precompiled wheel base
`4dcd10eb0d223a3ec4b2c96deaf3a48a96c8dcaa`, matching the prior clean Qwen CUDA 13 build.
The accepted live packet installs with `python3 -m pip install --no-build-isolation --no-deps
-e .` and copies the ABI-matched FA2 extension from `/opt/jethac-vllm` into the source
overlay. Do not allow pip dependency resolution to downgrade Torch/FlashInfer in this lane.

Setup-only check passed:
`results/vllm_gemma3_27b_rung1_setup_only_20260608T1855JST.md`. This artifact is retained
only for the metadata-404 fix; its dependency-resolving install downgraded packages and is
not accepted as the live runtime environment.

fp8 comparator row passed:
`results/vllm_gemma3_27b_rung1_fp8_20260608T1924JST.md`. Gemma 3 27B text-only serves with
`--kv-cache-dtype fp8`, FlashInfer decoder attention, and the measured running-model
geometry required for Rung 1: `62` decoder layers, `52` local SWA layers, `10` full/global
layers, uniform `heads=32`, `kv_heads=16`, `head_dim=128`, `head_dim_v=128`, and the
expected `0-4 local / 5 full` repeating SWA pattern. vLLM reports `882,851` fp8 KV tokens
and `6.74x` maximum concurrency at `131,072` tokens/request. All three OpenAI benchmark
cases completed with unflagged output.

NVFP4 candidate row is red:
`results/vllm_gemma3_27b_rung1_nvfp4_20260608T1924JST.md`. The row routes through
FlashInfer FA2 with vLLM V-scale-factor deswizzle, records the same 62-layer Gemma geometry,
and reports `1,568,861` KV tokens / `11.97x` concurrency, a `1.777x` capacity gain over
fp8 at decode-speed parity. It is not correct: strict `spark-ok` smoke returns nonsensical
mixed-script text, and the benchmark generations are also corrupted. Treat Gemma 3 NVFP4-KV
as a routing/capacity proof plus quality failure, not a green Rung 1.

## Expected Log Lines

Both rows:

- `model='google/gemma-3-27b-it'`
- `kv_cache_dtype='fp8'` or `kv_cache_dtype='nvfp4'`
- `Using fp8 data type to store kv cache` or `Using nvfp4 data type to store kv cache`
- `Using AttentionBackendEnum.FLASHINFER backend`
- `GPU KV cache size: <tokens> tokens`
- `Maximum concurrency for 131,072 tokens per request: <ratio>x`

NVFP4 row only:

- `Using FlashInfer FA2 backend for NVFP4 KV cache on SM12x with vLLM V-scale-factor deswizzle enabled.`

Geometry hook lines must show every decoder layer, including:

- `SPARK_GEMMA_KV_GEOMETRY layer=model.layers.0.self_attn.attn heads=32 kv_heads=16 head_dim=128 head_dim_v=128 sliding_window=1024 kv_cache_dtype=<fp8|nvfp4>`
- `SPARK_GEMMA_KV_GEOMETRY layer=model.layers.5.self_attn.attn heads=32 kv_heads=16 head_dim=128 head_dim_v=128 sliding_window=None kv_cache_dtype=<fp8|nvfp4>`
- `SPARK_GEMMA_KV_SPEC layer=model.layers.0.self_attn.attn spec=SlidingWindowSpec block_size=16 num_kv_heads=16 head_size=128 head_size_v=128 dtype=<...> kv_quant_mode=<...> sliding_window=1024 real_page_size_bytes=<...> page_size_bytes=<...> bytes_per_token=<...>`
- `SPARK_GEMMA_KV_SPEC layer=model.layers.5.self_attn.attn spec=FullAttentionSpec block_size=16 num_kv_heads=16 head_size=128 head_size_v=128 dtype=<...> kv_quant_mode=<...> sliding_window=None real_page_size_bytes=<...> page_size_bytes=<...> bytes_per_token=<...>`

Expected measured SWA pattern: local/sliding layers `0-4`, then full layer `5`, repeated
every six layers. For a default `block_size=16`, expected per-layer page sizes are
`65536` bytes for fp8 KV and `36864` bytes for NVFP4 KV, because
`nvfp4_kv_cache_full_dim(128) = 128 / 2 + 128 / 16 = 72`. The theoretical per-layer
KV capacity ratio is therefore `65536 / 36864 = 1.7778x`; the live row should record the
actual vLLM-reported pool/concurrency ratio from logs.

## Minimal Geometry Hook

Apply this only in the vLLM overlay worktree when preparing the live run. Keep it as
temporary evidence logging unless it is later upstream-shaped behind an env flag.

```diff
diff --git a/vllm/model_executor/layers/attention/attention.py b/vllm/model_executor/layers/attention/attention.py
--- a/vllm/model_executor/layers/attention/attention.py
+++ b/vllm/model_executor/layers/attention/attention.py
@@
-import torch
+import os
+import torch
@@
         self.num_kv_heads = num_kv_heads
         self.sliding_window = sliding_window
         self.has_sink = extra_impl_args.get("sinks") is not None
+        if os.environ.get("VLLM_SPARK_KV_GEOMETRY_LOG") == "1":
+            logger.info(
+                "SPARK_GEMMA_KV_GEOMETRY layer=%s heads=%s kv_heads=%s "
+                "head_dim=%s head_dim_v=%s sliding_window=%s "
+                "kv_cache_dtype=%s torch_dtype=%s",
+                prefix,
+                self.num_heads,
+                self.num_kv_heads,
+                self.head_size,
+                self.head_size_v,
+                self.sliding_window,
+                self.kv_cache_dtype,
+                self.kv_cache_torch_dtype,
+            )
@@
-            return SlidingWindowSpec(
+            spec = SlidingWindowSpec(
                 block_size=block_size,
                 num_kv_heads=self.num_kv_heads,
                 head_size=self.head_size,
@@
                 kv_quant_mode=quant_mode,
                 sliding_window=self.sliding_window,
             )
+            if os.environ.get("VLLM_SPARK_KV_GEOMETRY_LOG") == "1":
+                logger.info(
+                    "SPARK_GEMMA_KV_SPEC layer=%s spec=%s block_size=%s "
+                    "num_kv_heads=%s head_size=%s head_size_v=%s dtype=%s "
+                    "kv_quant_mode=%s sliding_window=%s real_page_size_bytes=%s "
+                    "page_size_bytes=%s bytes_per_token=%.3f",
+                    self.layer_name, type(spec).__name__, spec.block_size,
+                    spec.num_kv_heads, spec.head_size,
+                    getattr(spec, "head_size_v", None), spec.dtype,
+                    spec.kv_quant_mode.name, getattr(spec, "sliding_window", None),
+                    spec.real_page_size_bytes, spec.page_size_bytes,
+                    spec.page_size_bytes / spec.block_size,
+                )
+            return spec
@@
-            return FullAttentionSpec(
+            spec = FullAttentionSpec(
                 block_size=block_size,
                 num_kv_heads=self.num_kv_heads,
                 head_size=self.head_size,
@@
                 dtype=self.kv_cache_torch_dtype,
                 kv_quant_mode=quant_mode,
             )
+            if os.environ.get("VLLM_SPARK_KV_GEOMETRY_LOG") == "1":
+                logger.info(
+                    "SPARK_GEMMA_KV_SPEC layer=%s spec=%s block_size=%s "
+                    "num_kv_heads=%s head_size=%s head_size_v=%s dtype=%s "
+                    "kv_quant_mode=%s sliding_window=%s real_page_size_bytes=%s "
+                    "page_size_bytes=%s bytes_per_token=%.3f",
+                    self.layer_name, type(spec).__name__, spec.block_size,
+                    spec.num_kv_heads, spec.head_size,
+                    getattr(spec, "head_size_v", None), spec.dtype,
+                    spec.kv_quant_mode.name, getattr(spec, "sliding_window", None),
+                    spec.real_page_size_bytes, spec.page_size_bytes,
+                    spec.page_size_bytes / spec.block_size,
+                )
+            return spec
```

## Gates For Green

- fp8 and NVFP4 rows use the same model, `max_model_len`, memory fraction, graph mode,
  batch token limit, prompt set, and source overlays.
- Runtime geometry log proves all Gemma 3 27B decoder layers are `head_dim=128`, with
  `num_attention_heads=32`, `num_kv_heads=16`, and the expected local/full SWA map.
- NVFP4 server log proves FlashInfer FA2 NVFP4 KV on SM12x, not fp8 fallback and not
  `trtllm-gen`. This gate is passed for routing, but quality is still red.
- Capacity report records `GPU KV cache size` and `Maximum concurrency` for both rows and
  computes NVFP4/fp8 ratio. Expect near `1.78x`, but record the actual number.
- `chat_smoke.json` returns exactly `spark-ok`. This gate failed for the NVFP4 row.
- `openai_benchmark.json` has all benchmark cases `ok=true`, nonempty normal content,
  TTFT, and decode tokens/sec.
- `quality.json` has no flags for either row, and `quality_compare.json` is `ok=true`.
  The current heuristic is too weak: the NVFP4 row produced garbage while the simple
  heuristic remained unflagged. Add a stronger correctness comparator before treating this
  rung as green.

Untested at this checkpoint: TP>1, DCP, non-default page/block sizes, fp8-vs-NVFP4
logprob parity, and Gemma 4 `D=512`.
