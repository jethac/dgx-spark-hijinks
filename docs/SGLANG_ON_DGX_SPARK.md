# SGLang On DGX Spark

Status: BF16/auto and fp8 Qwen rows pass in NVIDIA 26.05 container; stock `fp4_e2m1` Qwen fails before serving; the current `jethac/sglang` FP4-KV branch proves the expected FP4 KV capacity gain under an auto-safe no-graph policy. The matched `d7d931f` row records `1.7769x` fp8 KV capacity and backend traces for decode plus `extend_merge_paged`, with raw/chat smoke passing. Do not bless SGLang FP4 KV for serving quality or speed because the standardized FP4 benchmark content still degrades.

Target: DGX Spark-class GB10 = compute capability 12.1 = `sm_121`.

Scope: one Spark-class unit only. No TP>1 or multi-Spark claims yet.

## Why Track SGLang

SGLang is a serious serving runtime and should not be hidden under the vLLM plan. The `hikarioyama/sglang-nvfp4-kv-sm120` repo is especially relevant because it demonstrates the design shape for NVFP4 KV:

- `fp4_e2m1` KV cache
- FlashInfer FA2 kernel patches
- native FP4 KV memory pool
- hybrid-SWA wiring
- per-layer global-scale auto-calibration before CUDA graph capture
- fp4-vs-fp8 comparison discipline

That repo reports Qwen2.5 and Step3.7-Flash 198B validation on SM120 RTX Blackwell systems, including a measured KV capacity improvement versus fp8 and roughly fp8-like decode speed on the large-model path. It also reports small-model incoherence under NVFP4 KV and recommends fp8 KV for that regime. This is important reference work, but it is not Spark validation. Our target is GB10 `sm_121`, and our current SGLang proof is BF16 KV only.

AEON's Qwen and Gemma GB10 work is also relevant, but it is vLLM prior art. Its strongest signal is NVFP4 weight serving plus backend policy and DFlash. It does not prove SGLang `fp4_e2m1` KV, and its Gemma path reinforces that Gemma 4 can require Triton attention because of heterogeneous local/global head dimensions.

## Evidence Classes

Keep these separate in reports:

- NVIDIA 26.05 BF16 Spark evidence: `nvcr.io/nvidia/sglang:26.05-py3` serves `Qwen/Qwen2.5-1.5B-Instruct` on our GB10 with BF16 KV.
- hikarioyama SM120 NVFP4 KV evidence: `hikarioyama/sglang-nvfp4-kv-sm120` at `9b2160f0fb8e11dbbb5171a57f06a02b0e9ba6e2` reports `fp4_e2m1` KV, FlashInfer FA2 patches, FP4 pools, hybrid-SWA support, and global-scale calibration on SM120 hardware.
- jethac FlashInfer SM121 `mm_fp4` dispatch evidence: our FlashInfer fork changes dense/MoE GEMM auto-dispatch on GB10, but it does not validate SGLang KV attention.
- jethac FlashInfer FA2 NVFP4 KV SGLang-layout evidence: `results/flashinfer_sglang_linear_nvfp4_kv_probe_genpatch_20260608.json` passes a standalone FlashInfer FA2 paged NVFP4 KV probe on GB10 `sm_121` with tuple-packed `uint8` K/V, SGLang-style linear V scale factors, NHD/HND layouts, and prefill/decode cosine near `0.999999`. This proves the synthetic kernel/scale-buffer signature, not SGLang serving, graph replay, model output quality, KV capacity, or throughput.
- AEON vLLM evidence: useful recipe evidence for NVFP4 weights and DFlash on GB10, not evidence that Gemma 4 works with FP4 KV in SGLang.

Do not merge those into "SGLang NVFP4 works on Spark" until the GB10 `fp4_e2m1` serving row exists.

For Gemma 4, first prove SGLang with NVFP4 weights and BF16/fp8 KV. Test `fp4_e2m1` KV only after that baseline works.

## Baseline First

Before NVFP4:

- install or run SGLang on the single GB10 unit: done for `nvcr.io/nvidia/sglang:26.05-py3`
- capture `spark_doctor`: done for `sglang_20260607T115213Z`
- start an OpenAI-compatible server: done on port `30000`
- run `scripts/openai_chat_smoke.py`: passed
- establish BF16 or fp8 KV quality and speed: BF16/auto and fp8 baselines are captured for Qwen on the GB10 SGLang path

Only then test `fp4_e2m1`.

## 2026-06-08 Qwen fp8-vs-fp4 KV Probe

Artifacts:

- `results/sglang_qwen25_1_5b_fp8_vs_fp4kv_20260608T0332JST_summary.md`
- `results/sglang_qwen25_1_5b_bf16auto_040mem_20260608T0409JST_openai_benchmark.json`
- `results/sglang_qwen25_1_5b_fp8kv_20260608T0332JST_openai_benchmark.json`
- `results/sglang_qwen25_1_5b_fp4kv_20260608T0336JST_startup.log`
- `results/sglang_qwen25_1_5b_fp4kv_triton_20260608T0338JST_startup.log`
- `results/sglang_qwen25_1_5b_fp4kv_patched_flashinfer_20260608T0349JST_startup.log`
- `results/sglang_qwen25_1_5b_fp4kv_patched_triton_nographs_20260608T0404JST_openai_benchmark.json`
- `results/flashinfer_sglang_linear_nvfp4_kv_probe_genpatch_20260608.json`

Result:

- BF16/auto KV with FlashInfer attention serves at the same `mem_fraction_static=0.40` and records `1,557,709` KV tokens.
- BF16/auto decode is `58.89`, `58.59`, and `57.73 tok/s` across the standard short, medium, and long-prefill cases.
- fp8 KV with FlashInfer attention serves `Qwen/Qwen2.5-1.5B-Instruct` on GB10 and records `NVIDIA_GB10:sm_121:sms_48`.
- fp8 KV allocated `3,113,713` KV tokens and decoded around `58-59 tok/s`.
- stock `fp4_e2m1` with FlashInfer attention fails before health because `KV4Compatibility` rejects FlashInfer for MHA FP4 KV.
- stock `fp4_e2m1` with Triton attention allocates `5,534,509` KV tokens, about `1.78x` fp8 capacity, then fails before health because `KVFP4QuantizeUtil` cannot be imported from `kvfp4_tensor.py`.
- patched FlashInfer attention clears the SGLang gate and alias blockers, allocates `5,539,718` FP4 KV tokens, targets `compute_121a,code=sm_121a`, then fails compiling FlashInfer FP4 decode at `vec_dtypes.cuh(117)`.
- patched Triton attention with normal graph capture hangs at the first CUDA graph batch. With only `--disable-cuda-graph`, it still enters piecewise graph capture and stalls. With both `--disable-cuda-graph` and `--disable-piecewise-cuda-graph`, it serves and allocates `5,541,103` FP4 KV tokens, but short decode is only `0.276 tok/s` and output quality is visibly poor.
- A standalone FlashInfer FA2 NVFP4 KV probe now passes the SGLang-style linear V-scale signature on GB10: tuple-packed `uint8` K/V, BF16 Q/output, `head_dim=128`, `page_size=16`, NHD/HND prefill and decode, with cosine about `0.999999` and max absolute error at or below `0.015625`.

Interpretation:

- This confirms fp8 is a valid SGLang Qwen comparator on our GB10.
- This also confirms stock NVIDIA 26.05 SGLang does not yet provide a working `fp4_e2m1` serving row for this Qwen model.
- The SGLang fork fixes are necessary but not sufficient. The remaining blockers are FlashInfer FP4 E2M1 decode support for the FlashInfer-attention path and graph-compatible Triton FP4 KV serving with sane output.
- The standalone FlashInfer probe retires the synthetic decode blocker for the SGLang-style linear V-scale path. It does not prove clean SGLang `fp4_e2m1` serving; SGLang still has to pass the packed K/V plus scale buffers through its FlashInfer backend, serve under an accepted graph policy, and pass fp8-vs-fp4 quality/speed/capacity checks.

Smallest next proof:

- The FlashInfer-only JIT/synthetic decode probe is done for the SGLang-style linear V-scale signature.
- Next run the clean SGLang FlashInfer FP4 KV after-row, not the site-package overlay.
- Verify SGLang passes tuple-packed FP4 K/V plus per-block `kv_cache_sf`/linear V scale factors into FlashInfer, then require a row manifest, OpenAI benchmark, graph policy, fp8 comparator, and quality check before blessing the row.

## 2026-06-08 Clean Source FP4 KV Graph Attempt

Artifacts:

- `results/sglang_qwen_fp4kv_clean_20260608_precapture_missing.log`
- `results/sglang_qwen_fp4kv_clean_graph_calibrated_bad_output_20260608_server.log`
- `results/sglang_qwen_fp4kv_clean_sm120fallback_bad_output_20260608_server.log`
- `results/sglang_qwen_fp4kv_clean_chat_smoke_20260608.json`
- `results/sglang_qwen_fp4kv_clean_sm120fallback_chat_smoke_20260608.json`
- `results/sglang_qwen_fp4kv_clean_sm120fallback_raw_generate_20260608.json`
- `results/sglang_qwen_bf16_clean_good_output_20260608_server.log`
- `results/sglang_qwen_bf16_clean_chat_smoke_20260608.json`
- `results/sglang_qwen_bf16_clean_raw_generate_20260608.json`
- `results/sglang_nvfp4_kv_layout_probe_20260608.json`
- `results/sglang_qwen_fp4kv_decode_dtype_bad_output_20260608_server.log`
- `results/sglang_qwen_fp4kv_decode_dtype_chat_smoke_20260608.json`
- `results/sglang_qwen_fp4kv_decode_dtype_raw_generate_20260608.json`
- `results/sglang_qwen_fp4kv_eager_only_server_20260608.log`
- `results/sglang_qwen_fp4kv_eager_only_chat_smoke_20260608.json`
- `results/sglang_qwen_fp4kv_eager_only_raw_generate_20260608.json`
- `results/sglang_qwen_fp4kv_autosafe_server_20260608.log`
- `results/sglang_qwen_fp4kv_autosafe_chat_smoke_20260608.json`
- `results/sglang_qwen_fp4kv_autosafe_raw_generate_20260608.json`

Setup:

- base image: `nvcr.io/nvidia/sglang:26.05-py3`
- model: `Qwen/Qwen2.5-1.5B-Instruct`
- source install: editable `jethac/sglang` source overlay with installed NVIDIA FlashInfer plus local FlashInfer headers/JIT source
- required escape hatch: `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`, because the source tree requires FlashInfer `0.6.12` while NVIDIA 26.05 ships `0.6.10+cf494fca.nv26.05.cu132.50619265`
- FP4 launch: `--attention-backend flashinfer --kv-cache-dtype fp4_e2m1 --page-size 1 --mem-fraction-static 0.40`

Result:

- The first clean source attempt failed during CUDA graph capture because no-scale Qwen models had not calibrated NVFP4 KV global scales before graph capture.
- Porting the reference pre-capture calibration hook fixed that startup blocker. The log shows `NVFP4 KV cache calibrated 28 layers from 4096 eager prefill tokens`, then normal CUDA graph capture and piecewise CUDA graph capture both completed.
- With the writer using FlashInfer `nvfp4_kv_quantize` on SM121, the server reached readiness but failed quality: chat smoke returned `header: Re bywen, are you`, and raw generation answered `2+2 is a 20.0000000`.
- Aligning the SM120/121 writer with the reference repo, so SM120-family devices use `fp4_quantize(..., global_scale_inv, sf_vec_size=16, ...)` instead of `nvfp4_kv_quantize`, kept graph-enabled serving working but still failed quality: chat smoke returned `Placeholder\n\nI'm here to to.`, and raw generation answered `2+2 is a 2-digit number number`.
- A BF16 KV comparator using the same clean source overlay, same model, FlashInfer attention, and graph modes passed: chat smoke returned `spark-ok`, and raw generation answered `2+2 is 4`.
- A scale-factor layout probe showed the installed NVIDIA FlashInfer FA2 path accepts both 4D `[tokens, 1, heads, sf_cols]` and 3D `[tokens, heads, sf_cols]` scale-factor tensors for this page-size-1 shape, both at cosine `0.9999957` versus a faithful dequant reference. Scale-rank mismatch is not the current corruption root cause.
- Matching the reference decode planning call shape, so FlashInfer decode planning uses the packed KV storage dtype instead of the logical FP4 dtype, did not fix graph-enabled output. The graph-enabled run still returned `Placeholder\n\nI'm here to to.` and `2+2 is a 2-digit number number`.
- Disabling both normal CUDA graph and piecewise CUDA graph initially improved the FP4 KV path enough for ad hoc smoke checks, but the later standardized fp8-vs-FP4 row still failed deterministic output quality.
- The fork now disables CUDA graph capture automatically for native FP4 KV unless `SGLANG_FP4_KV_ENABLE_CUDA_GRAPH=1` is set. The matched autosafe row logs `Disabling CUDA graph capture for native FP4 KV cache`, reaches readiness, and returns `spark-ok`, but its standardized raw `2+2` artifact is malformed and the benchmark text degenerates.

Interpretation:

- The pre-capture calibration port is a real compatibility fix: it moves SGLang FP4 KV from graph-capture failure to serving.
- The SM120-family quantizer routing fix removes a concrete divergence from the `hikarioyama/sglang-nvfp4-kv-sm120` implementation, but it is not sufficient for correctness on GB10.
- The remaining blocker is FP4 KV output quality under serving, plus graph-safe FP4 KV if we want the normal high-performance path. The packed writer and capacity allocation are real, but they are not sufficient.
- SGLang FP4 KV is now a capacity-proven lane only when graph capture is disabled. Do not claim graph-enabled speed or no-graph serving quality until the deterministic sanity and quality comparator pass.

Next debugging target:

- Add an end-to-end graph replay probe before re-enabling graphs, but keep quality debugging first because the autosafe no-graph row still corrupts output.
- Build the next matched fp8-vs-fp4 row only after the FP4 side passes deterministic sanity. The first blessed SGLang FP4 KV row must record graph policy, capacity tokens, throughput, and quality.

## 2026-06-08 Autosafe fp8-vs-FP4 KV Row

Artifacts:

- `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_summary.md`
- `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_fp8_row_manifest.json`
- `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_fp8_openai_benchmark.json`
- `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_fp8_raw_2plus2.json`
- `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_row_manifest.json`
- `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_openai_benchmark.json`
- `results/sglang_qwen_fp4kv_autosafe_20260608T1315JST_raw_2plus2.json`

Result:

- fp8 comparator: `3,101,822` KV tokens, CUDA graphs disabled to match the FP4 policy, decode `56.73`, `56.81`, and `57.10 tok/s`, raw `2+2` returns `4`.
- FP4 KV: `5,519,481` KV tokens, auto-safe no-graph policy, `NVFP4 KV cache calibrated 28 layers from 4096 eager prefill tokens`, capacity ratio `1.779x` versus fp8.
- FP4 output quality fails: raw `2+2` returns malformed text, and the compact benchmark output is repetitive or incoherent even though the row manifest is mechanically `ok=true`.
- The build-target audit for these row logs still does not provide accepted native build-target evidence.

Interpretation:

- This is the cleanest SGLang FP4 KV capacity proof so far.
- It is not a blessed serving row and not a speed claim. The FP4 throughput numbers are not comparable because the output is wrong.
- The next SGLang target is a quality fix or a larger/model-shaped FP4 KV row that passes deterministic sanity and quality checks while preserving the `1.78x` capacity gain.

## 2026-06-08 d7d931f Matched Backend Trace Row

Artifacts:

- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_summary.md`
- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp8_openai_benchmark.json`
- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp8_raw_2plus2.json`
- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp8_chat_smoke.json`
- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp8_server.log`
- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp8_trace_excerpt.txt`
- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp4_openai_benchmark.json`
- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp4_raw_2plus2.json`
- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp4_chat_smoke.json`
- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp4_server.log`
- `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_fp4_trace_excerpt.txt`

Result:

- fp8 comparator: `3,105,240` KV tokens, decode `56.996`, `57.034`, and `57.266 tok/s`, raw `2+2` and chat smoke pass with normal benchmark content.
- FP4 KV: `5,517,572` KV tokens, capacity ratio `1.7769x` versus fp8, raw `2+2` returns `4`, and chat smoke returns `spark-ok`.
- FP4 backend trace: all 28 decode layers and all 28 `extend_merge_paged` layers use packed `uint8` K/V plus FP8 scale buffers.
- FP4 benchmark quality still fails: `short_decode` stops at `,explain why.`, and medium/long outputs are repetitive or garbled.

Interpretation:

- This retires the "repeat a matched row with request-path trace" task.
- It proves capacity, backend routing, and smoke-level sanity on `jethac/sglang@d7d931f`.
- It is still not a blessed serving-quality or speed row. The next SGLang work is quality localization on the degraded benchmark prompts, not another capacity rerun.

## 2026-06-08 d7d931f Logprob Quality Probe

Artifacts:

- `results/sglang_qwen_fp4kv_d7d931f_logprob_quality_20260608T1609JST_summary.md`
- `results/sglang_qwen_fp4kv_d7d931f_logprob_quality_20260608T1609JST_fp8_quality_probe.json`
- `results/sglang_qwen_fp4kv_d7d931f_logprob_quality_20260608T1609JST_fp4_quality_probe.json`
- `results/sglang_qwen_fp4kv_d7d931f_logprob_quality_20260608T1609JST_compare.json`

Result:

- fp8 passed `short_decode` and `medium_decode` quality probes with generated-token logprobs.
- FP4 `short_decode` began with the same high-confidence English prefix as fp8, then drifted into mixed Chinese text and repetition; similarity to fp8 was only `0.0883`.
- FP4 `medium_decode` diverged immediately: first generated tokens were `the following code:` instead of fp8's `**Engineering Note:`, then the output collapsed into repeated `import` text. Similarity to fp8 was `0.0084`.
- The comparison artifact has `ok=false`.

Interpretation:

- This localizes the FP4 quality failure more tightly than the earlier benchmark row. The corruption can be present at token one for a normal prompt, but another prompt can start plausibly and degrade later.
- This is still not a serving win. It points the next SGLang work at prefill/decode state, per-layer calibration application, wrapper metadata, and scale/layout coupling around the failing prompt.

Likely code locations:

- `third_party/flashinfer/include/flashinfer/vec_dtypes.cuh`: generic `vec_cast` lacks the needed FP4 E2M1 to float conversion path.
- `third_party/flashinfer/include/flashinfer/attention/decode.cuh`: K/V loads go through `vec_t<float>::cast_load`.
- `third_party/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py`: after FlashInfer compiles, verify SGLang passes the FP4 per-block scale buffers the kernel expects, not only scalar `k_scale`/`v_scale`.

## 2026-06-07 Smoke Result

Image:

- `nvcr.io/nvidia/sglang:26.05-py3`
- manifest included `linux/arm64`
- in-container versions: SGLang `0.5.11+nv26.5.51621272`, `sglang-kernel` `0.4.2+nv26.5.51621272`, FlashInfer `0.6.10+cf494fca.nv26.5.cu132.50619265`, PyTorch `2.12.0a0+5aff3928d8.nv26.5.50603568`

Model:

- `Qwen/Qwen2.5-1.5B-Instruct`
- dtype `bfloat16`
- KV cache dtype `torch.bfloat16`
- attention backend `flashinfer`
- CUDA graphs enabled

Artifacts:

- `results/sglang_20260607T115213Z_chat_smoke.json`
- `results/sglang_20260607T115213Z_python_versions.txt`
- `results/sglang_20260607T115213Z_cuda_so_audit_sglang.json`
- `results/sglang_20260607T115213Z_server.log`
- `results/sglang_bench_20260607T120315Z_openai_benchmark.json`
- `results/sglang_bench_longprefill_20260607T120614Z_openai_benchmark.json`

Interpretation:

- Basic OpenAI-compatible serving works on the GB10.
- Short and medium decode measured around 60 tok/s with `mem_fraction_static=0.20`.
- The first long-prefill benchmark failed because the server exposed too small a KV token budget in that run.
- Retrying only long-prefill with `mem_fraction_static=0.40` succeeded: 2,369 prompt tokens, 64 completion tokens, TTFT 0.683 s, total 1.763 s, decode 59.23 tok/s.
- This is a SGLang runtime baseline, not a Gemma baseline and not an NVFP4 validation.

Remaining sm121-specific concern:

- The container reports the device as `NVIDIA GB10 (12, 1)`, but `torch.cuda.get_arch_list()` has `sm_120` and `compute_120`, not explicit `sm_121`.
- The CUDA shared-object audit found `objects_with_sm_121: 0` and `objects_with_sm_120: 3`.
- The server log says `SM120 (Blackwell) detected: auto-selecting fp4-gemm-backend=flashinfer_cudnn` on a GB10 `sm_121` device.
- Treat this as a dispatch/packaging validation issue before calling the path fully Spark-native.

## 2026-06-07 Gemma 4 E2B Result

Model:

- `google/gemma-4-E2B-it-qat-w4a16-ct`

Default launch:

- artifact prefix: `results/sglang_gemma4_e2b_w4a16_20260607T121536Z`
- result: container exited before health
- attention backend selection: `Use triton as default attention backend for Gemma4`
- failure: SGLang selects the Gemma4 multimodal path, constructs the audio tower, then crashes in `Gemma4AudioConformerLightConv1d`
- concrete exception: `AttributeError: 'MergedColumnParallelLinear' object has no attribute 'weight'`

Language-only retry:

- artifact prefix: `results/sglang_gemma4_e2b_w4a16_language_only_20260607T121751Z`
- result: container exited before health
- command added `--language-only`
- failure: argument/config validation raises `ValueError: requires at least one encoder urls to be set via --encoder-urls`

Interpretation:

- The NVIDIA SGLang 26.05 container is functional on GB10 for a supported Qwen BF16 model.
- The same container is not currently a working Gemma 4 E2B text-serving path in our test.
- This is not an sm121 kernel failure yet; it fails before serving, during SGLang's Gemma4 model setup.
- The failure is still a Spark campaign issue because the mission requires Gemma-class models to run through SGLang or have an explicit go/no-go.

## Preferred First Container

As of 2026-06-07, the preferred first smoke path is container-based, not bare-metal pip:

- primary image: `nvcr.io/nvidia/sglang:26.05-py3`
- reason: NVIDIA's 26.05 SGLang release notes list DGX Spark support, CUDA 13.2.1, SGLang 0.5.11, FlashInfer 0.6.10, and NVFP4 support on Blackwell including DGX Spark
- manifest check: verify the image has `linux/arm64` before running it
- fallback image: `lmsysorg/sglang:latest-cu130-runtime`
- fallback reason: upstream SGLang recommends CUDA 13 Docker images, and the host already has CUDA 13.0-oriented images in use

Use a small public model such as `Qwen/Qwen2.5-1.5B-Instruct` for the first container smoke. Avoid gated or large models until the runtime itself is proven.

References:

- https://docs.nvidia.com/deeplearning/frameworks/sglang-release-notes/rel-26-05.html
- https://sgl-project.github.io/get_started/install.html
- https://build.nvidia.com/spark/sglang/instructions

## Container Evidence To Capture

For the first smoke, capture:

- `docker manifest inspect` for the selected image
- `docker image inspect` after pull
- `nvidia-smi` from inside the container
- `spark_doctor` before server start
- `/v1/models` response
- `openai_chat_smoke.py` result
- runtime process probe matching `sglang`
- container Python versions for `sglang`, `sglang-kernel`, `flashinfer-python`, `torch`, and `triton`
- `cuda_so_audit.py` against `sglang`, `sgl_kernel`, and `flashinfer`
- full server log

Known risk: on `aarch64`, SGLang may JIT some kernels at first launch instead of using prebuilt cubins. Treat slow first start or JIT cache failures as packaging evidence, not model evidence.

## NVFP4 Rule

Use BF16/auto and fp8 as the proven SGLang Qwen baselines on our Spark today. Keep NVFP4/FP4 KV unblessed until it passes on Spark with graph-compatible serving and a quality check.

Current fork verification:

- `jethac/sglang@98ad46961` has passed Linux `aarch64` syntax compilation, targeted `KV4Compatibility` pytest for the touched FP4 KV gate files, and a targeted import-alias test for the historical `KVFP4QuantizeUtil` name.
- A CPU-only Docker route was attempted to avoid spending GPU time on Python-level `KV4Compatibility` tests, but `docker/arm64.Dockerfile` failed before pytest while building `sglang-kernel-cpu`; see `results/sglang_fp4_kv_sm121_cpu_docker_verify_20260608T0243JST.md`.
- A no-kernel scratch-venv pytest route passed; see `results/sglang_fp4_kv_sm121_pytest_20260608T0320JST.md`.
- The site-package overlay after-row proves `fp4_e2m1` can serve only when both CUDA graph modes are disabled; that row is not blessed because it is overlay-based, very slow, and low quality.
- The standalone convention bridge passed on GB10; see `results/sglang_nvfp4_kv_convention_probe_20260608.json`.
  - `fp4_quantize` with encode scale (`1 / decode_scale`) plus FA2 reader decode scale: passed (`attention_cosine_vs_source=0.9950249`, `attention_cosine_vs_dequant=0.9999955`).
  - `nvfp4_kv_quantize` with decode scale plus FA2 reader decode scale: passed with the same cosine values.
  - `nvfp4_kv_quantize` with encode scale plus FA2 reader decode scale: failed (`attention_cosine_vs_source=0.0`).
  - This clears raw FA2 reader convention math for the viable pairs, but it does not bless SGLang serving quality. Since the serving overlay had already tried the `fp4_quantize` convention and still produced corrupt text, keep debugging calibration scale plumbing, memory-pool/backend scale application, V-scale layout, and the forced-compiled decode path.
- The SGLang pool bridge passed on GB10; see `results/sglang_fp4_pool_bridge_probe_20260608.json` and `results/sglang_fp4_pool_bridge_probe_prefill_20260608.json`.
  - It writes synthetic K/V through `MHATokenToKVPoolFP4.set_kv_buffer()`, then feeds `get_kv_buffer()`, `get_kv_scale_buffer()`, and `get_kv_global_scale()` directly into FlashInfer FA2.
  - Decode result: `attention_cosine_vs_dequant=0.9999946`, `attention_cosine_vs_source=0.99585`, finite output, and `passed=true`.
  - Paged prefill result: `attention_cosine_vs_dequant=0.9999946`, `attention_cosine_vs_source=0.99576`, finite output, and `passed=true`.
  - This clears the basic pool layout/global-scale contract for decode and paged prefill. The remaining quality bug is downstream: backend wrapper metadata, graph/capture state, stale calibration state, or a model-serving path not covered by the synthetic pool bridge.
- The SGLang matched backend trace row passed smoke on GB10; see `results/sglang_qwen_fp4kv_d7d931f_matched_20260608T1548JST_summary.md`.
  - `jethac/sglang@d7d931f` adds `SGLANG_FP4_KV_TRACE_BACKEND=1` to log native FP4-KV backend calls.
  - The source-overlay run used NVIDIA SGLang 26.05 with `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`, `--kv-cache-dtype fp4_e2m1`, FlashInfer attention, page size 1, and both CUDA graph modes disabled.
  - It allocated `5,517,572` FP4 KV tokens versus `3,105,240` fp8 tokens (`1.7769x`), calibrated 28 layers, and traced all 28 decode plus all 28 `extend_merge_paged` layers using packed `uint8` K/V plus FP8 scale buffers.
  - Raw `2+2 is` returned ` 4, 2+2 is 4, 2+2 is`, and chat smoke returned exactly `spark-ok`.
  - This is stronger capacity/routing evidence, but still not a blessed SGLang row because standardized FP4 benchmark content remains degraded versus fp8.

For NVFP4 validation, record:

- SGLang version/image/commit
- SGLang upstream base commit if using a fork or overlay
- FlashInfer version or patch source
- FlashInfer upstream base commit if using a fork or overlay
- model id/revision
- attention backend
- `--kv-cache-dtype fp4_e2m1`
- page size
- CUDA graph mode
- fresh JIT cache path
- `SGLANG_FP4_KV_AUTOCALIB` value
- whether checkpoint `k_scale`/`v_scale` values are present or calibration is generated at startup
- deterministic prompt output
- fp4-vs-fp8 quality comparison
- prefill/decode speed
- memory/KV capacity difference
- whether patches are SM120-derived or SM121-specific
- whether the model is large enough for NVFP4 KV quality to be meaningful; the SM120 reference repo warns that small models can fail quality even when the kernel is numerically correct

## Fork Rule

If SGLang needs source changes, fork `sgl-project/sglang` to `jethac/sglang`, add it as `third_party/sglang`, and do the patch in an issue-named worktree.

If FlashInfer needs source changes for the SGLang path, fork `flashinfer-ai/flashinfer` to `jethac/flashinfer`, add it as `third_party/flashinfer`, and use a separate worktree.

Use the supplied SM120 repo as reference context and prior art. Unless a better reason appears, build on its design and tests, but port productionable changes into `jethac/sglang` and `jethac/flashinfer` branches with upstream-base commits recorded, contributing guidelines followed, and GB10 proof artifacts attached.
