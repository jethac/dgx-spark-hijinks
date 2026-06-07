# SGLang On DGX Spark

Status: BF16/auto and fp8 Qwen rows pass in NVIDIA 26.05 container; stock `fp4_e2m1` Qwen fails before serving; `jethac/sglang@98ad46961` clears the first gate/alias blockers; patched Triton FP4 KV can serve only with graph paths disabled and is too slow to bless.

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

Result:

- BF16/auto KV with FlashInfer attention serves at the same `mem_fraction_static=0.40` and records `1,557,709` KV tokens.
- BF16/auto decode is `58.89`, `58.59`, and `57.73 tok/s` across the standard short, medium, and long-prefill cases.
- fp8 KV with FlashInfer attention serves `Qwen/Qwen2.5-1.5B-Instruct` on GB10 and records `NVIDIA_GB10:sm_121:sms_48`.
- fp8 KV allocated `3,113,713` KV tokens and decoded around `58-59 tok/s`.
- stock `fp4_e2m1` with FlashInfer attention fails before health because `KV4Compatibility` rejects FlashInfer for MHA FP4 KV.
- stock `fp4_e2m1` with Triton attention allocates `5,534,509` KV tokens, about `1.78x` fp8 capacity, then fails before health because `KVFP4QuantizeUtil` cannot be imported from `kvfp4_tensor.py`.
- patched FlashInfer attention clears the SGLang gate and alias blockers, allocates `5,539,718` FP4 KV tokens, targets `compute_121a,code=sm_121a`, then fails compiling FlashInfer FP4 decode at `vec_dtypes.cuh(117)`.
- patched Triton attention with normal graph capture hangs at the first CUDA graph batch. With only `--disable-cuda-graph`, it still enters piecewise graph capture and stalls. With both `--disable-cuda-graph` and `--disable-piecewise-cuda-graph`, it serves and allocates `5,541,103` FP4 KV tokens, but short decode is only `0.276 tok/s` and output quality is visibly poor.

Interpretation:

- This confirms fp8 is a valid SGLang Qwen comparator on our GB10.
- This also confirms stock NVIDIA 26.05 SGLang does not yet provide a working `fp4_e2m1` serving row for this Qwen model.
- The SGLang fork fixes are necessary but not sufficient. The remaining blockers are FlashInfer FP4 E2M1 decode support for the FlashInfer-attention path and graph-compatible Triton FP4 KV serving with sane output.

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
