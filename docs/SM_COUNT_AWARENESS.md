# GB10 SM Count Awareness

Status: active benchmark and tuning rule.

DGX Spark-class GB10 devices are `sm_121`, but performance tuning must not assume every `sm_121` device has the same SM count. The current GB10 unit reports 48 CUDA multiprocessors. Future or binned GB10 systems may share the same ISA while exposing fewer SMs.

## Rule

- Correctness and routing gates may use the SM12x / compute-capability family when the kernel path is family-compatible.
- Performance heuristics must use runtime SM count when tile, occupancy, split-K, graph-size, grid, or "is this path worth it" decisions depend on available parallelism.
- Before/after benchmark rows are only directly comparable when GPU name, compute capability, and `multi_processor_count` match.

## Harness Changes

The following scripts now record CUDA hardware identity and a comparison key:

- `scripts/spark_doctor.py`
- `scripts/run_with_telemetry.py`
- `scripts/openai_serving_benchmark.py`
- `scripts/spark_smoke_suite.py`

The shared helper is `scripts/spark_hardware.py`.

Current GB10 evidence:

- artifact: `results/spark_doctor_smcount_20260607T172142Z.md`
- GPU: `NVIDIA GB10`
- compute capability: `[12, 1]`
- SM count: `48`
- comparison key: `NVIDIA_GB10:sm_121:sms_48`

The helper warns when a run reports an SM count different from the observed 48-SM GB10 reference run so performance rows are not silently compared across bins.

## Fork Audit

Read-only audit of the current vLLM, FlashInfer, and SGLang fork branches found no hardcoded 48-SM performance assumption in our patches.

Findings:

- vLLM NVFP4 KV routing is keyed on `kv_cache_dtype == "nvfp4"` plus the SM12x family helper. That is a correctness/routing decision, not an SM-count heuristic.
- FlashInfer FA2 KV stride/page changes add scale-factor stride plumbing and V-offset handling. The changed files do not bake in a 48-SM assumption.
- FlashInfer performance helpers touched by the earlier `mm_fp4` lane use runtime SM count where relevant; the audited call path passes `get_device_sm_count(...)`.
- Existing FlashInfer page/prefill split/grid helpers use CUDA device attributes such as `cudaDevAttrMultiProcessorCount` dynamically.
- SGLang gate changes are architecture/backend availability checks, not SM-count tuning.

False positives checked:

- `0.48` thresholds in SGLang are numeric thresholds, not SM counts.
- `range(48, 257, 16)` in SGLang batch/capture configuration is unrelated to CUDA SM count.
- FlashInfer comments containing `32-48` are tile-coordinate diagrams, not device-SM assumptions.

## Remaining Work

- Keep auditing any future tile, split-K, graph-size, and occupancy changes for runtime SM-count use.
- Do not claim performance portability from the observed 48-SM reference system to smaller `sm_121` bins without new measurements or an explicit normalization model.
