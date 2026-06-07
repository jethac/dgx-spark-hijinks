# SGLang FP4 KV SM121 CPU Docker Verification Attempt

Date: 2026-06-08 JST

Target branch:

- repo: `https://github.com/jethac/sglang.git`
- branch/commit: `spark/hijinks-018-fp4-e2m1-kv-sm121` / `67c7967a1913960055e64c49c26c5f622c1f1ff1`
- reference: `hikarioyama/sglang-nvfp4-kv-sm120@9b2160f0fb8e11dbbb5171a57f06a02b0e9ba6e2`

Intent:

- Use a Linux `aarch64` Docker build as a cheap, CPU-only route to run the SGLang `KV4Compatibility` unit test without consuming GB10 GPU time.
- Validate Python-level routing/compatibility behavior before attempting any GPU serving or native FP4 KV memory-pool work.

Command:

```bash
cd /root/spark-validation/sglang-fp4-kv-sm121
docker build \
  -t sglang-fp4-kv-sm121-cpu:67c7967a1 \
  -f docker/arm64.Dockerfile \
  --build-arg SGLANG_REPO=https://github.com/jethac/sglang.git \
  --build-arg VER_SGLANG=67c7967a1913960055e64c49c26c5f622c1f1ff1 \
  .
```

Outcome:

- Docker was available and the `docker/arm64.Dockerfile` build started.
- The build cloned `jethac/sglang` and checked out `67c7967a1913960055e64c49c26c5f622c1f1ff1`.
- The build failed before pytest while building `sglang-kernel-cpu`.

Failure:

```text
/sgl-workspace/sglang/sgl-kernel/csrc/cpu/aarch64/shm.h:63:28:
error: inlining failed in call to always_inline float16x8_t vaddq_f16(float16x8_t, float16x8_t):
target specific option mismatch
```

The failing translation unit was:

```text
/sgl-workspace/sglang/sgl-kernel/csrc/cpu/shm.cpp
```

Interpretation:

- This does not contradict the SGLang FP4 KV compatibility-gate patch. The targeted tests did not run.
- The current ARM64 CPU Docker build is not yet a cheap verification route because the CPU kernel package fails to compile on this host/toolchain.
- Next options are either a no-kernel Python-only pytest image for `test_server_args.py`, or an upstream-shaped ARM64 CPU build-flag fix for the `vaddq_f16` path.
