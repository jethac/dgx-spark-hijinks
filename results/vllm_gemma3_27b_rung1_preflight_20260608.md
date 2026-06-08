# vLLM Gemma 3 27B Rung 1 Live Preflight, 2026-06-08

Status: host reachable; do not start the live row until source/cache prerequisites are
cleaned up.

## Connectivity

- SSH to the Spark-class Linux endpoint as `root` works.
- The device reports `NVIDIA GB10`, compute capability `12.1`, and `0 %` GPU utilization
  at preflight time.
- `docker ps` reported no running containers.
- The target vLLM image exists locally:
  `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`.

## Path Findings

The original packet assumptions are not valid on the Linux endpoint:

- missing: `/home/jethac/src/vllm`
- missing: `/home/jethac/src/flashinfer`
- present: `/home/jethac/.cache/huggingface`
- present: `/home/jethac/dgx-spark-hijinks`
- present: `/home/jethac/spark_tmp`

The existing `/home/jethac/dgx-spark-hijinks` checkout is an older `main` checkout with
local dirty files and many untracked historical result artifacts. Do not update or reuse it
as the live command source without first making an intentional clean run checkout. The safer
next live run should use a fresh checkout of `docs/codex-direction-nvfp4-kv` or initialized
submodules from that branch.

## Model Cache

The Hugging Face cache is large and contains Gemma 4 and Qwen artifacts, but no
`google/gemma-3-27b-it` cache directory was found in the bounded search. The first Gemma 3
live row likely needs gated Hugging Face access plus download time and disk headroom.

## Immediate Fix Applied Locally

The prep packet generator was updated after this preflight so generated packets:

- stream `docker logs -f` into `_server.log`
- write container IDs
- wait for `/v1/models` before recording the OpenAI serving row
- remove row containers on failure or normal exit
- document submodule-based source paths instead of the missing `/home/jethac/src/*` paths

## Next Gate

Create or sync a clean Linux run checkout, initialize the `third_party/vllm` and
`third_party/flashinfer` submodules, ensure `HF_TOKEN`/Gemma 3 access is available, then
run the regenerated command packet for the fp8 comparator row first. Only start the NVFP4
candidate after the fp8 row produces server logs, runtime geometry, smoke, benchmark, and
quality artifacts.
