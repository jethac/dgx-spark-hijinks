# vLLM Gemma 3 27B Rung 1 Checkout Setup, 2026-06-08

Status: clean Linux run checkout prepared; live serving not started.

## Checkout

- path: `/home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608`
- branch: `docs/codex-direction-nvfp4-kv`
- superproject commit: `595dfb6dba863088707afadbad816a511b803f81`
- generated packet:
  `docs/results/vllm_gemma3_27b_rung1_20260608TCHECKOUTJST_command_packet.sh`

The checkout was cloned from `https://github.com/jethac/dgx-spark-hijinks.git`. The
needed submodules were fetched over HTTPS to avoid relying on GitHub SSH host-key setup on
the Spark-class host.

## Source Overlays

The live Gemma 3 Rung 1 plan requires the same source overlay commits used by the proven
Qwen NVFP4-KV path, not the branch's general-purpose submodule pointers. The run checkout
therefore intentionally has these submodules checked out at detached commits:

```text
+e152cf4da4ab2a9d093b7d9d4b499198b0211c61 third_party/flashinfer (v0.2.11.post3-1038-ge152cf4d)
+8916796bc50926fd61e606718b194a71e2e31a24 third_party/vllm (v0.13.0rc1-5247-g8916796bc)
```

The leading `+` in `git submodule status` is expected here: it means the run checkout is
using the exact Gemma 3 packet overlay commits rather than the superproject's broader
campaign submodule pins.

## Validation

The generated command packet was checked with:

```bash
bash -n docs/results/vllm_gemma3_27b_rung1_20260608TCHECKOUTJST_command_packet.sh
```

The packet contains 223 lines and uses these source mounts:

```text
VLLM_SRC=/home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608/third_party/vllm
FLASHINFER_SRC=/home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608/third_party/flashinfer
HF_CACHE=/home/jethac/.cache/huggingface
```

## Remaining Gate Before GPU Run

`google/gemma-3-27b-it` was not found in the bounded Hugging Face cache search during the
previous preflight. Before running the fp8 comparator row, confirm `HF_TOKEN`/gated access
and disk headroom for the model download. Start the fp8 comparator first and only proceed
to the NVFP4 candidate after fp8 produces server log, import probe, runtime geometry, chat
smoke, benchmark, build-target audit, and quality artifacts.
