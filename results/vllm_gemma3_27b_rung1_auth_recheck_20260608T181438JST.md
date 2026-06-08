# vLLM Gemma 3 27B Rung 1 Auth Recheck

Date: 2026-06-08 18:14 JST

Purpose: recheck whether the prepared Gemma 3 27B Rung 1 vLLM packet can be run without
wasting GPU time on a gated Hugging Face download failure, and close the runtime geometry
hook precondition required by the evidence gate.

## Remote Checks

Target host:

- `thinkstationpgx-00b4.tail740c8d.ts.net`

Commands run from the Windows workstation:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=8 root@thinkstationpgx-00b4.tail740c8d.ts.net "hostname; whoami; nvidia-smi --query-gpu=name,compute_cap,utilization.gpu,memory.used,memory.total --format=csv,noheader; docker ps --format '{{.Names}} {{.Status}}' | head -20"

ssh -o BatchMode=yes -o ConnectTimeout=8 root@thinkstationpgx-00b4.tail740c8d.ts.net "bash -lc 'env | grep -E ^HF_TOKEN= >/dev/null && echo HF_TOKEN_present=yes || echo HF_TOKEN_present=no; test -f /root/.cache/huggingface/token && echo root_hf_token_file=yes || echo root_hf_token_file=no; test -f /home/jethac/.cache/huggingface/token && echo jethac_hf_token_file=yes || echo jethac_hf_token_file=no; test -d /home/jethac/.cache/huggingface/hub/models--google--gemma-3-27b-it && echo gemma3_cache=yes || echo gemma3_cache=no; ls -ld /home/jethac/.cache/huggingface /home/jethac/.cache/huggingface/hub 2>/dev/null || true'"

ssh -o BatchMode=yes -o ConnectTimeout=8 root@thinkstationpgx-00b4.tail740c8d.ts.net "timeout 5s grep -n SPARK_GEMMA_KV_GEOMETRY /home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608/third_party/vllm/vllm/model_executor/layers/attention/attention.py || true; timeout 5s grep -n SPARK_GEMMA_KV_SPEC /home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608/third_party/vllm/vllm/model_executor/layers/attention/attention.py || true; timeout 5s grep -n VLLM_SPARK_KV_GEOMETRY_LOG /home/jethac/spark_tmp/dgx-spark-hijinks-vllm-gemma3-rung1-20260608/third_party/vllm/vllm/model_executor/layers/attention/attention.py || true"
```

## Result

The host is reachable and idle:

```text
thinkstationpgx-00b4
root
NVIDIA GB10, 12.1, 0 %, [N/A], [N/A]
```

No running Docker containers were printed by the `docker ps` query.

The Gemma 3 access/cache gate is still red:

```text
HF_TOKEN_present=no
root_hf_token_file=no
jethac_hf_token_file=no
gemma3_cache=no
drwxrwxr-x  5 jethac jethac 4096 Jun  8 17:31 /home/jethac/.cache/huggingface
drwxr-xr-x 19 jethac jethac 4096 Jun  8 16:40 /home/jethac/.cache/huggingface/hub
```

The prepared vLLM overlay does not yet contain the runtime geometry hook. Grepping the
remote overlay for `SPARK_GEMMA_KV_GEOMETRY`, `SPARK_GEMMA_KV_SPEC`, and
`VLLM_SPARK_KV_GEOMETRY_LOG` returned no matches.

Follow-up in this same checkpoint: the missing hook was added to
`jethac/vllm@spark/hijinks-007-nvfp4-kv-sm121` and pushed as
`3658ba7123c3eb2211f18a882af1b993112fadb1`. The remote run checkout was updated to that
commit and now contains the required geometry log strings:

```text
+3658ba7123c3eb2211f18a882af1b993112fadb1 third_party/vllm (v0.13.0rc1-5248-g3658ba712)
298:                "SPARK_GEMMA_KV_GEOMETRY layer=%s heads=%s kv_heads=%s "
593:            "SPARK_GEMMA_KV_SPEC layer=%s spec=%s block_size=%s "
```

The remote run checkout is otherwise the intended detached overlay checkout when inspected
as `jethac`:

```text
+e152cf4da4ab2a9d093b7d9d4b499198b0211c61 third_party/flashinfer (v0.2.11.post3-1038-ge152cf4d)
+3658ba7123c3eb2211f18a882af1b993112fadb1 third_party/vllm (v0.13.0rc1-5248-g3658ba712)
```

It also has untracked prep artifacts from the previous setup. Do not treat that checkout as
a clean source of superproject changes without reviewing those files separately.

## Interpretation

Do not start the prepared Gemma 3 27B fp8 comparator row yet. The remaining blocker is
gated Hugging Face authentication or a missing local cache for `google/gemma-3-27b-it`.
The geometry hook blocker is closed by
`jethac/vllm@3658ba7123c3eb2211f18a882af1b993112fadb1`.

Once a valid token or local snapshot is present, run the fp8 comparator first from:

```text
docs/results/vllm_gemma3_27b_rung1_20260608TCHECKOUTJST_command_packet.sh
```

Only run the NVFP4 candidate after the fp8 row produces server logs, runtime geometry,
smoke, benchmark, build-target audit, and quality artifacts.
