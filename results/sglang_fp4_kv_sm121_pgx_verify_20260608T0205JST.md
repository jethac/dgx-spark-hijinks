# SGLang FP4 KV SM121 PGX Verification

Date: 2026-06-08 JST
Host: `root@thinkstationpgx-00b4`
Local repo: `B:\workshop\dgx-spark-hijinks`
Target SGLang branch: `jethac/sglang spark/hijinks-018-fp4-e2m1-kv-sm121`
Target commit: `67c7967a1913960055e64c49c26c5f622c1f1ff1`
Remote clone path: `/root/spark-validation/sglang-fp4-kv-sm121`
Remote worktree pattern: `/root/spark-validation/sglang-fp4-kv-sm121-worktrees/verify-*`

## Local Precheck

Command:

```powershell
git submodule status -- third_party/sglang
git -C third_party/sglang show --stat --oneline --decorate --name-only 67c7967a1913960055e64c49c26c5f622c1f1ff1
git -C third_party/sglang remote -v
git -C third_party/sglang branch -a --contains 67c7967a1913960055e64c49c26c5f622c1f1ff1
```

Result:

```text
67c7967a1913960055e64c49c26c5f622c1f1ff1 third_party/sglang (gateway-v0.3.1-4951-g67c7967a1)

67c7967a1 (HEAD, origin/spark/hijinks-018-fp4-e2m1-kv-sm121, spark/hijinks-018-fp4-e2m1-kv-sm121) Allow SM12x FlashInfer FP4 KV gates
python/sglang/srt/layers/quantization/kvfp4_tensor.py
python/sglang/srt/server_args.py
test/registered/unit/server_args/test_server_args.py

origin git@github.com:jethac/sglang.git (fetch)
origin git@github.com:jethac/sglang.git (push)
upstream https://github.com/sgl-project/sglang.git (fetch)
upstream https://github.com/sgl-project/sglang.git (push)

* (HEAD detached at 67c7967a1)
+ spark/hijinks-018-fp4-e2m1-kv-sm121
  remotes/origin/spark/hijinks-018-fp4-e2m1-kv-sm121
```

## PGX Host Check

Command:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 root@thinkstationpgx-00b4 "hostname && date -Is && uname -a"
```

Result:

```text
thinkstationpgx-00b4
2026-06-08T02:03:58+09:00
Linux thinkstationpgx-00b4 6.17.0-1021-nvidia #21-Ubuntu SMP PREEMPT_DYNAMIC Wed May 27 19:14:05 UTC 2026 aarch64 aarch64 aarch64 GNU/Linux
```

## Clone, Fetch, And Worktree

Commands run non-interactively over SSH:

```bash
mkdir -p /root/spark-validation /root/spark-validation/sglang-fp4-kv-sm121-worktrees
git clone --filter=blob:none https://github.com/jethac/sglang.git /root/spark-validation/sglang-fp4-kv-sm121
git -C /root/spark-validation/sglang-fp4-kv-sm121 remote -v
git -C /root/spark-validation/sglang-fp4-kv-sm121 fetch origin +refs/heads/spark/hijinks-018-fp4-e2m1-kv-sm121:refs/remotes/origin/spark/hijinks-018-fp4-e2m1-kv-sm121
git -C /root/spark-validation/sglang-fp4-kv-sm121 rev-parse origin/spark/hijinks-018-fp4-e2m1-kv-sm121
git -C /root/spark-validation/sglang-fp4-kv-sm121 cat-file -e 67c7967a1913960055e64c49c26c5f622c1f1ff1^{commit}
git -C /root/spark-validation/sglang-fp4-kv-sm121 worktree prune
git -C /root/spark-validation/sglang-fp4-kv-sm121 worktree add --detach /root/spark-validation/sglang-fp4-kv-sm121-worktrees/verify-67c7967a1-20260608T020430+0900 67c7967a1913960055e64c49c26c5f622c1f1ff1
git -C /root/spark-validation/sglang-fp4-kv-sm121-worktrees/verify-67c7967a1-20260608T020430+0900 rev-parse HEAD
git -C /root/spark-validation/sglang-fp4-kv-sm121-worktrees/verify-67c7967a1-20260608T020430+0900 status --short
```

Result:

```text
Clone succeeded under /root/spark-validation/sglang-fp4-kv-sm121.
origin/spark/hijinks-018-fp4-e2m1-kv-sm121 resolved to 67c7967a1913960055e64c49c26c5f622c1f1ff1.
Detached worktree HEAD resolved to 67c7967a1913960055e64c49c26c5f622c1f1ff1.
Worktree status --short produced no output.
```

## Exact Python Command Attempt

Command:

```bash
python --version
python -m py_compile python/sglang/srt/layers/quantization/kvfp4_tensor.py python/sglang/srt/server_args.py test/registered/unit/server_args/test_server_args.py
```

Result:

```text
main: line 14: python: command not found
[exit 127] python --version
main: line 14: python: command not found
[exit 127] python -m py_compile python/sglang/srt/layers/quantization/kvfp4_tensor.py python/sglang/srt/server_args.py test/registered/unit/server_args/test_server_args.py
```

Limitation: PGX does not currently have a `python` executable on PATH.

## Python3 Compile

Commands:

```bash
git -C /root/spark-validation/sglang-fp4-kv-sm121 fetch origin +refs/heads/spark/hijinks-018-fp4-e2m1-kv-sm121:refs/remotes/origin/spark/hijinks-018-fp4-e2m1-kv-sm121
git -C /root/spark-validation/sglang-fp4-kv-sm121 rev-parse origin/spark/hijinks-018-fp4-e2m1-kv-sm121
git -C /root/spark-validation/sglang-fp4-kv-sm121 worktree add --detach /root/spark-validation/sglang-fp4-kv-sm121-worktrees/verify-python3-67c7967a1-20260608T020519+0900 67c7967a1913960055e64c49c26c5f622c1f1ff1
cd /root/spark-validation/sglang-fp4-kv-sm121-worktrees/verify-python3-67c7967a1-20260608T020519+0900
python3 --version
python3 -m py_compile python/sglang/srt/layers/quantization/kvfp4_tensor.py python/sglang/srt/server_args.py test/registered/unit/server_args/test_server_args.py
```

Result:

```text
origin/spark/hijinks-018-fp4-e2m1-kv-sm121 resolved to 67c7967a1913960055e64c49c26c5f622c1f1ff1.
Python 3.12.3
[exit 0] python3 -m py_compile python/sglang/srt/layers/quantization/kvfp4_tensor.py python/sglang/srt/server_args.py test/registered/unit/server_args/test_server_args.py
```

Status: PASS with `python3`.

## Targeted Pytest

Requested command:

```bash
PYTHONPATH=python python -m pytest test/registered/unit/server_args/test_server_args.py -k KV4Compatibility -q
```

Availability checks:

```bash
command -v python || true
command -v python3 || true
python3 --version
python3 -m pytest --version
```

Result:

```text
/usr/bin/python3
Python 3.12.3
/usr/bin/python3: No module named pytest
[exit 1] python3 -m pytest --version
SKIP pytest: python3 -m pytest is unavailable
```

Status: NOT RUN. The exact requested command is blocked because `python` is missing, and the closest available interpreter `python3` does not have `pytest` installed.

## Cleanup Check

Commands:

```bash
git -C /root/spark-validation/sglang-fp4-kv-sm121 worktree remove /root/spark-validation/sglang-fp4-kv-sm121-worktrees/verify-67c7967a1-20260608T020430+0900
git -C /root/spark-validation/sglang-fp4-kv-sm121 worktree remove /root/spark-validation/sglang-fp4-kv-sm121-worktrees/verify-python3-67c7967a1-20260608T020519+0900
git -C /root/spark-validation/sglang-fp4-kv-sm121 worktree list --porcelain
git -C /root/spark-validation/sglang-fp4-kv-sm121 status --short
```

Result:

```text
worktree /root/spark-validation/sglang-fp4-kv-sm121
HEAD 02be2e71899491b7aaf2849dce6431f61fc190b6
branch refs/heads/main

status --short produced no output.
```

Status: temporary detached worktrees removed. The reusable clone remains at `/root/spark-validation/sglang-fp4-kv-sm121`.

## Summary

PASS:

```text
Branch fetch confirmed origin/spark/hijinks-018-fp4-e2m1-kv-sm121 == 67c7967a1913960055e64c49c26c5f622c1f1ff1.
Linux/aarch64 PGX host reachable over non-interactive SSH.
Detached PGX worktree checkout of the target commit succeeded.
python3 -m py_compile passed for:
  python/sglang/srt/layers/quantization/kvfp4_tensor.py
  python/sglang/srt/server_args.py
  test/registered/unit/server_args/test_server_args.py
```

BLOCKED:

```text
The exact `python` executable is absent on PGX.
`python3 -m pytest` is unavailable because the `pytest` module is not installed.
The targeted KV4Compatibility pytest was not run.
```

Wrapper note:

```text
The second SSH wrapper printed the summary and removed the temporary worktree, then exited 1 due to a missing final shell `fi` in the wrapper script. The validation statuses above use the per-command exit codes printed before that wrapper error.
```
