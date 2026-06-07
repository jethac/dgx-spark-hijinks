#!/usr/bin/env python3
"""Run a command while sampling process and memory telemetry.

This is intended for fragile fallback paths, especially HF fallback rows where a
plain return code such as -9 is not enough to distinguish resource pressure from
a real model/runtime error.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from spark_hardware import collect_cuda_hardware


def run_capture(cmd: list[str], timeout: int = 10) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"cmd": cmd, "error": repr(exc)}


def read_status(pid: int) -> dict[str, Any] | None:
    path = Path("/proc") / str(pid) / "status"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    out: dict[str, Any] = {"pid": pid}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if key in {"Name", "State", "PPid"}:
            out[key.lower()] = value
        elif key in {"VmRSS", "VmHWM", "VmSize", "RssAnon", "RssFile", "VmSwap"}:
            out[key.lower()] = parse_kib(value)
    return out


def parse_kib(value: str) -> int | None:
    match = value.split()
    if not match:
        return None
    try:
        return int(match[0])
    except ValueError:
        return None


def list_children(root_pid: int) -> list[int]:
    children: list[int] = []
    proc_root = Path("/proc")
    if not proc_root.exists():
        return children
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        status = read_status(int(path.name))
        if not status:
            continue
        try:
            ppid = int(str(status.get("ppid", "0")).split()[0])
        except ValueError:
            continue
        if ppid == root_pid:
            children.append(int(path.name))
    return children


def process_tree(root_pid: int) -> list[int]:
    seen = {root_pid}
    queue = [root_pid]
    while queue:
        pid = queue.pop(0)
        for child in list_children(pid):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return sorted(seen)


def sample_processes(root_pid: int) -> dict[str, Any]:
    pids = process_tree(root_pid)
    statuses = [status for pid in pids if (status := read_status(pid))]
    rss_kib = sum(int(status.get("vmrss") or 0) for status in statuses)
    hwm_kib = sum(int(status.get("vmhwm") or 0) for status in statuses)
    swap_kib = sum(int(status.get("vmswap") or 0) for status in statuses)
    return {
        "timestamp": time.time(),
        "pids": pids,
        "rss_kib": rss_kib,
        "hwm_kib": hwm_kib,
        "swap_kib": swap_kib,
        "processes": statuses,
    }


def memory_snapshot() -> dict[str, Any]:
    return {
        "free": run_capture(["free", "-b"], timeout=5),
        "nvidia_smi": run_capture(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
        ),
    }


def oom_snapshot(since: str | None = None) -> dict[str, Any]:
    journal_cmd = ["journalctl", "-k", "--no-pager"]
    if since:
        journal_cmd.extend(["--since", since])
    journal_cmd.extend(["-g", "Out of memory|Killed process|oom-kill"])
    return {
        "dmesg_oom_tail": run_capture(
            [
                "sh",
                "-lc",
                "dmesg -T | grep -Ei 'out of memory|killed process|oom-kill' | tail -80",
            ],
            timeout=10,
        ),
        "journal_oom": run_capture(journal_cmd, timeout=15),
    }


def terminate_process(proc: subprocess.Popen[str], grace_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.terminate()
    deadline = time.time() + grace_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        proc.kill()


def classify_returncode(returncode: int | None) -> str:
    if returncode is None:
        return "timeout_or_interrupted"
    if returncode == 0:
        return "ok"
    if returncode == -9 or returncode == 137:
        return "process_killed"
    if returncode == -11 or returncode == 139:
        return "process_crash"
    if returncode == -15 or returncode == 143:
        return "terminated"
    return "process_error"


def run_with_telemetry(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    hardware = collect_cuda_hardware()
    pre_memory = memory_snapshot()
    proc = subprocess.Popen(
        args.command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    samples: list[dict[str, Any]] = []
    stop_sampling = threading.Event()

    def sampler() -> None:
        while not stop_sampling.is_set():
            try:
                sample = sample_processes(proc.pid)
                sample["memory"] = memory_snapshot()
                samples.append(sample)
            except Exception as exc:
                samples.append({"timestamp": time.time(), "error": repr(exc)})
            stop_sampling.wait(args.interval_s)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=args.timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        terminate_process(proc)
        stdout, stderr = proc.communicate()
    finally:
        stop_sampling.set()
        thread.join(timeout=5)

    finished = time.time()
    report = {
        "schema": "spark-run-with-telemetry/v1",
        "run_id": args.run_id,
        "backend": args.backend,
        "model": args.model,
        "command": args.command,
        "hardware": hardware,
        "started_utc": started_utc,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished)),
        "elapsed_s": finished - started,
        "timeout_s": args.timeout_s,
        "timed_out": timed_out,
        "returncode": proc.returncode,
        "failure_class": "timeout" if timed_out else classify_returncode(proc.returncode),
        "stdout_tail": stdout[-args.tail_chars :],
        "stderr_tail": stderr[-args.tail_chars :],
        "samples": samples,
        "peak_process_tree_rss_kib": max((s.get("rss_kib", 0) for s in samples), default=0),
        "peak_process_tree_swap_kib": max((s.get("swap_kib", 0) for s in samples), default=0),
        "pre_memory": pre_memory,
        "post_memory": memory_snapshot(),
        "oom_evidence": oom_snapshot(since=started_utc),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--backend", default="unknown")
    parser.add_argument("--model")
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--interval-s", type=float, default=5.0)
    parser.add_argument("--tail-chars", type=int, default=12000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("command is required after --")

    report = run_with_telemetry(args)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("run_id", "failure_class", "returncode", "elapsed_s")}, indent=2))
    return 0 if report.get("returncode") == 0 and not report.get("timed_out") else 2


if __name__ == "__main__":
    raise SystemExit(main())
