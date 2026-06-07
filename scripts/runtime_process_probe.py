#!/usr/bin/env python3
"""Collect runtime evidence for a local model server process."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_ENV_PREFIXES = (
    "CUDA",
    "NCCL",
    "VLLM",
    "SGLANG",
    "TORCH",
    "PYTHON",
    "FLASH",
    "TRITON",
)

DEFAULT_MAP_PATTERNS = (
    "vllm",
    "flashinfer",
    "flash_attn",
    "flash",
    "triton",
    "torch",
    "cuda",
    "cublas",
    "cudnn",
    "nccl",
    "cutlass",
)


def run(cmd: list[str], timeout: int = 20) -> dict[str, Any]:
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


def get_json(url: str, timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def proc_cmdline(pid: int) -> list[str] | None:
    path = Path(f"/proc/{pid}/cmdline")
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]


def proc_environ(pid: int) -> dict[str, Any]:
    path = Path(f"/proc/{pid}/environ")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return {"error": repr(exc)}
    env: dict[str, str] = {}
    for part in raw.split(b"\0"):
        if not part or b"=" not in part:
            continue
        key, value = part.split(b"=", 1)
        key_text = key.decode("utf-8", errors="replace")
        if key_text.startswith(DEFAULT_ENV_PREFIXES):
            env[key_text] = value.decode("utf-8", errors="replace")
    return {"values": dict(sorted(env.items()))}


def proc_status(pid: int) -> dict[str, str] | None:
    path = Path(f"/proc/{pid}/status")
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    wanted = {
        "Name",
        "State",
        "Tgid",
        "Pid",
        "PPid",
        "Uid",
        "Gid",
        "Threads",
        "VmRSS",
        "VmHWM",
        "VmSize",
    }
    result: dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in wanted:
            result[key] = value.strip()
    return result


def proc_cwd(pid: int) -> dict[str, Any]:
    try:
        return {"path": os.readlink(f"/proc/{pid}/cwd")}
    except OSError as exc:
        return {"error": repr(exc)}


def proc_maps(pid: int, patterns: tuple[str, ...]) -> dict[str, Any]:
    path = Path(f"/proc/{pid}/maps")
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return {"error": repr(exc)}
    regex = re.compile("|".join(re.escape(p) for p in patterns), re.IGNORECASE)
    libs: set[str] = set()
    for line in lines:
        if "/" not in line:
            continue
        lib = line.split(None, 5)[-1]
        if regex.search(lib):
            libs.add(lib)
    return {
        "matched_count": len(libs),
        "matched_paths": sorted(libs),
    }


def find_pids(match: str) -> list[int]:
    proc_root = Path("/proc")
    pids = []
    self_pid = os.getpid()
    if not proc_root.exists():
        return pids
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == self_pid:
            continue
        cmdline = proc_cmdline(pid)
        if not cmdline:
            continue
        text = " ".join(cmdline)
        if "runtime_process_probe.py" in text:
            continue
        if match.lower() in text.lower():
            pids.append(pid)
    return sorted(pids)


def module_version(name: str) -> dict[str, Any]:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    return {
        "available": True,
        "version": getattr(mod, "__version__", "unknown"),
        "file": getattr(mod, "__file__", None),
    }


def collect_process(pid: int) -> dict[str, Any]:
    return {
        "pid": pid,
        "cmdline": proc_cmdline(pid),
        "cwd": proc_cwd(pid),
        "status": proc_status(pid),
        "environment": proc_environ(pid),
        "maps": proc_maps(pid, DEFAULT_MAP_PATTERNS),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--match", default="vllm")
    parser.add_argument("--pid", action="append", type=int, default=[])
    parser.add_argument("--output")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema": "runtime-process-probe/v1",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version,
            "executable": sys.executable,
        },
        "url": args.url,
        "match": args.match,
        "commands": {},
        "packages": {},
        "server": {},
        "processes": [],
    }

    for name in ["vllm", "sglang", "flashinfer", "torch", "triton", "transformers"]:
        report["packages"][name] = module_version(name)

    report["commands"]["nvidia_smi_query"] = run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,utilization.gpu",
            "--format=csv,noheader",
        ]
    )

    try:
        report["server"]["models"] = get_json(args.url.rstrip("/") + "/v1/models", args.timeout)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        report["server"]["models_error"] = repr(exc)

    pids = list(args.pid)
    if not pids:
        pids = find_pids(args.match)
    report["pids"] = pids
    report["processes"] = [collect_process(pid) for pid in pids]
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0 if report["processes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
