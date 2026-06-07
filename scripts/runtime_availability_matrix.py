#!/usr/bin/env python3
"""Collect availability evidence for candidate DGX Spark runtimes."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import time
from typing import Any


MODULES = [
    "vllm",
    "sglang",
    "litert_lm",
    "litert",
    "tensorflow",
    "torch",
    "flashinfer",
    "triton",
    "transformers",
    "llama_cpp",
]

COMMANDS = [
    "docker",
    "podman",
    "uv",
    "pip",
    "python3",
    "vllm",
    "sglang",
    "litert_lm",
    "litertlm",
    "ollama",
    "llama-server",
    "llama-bench",
]

PROCESS_PATTERNS = [
    "vllm",
    "VLLM::EngineCore",
    "sglang",
    "litert",
    "ollama",
    "llama-server",
]


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


def module_info(name: str) -> dict[str, Any]:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    return {
        "available": True,
        "version": getattr(mod, "__version__", "unknown"),
        "file": getattr(mod, "__file__", None),
    }


def command_info(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    if not path:
        return {"available": False}
    info: dict[str, Any] = {"available": True, "path": path}
    if name in {"docker", "podman"}:
        info["version"] = run([path, "--version"], timeout=10)
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--include-docker-images", action="store_true")
    parser.add_argument("--include-raw-ps", action="store_true")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema": "runtime-availability-matrix/v1",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "modules": {name: module_info(name) for name in MODULES},
        "commands": {name: command_info(name) for name in COMMANDS},
        "processes": {},
        "ports": run(["ss", "-ltnp"], timeout=10),
    }

    ps = run(["ps", "-eo", "pid,ppid,user,etime,cmd"], timeout=10)
    if args.include_raw_ps:
        report["processes"]["raw"] = ps
    if ps.get("stdout"):
        matches = []
        for line in ps["stdout"].splitlines():
            lower = line.lower()
            if any(pattern.lower() in lower for pattern in PROCESS_PATTERNS):
                if "runtime_availability_matrix.py" not in line:
                    matches.append(line)
        report["processes"]["matches"] = matches

    if args.include_docker_images and report["commands"]["docker"].get("available"):
        report["docker_images"] = run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}} {{.ID}}"],
            timeout=30,
        )

    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
