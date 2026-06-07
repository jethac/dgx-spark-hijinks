#!/usr/bin/env python3
"""Collect DGX Spark / GB10 environment evidence.

The goal is to answer one question before benchmarks start:
are we actually on a stack that can target sm_121?
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from spark_hardware import collect_cuda_hardware


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
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"cmd": cmd, "error": repr(exc)}


def find_executable(name: str, extra_paths: list[str] | None = None) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for raw in extra_paths or []:
        path = Path(raw)
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return None


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


def torch_info() -> dict[str, Any]:
    info = module_version("torch")
    if not info.get("available"):
        return info

    import torch  # type: ignore

    cuda: dict[str, Any] = {
        "built_cuda": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        cuda["device_count"] = torch.cuda.device_count()
        devices = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "capability": list(torch.cuda.get_device_capability(idx)),
                    "total_memory": props.total_memory,
                    "multi_processor_count": props.multi_processor_count,
                }
            )
        cuda["devices"] = devices
        cuda["arch_list"] = (
            torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else None
        )
    info["cuda"] = cuda
    return info


def inspect_so(paths: list[str]) -> list[dict[str, Any]]:
    out = []
    cuobjdump = find_executable(
        "cuobjdump",
        ["/usr/local/cuda/bin/cuobjdump", "/usr/local/cuda-13.0/bin/cuobjdump"],
    )
    for raw in paths:
        path = Path(raw)
        item: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            out.append(item)
            continue
        if cuobjdump:
            result = run([cuobjdump, "--list-elf", str(path)], timeout=60)
            text = "\n".join([result.get("stdout", ""), result.get("stderr", "")])
            item["cuobjdump"] = result
            item["mentions_sm_121"] = "sm_121" in text or "compute_121" in text
            item["mentions_sm_120"] = "sm_120" in text or "compute_120" in text
            item["mentions_sm_80"] = "sm_80" in text or "compute_80" in text
        else:
            item["cuobjdump"] = {"available": False}
        out.append(item)
    return out


def collect(args: argparse.Namespace) -> dict[str, Any]:
    commands: dict[str, Any] = {}
    command_specs = {
        "nvidia_smi": ("nvidia-smi", ["nvidia-smi"], []),
        "nvidia_smi_query": (
            "nvidia-smi",
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            [],
        ),
        "nvcc": (
            "nvcc",
            ["nvcc", "--version"],
            ["/usr/local/cuda/bin/nvcc", "/usr/local/cuda-13.0/bin/nvcc"],
        ),
        "cuobjdump": (
            "cuobjdump",
            ["cuobjdump", "--version"],
            ["/usr/local/cuda/bin/cuobjdump", "/usr/local/cuda-13.0/bin/cuobjdump"],
        ),
        "uname": ("uname", ["uname", "-a"], []),
    }
    for name, (exe_name, cmd, extra_paths) in command_specs.items():
        exe = find_executable(exe_name, extra_paths)
        if exe:
            resolved_cmd = [exe] + cmd[1:]
            commands[name] = run(resolved_cmd)
        else:
            commands[name] = {"available": False, "cmd": cmd}

    modules = {
        name: module_version(name)
        for name in [
            "vllm",
            "flashinfer",
            "triton",
            "transformers",
            "lm_eval",
            "llama_cpp",
        ]
    }
    modules["torch"] = torch_info()
    hardware = collect_cuda_hardware()

    env_keys = [
        "CUDA_VISIBLE_DEVICES",
        "TORCH_CUDA_ARCH_LIST",
        "VLLM_ATTENTION_BACKEND",
        "NCCL_P2P_DISABLE",
        "VLLM_USE_V1",
    ]

    report = {
        "schema": "spark-doctor/v1",
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
            "executable": sys.executable,
        },
        "environment": {key: os.environ.get(key) for key in env_keys},
        "commands": commands,
        "hardware": hardware,
        "modules": modules,
        "shared_objects": inspect_so(args.inspect_so),
        "findings": [],
    }

    findings = report["findings"]
    torch_cuda = modules["torch"].get("cuda", {})
    devices = torch_cuda.get("devices") or []
    if devices:
        first = devices[0]
        cap = first.get("capability")
        name = first.get("name", "")
        sm_count = first.get("multi_processor_count")
        if cap == [12, 1]:
            findings.append("OK: first CUDA device reports compute capability 12.1 / sm_121.")
        else:
            findings.append(f"WARN: first CUDA device is {name!r} with capability {cap}, not sm_121.")
        if sm_count is not None:
            findings.append(f"INFO: first CUDA device reports {sm_count} SMs.")
    else:
        findings.append("WARN: PyTorch did not report an available CUDA device.")

    arch_list = torch_cuda.get("arch_list") or []
    if devices and devices[0].get("capability") == [12, 1]:
        if "sm_121" in arch_list:
            findings.append("OK: PyTorch arch list explicitly includes sm_121.")
        elif "sm_120" in arch_list:
            findings.append(
                "NOTE: PyTorch arch list has sm_120 but not sm_121; verify this is an intentional compatible path for installed kernels."
            )
        elif arch_list:
            findings.append(
                f"WARN: PyTorch arch list does not include sm_121 or sm_120: {arch_list}"
            )

    machine = platform.machine().lower()
    if machine not in {"aarch64", "arm64"}:
        findings.append(f"WARN: host architecture is {machine}; DGX Spark target is ARM64/aarch64.")
    else:
        findings.append("OK: host architecture is ARM64/aarch64.")

    query = commands.get("nvidia_smi_query", {}).get("stdout", "")
    if "12.1" in query or "GB10" in query:
        findings.append("OK: nvidia-smi query is consistent with GB10 / compute capability 12.1.")
    elif query:
        findings.append("WARN: nvidia-smi query did not clearly identify GB10 / 12.1.")

    findings.extend(hardware.get("warnings", []))

    return report


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Spark Doctor", ""]
    lines.append("## Findings")
    for finding in report["findings"]:
        lines.append(f"- {finding}")
    lines.append("")
    lines.append("## Platform")
    for key, value in report["platform"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## CUDA Hardware")
    for device in report.get("hardware", {}).get("devices", []):
        lines.append(
            "- `{index}`: `{name}`, capability `{capability}`, SMs `{sms}`, key `{key}`".format(
                index=device.get("index"),
                name=device.get("name"),
                capability=device.get("capability"),
                sms=device.get("multi_processor_count"),
                key=device.get("comparison_key"),
            )
        )
    for warning in report.get("hardware", {}).get("warnings", []):
        lines.append(f"- warning: {warning}")
    lines.append("")
    lines.append("## Commands")
    for key, value in report["commands"].items():
        lines.append(f"### {key}")
        if value.get("available") is False:
            lines.append("- not available")
        else:
            lines.append(f"- returncode: `{value.get('returncode')}`")
            stdout = value.get("stdout") or ""
            stderr = value.get("stderr") or ""
            if stdout:
                lines.append("")
                lines.append("```text")
                lines.append(stdout)
                lines.append("```")
            if stderr:
                lines.append("")
                lines.append("stderr:")
                lines.append("```text")
                lines.append(stderr)
                lines.append("```")
    lines.append("")
    lines.append("## Python Modules")
    for key, value in report["modules"].items():
        if value.get("available"):
            lines.append(f"- `{key}`: `{value.get('version')}`")
        else:
            lines.append(f"- `{key}`: not available")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    parser.add_argument(
        "--inspect-so",
        action="append",
        default=[],
        help="CUDA extension .so to inspect with cuobjdump; may be repeated",
    )
    args = parser.parse_args()

    report = collect(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
