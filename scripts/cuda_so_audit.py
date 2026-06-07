#!/usr/bin/env python3
"""Audit CUDA extension shared objects for embedded architecture targets."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


ARCH_RE = re.compile(r"\b(?:sm|compute)_[0-9]+[a-z]?\b")


def find_executable(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for raw in ["/usr/local/cuda/bin/" + name, "/usr/local/cuda-13.0/bin/" + name]:
        path = Path(raw)
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return None


def module_root(name: str) -> Path:
    mod = importlib.import_module(name)
    file_name = getattr(mod, "__file__", None)
    if not file_name:
        raise RuntimeError(f"module {name!r} has no __file__")
    path = Path(file_name).resolve()
    return path.parent if path.name != "__init__.py" else path.parent


def run_cuobjdump(cuobjdump: str, so_path: Path, timeout: int) -> dict[str, Any]:
    proc = subprocess.run(
        [cuobjdump, "--list-elf", str(so_path)],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    text = "\n".join([proc.stdout, proc.stderr])
    return {
        "returncode": proc.returncode,
        "architectures": sorted(set(ARCH_RE.findall(text))),
        "stdout_head": proc.stdout[:2000],
        "stderr_head": proc.stderr[:2000],
    }


def iter_shared_objects(root: Path, max_files: int) -> list[Path]:
    files = sorted(root.rglob("*.so"))
    return files[:max_files]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", action="append", default=[])
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--max-files", type=int, default=80)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output")
    args = parser.parse_args()

    cuobjdump = find_executable("cuobjdump")
    report: dict[str, Any] = {
        "schema": "cuda-so-audit/v1",
        "cuobjdump": cuobjdump,
        "inputs": {"packages": args.package, "paths": args.path},
        "objects": [],
        "summary": {},
    }
    if not cuobjdump:
        report["error"] = "cuobjdump not found"
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    roots: list[Path] = []
    for package in args.package:
        try:
            roots.append(module_root(package))
        except Exception as exc:
            report["objects"].append({"package": package, "error": repr(exc)})
    for raw in args.path:
        roots.append(Path(raw).resolve())

    seen: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix == ".so":
            candidates = [root]
        else:
            candidates = iter_shared_objects(root, args.max_files)
        for so_path in candidates:
            if so_path in seen:
                continue
            seen.add(so_path)
            item: dict[str, Any] = {"path": str(so_path)}
            try:
                item.update(run_cuobjdump(cuobjdump, so_path, args.timeout))
            except Exception as exc:
                item["error"] = repr(exc)
            report["objects"].append(item)

    arch_counts: dict[str, int] = {}
    for item in report["objects"]:
        for arch in item.get("architectures", []):
            arch_counts[arch] = arch_counts.get(arch, 0) + 1
    report["summary"] = {
        "object_count": len(report["objects"]),
        "architecture_counts": dict(sorted(arch_counts.items())),
        "objects_with_sm_121": sum(
            1 for item in report["objects"] if "sm_121" in item.get("architectures", [])
        ),
        "objects_with_sm_120": sum(
            1 for item in report["objects"] if "sm_120" in item.get("architectures", [])
        ),
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

