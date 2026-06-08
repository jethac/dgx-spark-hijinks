#!/usr/bin/env python3
"""Audit CUDA artifacts for embedded architecture target strings.

Unlike cuda_so_audit.py, this also checks standalone .cubin and .ptx files.
Use it after a server has handled at least one request when JIT caches may have
materialized under package or user cache directories.
"""

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
ARTIFACT_SUFFIXES = {".so", ".cubin", ".ptx"}


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
    return Path(file_name).resolve().parent


def iter_artifacts(root: Path, max_files: int) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root] if root.suffix in ARTIFACT_SUFFIXES else []
    artifacts = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in ARTIFACT_SUFFIXES
    ]
    return sorted(artifacts)[:max_files]


def read_arch_strings(path: Path) -> list[str]:
    data = path.read_bytes()
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        text = ""
    return sorted(set(ARCH_RE.findall(text)))


def cuobjdump_arches(cuobjdump: str | None, path: Path, timeout: int) -> dict[str, Any]:
    if not cuobjdump:
        return {"returncode": None, "architectures": [], "error": "cuobjdump not found"}
    proc = subprocess.run(
        [cuobjdump, "--list-elf", str(path)],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    text = "\n".join([proc.stdout, proc.stderr])
    return {
        "returncode": proc.returncode,
        "architectures": sorted(set(ARCH_RE.findall(text))),
        "stdout_head": proc.stdout[:1000],
        "stderr_head": proc.stderr[:1000],
    }


def audit_artifact(cuobjdump: str | None, path: Path, timeout: int) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": str(path),
        "suffix": path.suffix,
        "text_architectures": read_arch_strings(path),
    }
    if path.suffix in {".so", ".cubin"}:
        try:
            item["cuobjdump"] = cuobjdump_arches(cuobjdump, path, timeout)
        except Exception as exc:
            item["cuobjdump"] = {"error": repr(exc), "architectures": []}
    architectures = set(item["text_architectures"])
    architectures.update(item.get("cuobjdump", {}).get("architectures", []))
    item["architectures"] = sorted(architectures)
    return item


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="append", default=[])
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--max-files", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output")
    args = parser.parse_args()

    cuobjdump = find_executable("cuobjdump")
    report: dict[str, Any] = {
        "schema": "cuda-artifact-arch-audit/v1",
        "cuobjdump": cuobjdump,
        "inputs": {"packages": args.package, "paths": args.path},
        "roots": [],
        "artifacts": [],
        "summary": {},
    }

    roots: list[Path] = []
    for package in args.package:
        try:
            root = module_root(package)
            roots.append(root)
            report["roots"].append({"package": package, "path": str(root), "exists": root.exists()})
        except Exception as exc:
            report["roots"].append({"package": package, "error": repr(exc), "exists": False})
    for raw in args.path:
        root = Path(raw).resolve()
        roots.append(root)
        report["roots"].append({"path": str(root), "exists": root.exists()})

    seen: set[Path] = set()
    per_root_limit = max(1, args.max_files)
    for root in roots:
        for artifact in iter_artifacts(root, per_root_limit):
            if artifact in seen:
                continue
            seen.add(artifact)
            try:
                report["artifacts"].append(audit_artifact(cuobjdump, artifact, args.timeout))
            except Exception as exc:
                report["artifacts"].append({"path": str(artifact), "error": repr(exc), "architectures": []})

    arch_counts: dict[str, int] = {}
    suffix_counts: dict[str, int] = {}
    for item in report["artifacts"]:
        suffix = item.get("suffix", "")
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        for arch in item.get("architectures", []):
            arch_counts[arch] = arch_counts.get(arch, 0) + 1

    report["summary"] = {
        "artifact_count": len(report["artifacts"]),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "architecture_counts": dict(sorted(arch_counts.items())),
        "artifacts_with_sm_121": sum(
            1 for item in report["artifacts"] if "sm_121" in item.get("architectures", [])
        ),
        "artifacts_with_sm_121a": sum(
            1 for item in report["artifacts"] if "sm_121a" in item.get("architectures", [])
        ),
        "artifacts_with_compute_121": sum(
            1 for item in report["artifacts"] if "compute_121" in item.get("architectures", [])
        ),
        "artifacts_with_compute_121a": sum(
            1 for item in report["artifacts"] if "compute_121a" in item.get("architectures", [])
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
