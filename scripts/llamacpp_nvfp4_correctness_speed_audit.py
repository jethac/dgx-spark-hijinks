#!/usr/bin/env python3
"""Audit llama.cpp native-FP4 correctness/speed packet artifacts.

This script intentionally audits compact text artifacts only. Large GGUF files,
saved logits, and profiler traces can stay outside the repository.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
BAD_PATTERNS = (
    re.compile(r"\bnan\b", re.IGNORECASE),
    re.compile(r"\binf(?:inity)?\b", re.IGNORECASE),
    re.compile(r"\bfailed\b", re.IGNORECASE),
    re.compile(r"error loading model", re.IGNORECASE),
    re.compile(r"cuda error", re.IGNORECASE),
    re.compile(r"segmentation fault", re.IGNORECASE),
    re.compile(r"\btraceback\b", re.IGNORECASE),
)


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_text(path: Path) -> tuple[str, str | None]:
    try:
        return path.read_text(encoding="utf-8", errors="replace"), None
    except FileNotFoundError:
        return "", f"missing artifact: {path}"
    except OSError as exc:
        return "", f"could not read {path}: {exc}"


def finite_numbers(line: str) -> list[float]:
    values: list[float] = []
    for match in NUMBER_RE.findall(line):
        try:
            value = float(match)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def extract_line_number(text: str, *needles: str, pick: str = "last") -> float | None:
    lowered_needles = [needle.lower() for needle in needles]
    for line in text.splitlines():
        lower = line.lower()
        if all(needle in lower for needle in lowered_needles):
            values = finite_numbers(line)
            if values:
                return values[0] if pick == "first" else values[-1]
    return None


def extract_kld(stdout: str) -> dict[str, Any]:
    mean_kld = extract_line_number(stdout, "mean", "kld", pick="first")
    same_top_p = extract_line_number(stdout, "same", "top", "p", pick="last")
    if same_top_p is not None and same_top_p <= 1.0:
        same_top_p *= 100.0
    return {
        "mean_kld": mean_kld,
        "same_top_p_percent": same_top_p,
        "mean_kld_finite": isinstance(mean_kld, float) and math.isfinite(mean_kld),
        "same_top_p_finite": isinstance(same_top_p, float) and math.isfinite(same_top_p),
    }


def scan_bad_patterns(path: Path, text: str) -> list[str]:
    findings: list[str] = []
    for pattern in BAD_PATTERNS:
        match = pattern.search(text)
        if match:
            findings.append(f"{path.name}: contains failure marker {match.group(0)!r}")
    return findings


def audit_text_artifact(path: Path, *, required: bool = True) -> dict[str, Any]:
    text, error = read_text(path)
    findings: list[str] = []
    if error:
        if required:
            findings.append(error)
        return {
            "path": str(path),
            "exists": False,
            "ok": not required,
            "findings": findings,
        }
    findings.extend(scan_bad_patterns(path, text))
    return {
        "path": str(path),
        "exists": True,
        "ok": not findings,
        "bytes": len(text.encode("utf-8", errors="replace")),
        "findings": findings,
    }


def text_contains(path: Path, needle: str) -> dict[str, Any]:
    text, error = read_text(path)
    findings: list[str] = []
    if error:
        findings.append(error)
    elif needle.lower() not in text.lower():
        findings.append(f"{path.name}: expected text {needle!r} not found")
    findings.extend(scan_bad_patterns(path, text))
    return {
        "path": str(path),
        "exists": error is None,
        "contains": error is None and needle.lower() in text.lower(),
        "ok": not findings,
        "findings": findings,
        "text_excerpt": text.strip()[:200] if text else "",
    }


def parse_bench(path: Path) -> dict[str, Any]:
    text, error = read_text(path)
    findings: list[str] = []
    if error:
        findings.append(error)
        return {"path": str(path), "exists": False, "ok": False, "findings": findings}

    findings.extend(scan_bad_patterns(path, text))
    lower = text.lower()
    has_cuda = "cuda" in lower
    has_gb10 = "nvidia gb10" in lower or "gb10" in lower
    if not has_cuda:
        findings.append(f"{path.name}: CUDA backend/device evidence not found")
    if not has_gb10:
        findings.append(f"{path.name}: NVIDIA GB10 device evidence not found")

    tok_values: list[float] = []
    for line in text.splitlines():
        if "tok/s" not in line.lower():
            continue
        tok_values.extend(finite_numbers(line))
    if not tok_values:
        findings.append(f"{path.name}: no tok/s bench values found")

    return {
        "path": str(path),
        "exists": True,
        "ok": not findings,
        "has_cuda": has_cuda,
        "has_gb10": has_gb10,
        "tok_s_numbers": tok_values,
        "findings": findings,
    }


def parse_run_info(path: Path) -> dict[str, Any]:
    text, error = read_text(path)
    findings: list[str] = []
    if error:
        findings.append(error)
        return {"path": str(path), "exists": False, "ok": False, "findings": findings}
    findings.extend(scan_bad_patterns(path, text))
    if "run_id=" not in text:
        findings.append("run_info.txt: missing run_id=")
    if "NVIDIA GB10" not in text and "GB10" not in text:
        findings.append("run_info.txt: missing NVIDIA GB10 evidence")
    return {
        "path": str(path),
        "exists": True,
        "ok": not findings,
        "has_gb10": "GB10" in text,
        "findings": findings,
    }


def audit_run_dir(
    run_dir: Path,
    *,
    min_same_top_p: float,
    require_speed: bool,
) -> dict[str, Any]:
    findings: list[str] = []
    artifacts: dict[str, Any] = {
        "run_info": parse_run_info(run_dir / "run_info.txt"),
        "ref_perplexity_stdout": audit_text_artifact(run_dir / "ref_perplexity.stdout"),
        "ref_perplexity_stderr": audit_text_artifact(run_dir / "ref_perplexity.stderr"),
        "nvfp4_kld_stderr": audit_text_artifact(run_dir / "nvfp4_kld.stderr"),
        "ref_tokyo": text_contains(run_dir / "ref_tokyo.txt", "Tokyo"),
        "nvfp4_tokyo": text_contains(run_dir / "nvfp4_tokyo.txt", "Tokyo"),
        "ref_2plus2": text_contains(run_dir / "ref_2plus2.txt", "4"),
        "nvfp4_2plus2": text_contains(run_dir / "nvfp4_2plus2.txt", "4"),
    }

    kld_stdout, kld_error = read_text(run_dir / "nvfp4_kld.stdout")
    if kld_error:
        artifacts["nvfp4_kld_stdout"] = {
            "path": str(run_dir / "nvfp4_kld.stdout"),
            "exists": False,
            "ok": False,
            "findings": [kld_error],
        }
    else:
        kld_findings = scan_bad_patterns(run_dir / "nvfp4_kld.stdout", kld_stdout)
        kld = extract_kld(kld_stdout)
        if not kld["mean_kld_finite"]:
            kld_findings.append("nvfp4_kld.stdout: finite Mean KLD not found")
        if not kld["same_top_p_finite"]:
            kld_findings.append("nvfp4_kld.stdout: finite Same top p not found")
        elif float(kld["same_top_p_percent"]) < min_same_top_p:
            kld_findings.append(
                "nvfp4_kld.stdout: Same top p "
                f"{float(kld['same_top_p_percent']):.6g}% is below {min_same_top_p:.6g}%"
            )
        artifacts["nvfp4_kld_stdout"] = {
            "path": str(run_dir / "nvfp4_kld.stdout"),
            "exists": True,
            "ok": not kld_findings,
            "findings": kld_findings,
            **kld,
        }

    speed_files = {
        "ref_bench_pp512_tg128": run_dir / "ref_bench_pp512_tg128.txt",
        "nvfp4_bench_pp512_tg128": run_dir / "nvfp4_bench_pp512_tg128.txt",
        "ref_bench_pp2048_tg128": run_dir / "ref_bench_pp2048_tg128.txt",
        "nvfp4_bench_pp2048_tg128": run_dir / "nvfp4_bench_pp2048_tg128.txt",
    }
    speed = {name: parse_bench(path) for name, path in speed_files.items()}
    speed_present = all(item.get("exists") for item in speed.values())
    speed_ok = all(item.get("ok") for item in speed.values())
    if require_speed:
        artifacts.update(speed)
    else:
        artifacts.update({name: {**item, "required": False} for name, item in speed.items()})

    correctness_keys = [
        "run_info",
        "ref_perplexity_stdout",
        "ref_perplexity_stderr",
        "nvfp4_kld_stdout",
        "nvfp4_kld_stderr",
        "ref_tokyo",
        "nvfp4_tokyo",
        "ref_2plus2",
        "nvfp4_2plus2",
    ]
    correctness_ok = all(artifacts[key].get("ok") for key in correctness_keys)
    if not correctness_ok:
        findings.append("correctness gate failed")
    if require_speed and not speed_ok:
        findings.append("speed gate failed")

    for name, artifact in artifacts.items():
        for finding in artifact.get("findings") or []:
            findings.append(f"{name}: {finding}")

    return {
        "schema": "llamacpp-nvfp4-correctness-speed-audit/v1",
        "run_dir": str(run_dir),
        "thresholds": {
            "min_same_top_p_percent": min_same_top_p,
            "require_speed": require_speed,
        },
        "correctness_ok": correctness_ok,
        "speed_present": speed_present,
        "speed_ok": speed_ok,
        "ok": correctness_ok and (speed_ok if require_speed else True),
        "artifacts": artifacts,
        "findings": findings,
        "notes": [
            "A green audit proves the small packet gate only, not a full lm-eval row.",
            "Native FP4 dispatch still requires separate profiler or kernel-name evidence.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Directory containing compact packet artifacts")
    parser.add_argument("--output")
    parser.add_argument("--min-same-top-p-percent", type=float, default=80.0)
    parser.add_argument(
        "--require-speed",
        action="store_true",
        help="Require all four llama-bench artifacts to pass CUDA/GB10/tok-s checks.",
    )
    args = parser.parse_args()

    result = audit_run_dir(
        Path(args.run_dir),
        min_same_top_p=args.min_same_top_p_percent,
        require_speed=args.require_speed,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
