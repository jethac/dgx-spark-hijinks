#!/usr/bin/env python3
"""Annotate benchmark and server failures from captured Spark artifacts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Annotation:
    source: str
    source_type: str
    failure_class: str
    confidence: str
    evidence: str
    backend: str | None = None
    row: str | None = None
    model: str | None = None
    task: str | None = None
    status: str | None = None
    returncode: int | None = None
    suspected_causes: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def classify_text(text: str, backend: str | None = None) -> tuple[str, str, list[str]]:
    lower = text.lower()
    causes: list[str] = []

    if "returncode=-9" in lower or "sigkill" in lower or re.search(r"\bkilled\b", lower):
        if backend == "hf":
            causes.append("oom_or_resource_pressure")
        return "process_killed", "high", causes
    if "cuda out of memory" in lower or "outofmemoryerror" in lower or "oom" in lower:
        return "oom", "high", ["memory_pressure"]
    if "token_logprobs" in lower or "logprobs schema" in lower or "echoed prompt/continuation" in lower:
        return "api_schema_mismatch", "high", ["gguf_loglikelihood_adapter"]
    if "max_tokens_per_mm_item" in lower and "max_num_batched_tokens" in lower:
        return "configuration_error", "high", ["batch_token_budget_too_low"]
    if "requires at least one encoder urls" in lower:
        return "configuration_error", "high", ["language_only_encoder_configuration"]
    if "input length" in lower and "exceeds the maximum allowed length" in lower:
        return "configuration_error", "high", ["token_budget_too_low"]
    if "mergedcolumnparallellinear" in lower and "has no attribute 'weight'" in lower:
        return "runtime_exception", "high", ["sglang_gemma4_audio_path"]
    if "attributeerror" in lower or "valueerror" in lower or "traceback" in lower:
        return "runtime_exception", "medium", []
    if "returncode=1" in lower:
        if backend == "vllm":
            causes.append("model_load_probe_failed")
        return "process_error", "medium", causes
    if "timeout" in lower or "timed out" in lower:
        return "timeout", "medium", []
    if "connection reset" in lower or "failed to connect" in lower:
        return "server_unavailable", "medium", []
    if "error downloading from huggingface" in lower:
        causes.append("huggingface_download_failed")
        if "permission denied" in lower:
            causes.append("cache_permission_denied")
        return "artifact_download_failed", "high", causes
    if "permission denied" in lower:
        return "permission_error", "medium", []
    return "unknown_failure", "low", []


def parse_returncode(text: str) -> int | None:
    match = re.search(r"returncode=(-?\d+)", text)
    if not match:
        return None
    return int(match.group(1))


def markdown_annotations(path: Path) -> list[Annotation]:
    annotations: list[Annotation] = []
    headers: list[str] | None = None
    in_table = False

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            in_table = False
            headers = None
            continue
        cells = split_markdown_row(line)
        if set("".join(cells)) <= {"-", ":"}:
            in_table = True
            continue
        if headers is None:
            headers = [cell.lower().replace(" ", "_") for cell in cells]
            continue
        if not in_table or len(cells) != len(headers):
            continue

        row = dict(zip(headers, cells))
        status = (row.get("status") or "").lower()
        reason = row.get("reason") or ""
        backend = (row.get("backend") or "").lower() or None
        if not any(row.get(key) for key in ("row", "task", "backend")):
            continue
        if status not in {"eval_failed", "loader_failed"} and not reason:
            continue
        evidence = reason or f"status={status}"
        failure_class, confidence, causes = classify_text(evidence, backend)
        annotations.append(
            Annotation(
                source=str(path),
                source_type="markdown_table",
                row=row.get("row"),
                task=row.get("task"),
                backend=backend,
                status=status or None,
                returncode=parse_returncode(evidence),
                failure_class=failure_class,
                confidence=confidence,
                suspected_causes=causes,
                evidence=evidence,
                details={k: v for k, v in row.items() if k not in {"row", "task", "backend", "status", "reason"}},
            )
        )

    return annotations


def json_annotations(path: Path) -> list[Annotation]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    if not isinstance(data, dict):
        return []

    annotations: list[Annotation] = []
    if data.get("schema") == "openai-serving-benchmark/v1":
        backend = data.get("backend")
        model = data.get("model")
        for case in data.get("cases") or []:
            if case.get("ok"):
                continue
            evidence = case.get("error") or case.get("content_preview") or "OpenAI benchmark case returned ok=false"
            if not case.get("error") and case.get("usage", {}).get("prompt_tokens") == 1:
                evidence = "case returned ok=false with prompt_tokens=1; likely request/token-budget handling failure"
            failure_class, confidence, causes = classify_text(evidence, backend)
            if failure_class == "unknown_failure" and case.get("decode_tok_s") is None:
                failure_class = "invalid_or_empty_response"
                confidence = "medium"
            annotations.append(
                Annotation(
                    source=str(path),
                    source_type="openai_serving_case",
                    row=data.get("run_id"),
                    task=case.get("case"),
                    backend=backend,
                    model=model,
                    status="case_failed",
                    failure_class=failure_class,
                    confidence=confidence,
                    suspected_causes=causes,
                    evidence=evidence,
                    details={
                        "usage": case.get("usage"),
                        "total_s": case.get("total_s"),
                        "ttft_s": case.get("ttft_s"),
                    },
                )
            )
    elif data.get("schema") == "openai-chat-smoke/v1" and not data.get("ok"):
        evidence = data.get("error") or data.get("content") or "chat smoke returned ok=false"
        failure_class, confidence, causes = classify_text(evidence)
        annotations.append(
            Annotation(
                source=str(path),
                source_type="openai_chat_smoke",
                model=data.get("model"),
                status="smoke_failed",
                failure_class=failure_class,
                confidence=confidence,
                suspected_causes=causes,
                evidence=evidence,
                details={"elapsed_s": data.get("elapsed_s")},
            )
        )
    elif data.get("schema") == "spark-run-with-telemetry/v1":
        stderr_tail = data.get("stderr_tail") or ""
        stdout_tail = data.get("stdout_tail") or ""
        evidence = stderr_tail.strip() or stdout_tail.strip() or f"returncode={data.get('returncode')}"
        output_class, output_confidence, output_causes = classify_text(evidence, data.get("backend"))
        failure_class = data.get("failure_class") or "unknown_failure"
        if failure_class == "ok" and output_class == "unknown_failure":
            return annotations
        causes = list(output_causes)
        if output_class != "unknown_failure":
            failure_class = output_class
            confidence = output_confidence
        else:
            if data.get("returncode") in {-11, 139}:
                failure_class = "process_crash"
                causes.append("segmentation_fault")
                stdout_lines = [line.strip() for line in stdout_tail.splitlines() if line.strip()]
                last_stdout = stdout_lines[-1] if stdout_lines else ""
                evidence = f"returncode={data.get('returncode')}; last stdout={last_stdout}"
            confidence = "high" if failure_class in {"process_crash", "process_killed", "timeout"} else "medium"
        oom_text = json.dumps(data.get("oom_evidence") or {}, sort_keys=True).lower()
        if "out of memory" in oom_text or "oom-kill" in oom_text or "killed process" in oom_text:
            causes.append("kernel_oom_evidence")
            confidence = "high"
        elif failure_class == "process_killed":
            causes.append("signal_without_kernel_oom_evidence")
        annotations.append(
            Annotation(
                source=str(path),
                source_type="telemetry_run",
                row=data.get("run_id"),
                backend=data.get("backend"),
                model=data.get("model"),
                status="run_failed",
                returncode=data.get("returncode"),
                failure_class=failure_class,
                confidence=confidence,
                suspected_causes=causes,
                evidence=first_matching_line(evidence) if output_class != "unknown_failure" else evidence[-500:],
                details={
                    "elapsed_s": data.get("elapsed_s"),
                    "timed_out": data.get("timed_out"),
                    "peak_process_tree_rss_kib": data.get("peak_process_tree_rss_kib"),
                    "peak_process_tree_swap_kib": data.get("peak_process_tree_swap_kib"),
                },
            )
        )
    elif data.get("schema") == "gguf-logprobs-probe/v1" and not data.get("ok"):
        classification = data.get("classification") or {}
        notes = classification.get("notes") or []
        evidence = "; ".join(notes) or "GGUF logprobs probe returned ok=false"
        failure_class, confidence, causes = classify_text(evidence)
        if failure_class == "unknown_failure":
            failure_class = "api_schema_mismatch"
            confidence = "high"
            causes = ["gguf_loglikelihood_adapter"]
        annotations.append(
            Annotation(
                source=str(path),
                source_type="gguf_logprobs_probe",
                backend="llama.cpp",
                model=(data.get("payload") or {}).get("model"),
                status="probe_failed",
                failure_class=failure_class,
                confidence=confidence,
                suspected_causes=causes,
                evidence=evidence,
                details=classification,
            )
        )
    elif data.get("schema") == "spark-smoke-suite/v1" and not data.get("ok"):
        for step in data.get("steps") or []:
            if step.get("ok") is True:
                continue
            if step.get("ok") is None and not step.get("required"):
                continue
            evidence = (
                step.get("skip_reason")
                or step.get("error")
                or step.get("stderr_tail")
                or step.get("stdout_tail")
                or f"returncode={step.get('returncode')}"
            )
            failure_class, confidence, causes = classify_text(evidence, step.get("name"))
            if not step.get("configured") and step.get("required"):
                failure_class = "missing_required_smoke"
                confidence = "high"
                causes = ["required_smoke_not_configured"]
            elif step.get("timed_out"):
                failure_class = "timeout"
                confidence = "high"
            elif failure_class == "unknown_failure":
                failure_class = "smoke_step_failed"
                confidence = "medium"
            annotations.append(
                Annotation(
                    source=str(path),
                    source_type="spark_smoke_suite",
                    row=data.get("run_id"),
                    task=step.get("name"),
                    backend=step.get("name"),
                    status="smoke_failed",
                    returncode=step.get("returncode"),
                    failure_class=failure_class,
                    confidence=confidence,
                    suspected_causes=causes,
                    evidence=str(evidence)[-500:],
                    details={
                        "required": step.get("required"),
                        "configured": step.get("configured"),
                        "timed_out": step.get("timed_out"),
                        "artifact": step.get("artifact"),
                    },
                )
            )
    return annotations


def first_matching_line(text: str) -> str:
    patterns = [
        "ValueError:",
        "AttributeError:",
        "RuntimeError:",
        "CUDA out of memory",
        "OutOfMemoryError",
        "Traceback",
        "Received sigquit",
        "returncode=",
        "Connection reset",
        "Failed to connect",
    ]
    lines = text.splitlines()
    for pattern in patterns:
        for line in lines:
            if pattern.lower() in line.lower():
                return line.strip()
    return "\n".join(lines[-3:]).strip()


def log_annotations(path: Path) -> list[Annotation]:
    text = path.read_text(encoding="utf-8", errors="replace")
    failure_lines = [
        line.strip()
        for line in text.splitlines()
        if any(
            pattern in line.lower()
            for pattern in (
                "traceback",
                "valueerror:",
                "attributeerror:",
                "runtimeerror:",
                "cuda out of memory",
                "outofmemoryerror",
                "received sigquit",
                "returncode=",
                "connection reset",
                "failed to connect",
                "timed out",
                "timeouterror",
                "exceeds the maximum allowed length",
            )
        )
    ]
    if not failure_lines:
        return []
    failure_text = "\n".join(failure_lines)
    failure_class, confidence, causes = classify_text(failure_text)
    if failure_class == "unknown_failure":
        return []
    return [
        Annotation(
            source=str(path),
            source_type="server_log",
            failure_class=failure_class,
            confidence=confidence,
            suspected_causes=causes,
            evidence=first_matching_line(failure_text),
        )
    ]


def collect_annotations(results_dir: Path) -> list[Annotation]:
    annotations: list[Annotation] = []
    for path in sorted(results_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".md":
            annotations.extend(markdown_annotations(path))
        elif path.suffix.lower() == ".json":
            annotations.extend(json_annotations(path))
        elif path.suffix.lower() == ".log":
            annotations.extend(log_annotations(path))
    return annotations


def write_markdown(path: Path, annotations: list[Annotation]) -> None:
    by_class = Counter(item.failure_class for item in annotations)
    by_backend = Counter(item.backend or "unknown" for item in annotations)
    lines = [
        "# Benchmark Failure Annotations",
        "",
        "Generated by `scripts/failure_annotator.py` from captured artifacts.",
        "",
        "## Summary",
        "",
        "| failure class | count |",
        "|---|---:|",
    ]
    for key, value in sorted(by_class.items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "| backend | count |", "|---|---:|"])
    for key, value in sorted(by_backend.items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Annotations",
            "",
            "| class | confidence | backend | row/task | evidence | source |",
            "|---|---|---|---|---|---|",
        ]
    )
    for item in annotations:
        row_task = " / ".join(part for part in [item.row, item.task] if part) or ""
        evidence = item.evidence.replace("|", "\\|").replace("\n", " ")[:240]
        lines.append(
            f"| `{item.failure_class}` | {item.confidence} | {item.backend or ''} | "
            f"{row_task} | {evidence} | `{item.source}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    annotations = collect_annotations(args.results_dir)
    report = {
        "schema": "spark-failure-annotations/v1",
        "results_dir": str(args.results_dir),
        "count": len(annotations),
        "summary": {
            "by_failure_class": dict(sorted(Counter(item.failure_class for item in annotations).items())),
            "by_backend": dict(sorted(Counter(item.backend or "unknown" for item in annotations).items())),
        },
        "annotations": [asdict(item) for item in annotations],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.output_md:
        write_markdown(args.output_md, annotations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
