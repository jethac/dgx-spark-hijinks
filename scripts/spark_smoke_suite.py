#!/usr/bin/env python3
"""Compact DGX Spark smoke-suite orchestrator.

This script intentionally does not launch heavyweight model servers. It records
what is currently available, tests configured endpoints/commands, and leaves a
single JSON artifact that can be used before and after stack changes.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from spark_hardware import collect_cuda_hardware


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_command(
    *,
    name: str,
    command: list[str],
    timeout_s: int,
    artifact: Path | None = None,
    stdout_artifact: bool = False,
    required: bool = False,
    cwd: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "name": name,
        "required": required,
        "configured": True,
        "command": command,
        "artifact": str(artifact) if artifact else None,
        "ok": False,
        "timed_out": False,
    }
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        result.update(
            {
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
                "ok": proc.returncode == 0,
            }
        )
        if artifact and stdout_artifact:
            artifact.write_text(proc.stdout, encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        result.update(
            {
                "timed_out": True,
                "returncode": None,
                "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            }
        )
    except OSError as exc:
        result["error"] = repr(exc)
    result["elapsed_s"] = round(time.perf_counter() - started, 3)
    return result


def skipped(name: str, reason: str, *, required: bool = False) -> dict[str, Any]:
    return {
        "name": name,
        "required": required,
        "configured": False,
        "ok": False if required else None,
        "skip_reason": reason,
    }


def add_openai_step(
    steps: list[dict[str, Any]],
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
    timeout_s: int,
    backend: str,
    url: str | None,
    model: str | None,
    required: bool,
    skip: bool,
) -> None:
    if skip:
        steps.append(skipped(f"{backend}-openai-smoke", f"--skip-{backend} provided", required=False))
        return
    if not url:
        steps.append(skipped(f"{backend}-openai-smoke", f"--{backend}-url not provided", required=required))
        return
    artifact = results_dir / f"{run_id}_{backend}_openai_chat_smoke.json"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "openai_chat_smoke.py"),
        "--url",
        url,
        "--timeout",
        str(timeout_s),
        "--output",
        str(artifact),
    ]
    if not model:
        steps.append(skipped(f"{backend}-openai-smoke", f"--{backend}-model not provided", required=required))
        return
    cmd.extend(["--model", model])
    steps.append(
        run_command(
            name=f"{backend}-openai-smoke",
            command=cmd,
            timeout_s=timeout_s,
            artifact=artifact,
            required=required,
            cwd=repo_root,
        )
    )


def add_litert_steps(
    steps: list[dict[str, Any]],
    *,
    results_dir: Path,
    run_id: str,
    timeout_s: int,
    skip_litert: bool,
    litert_repo: str,
    litert_model_file: str,
) -> None:
    if skip_litert:
        steps.append(skipped("litert-lm-cpu-chat", "--skip-litert-lm provided"))
        return
    artifact = results_dir / f"{run_id}_litert_lm_cpu_chat_telemetry.json"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_with_telemetry.py"),
        "--run-id",
        f"{run_id}-litert-lm-cpu-chat",
        "--backend",
        "litert-lm-cpu",
        "--model",
        f"{litert_repo}/{litert_model_file}",
        "--timeout-s",
        str(timeout_s),
        "--interval-s",
        "5",
        "--output",
        str(artifact),
        "--",
        "litert-lm",
        "run",
        "--from-huggingface-repo",
        litert_repo,
        "--backend",
        "cpu",
        "--temperature",
        "0",
        "--prompt",
        "Reply with exactly this text: spark-ok",
        litert_model_file,
    ]
    steps.append(
        run_command(
            name="litert-lm-cpu-chat",
            command=cmd,
            timeout_s=timeout_s + 30,
            artifact=artifact,
        )
    )


def add_optional_shell_step(
    steps: list[dict[str, Any]],
    *,
    name: str,
    raw_command: str | None,
    timeout_s: int,
    required: bool = False,
    skip: bool = False,
) -> None:
    if skip:
        steps.append(skipped(name, f"--skip-{name} provided", required=False))
        return
    if not raw_command:
        steps.append(skipped(name, f"--{name}-command not provided", required=required))
        return
    steps.append(
        run_command(
            name=name,
            command=shlex.split(raw_command),
            timeout_s=timeout_s,
            required=required,
        )
    )


def add_hf_step(
    steps: list[dict[str, Any]],
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
    raw_command: str | None,
    timeout_s: int,
    skip: bool,
) -> None:
    if skip:
        steps.append(skipped("hf", "--skip-hf provided", required=False))
        return
    if not raw_command:
        steps.append(skipped("hf", "--hf-command not provided", required=True))
        return
    artifact = results_dir / f"{run_id}_hf_telemetry.json"
    command = [
        sys.executable,
        str(repo_root / "scripts" / "run_with_telemetry.py"),
        "--run-id",
        f"{run_id}-hf",
        "--backend",
        "hf",
        "--timeout-s",
        str(timeout_s),
        "--interval-s",
        "5",
        "--output",
        str(artifact),
        "--",
        *shlex.split(raw_command),
    ]
    steps.append(
        run_command(
            name="hf",
            command=command,
            timeout_s=timeout_s + 30,
            artifact=artifact,
            required=True,
            cwd=repo_root,
        )
    )


def add_mtp_step(
    steps: list[dict[str, Any]],
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
    raw_command: str | None,
    timeout_s: int,
    skip: bool,
    model: str | None,
) -> None:
    if skip:
        steps.append(skipped("mtp", "--skip-mtp provided", required=False))
        return
    if not raw_command:
        steps.append(skipped("mtp", "--mtp-command not provided", required=True))
        return
    artifact = results_dir / f"{run_id}_mtp_telemetry.json"
    command = [
        sys.executable,
        str(repo_root / "scripts" / "run_with_telemetry.py"),
        "--run-id",
        f"{run_id}-mtp",
        "--backend",
        "mtp",
        "--timeout-s",
        str(timeout_s),
        "--interval-s",
        "5",
        "--output",
        str(artifact),
    ]
    if model:
        command.extend(["--model", model])
    command.extend(["--", *shlex.split(raw_command)])
    steps.append(
        run_command(
            name="mtp",
            command=command,
            timeout_s=timeout_s + 30,
            artifact=artifact,
            required=True,
            cwd=repo_root,
        )
    )


def add_nvfp4_step(
    steps: list[dict[str, Any]],
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
    raw_command: str | None,
    timeout_s: int,
    skip: bool,
    preset: str,
    iterations: int,
) -> None:
    if skip:
        steps.append(skipped("nvfp4", "--skip-nvfp4 provided", required=False))
        return
    if raw_command:
        command = shlex.split(raw_command)
        artifact = None
    else:
        artifact = results_dir / f"{run_id}_flashinfer_mm_fp4_smoke.json"
        command = [
            sys.executable,
            str(repo_root / "scripts" / "flashinfer_mm_fp4_microbench.py"),
            "--phase",
            "exploratory",
            "--run-id",
            f"{run_id}-flashinfer-mm-fp4",
            "--container",
            "host",
            "--preset",
            preset,
            "--iterations",
            str(iterations),
            "--output",
            str(artifact),
        ]
    steps.append(
        run_command(
            name="nvfp4",
            command=command,
            timeout_s=timeout_s,
            artifact=artifact,
            required=True,
            cwd=repo_root,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"spark-smoke-suite-{utc_stamp()}")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--vllm-url")
    parser.add_argument("--vllm-model")
    parser.add_argument("--sglang-url")
    parser.add_argument("--sglang-model")
    parser.add_argument("--llamacpp-url")
    parser.add_argument("--llamacpp-model")
    parser.add_argument("--include-litert-lm", action="store_true")
    parser.add_argument(
        "--litert-repo",
        default="litert-community/gemma-4-E2B-it-litert-lm",
    )
    parser.add_argument("--litert-model-file", default="gemma-4-E2B-it.litertlm")
    parser.add_argument("--hf-command", help="HF fallback command to run under telemetry.")
    parser.add_argument("--mtp-command", help="MTP/spec decode command to run under this suite.")
    parser.add_argument("--mtp-model", help="Model label to store in the MTP telemetry artifact.")
    parser.add_argument("--nvfp4-command", help="NVFP4/fp8 probe command to run under this suite.")
    parser.add_argument("--nvfp4-preset", default="smoke", choices=["smoke", "dense_decode", "moe_expert"])
    parser.add_argument("--nvfp4-iterations", type=int, default=10)
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument("--skip-llamacpp", action="store_true")
    parser.add_argument("--skip-sglang", action="store_true")
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-mtp", action="store_true")
    parser.add_argument("--skip-nvfp4", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = repo_root / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or results_dir / f"{args.run_id}.json"
    if not output.is_absolute():
        output = repo_root / output

    steps: list[dict[str, Any]] = []
    doctor_json = results_dir / f"{args.run_id}_spark_doctor.json"
    steps.append(
        run_command(
            name="spark-doctor",
            command=[
                sys.executable,
                str(repo_root / "scripts" / "spark_doctor.py"),
                "--json",
            ],
            timeout_s=args.timeout_s,
            artifact=doctor_json,
            stdout_artifact=True,
            required=True,
            cwd=repo_root,
        )
    )

    add_openai_step(
        steps,
        repo_root=repo_root,
        results_dir=results_dir,
        run_id=args.run_id,
        timeout_s=args.timeout_s,
        backend="vllm",
        url=args.vllm_url,
        model=args.vllm_model,
        required=True,
        skip=args.skip_vllm,
    )
    add_openai_step(
        steps,
        repo_root=repo_root,
        results_dir=results_dir,
        run_id=args.run_id,
        timeout_s=args.timeout_s,
        backend="sglang",
        url=args.sglang_url,
        model=args.sglang_model,
        required=True,
        skip=args.skip_sglang,
    )
    add_openai_step(
        steps,
        repo_root=repo_root,
        results_dir=results_dir,
        run_id=args.run_id,
        timeout_s=args.timeout_s,
        backend="llamacpp",
        url=args.llamacpp_url,
        model=args.llamacpp_model,
        required=True,
        skip=args.skip_llamacpp,
    )
    if args.include_litert_lm:
        add_litert_steps(
            steps,
            results_dir=results_dir,
            run_id=args.run_id,
            timeout_s=args.timeout_s,
            skip_litert=False,
            litert_repo=args.litert_repo,
            litert_model_file=args.litert_model_file,
        )
    else:
        steps.append(skipped("litert-lm-cpu-chat", "LiteRT-LM is opt-in; pass --include-litert-lm"))
    add_hf_step(
        steps,
        repo_root=repo_root,
        results_dir=results_dir,
        run_id=args.run_id,
        raw_command=args.hf_command,
        timeout_s=args.timeout_s,
        skip=args.skip_hf,
    )
    add_mtp_step(
        steps,
        repo_root=repo_root,
        results_dir=results_dir,
        run_id=args.run_id,
        raw_command=args.mtp_command,
        timeout_s=args.timeout_s,
        skip=args.skip_mtp,
        model=args.mtp_model,
    )
    add_nvfp4_step(
        steps,
        repo_root=repo_root,
        results_dir=results_dir,
        run_id=args.run_id,
        raw_command=args.nvfp4_command,
        timeout_s=args.timeout_s,
        skip=args.skip_nvfp4,
        preset=args.nvfp4_preset,
        iterations=args.nvfp4_iterations,
    )

    required_failures = [step for step in steps if step.get("required") and step.get("ok") is not True]
    configured_failures = [
        step
        for step in steps
        if step.get("configured") and step.get("ok") is False
    ]
    report = {
        "schema": "spark-smoke-suite/v1",
        "run_id": args.run_id,
        "started_utc": iso_utc(),
        "hardware": collect_cuda_hardware(),
        "host": os.uname().nodename if hasattr(os, "uname") else None,
        "repo_root": str(repo_root),
        "results_dir": str(results_dir),
        "steps": steps,
        "summary": {
            "total": len(steps),
            "configured": sum(1 for step in steps if step.get("configured")),
            "skipped": sum(1 for step in steps if not step.get("configured")),
            "ok_steps": sum(1 for step in steps if step.get("ok") is True),
            "failed": len(configured_failures),
            "required_failed": len(required_failures),
        },
    }
    report["finished_utc"] = iso_utc()
    report["ok"] = not required_failures and not configured_failures

    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": args.run_id, "ok": report["ok"], **report["summary"]}, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
