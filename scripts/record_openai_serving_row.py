#!/usr/bin/env python3
"""Record one OpenAI-compatible serving row with a manifest."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from spark_hardware import collect_cuda_hardware


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def sanitize_command(cmd: list[str], repo_root: Path) -> list[str]:
    sanitized: list[str] = []
    for item in cmd:
        if item == sys.executable:
            sanitized.append("python")
            continue
        path = Path(item)
        if path.is_absolute():
            sanitized.append(rel(path, repo_root))
        else:
            sanitized.append(item)
    return sanitized


def run_command(cmd: list[str], *, dry_run: bool, repo_root: Path) -> dict[str, Any]:
    started = time.time()
    record: dict[str, Any] = {
        "command": sanitize_command(cmd, repo_root),
        "started_unix": started,
        "returncode": None,
        "ok": False,
    }
    if dry_run:
        record["dry_run"] = True
        record["ok"] = True
        return record

    proc = subprocess.run(cmd)
    record["returncode"] = proc.returncode
    record["finished_unix"] = time.time()
    record["ok"] = proc.returncode == 0
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["vllm", "sglang", "llamacpp"])
    parser.add_argument(
        "--phase", required=True, choices=["before", "after", "exploratory"]
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--model-revision")
    parser.add_argument("--runtime-ref")
    parser.add_argument("--container-image")
    parser.add_argument("--container-digest")
    parser.add_argument("--quantization")
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--attention-backend")
    parser.add_argument("--cuda-graph-mode")
    parser.add_argument("--server-log")
    parser.add_argument("--process-match")
    parser.add_argument("--cuda-so-package", action="append", default=[])
    parser.add_argument("--llama-bench-command")
    parser.add_argument("--run-gguf-logprobs-probe", action="store_true")
    parser.add_argument("--chat-smoke-prompt", default="Reply with exactly this text: spark-ok")
    parser.add_argument("--chat-smoke-max-tokens", type=int, default=8)
    parser.add_argument("--benchmark-prompt-suffix", default="")
    parser.add_argument("--chat-template-kwargs-json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}
    commands: dict[str, Any] = {}

    chat_out = results_dir / f"{args.run_id}_chat_smoke.json"
    bench_out = results_dir / f"{args.run_id}_openai_benchmark.json"
    artifacts["chat_smoke"] = rel(chat_out, repo_root)
    artifacts["openai_benchmark"] = rel(bench_out, repo_root)

    commands["chat_smoke"] = run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "openai_chat_smoke.py"),
            "--url",
            args.url,
            "--model",
            args.model,
            "--prompt",
            args.chat_smoke_prompt,
            "--max-tokens",
            str(args.chat_smoke_max_tokens),
            *(
                ["--chat-template-kwargs-json", args.chat_template_kwargs_json]
                if args.chat_template_kwargs_json
                else []
            ),
            "--output",
            str(chat_out),
        ],
        dry_run=args.dry_run,
        repo_root=repo_root,
    )
    commands["openai_benchmark"] = run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "openai_serving_benchmark.py"),
            "--url",
            args.url,
            "--model",
            args.model,
            "--backend",
            args.backend,
            "--phase",
            args.phase,
            "--run-id",
            args.run_id,
            "--prompt-suffix",
            args.benchmark_prompt_suffix,
            *(
                ["--chat-template-kwargs-json", args.chat_template_kwargs_json]
                if args.chat_template_kwargs_json
                else []
            ),
            "--output",
            str(bench_out),
        ],
        dry_run=args.dry_run,
        repo_root=repo_root,
    )

    if args.process_match:
        process_out = results_dir / f"{args.run_id}_runtime_probe.json"
        artifacts["runtime_probe"] = rel(process_out, repo_root)
        commands["runtime_probe"] = run_command(
            [
                sys.executable,
                str(repo_root / "scripts" / "runtime_process_probe.py"),
                "--url",
                args.url,
                "--match",
                args.process_match,
                "--output",
                str(process_out),
            ],
            dry_run=args.dry_run,
            repo_root=repo_root,
        )

    if args.server_log:
        log_path = Path(args.server_log)
        artifacts["server_log"] = rel(log_path, repo_root)
        build_out = results_dir / f"{args.run_id}_build_target_audit.json"
        artifacts["build_target_audit"] = rel(build_out, repo_root)
        commands["build_target_audit"] = run_command(
            [
                sys.executable,
                str(repo_root / "scripts" / "cuda_build_target_audit.py"),
                "--log",
                str(log_path),
                "--output",
                str(build_out),
            ],
            dry_run=args.dry_run,
            repo_root=repo_root,
        )

    if args.cuda_so_package:
        cuda_so_out = results_dir / f"{args.run_id}_cuda_so_audit.json"
        artifacts["cuda_so_audit"] = rel(cuda_so_out, repo_root)
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "cuda_so_audit.py"),
            "--output",
            str(cuda_so_out),
        ]
        for package in args.cuda_so_package:
            cmd.extend(["--package", package])
        commands["cuda_so_audit"] = run_command(
            cmd, dry_run=args.dry_run, repo_root=repo_root
        )

    if args.llama_bench_command:
        llama_bench_out = results_dir / f"{args.run_id}_llama_bench.txt"
        artifacts["llama_bench"] = rel(llama_bench_out, repo_root)
        command = args.llama_bench_command + f" > {shlex.quote(str(llama_bench_out))}"
        commands["llama_bench"] = run_command(
            ["sh", "-lc", command], dry_run=args.dry_run, repo_root=repo_root
        )

    if args.run_gguf_logprobs_probe:
        logprobs_out = results_dir / f"{args.run_id}_gguf_logprobs_probe.json"
        artifacts["gguf_logprobs_probe"] = rel(logprobs_out, repo_root)
        commands["gguf_logprobs_probe"] = run_command(
            [
                sys.executable,
                str(repo_root / "scripts" / "gguf_logprobs_probe.py"),
                "--url",
                args.url,
                "--output",
                str(logprobs_out),
            ],
            dry_run=args.dry_run,
            repo_root=repo_root,
        )

    manifest = {
        "schema": "openai-serving-row-manifest/v1",
        "run_id": args.run_id,
        "backend": args.backend,
        "phase": args.phase,
        "model": args.model,
        "metadata": {
            "model_revision": args.model_revision,
            "runtime_ref": args.runtime_ref,
            "container_image": args.container_image,
            "container_digest": args.container_digest,
            "quantization": args.quantization,
            "kv_cache_dtype": args.kv_cache_dtype,
            "attention_backend": args.attention_backend,
            "cuda_graph_mode": args.cuda_graph_mode,
        },
        "hardware": collect_cuda_hardware() if not args.dry_run else {},
        "artifacts": artifacts,
        "commands": commands,
        "ok": all(command.get("ok") for command in commands.values()),
        "dry_run": args.dry_run,
    }

    manifest_out = results_dir / f"{args.run_id}_row_manifest.json"
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
