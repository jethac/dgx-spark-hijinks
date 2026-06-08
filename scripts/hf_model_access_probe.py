#!/usr/bin/env python3
"""Probe Hugging Face model access without downloading model weights."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


DEFAULT_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def disk_usage(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def exception_record(exc: BaseException) -> dict[str, Any]:
    return {
        "ok": False,
        "exception_type": type(exc).__name__,
        "message": str(exc)[:2000],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--allow-pattern", action="append", default=[])
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    started = time.time()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    allow_patterns = args.allow_pattern or DEFAULT_ALLOW_PATTERNS

    record: dict[str, Any] = {
        "schema": "hf-model-access-probe/v1",
        "model": args.model,
        "started_unix": started,
        "environment": {
            "hf_home": os.environ.get("HF_HOME"),
            "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
            "hf_token_env_present": bool(os.environ.get("HF_TOKEN")),
            "cache_dir": str(cache_dir) if cache_dir else None,
        },
        "allow_patterns": allow_patterns,
        "local_files_only": args.local_files_only,
        "checks": {},
    }

    disk_path = cache_dir or Path(os.environ.get("HF_HOME") or ".")
    try:
        record["disk"] = disk_usage(disk_path)
    except Exception as exc:  # pragma: no cover - best-effort telemetry
        record["disk"] = exception_record(exc)

    try:
        import huggingface_hub
        from huggingface_hub import HfApi

        record["huggingface_hub_version"] = getattr(
            huggingface_hub, "__version__", None
        )
        info = HfApi().model_info(args.model)
        record["checks"]["model_info"] = {
            "ok": True,
            "private": getattr(info, "private", None),
            "gated": getattr(info, "gated", None),
            "sha": getattr(info, "sha", None),
            "siblings": len(getattr(info, "siblings", []) or []),
        }
    except Exception as exc:
        record["checks"]["model_info"] = exception_record(exc)

    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            args.model,
            allow_patterns=allow_patterns,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=args.local_files_only,
        )
        root = Path(path)
        record["checks"]["small_snapshot"] = {
            "ok": True,
            "path": str(root),
            "files": sorted(
                item.relative_to(root).as_posix()
                for item in root.rglob("*")
                if item.is_file()
            ),
        }
    except Exception as exc:
        record["checks"]["small_snapshot"] = exception_record(exc)

    record["finished_unix"] = time.time()
    record["ok"] = all(
        check.get("ok") for check in record.get("checks", {}).values()
    )
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
