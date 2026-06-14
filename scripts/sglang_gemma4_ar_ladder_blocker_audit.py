#!/usr/bin/env python3
"""Audit whether the SGLang Gemma 4 AR ladder has a new dependency to test.

This is an offline safety check. It does not prove that a blocker is fixed; it
only reports whether the dependency refs have moved away from the refs that
produced the known-red SGLang rows.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import re
import subprocess
import sys
from typing import Any


KNOWN_BLOCKED_FLASHINFER_REF = "3fa0775cafaf88da5e0fc3b987afa6bd75d9510c"
KNOWN_BLOCKED_SGLANG_REF = "f920e2d88af68031b745494f5435efb71ac93562"
FLASHINFER_BRANCH = "spark/hijinks-022-fa2-d512"
SGLANG_BRANCH = "spark/hijinks-025-sglang-0.5.13-rebase"


def run_git(args: list[str], cwd: pathlib.Path) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {cwd}: {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def ls_remote(repo: pathlib.Path, branch: str) -> str:
    out = run_git(["ls-remote", "origin", branch], repo)
    if not out:
        raise RuntimeError(f"no ls-remote result for {repo} {branch}")
    return out.split()[0]


def latest_mail(repo_root: pathlib.Path) -> dict[str, Any]:
    mail_dir = repo_root / "mail"
    latest: tuple[int, pathlib.Path] | None = None
    for path in mail_dir.glob("*.md"):
        match = re.match(r"^(\d{4})_", path.name)
        if not match:
            continue
        number = int(match.group(1))
        if latest is None or number > latest[0]:
            latest = (number, path)
    if latest is None:
        return {"number": None, "path": None}
    return {
        "number": latest[0],
        "path": str(latest[1].relative_to(repo_root)).replace("\\", "/"),
    }


def blocker_state(name: str, current: str, known_blocked: str) -> dict[str, Any]:
    changed = current != known_blocked
    return {
        "name": name,
        "current_ref": current,
        "known_blocked_ref": known_blocked,
        "dependency_changed": changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--flashinfer-ref", help="override FlashInfer ref")
    parser.add_argument("--sglang-ref", help="override SGLang ref")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    flashinfer_ref = args.flashinfer_ref or ls_remote(
        repo_root / "third_party" / "flashinfer", FLASHINFER_BRANCH
    )
    sglang_ref = args.sglang_ref or ls_remote(
        repo_root / "third_party" / "sglang", SGLANG_BRANCH
    )

    deps = [
        blocker_state(
            "flashinfer",
            flashinfer_ref,
            KNOWN_BLOCKED_FLASHINFER_REF,
        ),
        blocker_state(
            "sglang",
            sglang_ref,
            KNOWN_BLOCKED_SGLANG_REF,
        ),
    ]
    any_dependency_changed = any(item["dependency_changed"] for item in deps)

    if any_dependency_changed:
        ladder_status = "dependency-changed-review-before-rerun"
        can_run_claim_ladder = False
        diagnostic_override_allowed = True
        next_action = (
            "Review the dependency delta, build or select a packaged image with "
            "the changed dependency, then rerun only the smallest matched red row "
            "with SGLANG_AR_LADDER_OVERRIDE_REASON naming the fix/ref."
        )
    else:
        ladder_status = "blocked-known-red-dependencies"
        can_run_claim_ladder = False
        diagnostic_override_allowed = False
        next_action = (
            "Do not rerun the known-red SGLang Gemma 4 AR ladder rows. Wait for "
            "the shared FlashInfer large-prefill accumulation fix for the 12B "
            "ctx8185 artifact, an explicitly scoped chunked/merge diagnostic, "
            "or the D512/VO256 fp8 dispatcher fix."
        )

    now = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9)))
    result: dict[str, Any] = {
        "schema": "sglang-gemma4-ar-ladder-blocker-audit/v1",
        "timestamp_jst": now.isoformat(),
        "repo_root": str(repo_root),
        "mail_latest": latest_mail(repo_root),
        "dependency_branches": {
            "flashinfer": FLASHINFER_BRANCH,
            "sglang": SGLANG_BRANCH,
        },
        "dependencies": deps,
        "known_blockers": {
            "full_nvfp4_12b_26b_31b": (
                "12B ctx8185/prefix4096 full-NVFP4 matched row is red by "
                "+0.402969 nats/token, but mail/0140 classifies that as a "
                "FlashInfer single-/large-prefill accumulation artifact. Exact "
                "SDPA and vLLM chunked/reuse put the true NVFP4 cost near +0.19."
            ),
            "e4b_fp8_comparator": (
                "E4B fp8 comparator is red in FlashInfer D512/VO256 "
                "1-byte-KV paged-prefill dispatcher scheduling."
            ),
        },
        "ladder_status": ladder_status,
        "can_run_claim_ladder": can_run_claim_ladder,
        "diagnostic_override_allowed": diagnostic_override_allowed,
        "next_action": next_action,
    }

    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
