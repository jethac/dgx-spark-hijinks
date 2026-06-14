#!/usr/bin/env python3
"""Poll SGLang lane coordination state before spending a Spark window.

This is an offline coordination guard. It checks the local/remote mail ledger,
the dependency refs that gate the Gemma 4 AR ladder, and the local git state.
It does not prove a runtime row; it answers whether there is new coordination or
a dependency movement that justifies leaving the known-red hold pattern.
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


MAIL_RE = re.compile(r"^mail/(\d{4})_([^_]+)-to-([^_]+)_(.+)\.md$")
KNOWN_BLOCKED_FLASHINFER_REF = "3fa0775cafaf88da5e0fc3b987afa6bd75d9510c"
KNOWN_BLOCKED_SGLANG_REF = "f920e2d88af68031b745494f5435efb71ac93562"
FLASHINFER_BRANCH = "spark/hijinks-022-fa2-d512"
SGLANG_BRANCH = "spark/hijinks-025-sglang-0.5.13-rebase"


def run_git(repo_root: pathlib.Path, args: list[str], *, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {repo_root}: {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def run_submodule_git(repo_root: pathlib.Path, subdir: str, args: list[str]) -> str:
    return run_git(repo_root / "third_party" / subdir, args)


def ls_remote_ref(repo_root: pathlib.Path, subdir: str, branch: str) -> str:
    out = run_submodule_git(repo_root, subdir, ["ls-remote", "origin", branch])
    if not out:
        raise RuntimeError(f"no ls-remote result for {subdir} {branch}")
    return out.split()[0]


def local_mail_paths(repo_root: pathlib.Path) -> list[str]:
    mail_dir = repo_root / "mail"
    if not mail_dir.exists():
        return []
    return [path.relative_to(repo_root).as_posix() for path in mail_dir.glob("*.md")]


def mail_paths_in_tree(repo_root: pathlib.Path, treeish: str) -> list[str]:
    out = run_git(
        repo_root,
        ["ls-tree", "-r", "--name-only", treeish, "--", "mail/"],
        check=False,
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def origin_refs(repo_root: pathlib.Path) -> list[str]:
    out = run_git(
        repo_root,
        ["for-each-ref", "--format=%(refname:short)", "refs/remotes/origin"],
    )
    return sorted(
        line
        for line in out.splitlines()
        if line.startswith("origin/") and line != "origin/HEAD"
    )


def parse_mail(path: str) -> dict[str, Any] | None:
    match = MAIL_RE.match(path)
    if not match:
        return None
    return {
        "number": int(match.group(1)),
        "path": path,
        "sender": match.group(2),
        "recipient": match.group(3),
        "slug": match.group(4),
    }


def latest_mail(paths: list[str], *, sender: str | None = None) -> dict[str, Any] | None:
    parsed = [item for path in paths if (item := parse_mail(path))]
    if sender:
        parsed = [item for item in parsed if item["sender"] == sender]
    if not parsed:
        return None
    item = max(parsed, key=lambda value: (value["number"], value["path"]))
    result = dict(item)
    result["number"] = f"{item['number']:04d}"
    return result


def remote_mail_by_ref(repo_root: pathlib.Path, refs: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for ref in refs:
        paths = mail_paths_in_tree(repo_root, ref)
        if paths:
            result[ref] = paths
    return result


def dependency_state(name: str, current: str, known_blocked: str) -> dict[str, Any]:
    return {
        "name": name,
        "current_ref": current,
        "known_blocked_ref": known_blocked,
        "dependency_changed": current != known_blocked,
    }


def git_status_short(repo_root: pathlib.Path) -> str:
    try:
        return run_git(repo_root, ["status", "--short"])
    except RuntimeError as exc:
        return f"unavailable: {exc}"


def audit(
    repo_root: pathlib.Path,
    *,
    scan_all_origin_mail: bool,
    flashinfer_ref_override: str | None = None,
    sglang_ref_override: str | None = None,
) -> dict[str, Any]:
    local_paths = local_mail_paths(repo_root)
    epoch2_paths = mail_paths_in_tree(repo_root, "origin/epoch2")
    refs = origin_refs(repo_root) if scan_all_origin_mail else ["origin/epoch2"]
    remote_by_ref = remote_mail_by_ref(repo_root, refs)
    all_remote_paths = sorted({path for paths in remote_by_ref.values() for path in paths})

    local_latest = latest_mail(local_paths)
    remote_epoch2_latest = latest_mail(epoch2_paths)
    remote_any_latest = latest_mail(all_remote_paths)
    remote_claude_latest = latest_mail(all_remote_paths, sender="claude")

    local_max = int(local_latest["number"]) if local_latest else 0
    remote_max = int(remote_any_latest["number"]) if remote_any_latest else 0
    new_remote_mail = remote_max > local_max

    flashinfer_ref = flashinfer_ref_override or ls_remote_ref(
        repo_root, "flashinfer", FLASHINFER_BRANCH
    )
    sglang_ref = sglang_ref_override or ls_remote_ref(
        repo_root, "sglang", SGLANG_BRANCH
    )
    dependencies = [
        dependency_state("flashinfer", flashinfer_ref, KNOWN_BLOCKED_FLASHINFER_REF),
        dependency_state("sglang", sglang_ref, KNOWN_BLOCKED_SGLANG_REF),
    ]
    dependency_changed = any(item["dependency_changed"] for item in dependencies)

    if new_remote_mail:
        next_action = "Read and merge/fetch the newer remote mail before running anything."
        lane_status = "new-remote-mail"
    elif dependency_changed:
        next_action = (
            "Review the dependency delta, select/build the package image, then rerun "
            "the smallest matched known-red row with an explicit override reason."
        )
        lane_status = "dependency-changed-review-before-rerun"
    else:
        next_action = (
            "Do not rerun known-red SGLang Gemma 4 AR rows. Wait for Claude's "
            "FlashInfer/numerics or D512 fp8 dispatcher fix."
        )
        lane_status = "blocked-known-red-dependencies"

    now = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9)))
    return {
        "schema": "sglang-lane-state-poll/v1",
        "timestamp_jst": now.isoformat(),
        "repo_root": str(repo_root),
        "git_status_short": git_status_short(repo_root),
        "mail": {
            "local_latest": local_latest,
            "remote_epoch2_latest": remote_epoch2_latest,
            "remote_any_latest": remote_any_latest,
            "remote_claude_latest": remote_claude_latest,
            "new_remote_mail": new_remote_mail,
            "remote_refs_scanned": refs,
        },
        "dependency_branches": {
            "flashinfer": FLASHINFER_BRANCH,
            "sglang": SGLANG_BRANCH,
        },
        "dependencies": dependencies,
        "lane_status": lane_status,
        "next_action": next_action,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="write JSON to this path instead of stdout",
    )
    parser.add_argument(
        "--epoch2-only-mail",
        action="store_true",
        help="skip scanning all origin branches for accidentally committed mail",
    )
    parser.add_argument(
        "--flashinfer-ref",
        help="override FlashInfer ref; intended for offline transition tests",
    )
    parser.add_argument(
        "--sglang-ref",
        help="override SGLang ref; intended for offline transition tests",
    )
    args = parser.parse_args()

    result = audit(
        args.repo_root.resolve(),
        scan_all_origin_mail=not args.epoch2_only_mail,
        flashinfer_ref_override=args.flashinfer_ref,
        sglang_ref_override=args.sglang_ref,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
