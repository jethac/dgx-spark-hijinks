#!/usr/bin/env python3
"""Print the next safe epoch2 mail number or validate a requested number."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from collections import defaultdict
from typing import Iterable


MAIL_RE = re.compile(r"^mail/(\d{4})_.*\.md$")


def git_ls_tree(repo_root: pathlib.Path, treeish: str) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", treeish, "--", "mail/"],
        cwd=str(repo_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git ls-tree failed for {treeish}: {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def local_mail_paths(repo_root: pathlib.Path) -> list[str]:
    mail_dir = repo_root / "mail"
    if not mail_dir.exists():
        return []
    return [
        path.relative_to(repo_root).as_posix()
        for path in sorted(mail_dir.glob("*.md"))
    ]


def numbered(paths: Iterable[str]) -> dict[int, list[str]]:
    result: dict[int, list[str]] = defaultdict(list)
    for path in paths:
        match = MAIL_RE.match(path)
        if match:
            result[int(match.group(1))].append(path)
    return dict(result)


def merge_number_maps(*maps: dict[int, list[str]]) -> dict[int, list[str]]:
    merged: dict[int, list[str]] = defaultdict(list)
    for mapping in maps:
        for number, paths in mapping.items():
            merged[number].extend(paths)
    return dict(merged)


def format_number(number: int) -> str:
    return f"{number:04d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--treeish",
        default="origin/epoch2",
        help="remote tree to inspect before choosing a number",
    )
    parser.add_argument("--sender", help="sender token for a suggested path")
    parser.add_argument("--recipient", help="recipient token for a suggested path")
    parser.add_argument("--slug", help="subject slug for a suggested path")
    parser.add_argument(
        "--check-number",
        type=int,
        help="fail if this mail number is already occupied locally or remotely",
    )
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    remote_numbers = numbered(git_ls_tree(repo_root, args.treeish))
    local_numbers = numbered(local_mail_paths(repo_root))
    all_numbers = merge_number_maps(remote_numbers, local_numbers)

    max_number = max(all_numbers, default=0)
    next_number = max_number + 1
    duplicate_numbers = {
        format_number(number): sorted(set(paths))
        for number, paths in sorted(all_numbers.items())
        if len(set(paths)) > 1
    }

    requested_occupied = False
    requested_paths: list[str] = []
    if args.check_number is not None:
        requested_paths = sorted(set(all_numbers.get(args.check_number, [])))
        requested_occupied = bool(requested_paths)

    suggested_path = None
    if args.sender or args.recipient or args.slug:
        missing = [
            name
            for name, value in (
                ("sender", args.sender),
                ("recipient", args.recipient),
                ("slug", args.slug),
            )
            if not value
        ]
        if missing:
            parser.error(
                "--sender, --recipient, and --slug must be provided together"
            )
        suggested_path = (
            f"mail/{format_number(next_number)}_"
            f"{args.sender}-to-{args.recipient}_{args.slug}.md"
        )

    result = {
        "treeish": args.treeish,
        "max_number": format_number(max_number),
        "next_number": format_number(next_number),
        "suggested_path": suggested_path,
        "duplicate_numbers": duplicate_numbers,
        "check_number": (
            format_number(args.check_number)
            if args.check_number is not None
            else None
        ),
        "check_number_occupied": requested_occupied,
        "check_number_paths": requested_paths,
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif suggested_path:
        print(suggested_path)
    else:
        print(format_number(next_number))
        if duplicate_numbers:
            dupes = ", ".join(duplicate_numbers)
            print(f"warning: existing duplicate mail numbers: {dupes}", file=sys.stderr)

    if requested_occupied:
        print(
            f"mail number {format_number(args.check_number)} is already occupied:",
            file=sys.stderr,
        )
        for path in requested_paths:
            print(f"  {path}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
