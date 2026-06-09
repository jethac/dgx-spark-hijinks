#!/usr/bin/env python3
"""Audit a llama.cpp checkout for supplied-token loglikelihood support.

This is a source-level companion to the live llama-server probes. It answers a
different question: whether the checkout appears to expose a server API capable
of scoring supplied continuation tokens, and where existing non-server scoring
logic can be reused if it does not.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Match:
    path: str
    line: int
    text: str

    def to_json(self) -> dict[str, Any]:
        return {"path": self.path, "line": self.line, "text": self.text}


def run_git(src: Path, args: list[str]) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(src), *args],
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def find_literals(src: Path, relpath: str, literals: list[str]) -> list[Match]:
    path = src / relpath
    matches: list[Match] = []
    for line_no, line in enumerate(read_lines(path), start=1):
        lower = line.lower()
        if any(literal.lower() in lower for literal in literals):
            matches.append(Match(relpath.replace("\\", "/"), line_no, line.strip()))
    return matches


def count_tree_literals(root: Path, literals: list[str]) -> dict[str, list[Match]]:
    out: dict[str, list[Match]] = {}
    if not root.exists():
        return out
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".cpp", ".h", ".hpp", ".md", ".py"}:
            continue
        relpath = path.relative_to(root.parent.parent).as_posix()
        matches: list[Match] = []
        for line_no, line in enumerate(read_lines(path), start=1):
            lower = line.lower()
            if any(literal.lower() in lower for literal in literals):
                matches.append(Match(relpath, line_no, line.strip()))
        if matches:
            out[relpath] = matches
    return out


def first(items: list[Match], limit: int = 8) -> list[dict[str, Any]]:
    return [item.to_json() for item in items[:limit]]


def flatten(grouped: dict[str, list[Match]]) -> list[Match]:
    matches: list[Match] = []
    for group in grouped.values():
        matches.extend(group)
    return matches


def build_report(src: Path) -> dict[str, Any]:
    src = src.resolve()
    server_dir = src / "tools" / "server"
    server_prompt_logprob_terms = [
        "token_logprobs",
        "prompt_logprobs",
        "prompt_token_logprobs",
        "continuation_token_logprobs",
    ]
    server_prompt_logprob_matches = flatten(
        count_tree_literals(server_dir, server_prompt_logprob_terms)
    )

    server_generated_logprob_matches = (
        find_literals(
            src,
            "tools/server/server-common.cpp",
            ["logprobs", "top_logprobs", "n_probs", "only no echo is supported"],
        )
        + find_literals(
            src,
            "tools/server/server-task.cpp",
            ["probs_output", "completion_probabilities", "top_logprobs", "logprobs"],
        )
        + find_literals(
            src,
            "tools/server/README.md",
            ["n_probs", "completion_probabilities", "post_sampling_probs"],
        )
    )
    prompt_eval_matches = find_literals(
        src,
        "tools/server/server-context.cpp",
        [
            "set logits only for the last prompt token",
            "extract the logits only for the last token",
            "llama_get_logits_ith",
            "generated_token_probs",
            "populate_token_probs",
        ],
    )

    scoring_matches = (
        find_literals(
            src,
            "tools/perplexity/perplexity.cpp",
            [
                "compute_logprobs",
                "multiple_choice_score",
                "hellaswag_score",
                "llama_get_logits",
                "llama_get_logits_ith",
                "softmax",
                "log_softmax",
            ],
        )
        + find_literals(
            src,
            "tools/imatrix/imatrix.cpp",
            ["log_softmax", "llama_get_logits"],
        )
    )

    route_matches = find_literals(
        src,
        "tools/server/server.cpp",
        ["/completion", "/v1/completions", "/chat/completions"],
    )
    route_names = [match.text for match in route_matches]
    stock_server_contract_capable = bool(server_prompt_logprob_matches)

    findings: list[str] = []
    if not stock_server_contract_capable:
        findings.append(
            "No server-source evidence of prompt-token token_logprobs arrays or a supplied-token loglikelihood endpoint."
        )
    if server_generated_logprob_matches:
        findings.append(
            "Server logprob evidence is generated-token/top-N oriented via n_probs/probs_output."
        )
    if prompt_eval_matches:
        findings.append(
            "Server prompt evaluation does not appear to retain full prompt-token logits for supplied-token scoring."
        )
    if scoring_matches:
        findings.append(
            "Non-server tools already compute supplied-token logprobs from logits and token ids."
        )
    else:
        findings.append("No reusable non-server supplied-token scoring primitive was found.")

    return {
        "schema": "llamacpp-source-loglikelihood-audit/v1",
        "source": str(src),
        "git": {
            "commit": run_git(src, ["rev-parse", "HEAD"]),
            "branch": run_git(src, ["branch", "--show-current"]),
            "describe": run_git(src, ["describe", "--always", "--dirty"]),
        },
        "stock_server_contract_capable": stock_server_contract_capable,
        "endpoint_patch_needed": not stock_server_contract_capable,
        "server_routes_seen": route_names,
        "server_prompt_logprob_evidence": first(server_prompt_logprob_matches),
        "server_generated_topn_logprob_evidence": first(server_generated_logprob_matches, limit=14),
        "server_prompt_eval_logits_evidence": first(prompt_eval_matches, limit=14),
        "reusable_supplied_token_scoring_evidence": first(scoring_matches, limit=14),
        "recommended_patch_shape": [
            "Add a server endpoint or request mode that accepts context plus continuation.",
            "Tokenize context and continuation separately, preserving normal BOS/chat-template behavior in the response.",
            "Decode context+continuation with logits for every continuation prediction position.",
            "For each continuation token id, compute log_softmax(logits)[token_id] directly, independent of top-N rank.",
            "Return continuation_token_ids, continuation_token_logprobs, target_logprob_sum, all_tokens_greedy, and lm_eval_loglikelihood_tuple.",
            "Keep generated-token n_probs/top_logprobs output separate from this supplied-token scoring contract.",
        ],
        "findings": findings,
        "ok": True,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# llama.cpp Source Loglikelihood Audit",
        "",
        f"Source: `{report.get('source')}`",
        f"Commit: `{report.get('git', {}).get('commit')}`",
        "",
        f"Stock server contract capable: `{report.get('stock_server_contract_capable')}`",
        f"Endpoint patch needed: `{report.get('endpoint_patch_needed')}`",
        "",
        "## Findings",
        "",
    ]
    for finding in report.get("findings", []):
        lines.append(f"- {finding}")
    lines.extend(["", "## Server Logprob Evidence", ""])
    for item in report.get("server_generated_topn_logprob_evidence", []):
        lines.append(f"- `{item['path']}:{item['line']}` {item['text']}")
    if not report.get("server_generated_topn_logprob_evidence"):
        lines.append("- none")
    lines.extend(["", "## Prompt-Eval Logits Evidence", ""])
    for item in report.get("server_prompt_eval_logits_evidence", []):
        lines.append(f"- `{item['path']}:{item['line']}` {item['text']}")
    if not report.get("server_prompt_eval_logits_evidence"):
        lines.append("- none")
    lines.extend(["", "## Prompt-Token Server Evidence", ""])
    for item in report.get("server_prompt_logprob_evidence", []):
        lines.append(f"- `{item['path']}:{item['line']}` {item['text']}")
    if not report.get("server_prompt_logprob_evidence"):
        lines.append("- none")
    lines.extend(["", "## Reusable Scoring Evidence", ""])
    for item in report.get("reusable_supplied_token_scoring_evidence", []):
        lines.append(f"- `{item['path']}:{item['line']}` {item['text']}")
    if not report.get("reusable_supplied_token_scoring_evidence"):
        lines.append("- none")
    lines.extend(["", "## Recommended Patch Shape", ""])
    for item in report.get("recommended_patch_shape", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-src", default="third_party/llama.cpp")
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    parser.add_argument(
        "--require-stock-server-capable",
        action="store_true",
        help="exit non-zero if the current server source cannot satisfy the supplied-token contract",
    )
    args = parser.parse_args()

    report = build_report(Path(args.llama_src))
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    if args.output_md:
        Path(args.output_md).write_text(markdown(report), encoding="utf-8")
    print(text, end="")
    if args.require_stock_server_capable and not report["stock_server_contract_capable"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
