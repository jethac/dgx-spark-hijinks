#!/usr/bin/env python3
"""Long-context needle-in-a-haystack retrieval probe through a vLLM OpenAI server.

This is the retrieval arm of task #38 (NVFP4 KV vs fp8 vs bf16 on Gemma 4). The
prompt-PPL sweep (scripts/vllm_prompt_ppl_sweep.py) measures *average* next-token
quality; this script instead measures whether a single fact buried in a long
context can be *retrieved* exactly, per (context_length, depth) cell. The
per-token stratification showed NVFP4's prose error grows mildly with position;
that is the exact signature that could nip deep-context retrieval, so this probe
exists to either falsify (null = "retrieval holds, strong claim") or confirm
(positive = "the real depth limit of the capacity win") the H-late hypothesis.

Design (matches the campaign's PPL-sweep client conventions):
  * Deterministic filler haystack built by tokenizing a repo corpus slice (or a
    deterministic synthetic generator) ONCE and slicing token windows -- NOT
    random per run, so a double-run identity check is meaningful.
  * A unique needle fact is inserted at a controllable token DEPTH (fraction of
    context: 0.0 .. 1.0) inside a controllable total CONTEXT LENGTH.
  * The model is queried at temperature 0 and scored for EXACT retrieval of the
    needle value, per (context_len, depth) cell -> an accuracy grid.
  * A RULER-style multi-needle mode (K facts, retrieve all) is available as a
    second mode.
  * Output schema `vllm-needle-retrieval/v1`: per-cell verdicts + accuracy grid,
    kv_cache_dtype, model, container image, a boot-profile note field (fp8 is
    per-boot bistable -- record it), and spark_hardware.py provenance.

The server configuration (KV dtype, knobs, image) is set OUTSIDE this script by
the runner; this script only records what it was told via flags.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from spark_hardware import collect_cuda_hardware

# Context lengths and depths for the standard grid. The runner caps context
# lengths at the served model's max-model-len; depths are context fractions.
DEFAULT_CONTEXT_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_DEPTHS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


# ---------------------------------------------------------------------------
# HTTP (mirrors vllm_prompt_ppl_sweep.post_json: deterministic, retrying)
# ---------------------------------------------------------------------------
def post_json(
    url: str,
    payload: dict[str, Any],
    timeout: int,
    *,
    attempts: int,
    retry_sleep_s: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    errors = []
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (TimeoutError, urllib.error.URLError) as exc:
            errors.append(f"attempt {attempt}/{attempts}: {exc!r}")
            if attempt == attempts:
                break
            time.sleep(retry_sleep_s)
    raise TimeoutError("; ".join(errors))


# ---------------------------------------------------------------------------
# Tokenizer / deterministic filler
# ---------------------------------------------------------------------------
def load_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def read_text(paths: list[Path]) -> str:
    parts = []
    for path in paths:
        parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(parts)


def synthetic_filler(min_chars: int) -> str:
    """Deterministic Paul-Graham-essay-style filler.

    Used only when no --text-file corpus is supplied. The text is fixed (no RNG),
    so tokenization is reproducible across runs and a double-run identity check is
    meaningful. The sentences are mundane prose deliberately unrelated to the
    needle fact, so the needle is the only place the access code appears.
    """
    sentences = [
        "The early engineers preferred small tools they could fully understand.",
        "A good essay is one that changes how the reader sees an ordinary thing.",
        "Most progress comes from people who kept working after it stopped being fun.",
        "Measurement without a baseline tells you almost nothing useful.",
        "The library was quiet in the afternoon and the lamps were always on.",
        "He counted the boxes twice and still wrote the wrong number down.",
        "When the cache is warm the second run is the only honest one.",
        "They walked along the river until the path turned back toward the town.",
        "A claim about speed is not a claim about quality, and the two should stay apart.",
        "The map on the wall had been redrawn so many times it was hard to read.",
        "Determinism is a courtesy you pay to the person who reproduces your work.",
        "She kept a notebook of the questions she had not yet been able to answer.",
        "The corridor smelled of old paper and the faint warmth of running machines.",
        "Every benchmark hides an assumption about what the reader already believes.",
        "He set the kettle on and went back to reading the same paragraph again.",
    ]
    buf: list[str] = []
    total = 0
    idx = 0
    # Deterministic cycle through the fixed sentence list.
    while total < min_chars:
        s = sentences[idx % len(sentences)]
        buf.append(s)
        total += len(s) + 1
        idx += 1
    return " ".join(buf)


def build_filler_tokens(
    tokenizer: Any,
    text_files: list[str],
    needed_tokens: int,
) -> list[int]:
    """Tokenize the filler corpus once and ensure it is long enough.

    The filler is repeated (deterministically) until it has at least
    `needed_tokens` tokens, so the largest context length in the grid fits.
    """
    if text_files:
        base_text = read_text([Path(p) for p in text_files])
    else:
        # ~5 chars/token is a safe over-estimate for the char budget.
        base_text = synthetic_filler(min_chars=needed_tokens * 6 + 4096)
    tokens = tokenizer.encode(base_text, add_special_tokens=False)
    if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
        raise TypeError("tokenizer.encode did not return a list[int]")
    if not tokens:
        raise ValueError("filler corpus tokenized to zero tokens")
    # Deterministically tile to reach the needed length.
    while len(tokens) < needed_tokens:
        tokens = tokens + tokens
    return tokens


# ---------------------------------------------------------------------------
# Needle insertion + prompt assembly
# ---------------------------------------------------------------------------
def needle_sentence(label: str, code: str) -> str:
    """The unique fact. Deliberately formatted so the value is unambiguous."""
    return f" The special access code for {label} is {code}. "


def question_text(labels: list[str]) -> str:
    if len(labels) == 1:
        return (
            f"What is the special access code for {labels[0]}? "
            "Answer with only the code."
        )
    listed = ", ".join(labels)
    return (
        f"What are the special access codes for the following, in order: {listed}? "
        "Answer with only the codes, separated by commas, in the same order."
    )


def insert_needles_at_depths(
    filler_tokens: list[int],
    context_len: int,
    needle_token_lists: list[list[int]],
    depths: list[float],
) -> tuple[list[int], list[int]]:
    """Assemble a context of `context_len` tokens with needles at given depths.

    Returns (context_token_ids, needle_offsets) where needle_offsets[i] is the
    token offset at which needle i was inserted.

    Construction is deterministic: a fixed slice of the filler is taken, then each
    needle is spliced in at floor(depth * (filler_len)). Inserting earliest depth
    last keeps earlier offsets stable as later ones shift; we sort by depth and
    track cumulative shift so reported offsets are exact post-insertion positions.
    """
    total_needle_tokens = sum(len(n) for n in needle_token_lists)
    filler_budget = context_len - total_needle_tokens
    if filler_budget < 1:
        raise ValueError(
            f"context_len={context_len} too small for needles "
            f"({total_needle_tokens} needle tokens)"
        )
    if len(filler_tokens) < filler_budget:
        raise ValueError(
            f"filler has {len(filler_tokens)} tokens, needs {filler_budget}"
        )
    base = list(filler_tokens[:filler_budget])

    # Pair each needle with its depth, sort by insertion point ascending so we can
    # account for the cumulative shift of earlier insertions on later ones.
    order = sorted(range(len(needle_token_lists)), key=lambda i: depths[i])
    # Base insertion point in the ORIGINAL filler coordinate space.
    base_points = {
        i: min(filler_budget, max(0, int(round(depths[i] * filler_budget))))
        for i in order
    }
    out = base
    shift = 0
    offsets = [0] * len(needle_token_lists)
    last_orig_point = -1
    # Insert in ascending original-coordinate order.
    for i in order:
        orig_point = base_points[i]
        # Two needles at the same depth must not collide: nudge to keep order.
        if orig_point <= last_orig_point:
            orig_point = last_orig_point
        last_orig_point = orig_point
        actual_point = orig_point + shift
        out = out[:actual_point] + needle_token_lists[i] + out[actual_point:]
        offsets[i] = actual_point
        shift += len(needle_token_lists[i])

    if len(out) != context_len:
        raise ValueError(
            f"assembled context is {len(out)} tokens, expected {context_len}"
        )
    return out, offsets


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def extract_codes(answer: str, codes: list[str]) -> dict[str, bool]:
    """Exact retrieval: each expected code must appear verbatim in the answer.

    Codes are zero-padded numeric strings (e.g. "48217"); we test substring
    presence so trailing punctuation / formatting from the model does not cause a
    false negative, while a wrong or missing code is correctly scored as a miss.
    """
    return {code: (code in answer) for code in codes}


def score_answer(answer: str, expected_codes: list[str]) -> dict[str, Any]:
    found = extract_codes(answer, expected_codes)
    hits = sum(1 for v in found.values() if v)
    return {
        "expected_codes": expected_codes,
        "found": found,
        "num_expected": len(expected_codes),
        "num_found": hits,
        # Single-needle: retrieved iff the one code is present.
        # Multi-needle (RULER): retrieved iff ALL codes present (strict recall).
        "retrieved": hits == len(expected_codes) and len(expected_codes) > 0,
        "answer_preview": answer[:200],
    }


def chat_answer(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("chat response has no choices")
    message = choices[0].get("message", {})
    content = message.get("content")
    return content if isinstance(content, str) else ""


# ---------------------------------------------------------------------------
# Deterministic per-cell needle/code generation (seedable but reproducible)
# ---------------------------------------------------------------------------
def cell_label_and_code(seed: int, context_len: int, depth: float, k: int) -> str:
    """Deterministic 5-digit code from the cell identity (no RNG state)."""
    # A simple reproducible hash over the cell coordinates; same inputs -> same
    # code on every run, which is what the determinism gate relies on.
    basis = (
        seed * 1_000_003
        + context_len * 9_176
        + int(round(depth * 1000)) * 131
        + k * 7
    )
    return f"{basis % 100000:05d}"


def make_cell_needles(
    seed: int,
    context_len: int,
    depths: list[float],
    tokenizer: Any,
) -> tuple[list[str], list[str], list[list[int]]]:
    """Build labels, codes, and tokenized needle sentences for one cell.

    Labels are stable per index (vault-0, vault-1, ...). Codes are deterministic
    per (seed, context_len, depth, k).
    """
    labels: list[str] = []
    codes: list[str] = []
    needle_tokens: list[list[int]] = []
    for k, depth in enumerate(depths):
        label = f"vault-{k}"
        code = cell_label_and_code(seed, context_len, depth, k)
        sentence = needle_sentence(label, code)
        toks = tokenizer.encode(sentence, add_special_tokens=False)
        labels.append(label)
        codes.append(code)
        needle_tokens.append(toks)
    return labels, codes, needle_tokens


# ---------------------------------------------------------------------------
# One cell (single needle) and one multi-needle cell
# ---------------------------------------------------------------------------
def run_single_cell(
    args: argparse.Namespace,
    tokenizer: Any,
    filler_tokens: list[int],
    context_len: int,
    depth: float,
) -> dict[str, Any]:
    label, code = "vault-0", cell_label_and_code(args.seed, context_len, depth, 0)
    needle_toks = tokenizer.encode(
        needle_sentence(label, code), add_special_tokens=False
    )
    context_ids, offsets = insert_needles_at_depths(
        filler_tokens, context_len, [needle_toks], [depth]
    )
    context_text = tokenizer.decode(context_ids)
    prompt = (
        context_text
        + "\n\n"
        + question_text([label])
    )
    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    started = time.perf_counter()
    response = post_json(
        endpoint,
        payload,
        args.timeout,
        attempts=args.request_attempts,
        retry_sleep_s=args.retry_sleep_s,
    )
    elapsed_s = time.perf_counter() - started
    answer = chat_answer(response)
    score = score_answer(answer, [code])
    return {
        "context_len": context_len,
        "depth": depth,
        "needle_offset": offsets[0],
        "needle_label": label,
        "expected_code": code,
        "context_token_count": len(context_ids),
        "elapsed_s": elapsed_s,
        "retrieved": score["retrieved"],
        "score": score,
        "usage": response.get("usage"),
    }


def run_multi_cell(
    args: argparse.Namespace,
    tokenizer: Any,
    filler_tokens: list[int],
    context_len: int,
    depths: list[float],
) -> dict[str, Any]:
    labels, codes, needle_toks = make_cell_needles(
        args.seed, context_len, depths, tokenizer
    )
    context_ids, offsets = insert_needles_at_depths(
        filler_tokens, context_len, needle_toks, depths
    )
    context_text = tokenizer.decode(context_ids)
    prompt = context_text + "\n\n" + question_text(labels)
    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    started = time.perf_counter()
    response = post_json(
        endpoint,
        payload,
        args.timeout,
        attempts=args.request_attempts,
        retry_sleep_s=args.retry_sleep_s,
    )
    elapsed_s = time.perf_counter() - started
    answer = chat_answer(response)
    score = score_answer(answer, codes)
    return {
        "context_len": context_len,
        "depths": depths,
        "needle_offsets": offsets,
        "needle_labels": labels,
        "expected_codes": codes,
        "context_token_count": len(context_ids),
        "elapsed_s": elapsed_s,
        "retrieved_all": score["retrieved"],
        "score": score,
        "usage": response.get("usage"),
    }


# ---------------------------------------------------------------------------
# Grid assembly
# ---------------------------------------------------------------------------
def assemble_grid(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a (context_len x depth) accuracy grid from single-needle cells."""
    by_ctx: dict[int, dict[str, Any]] = {}
    for cell in cells:
        ctx = cell["context_len"]
        depth = cell["depth"]
        by_ctx.setdefault(ctx, {})[f"{depth}"] = bool(cell["retrieved"])
    context_lens = sorted(by_ctx.keys())
    total = len(cells)
    hits = sum(1 for c in cells if c["retrieved"])
    return {
        "context_lengths": context_lens,
        "grid": by_ctx,
        "num_cells": total,
        "num_retrieved": hits,
        "overall_accuracy": (hits / total) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------
def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model:
        raise ValueError("--model is required unless --self-test")
    if not args.tokenizer:
        raise ValueError("--tokenizer is required unless --self-test")
    context_lengths = args.context_len or list(DEFAULT_CONTEXT_LENGTHS)
    depths = args.depth or list(DEFAULT_DEPTHS)
    if args.max_context_len:
        context_lengths = [c for c in context_lengths if c <= args.max_context_len]
        if not context_lengths:
            raise ValueError(
                f"no context length is <= --max-context-len {args.max_context_len}"
            )
    tokenizer = load_tokenizer(args.tokenizer)
    needed = max(context_lengths)
    filler_tokens = build_filler_tokens(tokenizer, args.text_file, needed)

    report: dict[str, Any] = {
        "schema": "vllm-needle-retrieval/v1",
        "run_id": args.run_id,
        "mode": args.mode,
        "scope": args.scope,
        "url": args.url,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "kv_cache_dtype": args.kv_cache_dtype,
        "runtime_ref": args.runtime_ref,
        "container_image": args.container_image,
        "boot_profile_note": args.boot_profile_note,
        "seed": args.seed,
        "context_lengths": context_lengths,
        "depths": depths,
        "filler_source": (args.text_file or ["<synthetic-deterministic>"]),
        "filler_token_count": len(filler_tokens),
        "max_tokens": args.max_tokens,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": collect_cuda_hardware(),
        "cells": [],
        "ok": False,
    }

    if args.mode == "single":
        for ctx in context_lengths:
            for depth in depths:
                report["cells"].append(
                    run_single_cell(args, tokenizer, filler_tokens, ctx, depth)
                )
        report["grid"] = assemble_grid(report["cells"])
    elif args.mode == "multi":
        for ctx in context_lengths:
            report["cells"].append(
                run_multi_cell(args, tokenizer, filler_tokens, ctx, depths)
            )
        total = len(report["cells"])
        hits = sum(1 for c in report["cells"] if c["retrieved_all"])
        report["grid"] = {
            "context_lengths": context_lengths,
            "num_cells": total,
            "num_all_retrieved": hits,
            "overall_all_accuracy": (hits / total) if total else 0.0,
        }
    else:
        raise ValueError(f"unknown mode {args.mode!r}")

    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["ok"] = True
    return report


# ---------------------------------------------------------------------------
# Self-test: insertion + scoring + grid logic against a MOCK server (no network)
# ---------------------------------------------------------------------------
class _MockTokenizer:
    """Deterministic char-level mock tokenizer (no model download)."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


def run_self_test() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    tok = _MockTokenizer()

    # --- 1. Needle inserted at the right token offset (depth 0.5). ---
    filler = list(range(1000))  # 1000 distinct filler tokens
    needle = tok.encode(needle_sentence("vault-0", "12345"))
    ctx_len = 200
    ids, offsets = insert_needles_at_depths(filler, ctx_len, [needle], [0.5])
    filler_budget = ctx_len - len(needle)
    expected_offset = int(round(0.5 * filler_budget))
    check(
        "single_needle_offset",
        offsets[0] == expected_offset,
        {"got": offsets[0], "expected": expected_offset},
    )
    check("single_needle_total_len", len(ids) == ctx_len, {"len": len(ids)})
    # The needle tokens appear contiguously at the reported offset.
    check(
        "single_needle_contiguous",
        ids[offsets[0] : offsets[0] + len(needle)] == needle,
    )

    # --- 2. Depth 0.0 inserts at the front, depth 1.0 at the end. ---
    ids0, off0 = insert_needles_at_depths(filler, ctx_len, [needle], [0.0])
    ids1, off1 = insert_needles_at_depths(filler, ctx_len, [needle], [1.0])
    check("depth0_at_front", off0[0] == 0, {"offset": off0[0]})
    check(
        "depth1_at_end",
        off1[0] == ctx_len - len(needle),
        {"offset": off1[0], "expected": ctx_len - len(needle)},
    )

    # --- 3. Multi-needle: K facts, distinct offsets, ascending, all present. ---
    needles = [tok.encode(needle_sentence(f"vault-{k}", f"{k}{k}{k}{k}{k}")) for k in range(3)]
    depths = [0.1, 0.5, 0.9]
    mids, moffsets = insert_needles_at_depths(filler, 300, needles, depths)
    check("multi_total_len", len(mids) == 300, {"len": len(mids)})
    check("multi_offsets_ascending", moffsets == sorted(moffsets), {"offsets": moffsets})
    all_present = all(
        mids[moffsets[k] : moffsets[k] + len(needles[k])] == needles[k]
        for k in range(3)
    )
    check("multi_all_needles_present", all_present)

    # --- 4. Scoring catches right / wrong / missing. ---
    right = score_answer("the code is 48217", ["48217"])
    wrong = score_answer("the code is 99999", ["48217"])
    missing = score_answer("I could not find the code.", ["48217"])
    check("score_right", right["retrieved"] is True)
    check("score_wrong", wrong["retrieved"] is False)
    check("score_missing", missing["retrieved"] is False)

    # --- 5. Multi-needle strict recall: all-present True, any-missing False. ---
    multi_all = score_answer("codes: 11111, 22222, 33333", ["11111", "22222", "33333"])
    multi_partial = score_answer("codes: 11111, 22222", ["11111", "22222", "33333"])
    check("multi_score_all", multi_all["retrieved"] is True)
    check("multi_score_partial", multi_partial["retrieved"] is False)

    # --- 6. Grid assembles correctly from mock cells. ---
    mock_cells = [
        {"context_len": 1024, "depth": 0.0, "retrieved": True},
        {"context_len": 1024, "depth": 1.0, "retrieved": False},
        {"context_len": 2048, "depth": 0.0, "retrieved": True},
        {"context_len": 2048, "depth": 1.0, "retrieved": True},
    ]
    grid = assemble_grid(mock_cells)
    check("grid_context_lengths", grid["context_lengths"] == [1024, 2048])
    check("grid_cell_value", grid["grid"][1024]["1.0"] is False)
    check("grid_accuracy", abs(grid["overall_accuracy"] - 0.75) < 1e-9, grid["overall_accuracy"])

    # --- 7. Determinism: same cell coordinates -> same code, twice. ---
    c_a = cell_label_and_code(7, 8192, 0.5, 0)
    c_b = cell_label_and_code(7, 8192, 0.5, 0)
    c_diff = cell_label_and_code(7, 8192, 0.75, 0)
    check("code_deterministic", c_a == c_b, {"a": c_a, "b": c_b})
    check("code_varies_by_depth", c_a != c_diff, {"a": c_a, "diff": c_diff})

    # --- 8. chat_answer extracts content from a mock server response. ---
    mock_response = {"choices": [{"message": {"content": "The code is 48217."}}]}
    answer = chat_answer(mock_response)
    check("chat_answer_extract", answer == "The code is 48217.")
    end_to_end = score_answer(answer, ["48217"])
    check("end_to_end_mock_retrieved", end_to_end["retrieved"] is True)

    # --- 9. Too-small context raises (guard against silent truncation). ---
    raised = False
    try:
        insert_needles_at_depths(filler, len(needle) - 1, [needle], [0.5])
    except ValueError:
        raised = True
    check("small_context_raises", raised)

    ok = all(c["ok"] for c in checks)
    return {
        "schema": "vllm-needle-retrieval-self-test/v1",
        "ok": ok,
        "num_checks": len(checks),
        "num_failed": sum(1 for c in checks if not c["ok"]),
        "checks": checks,
    }


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--tokenizer")
    parser.add_argument(
        "--text-file",
        action="append",
        default=[],
        help="filler corpus file(s); omit to use the deterministic synthetic filler",
    )
    parser.add_argument(
        "--context-len",
        action="append",
        type=int,
        default=[],
        help="total context length(s) in tokens; defaults to the standard grid",
    )
    parser.add_argument(
        "--depth",
        action="append",
        type=float,
        default=[],
        help="needle depth(s) as context fraction 0.0..1.0; defaults to standard grid",
    )
    parser.add_argument(
        "--max-context-len",
        type=int,
        help="cap: drop grid context lengths above this (model max-model-len)",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
        help="single needle per cell, or RULER-style K-needle (one per depth) per ctx",
    )
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--run-id", default="vllm_needle_retrieval")
    parser.add_argument("--kv-cache-dtype", default="unknown")
    parser.add_argument("--runtime-ref", default="")
    parser.add_argument("--container-image", default="")
    parser.add_argument(
        "--boot-profile-note",
        default="",
        help="fp8 is per-boot bistable; record which profile this boot is on",
    )
    parser.add_argument(
        "--scope",
        default=(
            "chat-completions retrieval; temperature 0; deterministic filler; "
            "exact-code scoring; prefix caching disabled"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--request-attempts", type=int, default=3)
    parser.add_argument("--retry-sleep-s", type=float, default=10.0)
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    try:
        if args.self_test:
            report = run_self_test()
        else:
            report = run_probe(args)
    except (
        ImportError,
        OSError,
        TimeoutError,
        TypeError,
        ValueError,
        urllib.error.URLError,
        json.JSONDecodeError,
    ) as exc:
        report = {
            "schema": "vllm-needle-retrieval/v1",
            "run_id": args.run_id,
            "ok": False,
            "error": repr(exc),
        }
    report["elapsed_s"] = round(time.perf_counter() - started, 3)

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
