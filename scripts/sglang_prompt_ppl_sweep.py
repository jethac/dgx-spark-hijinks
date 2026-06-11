#!/usr/bin/env python3
"""Measure supplied prompt-token PPL from an SGLang native /generate server.

SGLang's OpenAI endpoints expose generated-token logprobs, but its native
`/generate` endpoint can return input-token logprobs. This script uses token-ID
prompts so the scored token IDs are exact and independent of chat templates.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from spark_hardware import collect_cuda_hardware


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


def read_text(paths: list[Path]) -> str:
    return "\n\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in paths
    )


def load_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def tokenize_text(tokenizer: Any, text: str, add_special_tokens: bool) -> list[int]:
    tokens = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if not isinstance(tokens, list) or not all(isinstance(tok, int) for tok in tokens):
        raise TypeError("tokenizer.encode did not return list[int]")
    return tokens


def trim_tokens(tokens: list[int], ctx: int) -> list[int]:
    if ctx < 2:
        raise ValueError("context lengths must be at least 2 tokens")
    if len(tokens) < ctx:
        raise ValueError(f"text tokenized to {len(tokens)} tokens, shorter than ctx={ctx}")
    return tokens[:ctx]


def extract_logprob_entry(entry: Any) -> tuple[float | None, int | None]:
    if entry is None:
        return None, None
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        logprob, token_id = entry[0], entry[1]
        if isinstance(logprob, (int, float)) and isinstance(token_id, int):
            return float(logprob), token_id
    if isinstance(entry, dict):
        logprob = entry.get("logprob")
        token_id = entry.get("token_id")
        if isinstance(logprob, (int, float)) and isinstance(token_id, int):
            return float(logprob), token_id
    return None, None


def response_meta(response: dict[str, Any]) -> dict[str, Any]:
    if "meta_info" in response and isinstance(response["meta_info"], dict):
        return response["meta_info"]
    if "meta_info" in response and isinstance(response["meta_info"], list):
        if response["meta_info"] and isinstance(response["meta_info"][0], dict):
            return response["meta_info"][0]
    raise ValueError("SGLang response has no meta_info object")


def score_response(
    response: dict[str, Any],
    expected_token_ids: list[int],
    *,
    score_start_index: int,
    include_token_logprobs: bool = False,
) -> dict[str, Any]:
    meta = response_meta(response)
    input_logprobs = meta.get("input_token_logprobs")
    if not isinstance(input_logprobs, list):
        raise ValueError("meta_info has no input_token_logprobs list")

    expected_scored_token_ids = expected_token_ids[score_start_index:]
    relative_entries = False
    if len(input_logprobs) == len(expected_token_ids):
        entries = input_logprobs[score_start_index:]
    elif len(input_logprobs) == len(expected_scored_token_ids):
        entries = input_logprobs
        relative_entries = True
    else:
        raise ValueError(
            "input_token_logprobs length does not match prompt length or requested "
            "logprob span: "
            f"{len(input_logprobs)} vs full={len(expected_token_ids)} "
            f"span={len(expected_scored_token_ids)}"
        )

    scored: list[float] = []
    missing_positions: list[int] = []
    mismatched_positions: list[dict[str, int | None]] = []
    skipped_positions: list[int] = []
    token_logprobs: list[dict[str, int | float]] = []

    for rel_index, expected_token_id in enumerate(expected_scored_token_ids):
        abs_index = score_start_index + rel_index
        logprob, returned_token_id = extract_logprob_entry(entries[rel_index])
        if (
            rel_index == 0
            and returned_token_id is None
            and logprob is None
            and score_start_index > 0
        ):
            # SGLang's native input-logprob span can include a leading boundary
            # placeholder when logprob_start_len is nonzero. That token has no
            # returned target id/logprob, so it is not a scored continuation.
            skipped_positions.append(abs_index)
            continue
        if returned_token_id != expected_token_id:
            mismatched_positions.append(
                {
                    "position": abs_index,
                    "expected": expected_token_id,
                    "returned": returned_token_id,
                }
            )
            continue
        if logprob is None or not math.isfinite(logprob):
            missing_positions.append(abs_index)
            continue
        scored.append(logprob)
        if include_token_logprobs:
            token_logprobs.append(
                {
                    "position": abs_index,
                    "token_id": returned_token_id,
                    "logprob": logprob,
                }
            )

    if not scored:
        raise ValueError("no supplied prompt tokens had recoverable logprobs")

    mean_logprob = sum(scored) / len(scored)
    mean_nll = -mean_logprob
    result: dict[str, Any] = {
        "ok": not missing_positions and not mismatched_positions,
        "num_prompt_tokens": len(expected_token_ids),
        "score_start_index": score_start_index,
        "num_scored_tokens": len(scored),
        "num_skipped_boundary_tokens": len(skipped_positions),
        "skipped_boundary_positions": skipped_positions[:20],
        "relative_logprob_entries": relative_entries,
        "num_missing_tokens": len(missing_positions),
        "num_mismatched_tokens": len(mismatched_positions),
        "missing_positions_preview": missing_positions[:20],
        "mismatched_positions_preview": mismatched_positions[:20],
        "mean_logprob": mean_logprob,
        "mean_nll_nats": mean_nll,
        "ppl": math.exp(mean_nll) if mean_nll < 700 else float("inf"),
        "usage": meta.get("usage"),
        "cached_tokens": meta.get("cached_tokens"),
        "finish_reason": meta.get("finish_reason"),
    }
    if include_token_logprobs:
        result["token_logprobs"] = token_logprobs
    return result


def generate_payload(
    *,
    token_ids: list[int],
    max_new_tokens: int,
    logprob_start_len: int,
) -> dict[str, Any]:
    return {
        "input_ids": token_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
        "return_logprob": True,
        "logprob_start_len": logprob_start_len,
        "top_logprobs_num": 0,
        "return_text_in_logprobs": False,
        "stream": False,
    }


def warm_prefix(args: argparse.Namespace, token_ids: list[int], prefix_len: int) -> dict[str, Any]:
    if prefix_len <= 0:
        return {"enabled": False}
    if prefix_len >= len(token_ids):
        raise ValueError(
            f"--reuse-prefix-len must be smaller than ctx ({prefix_len} >= {len(token_ids)})"
        )
    endpoint = args.url.rstrip("/") + "/generate"
    payload = {
        "input_ids": token_ids[:prefix_len],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": args.warmup_max_new_tokens,
        },
        "return_logprob": False,
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
    meta = response_meta(response)
    return {
        "enabled": True,
        "prefix_len": prefix_len,
        "elapsed_s": elapsed_s,
        "cached_tokens": meta.get("cached_tokens"),
        "finish_reason": meta.get("finish_reason"),
    }


def run_context(args: argparse.Namespace, token_ids: list[int], ctx: int) -> dict[str, Any]:
    ctx_tokens = trim_tokens(token_ids, ctx)
    endpoint = args.url.rstrip("/") + "/generate"
    warmup = warm_prefix(args, ctx_tokens, args.reuse_prefix_len)
    score_start_index = max(1, args.logprob_start_len)
    payload = generate_payload(
        token_ids=ctx_tokens,
        max_new_tokens=args.max_new_tokens,
        logprob_start_len=args.logprob_start_len,
    )
    started = time.perf_counter()
    response = post_json(
        endpoint,
        payload,
        args.timeout,
        attempts=args.request_attempts,
        retry_sleep_s=args.retry_sleep_s,
    )
    elapsed_s = time.perf_counter() - started
    score = score_response(
        response,
        ctx_tokens,
        score_start_index=score_start_index,
        include_token_logprobs=args.include_token_logprobs,
    )
    return {
        "ctx": ctx,
        "endpoint": endpoint,
        "elapsed_s": elapsed_s,
        "warmup": warmup,
        "payload": {
            "prompt_token_count": len(ctx_tokens),
            "max_new_tokens": args.max_new_tokens,
            "logprob_start_len": args.logprob_start_len,
            "return_logprob": True,
            "top_logprobs_num": 0,
            "reuse_prefix_len": args.reuse_prefix_len,
            "score_start_index": score_start_index,
        },
        "score": score,
    }


def summarize_pair(fp8: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    fp8_by_ctx = {row["ctx"]: row for row in fp8.get("contexts", [])}
    rows = []
    for cand_row in candidate.get("contexts", []):
        ctx = cand_row["ctx"]
        fp8_row = fp8_by_ctx.get(ctx)
        if not fp8_row:
            continue
        fp8_score = fp8_row["score"]
        cand_score = cand_row["score"]
        rows.append(
            {
                "ctx": ctx,
                "ppl_fp8": fp8_score["ppl"],
                "ppl_candidate": cand_score["ppl"],
                "delta_ppl": cand_score["ppl"] - fp8_score["ppl"],
                "nats_per_token_fp8": fp8_score["mean_nll_nats"],
                "nats_per_token_candidate": cand_score["mean_nll_nats"],
                "delta_nats_per_token": cand_score["mean_nll_nats"]
                - fp8_score["mean_nll_nats"],
                "fp8_ok": fp8_score["ok"],
                "candidate_ok": cand_score["ok"],
            }
        )
    return rows


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    if not args.tokenizer:
        raise ValueError("--tokenizer is required unless --self-test or --compare-* is used")
    if not args.text_file:
        raise ValueError("--text-file is required unless --self-test or --compare-* is used")
    if not args.ctx:
        raise ValueError("--ctx is required unless --self-test or --compare-* is used")

    tokenizer = load_tokenizer(args.tokenizer)
    text = read_text([Path(path) for path in args.text_file])
    token_ids = tokenize_text(tokenizer, text, args.add_special_tokens)
    report: dict[str, Any] = {
        "schema": "sglang-prompt-ppl-sweep/v1",
        "run_id": args.run_id,
        "scope": args.scope,
        "url": args.url,
        "tokenizer": args.tokenizer,
        "kv_cache_dtype": args.kv_cache_dtype,
        "runtime_ref": args.runtime_ref,
        "container_image": args.container_image,
        "text_files": args.text_file,
        "text_chars": len(text),
        "available_tokens": len(token_ids),
        "add_special_tokens": args.add_special_tokens,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": collect_cuda_hardware(),
        "contexts": [],
        "ok": False,
    }
    for ctx in args.ctx:
        report["contexts"].append(run_context(args, token_ids, ctx))
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["ok"] = all(row["score"]["ok"] for row in report["contexts"])
    return report


def run_self_test() -> dict[str, Any]:
    token_ids = [11, 22, 33]
    response = {
        "meta_info": {
            "input_token_logprobs": [
                None,
                [-0.25, 22, None],
                [-1.25, 33, None],
            ]
        }
    }
    score = score_response(response, token_ids, score_start_index=1)
    missing_response = {
        "meta_info": {
            "input_token_logprobs": [
                None,
                [-0.25, 99, None],
                [-1.25, 33, None],
            ]
        }
    }
    missing_score = score_response(missing_response, token_ids, score_start_index=1)
    return {
        "schema": "sglang-prompt-ppl-sweep-self-test/v1",
        "ok": score["ok"] and not missing_score["ok"],
        "score": score,
        "missing_score": missing_score,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--tokenizer")
    parser.add_argument("--text-file", action="append", default=[])
    parser.add_argument("--ctx", action="append", type=int, default=[])
    parser.add_argument("--run-id", default="sglang_prompt_ppl_sweep")
    parser.add_argument("--kv-cache-dtype", default="unknown")
    parser.add_argument("--runtime-ref", default="")
    parser.add_argument("--container-image", default="")
    parser.add_argument(
        "--scope",
        default=(
            "SGLang native /generate supplied-token prompt-logprob PPL; "
            "matched fp8 versus mixed-KV sequential servers"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--warmup-max-new-tokens", type=int, default=1)
    parser.add_argument("--logprob-start-len", type=int, default=0)
    parser.add_argument("--reuse-prefix-len", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--request-attempts", type=int, default=3)
    parser.add_argument("--retry-sleep-s", type=float, default=10.0)
    parser.add_argument("--add-special-tokens", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--compare-fp8")
    parser.add_argument("--compare-candidate")
    parser.add_argument("--include-token-logprobs", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    try:
        if args.self_test:
            report = run_self_test()
        elif args.compare_fp8 and args.compare_candidate:
            fp8 = json.loads(Path(args.compare_fp8).read_text(encoding="utf-8"))
            candidate = json.loads(
                Path(args.compare_candidate).read_text(encoding="utf-8")
            )
            report = {
                "schema": "sglang-prompt-ppl-comparison/v1",
                "fp8_report": args.compare_fp8,
                "candidate_report": args.compare_candidate,
                "rows": summarize_pair(fp8, candidate),
            }
            report["ok"] = bool(report["rows"]) and all(
                row["fp8_ok"] and row["candidate_ok"] for row in report["rows"]
            )
        else:
            report = run_sweep(args)
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
            "schema": "sglang-prompt-ppl-sweep/v1",
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
