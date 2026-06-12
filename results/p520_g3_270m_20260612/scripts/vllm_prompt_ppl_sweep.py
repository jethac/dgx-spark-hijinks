#!/usr/bin/env python3
"""Measure prompt-token PPL from a vLLM OpenAI completion server.

This scores supplied prompt tokens with vLLM's `prompt_logprobs` field. It is
intended for matched fp8-KV versus NVFP4-KV quality sweeps where the server
configuration is changed outside this script.
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


def key_matches_token_id(key: Any, token_id: int) -> bool:
    if key == token_id:
        return True
    if isinstance(key, str):
        if key == str(token_id):
            return True
        if key.startswith("token_id:") and key.removeprefix("token_id:") == str(token_id):
            return True
    return False


def extract_logprob(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, dict):
        return None
    raw = value.get("logprob")
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def supplied_token_logprob(entry: Any, token_id: int) -> float | None:
    if entry is None or not isinstance(entry, dict):
        return None
    for key, value in entry.items():
        if key_matches_token_id(key, token_id):
            return extract_logprob(value)
    return None


def read_text(paths: list[Path]) -> str:
    parts = []
    for path in paths:
        parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(parts)


def load_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def tokenize_text(tokenizer: Any, text: str, add_special_tokens: bool) -> list[int]:
    tokens = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if not isinstance(tokens, list) or not all(isinstance(tok, int) for tok in tokens):
        raise TypeError("tokenizer.encode did not return a list[int]")
    return tokens


def trim_tokens(tokens: list[int], ctx: int) -> list[int]:
    if ctx < 2:
        raise ValueError("context lengths must be at least 2 tokens")
    if len(tokens) < ctx:
        raise ValueError(f"text tokenized to {len(tokens)} tokens, shorter than ctx={ctx}")
    return tokens[:ctx]


def completion_payload(
    *,
    model: str,
    token_ids: list[int],
    max_tokens: int,
    prompt_logprobs: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": token_ids,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "prompt_logprobs": prompt_logprobs,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
    }


def score_response(response: dict[str, Any], expected_token_ids: list[int]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("completion response has no choices")
    choice = choices[0]
    prompt_logprobs = choice.get("prompt_logprobs")
    if not isinstance(prompt_logprobs, list):
        raise ValueError("choice has no prompt_logprobs list")
    returned_token_ids = choice.get("prompt_token_ids")
    if not isinstance(returned_token_ids, list):
        raise ValueError("choice has no prompt_token_ids list")
    if returned_token_ids != expected_token_ids:
        raise ValueError(
            "returned prompt_token_ids do not match requested token ids: "
            f"requested={len(expected_token_ids)} returned={len(returned_token_ids)}"
        )
    if len(prompt_logprobs) != len(expected_token_ids):
        raise ValueError(
            "prompt_logprobs length does not match prompt_token_ids length: "
            f"{len(prompt_logprobs)} vs {len(expected_token_ids)}"
        )

    scored = []
    missing_positions = []
    # Per-position logprobs aligned with expected_token_ids; position 0 is not
    # a language-modeling target and stays None, as do unrecoverable positions.
    token_logprobs: list[float | None] = [None]
    # The first prompt token has no previous context, so it is not a language
    # modeling target. Score positions 1..N-1.
    for index, token_id in enumerate(expected_token_ids[1:], start=1):
        logprob = supplied_token_logprob(prompt_logprobs[index], token_id)
        if logprob is None or not math.isfinite(logprob):
            missing_positions.append(index)
            token_logprobs.append(None)
            continue
        scored.append(logprob)
        token_logprobs.append(logprob)

    if not scored:
        raise ValueError("no supplied prompt tokens had recoverable logprobs")

    mean_logprob = sum(scored) / len(scored)
    mean_nll = -mean_logprob
    return {
        "ok": not missing_positions,
        "num_prompt_tokens": len(expected_token_ids),
        "num_scored_tokens": len(scored),
        "num_missing_tokens": len(missing_positions),
        "missing_positions_preview": missing_positions[:20],
        "mean_logprob": mean_logprob,
        "mean_nll_nats": mean_nll,
        "ppl": math.exp(mean_nll) if mean_nll < 700 else float("inf"),
        "usage": response.get("usage"),
        "token_logprobs": token_logprobs,
    }


def run_context(
    args: argparse.Namespace,
    token_ids: list[int],
    ctx: int,
    tokenizer: Any = None,
) -> dict[str, Any]:
    ctx_tokens = trim_tokens(token_ids, ctx)
    endpoint = args.url.rstrip("/") + "/v1/completions"
    payload = completion_payload(
        model=args.model,
        token_ids=ctx_tokens,
        max_tokens=args.max_tokens,
        prompt_logprobs=args.prompt_logprobs,
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
    score = score_response(response, ctx_tokens)
    token_logprobs = score.pop("token_logprobs")
    dump_path = None
    if args.dump_token_logprobs:
        dump_dir = Path(args.dump_token_logprobs)
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_path = str(dump_dir / f"{args.run_id}_ctx{ctx}_tokens.json")
        token_strs = (
            tokenizer.convert_ids_to_tokens(ctx_tokens) if tokenizer is not None else None
        )
        Path(dump_path).write_text(
            json.dumps(
                {
                    "schema": "vllm-prompt-token-logprobs/v1",
                    "run_id": args.run_id,
                    "ctx": ctx,
                    "model": args.model,
                    "kv_cache_dtype": args.kv_cache_dtype,
                    "container_image": args.container_image,
                    "mean_logprob": score["mean_logprob"],
                    "token_ids": ctx_tokens,
                    "token_strs": token_strs,
                    "token_logprobs": token_logprobs,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "token_logprobs_dump": dump_path,
        "ctx": ctx,
        "endpoint": endpoint,
        "elapsed_s": elapsed_s,
        "payload": {
            "model": args.model,
            "prompt_token_count": len(ctx_tokens),
            "max_tokens": args.max_tokens,
            "prompt_logprobs": args.prompt_logprobs,
            "temperature": 0,
            "stream": False,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
        },
        "score": score,
    }


def summarize_pair(fp8: dict[str, Any], nvfp4: dict[str, Any]) -> list[dict[str, Any]]:
    fp8_by_ctx = {row["ctx"]: row for row in fp8.get("contexts", [])}
    rows = []
    for nv_row in nvfp4.get("contexts", []):
        ctx = nv_row["ctx"]
        fp8_row = fp8_by_ctx.get(ctx)
        if not fp8_row:
            continue
        fp8_score = fp8_row["score"]
        nv_score = nv_row["score"]
        rows.append(
            {
                "ctx": ctx,
                "ppl_fp8": fp8_score["ppl"],
                "ppl_nvfp4": nv_score["ppl"],
                "delta_ppl": nv_score["ppl"] - fp8_score["ppl"],
                "nats_per_token_fp8": fp8_score["mean_nll_nats"],
                "nats_per_token_nvfp4": nv_score["mean_nll_nats"],
                "delta_nats_per_token": nv_score["mean_nll_nats"]
                - fp8_score["mean_nll_nats"],
                "fp8_ok": fp8_score["ok"],
                "nvfp4_ok": nv_score["ok"],
            }
        )
    return rows


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model:
        raise ValueError("--model is required unless --self-test or --compare-* is used")
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
        "schema": "vllm-prompt-ppl-sweep/v1",
        "run_id": args.run_id,
        "scope": args.scope,
        "url": args.url,
        "model": args.model,
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
        report["contexts"].append(run_context(args, token_ids, ctx, tokenizer=tokenizer))
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["ok"] = all(row["score"]["ok"] for row in report["contexts"])
    return report


def run_self_test() -> dict[str, Any]:
    token_ids = [11, 22, 33]
    response = {
        "choices": [
            {
                "prompt_token_ids": token_ids,
                "prompt_logprobs": [
                    None,
                    {"22": {"logprob": -0.25}},
                    {"token_id:33": {"logprob": -1.25}},
                ],
            }
        ],
        "usage": {"prompt_tokens": 3},
    }
    score = score_response(response, token_ids)
    missing_response = {
        "choices": [
            {
                "prompt_token_ids": token_ids,
                "prompt_logprobs": [None, {"99": {"logprob": -0.25}}, {"33": -1.25}],
            }
        ]
    }
    missing_score = score_response(missing_response, token_ids)
    return {
        "schema": "vllm-prompt-ppl-sweep-self-test/v1",
        "ok": score["ok"] and not missing_score["ok"],
        "score": score,
        "missing_score": missing_score,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--text-file", action="append", default=[])
    parser.add_argument("--ctx", action="append", type=int, default=[])
    parser.add_argument("--run-id", default="vllm_prompt_ppl_sweep")
    parser.add_argument("--kv-cache-dtype", default="unknown")
    parser.add_argument("--runtime-ref", default="")
    parser.add_argument("--container-image", default="")
    parser.add_argument(
        "--scope",
        default=(
            "clean full-attention path; prefix caching disabled; Gemma/SWA and "
            "SGLang radix reuse paths excluded"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--prompt-logprobs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--request-attempts", type=int, default=3)
    parser.add_argument("--retry-sleep-s", type=float, default=10.0)
    parser.add_argument("--add-special-tokens", action="store_true")
    parser.add_argument(
        "--dump-token-logprobs",
        help="directory for per-context per-token logprob dumps (stratification arm)",
    )
    parser.add_argument("--output")
    parser.add_argument("--compare-fp8")
    parser.add_argument("--compare-nvfp4")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    try:
        if args.self_test:
            report = run_self_test()
        elif args.compare_fp8 and args.compare_nvfp4:
            fp8 = json.loads(Path(args.compare_fp8).read_text(encoding="utf-8"))
            nvfp4 = json.loads(Path(args.compare_nvfp4).read_text(encoding="utf-8"))
            report = {
                "schema": "vllm-prompt-ppl-comparison/v1",
                "fp8_report": args.compare_fp8,
                "nvfp4_report": args.compare_nvfp4,
                "rows": summarize_pair(fp8, nvfp4),
            }
            report["ok"] = bool(report["rows"]) and all(
                row["fp8_ok"] and row["nvfp4_ok"] for row in report["rows"]
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
            "schema": "vllm-prompt-ppl-sweep/v1",
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
