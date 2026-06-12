#!/usr/bin/env python3
"""Capture or compare SGLang servers for a greedy MTP identity gate.

The zero-bug gate is token identity. Capture mode records both OpenAI chat
responses and native /generate responses for one already-running server. Compare
mode compares two captured artifacts, so the two heavyweight servers can run
sequentially under GB10 memory guardrails.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROMPTS = [
    {
        "id": "capital_japan",
        "text": "In one short sentence, name the capital of Japan.",
        "max_tokens": 24,
    },
    {
        "id": "arithmetic",
        "text": "Answer with one sentence: what is 2 + 2?",
        "max_tokens": 24,
    },
    {
        "id": "spark_use",
        "text": "In one short sentence, say what a DGX Spark is useful for.",
        "max_tokens": 32,
    },
]


def request_json(
    url: str, payload: dict[str, Any] | None, timeout: float, method: str = "POST"
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return {"status": resp.status, "body": json.loads(resp.read().decode("utf-8"))}


def post_capture(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    started = time.time()
    try:
        result = request_json(url, payload, timeout)
        return {
            "ok": True,
            "elapsed_s": round(time.time() - started, 3),
            "status": result["status"],
            "response": result["body"],
        }
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "elapsed_s": round(time.time() - started, 3),
            "status": exc.code,
            "error": exc.read().decode("utf-8", errors="replace"),
        }
    except Exception as exc:  # noqa: BLE001 - artifact preserves raw failure.
        return {"ok": False, "elapsed_s": round(time.time() - started, 3), "error": repr(exc)}


def wait_ready(base_url: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            result = request_json(f"{base_url}/model_info", None, 5, method="GET")
            if result["status"] == 200:
                return result["body"]
        except Exception as exc:  # noqa: BLE001 - readiness loop.
            last_error = repr(exc)
        time.sleep(2)
    raise TimeoutError(f"server did not become ready within {timeout_s}s: {last_error}")


def chat_text(response: dict[str, Any]) -> str:
    choices = response.get("response", {}).get("choices", [])
    if not choices:
        return ""
    return (choices[0].get("message", {}).get("content") or "").strip()


def native_text(response: dict[str, Any]) -> str:
    body = response.get("response", {})
    if isinstance(body.get("text"), str):
        return body["text"].strip()
    if isinstance(body.get("text"), list) and body["text"]:
        return str(body["text"][0]).strip()
    return ""


def find_token_ids(obj: Any) -> list[int] | None:
    """Best-effort recursive token-id extraction across SGLang schemas."""
    keys = {
        "output_ids",
        "output_token_ids",
        "token_ids",
        "tokens",
        "completion_token_ids",
        "generated_token_ids",
    }
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in keys and isinstance(value, list) and all(
                isinstance(item, int) for item in value
            ):
                return value
            if key == "output_token_logprobs" and isinstance(value, list):
                ids: list[int] = []
                for item in value:
                    if (
                        isinstance(item, (list, tuple))
                        and len(item) >= 2
                        and isinstance(item[1], int)
                    ):
                        ids.append(item[1])
                    else:
                        ids = []
                        break
                if ids:
                    return ids
        for value in obj.values():
            found = find_token_ids(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_token_ids(value)
            if found is not None:
                return found
    return None


def run_one_server(
    label: str,
    base_url: str,
    model: str,
    ready_timeout_s: float,
    request_timeout_s: float,
) -> dict[str, Any]:
    model_info = wait_ready(base_url, ready_timeout_s)
    rows = []
    for prompt in PROMPTS:
        chat_payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt["text"]}],
            "temperature": 0.0,
            "max_tokens": prompt["max_tokens"],
            "logprobs": True,
            "top_logprobs": 0,
        }
        native_payload = {
            "text": prompt["text"],
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": prompt["max_tokens"],
            },
            "return_logprob": True,
            "return_text_in_logprobs": True,
            "top_logprobs_num": 0,
        }
        chat = post_capture(
            f"{base_url}/v1/chat/completions", chat_payload, request_timeout_s
        )
        native = post_capture(f"{base_url}/generate", native_payload, request_timeout_s)
        rows.append(
            {
                "prompt_id": prompt["id"],
                "prompt": prompt["text"],
                "chat_request": chat_payload,
                "chat_response": chat,
                "chat_text": chat_text(chat),
                "chat_token_ids": find_token_ids(chat),
                "native_request": native_payload,
                "native_response": native,
                "native_text": native_text(native),
                "native_token_ids": find_token_ids(native),
            }
        )
    return {"label": label, "base_url": base_url, "model_info": model_info, "rows": rows}


def compare_runs(spec_off: dict[str, Any], spec_on: dict[str, Any]) -> dict[str, Any]:
    checks = []
    any_token_ids = False
    for off, on in zip(spec_off["rows"], spec_on["rows"]):
        chat_text_match = bool(off["chat_text"]) and off["chat_text"] == on["chat_text"]
        native_text_match = bool(off["native_text"]) and off["native_text"] == on["native_text"]
        chat_ids_match = None
        native_ids_match = None
        if off["chat_token_ids"] is not None or on["chat_token_ids"] is not None:
            any_token_ids = True
            chat_ids_match = off["chat_token_ids"] == on["chat_token_ids"]
        if off["native_token_ids"] is not None or on["native_token_ids"] is not None:
            any_token_ids = True
            native_ids_match = off["native_token_ids"] == on["native_token_ids"]
        checks.append(
            {
                "prompt_id": off["prompt_id"],
                "chat_text_match": chat_text_match,
                "native_text_match": native_text_match,
                "chat_token_ids_match": chat_ids_match,
                "native_token_ids_match": native_ids_match,
                "off_chat_text": off["chat_text"],
                "on_chat_text": on["chat_text"],
                "off_native_text": off["native_text"],
                "on_native_text": on["native_text"],
            }
        )
    token_identity_checks = [
        check
        for check in checks
        if check["chat_token_ids_match"] is not None
        or check["native_token_ids_match"] is not None
    ]
    token_identity_ok = bool(token_identity_checks) and all(
        (
            check["chat_token_ids_match"] is not False
            and check["native_token_ids_match"] is not False
        )
        for check in token_identity_checks
    )
    text_identity_ok = all(
        check["chat_text_match"] and check["native_text_match"] for check in checks
    )
    return {
        "checks": checks,
        "any_token_ids": any_token_ids,
        "text_identity_ok": text_identity_ok,
        "token_identity_ok": token_identity_ok,
        "all_ok": text_identity_ok and token_identity_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["capture", "compare"], required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--label")
    parser.add_argument("--model")
    parser.add_argument("--spec-off-artifact", type=Path)
    parser.add_argument("--spec-on-artifact", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ready-timeout-s", type=float, default=1200)
    parser.add_argument("--request-timeout-s", type=float, default=300)
    args = parser.parse_args()

    if args.mode == "capture":
        if not args.base_url or not args.label or not args.model:
            raise SystemExit("capture mode requires --base-url, --label, and --model")
        artifact = {
            "schema": "sglang-mtp-identity-capture/v1",
            "model": args.model,
            "run": run_one_server(
                args.label,
                args.base_url.rstrip("/"),
                args.model,
                args.ready_timeout_s,
                args.request_timeout_s,
            ),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"label": args.label, "rows": len(artifact["run"]["rows"])}, sort_keys=True))
        return

    if not args.spec_off_artifact or not args.spec_on_artifact:
        raise SystemExit("compare mode requires --spec-off-artifact and --spec-on-artifact")
    spec_off_capture = json.loads(args.spec_off_artifact.read_text(encoding="utf-8"))
    spec_on_capture = json.loads(args.spec_on_artifact.read_text(encoding="utf-8"))
    spec_off = spec_off_capture["run"]
    spec_on = spec_on_capture["run"]
    comparison = compare_runs(spec_off, spec_on)
    model = spec_off_capture.get("model") or spec_on_capture.get("model")
    artifact = {
        "schema": "sglang-mtp-identity-comparison/v1",
        "model": model,
        "spec_off_artifact": str(args.spec_off_artifact),
        "spec_on_artifact": str(args.spec_on_artifact),
        "spec_off": spec_off,
        "spec_on": spec_on,
        "comparison": comparison,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2, sort_keys=True))
    if not comparison["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
