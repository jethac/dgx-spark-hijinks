#!/usr/bin/env python3
"""Run a small DiffusionGemma image prompt gate through OpenAI chat."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import struct
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any


def png_from_rgb(width: int, height: int, rgb_rows: list[list[tuple[int, int, int]]]) -> bytes:
    raw = bytearray()
    for row in rgb_rows:
        raw.append(0)
        for r, g, b in row:
            raw.extend((r, g, b))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk("IHDR".encode(), struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            chunk("IDAT".encode(), zlib.compress(bytes(raw), 9)),
            chunk("IEND".encode(), b""),
        ]
    )


def make_images() -> dict[str, dict[str, Any]]:
    width, height = 96, 64
    red = (235, 32, 42)
    blue = (32, 82, 210)
    green = (36, 170, 80)
    black = (0, 0, 0)

    red_blue_rows = []
    for _ in range(height):
        red_blue_rows.append([red if x < width // 2 else blue for x in range(width)])

    green_rows = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append(green if 16 <= x < 80 and 12 <= y < 52 else black)
        green_rows.append(row)

    specs = {
        "red_blue_halves": {
            "png": png_from_rgb(width, height, red_blue_rows),
            "prompt": (
                "Look at the image and describe its two main colors in one short phrase."
            ),
            "required_terms": ["red", "blue"],
        },
        "green_square": {
            "png": png_from_rgb(width, height, green_rows),
            "prompt": (
                "Look at the image and describe the main color of the square in one "
                "short phrase."
            ),
            "required_terms": ["green"],
        },
    }
    for spec in specs.values():
        spec["sha256"] = hashlib.sha256(spec["png"]).hexdigest()
        spec["data_url"] = "data:image/png;base64," + base64.b64encode(spec["png"]).decode("ascii")
    return specs


def request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return {"status": resp.status, "body": json.loads(resp.read().decode("utf-8"))}


def wait_ready(base_url: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            result = request_json(f"{base_url}/model_info", None, 5)
            if result["status"] == 200:
                return result["body"]
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
        time.sleep(2)
    raise TimeoutError(f"server did not become ready within {timeout_s}s: {last_error}")


def post_capture(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    started = time.time()
    try:
        result = request_json(url, payload, timeout)
        return {
            "ok": True,
            "elapsed_s": time.time() - started,
            "status": result["status"],
            "response": result["body"],
        }
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "elapsed_s": time.time() - started,
            "status": exc.code,
            "error": exc.read().decode("utf-8", errors="replace"),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "elapsed_s": time.time() - started, "error": repr(exc)}


def message_text(response: dict[str, Any]) -> str:
    if not response.get("ok"):
        return ""
    choices = response.get("response", {}).get("choices", [])
    if not choices:
        return ""
    return (choices[0].get("message", {}).get("content") or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30125")
    parser.add_argument("--model", default="google/diffusiongemma-26B-A4B-it")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--ready-timeout-s", type=float, default=1200)
    parser.add_argument("--request-timeout-s", type=float, default=900)
    args = parser.parse_args()

    model_info = wait_ready(args.base_url, args.ready_timeout_s)
    images = make_images()
    rows = []
    checks = []
    endpoint = f"{args.base_url}/v1/chat/completions"

    for image_id, spec in images.items():
        texts = []
        for repeat in range(args.repeats):
            payload = {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": spec["prompt"]},
                            {"type": "image_url", "image_url": {"url": spec["data_url"]}},
                        ],
                    }
                ],
                "max_tokens": 64,
                "temperature": 0.0,
            }
            response = post_capture(endpoint, payload, args.request_timeout_s)
            text = message_text(response)
            texts.append(text)
            request_for_record = dict(payload)
            request_for_record["messages"] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": spec["prompt"]},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,<sha256:{spec['sha256']}>"
                            },
                        },
                    ],
                }
            ]
            rows.append(
                {
                    "image_id": image_id,
                    "repeat": repeat,
                    "request": request_for_record,
                    "response": response,
                    "text": text,
                }
            )
        lower_texts = [text.lower() for text in texts]
        required_terms = spec["required_terms"]
        checks.append(
            {
                "image_id": image_id,
                "sha256": spec["sha256"],
                "required_terms": required_terms,
                "stable": len(set(texts)) == 1,
                "non_empty": all(bool(text) for text in texts),
                "answer_ok": all(
                    all(term in text for term in required_terms) for text in lower_texts
                ),
                "texts": texts,
            }
        )

    all_ok = all(
        check["stable"] and check["non_empty"] and check["answer_ok"]
        for check in checks
    )
    artifact = {
        "schema": "sglang-diffusiongemma-image-quality/v1",
        "base_url": args.base_url,
        "model": args.model,
        "model_info": model_info,
        "image_dimensions": {"width": 96, "height": 64},
        "repeats": args.repeats,
        "all_ok": all_ok,
        "checks": checks,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"all_ok": all_ok, "checks": checks}, indent=2, sort_keys=True))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
