#!/usr/bin/env python3
"""OpenAI-compatible Gemma 4 multimodal probe for SGLang serving rows."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_IMAGE_URL = (
    "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/"
    "images/man_ironing_on_back_of_suv.png"
)
DEFAULT_AUDIO_SERVER_PATH = (
    "/hijinks/results/p520_audio_mm_20260612/assets/"
    "speech_librispeech_1272-128104-0000.wav"
)
DEFAULT_AUDIO_TRANSCRIPT = (
    "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO "
    "WELCOME HIS GOSPEL"
)
DEFAULT_TEXT_PROMPT = "Answer with only the capital city of Japan."
DEFAULT_IMAGE_PROMPT = (
    "Describe the image in one short sentence. Mention the main object or activity."
)
DEFAULT_AUDIO_PROMPT = (
    "Transcribe this audio in English. Keep the original words as closely as possible."
)
DEFAULT_TEXT_KEYWORDS = ["tokyo"]
DEFAULT_IMAGE_KEYWORDS = ["iron", "car", "suv", "vehicle", "person"]
DEFAULT_AUDIO_KEYWORDS = ["quilter", "apostle", "middle", "gospel"]


def _read_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _post_json(url: str, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    elapsed_s = time.monotonic() - started
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {"raw_text": text}
    return {"status": status, "elapsed_s": elapsed_s, "body": parsed}


def _content_text(response: dict[str, Any]) -> str:
    body = response.get("body", {})
    try:
        return body["choices"][0]["message"]["content"] or ""
    except Exception:
        return ""


def _usage(response: dict[str, Any]) -> dict[str, Any] | None:
    body = response.get("body", {})
    usage = body.get("usage") if isinstance(body, dict) else None
    return usage if isinstance(usage, dict) else None


def _chat_payload(model: str, content: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def _keyword_hit(text: str, keywords: list[str], mode: str) -> bool:
    lower = text.lower()
    checks = [keyword.lower() in lower for keyword in keywords]
    return all(checks) if mode == "all" else any(checks)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    image_ref = _read_data_uri(args.image_path) if args.image_path else args.image_url
    audio_ref = args.audio_url
    rows = []

    probes: list[tuple[str, list[dict[str, Any]], list[str]]] = []
    if args.include_text:
        probes.append(
            (
                "text",
                [{"type": "text", "text": args.text_prompt}],
                args.text_keywords,
            )
        )
    if args.include_image:
        probes.append(
            (
                "image",
                [
                    {"type": "image_url", "image_url": {"url": image_ref}},
                    {"type": "text", "text": args.image_prompt},
                ],
                args.image_keywords,
            )
        )
    if args.include_audio:
        probes.append(
            (
                "audio",
                [
                    {"type": "audio_url", "audio_url": {"url": audio_ref}},
                    {"type": "text", "text": args.audio_prompt},
                ],
                args.audio_keywords,
            )
        )

    for label, content, keywords in probes:
        outputs = []
        for round_idx in range(args.repeat):
            payload = _chat_payload(args.model, content, args.max_tokens)
            response = _post_json(endpoint, payload, args.timeout_s)
            text = _content_text(response)
            outputs.append(
                {
                    "round": round_idx + 1,
                    "response": response,
                    "text": text,
                    "keyword_hit": _keyword_hit(text, keywords, args.keyword_mode),
                    "usage": _usage(response),
                }
            )
        rows.append(
            {
                "label": label,
                "keywords": keywords,
                "outputs": outputs,
                "all_http_200": all(item["response"]["status"] == 200 for item in outputs),
                "all_keyword_hit": all(item["keyword_hit"] for item in outputs),
                "content_equal": len({item["text"] for item in outputs}) == 1,
            }
        )

    return {
        "schema": "sglang-gemma4-multimodal-chat-probe/v1",
        "url": args.url,
        "model": args.model,
        "repeat": args.repeat,
        "image_ref_kind": "path_data_uri" if args.image_path else "url",
        "image_path": str(args.image_path) if args.image_path else None,
        "image_url": args.image_url if not args.image_path else None,
        "text_prompt": args.text_prompt,
        "image_prompt": args.image_prompt,
        "audio_prompt": args.audio_prompt,
        "keyword_mode": args.keyword_mode,
        "audio_url": audio_ref,
        "audio_reference_transcript": args.audio_reference_transcript,
        "rows": rows,
        "ok": all(row["all_http_200"] for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--image-path", type=Path)
    parser.add_argument("--audio-url", default=DEFAULT_AUDIO_SERVER_PATH)
    parser.add_argument("--audio-reference-transcript", default=DEFAULT_AUDIO_TRANSCRIPT)
    parser.add_argument("--text-prompt", default=DEFAULT_TEXT_PROMPT)
    parser.add_argument("--image-prompt", default=DEFAULT_IMAGE_PROMPT)
    parser.add_argument("--audio-prompt", default=DEFAULT_AUDIO_PROMPT)
    parser.add_argument(
        "--text-keywords",
        type=_split_csv,
        default=DEFAULT_TEXT_KEYWORDS,
        help="Comma-separated keyword substrings; any hit passes the text row.",
    )
    parser.add_argument(
        "--image-keywords",
        type=_split_csv,
        default=DEFAULT_IMAGE_KEYWORDS,
        help="Comma-separated keyword substrings; any hit passes the image row.",
    )
    parser.add_argument(
        "--audio-keywords",
        type=_split_csv,
        default=DEFAULT_AUDIO_KEYWORDS,
        help="Comma-separated keyword substrings; any hit passes the audio row.",
    )
    parser.add_argument(
        "--keyword-mode",
        choices=["any", "all"],
        default="any",
        help="Whether each row requires any listed keyword or all listed keywords.",
    )
    parser.add_argument("--no-text", dest="include_text", action="store_false")
    parser.add_argument("--no-image", dest="include_image", action="store_false")
    parser.add_argument("--no-audio", dest="include_audio", action="store_false")
    parser.set_defaults(include_text=True, include_image=True, include_audio=True)
    args = parser.parse_args()

    result = run_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(json.dumps({"ok": result["ok"], "rows": len(result["rows"])}, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
