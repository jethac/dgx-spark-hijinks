#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sglang-commit", required=True)
    parser.add_argument("--flashinfer-commit", required=True)
    args = parser.parse_args()

    out = Path(args.out_dir)
    server_log = (out / "server.log").read_text(encoding="utf-8", errors="replace")
    request_path = out / "generate.json"
    request_status_path = out / "request_status.txt"

    try:
        response = json.loads(request_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raw = (
            request_path.read_text(encoding="utf-8", errors="replace")
            if request_path.exists()
            else ""
        )
        response = {"parse_error": repr(exc), "raw": raw}

    try:
        request_status = int(request_status_path.read_text(encoding="utf-8").strip())
    except Exception:
        request_status = None

    text = json.dumps(response, ensure_ascii=False)
    coherent = "Tokyo" in text or "東京" in text
    geometry_lines = [
        line
        for line in server_log.splitlines()
        if "SGLang Gemma4 FlashInfer geometry" in line
    ]
    wrapper_lines = [
        line
        for line in server_log.splitlines()
        if "SGLang FlashInfer wrapper geometries" in line
    ]
    has_binary = "binary_md5 " in server_log
    has_source = "/flashinfer-src" in server_log
    has_vosplit = (
        "SGLang FlashInfer VO split enabled" in server_log
        and "extend_paged_vosplit" in server_log
    )
    unsupported = "Unsupported max_mma_kv: 0" in server_log
    decode_as_prefill_d512 = [
        line
        for line in geometry_lines
        if "label=decode_as_prefill" in line and "head_dim=512" in line
    ]
    standard_decode_d512 = [
        line
        for line in geometry_lines
        if "label=decode " in line and "head_dim=512" in line
    ]
    decode_d512 = [
        line
        for line in geometry_lines
        if "label=decode" in line and "head_dim=512" in line
    ]
    prefill_global = [line for line in geometry_lines if "extend_paged_vosplit" in line]
    prefill_swa = [
        line
        for line in geometry_lines
        if "label=extend_paged " in line and "head_dim=256" in line
    ]

    status = (
        "GREEN"
        if request_status == 0
        and coherent
        and geometry_lines
        and wrapper_lines
        and has_binary
        and has_source
        and has_vosplit
        and not unsupported
        else "RED"
    )

    summary = [
        "# SGLang Gemma 4 E4B Rung 0 Smoke",
        "",
        f"Status: {status}",
        "",
        f"- Run: `{args.run_id}`",
        f"- Model: `{args.model}`",
        f"- SGLang commit: `{args.sglang_commit}`",
        f"- FlashInfer commit: `{args.flashinfer_commit}`",
        f"- VO split requested: `{has_vosplit}`",
        f"- Geometry lines: `{len(geometry_lines)}`",
        f"- Wrapper geometry lines: `{len(wrapper_lines)}`",
        f"- Binary proof lines present: `{has_binary}`",
        f"- FlashInfer source paths present: `{has_source}`",
        f"- Request curl status: `{request_status}`",
        f"- Unsupported max_mma_kv: `{unsupported}`",
        f"- Coherent Tokyo answer: `{coherent}`",
        "",
        "## Result",
        "",
    ]
    if unsupported:
        if decode_as_prefill_d512 and not standard_decode_d512:
            summary += [
                "The SGLang Gemma 4 E4B text-only smoke remains RED, but the SGLang-side D=512 decode routing is now proven.",
                "SWA prefill plans at D=256, global prefill enters the two-pass VO-split path at D=512/VO=256, and global decode now reaches `decode_as_prefill_vosplit*` on `BatchPrefillWithPagedKVCacheWrapper`.",
                "The remaining blocker is FlashInfer dispatcher selection inside the VO-split paged-prefill path: it still fails with `Unsupported max_mma_kv: 0`. This is the r9/`jethac/flashinfer@76af7982` dispatcher-fix target, not a standard decode-wrapper routing failure.",
            ]
        else:
            summary += [
                "The SGLang Gemma 4 E4B text-only smoke remains RED after wrapper geometry was corrected.",
                "SWA prefill plans at D=256 and global prefill enters the two-pass VO-split path at D=512/VO=256.",
                "The remaining blocker is decode: the D=512 global layer still enters the standard decode wrapper, which instantiates a D=512/VO=512 paged module and fails in FlashInfer with `Unsupported max_mma_kv: 0`.",
            ]
    else:
        summary.append("See response and geometry samples below.")

    summary += [
        "",
        "## Response",
        "",
        "```json",
        json.dumps(response, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Geometry Samples",
        "",
    ]

    for label, lines in (
        ("SWA prefill", prefill_swa[:2]),
        ("Global VO-split prefill", prefill_global[:2]),
        ("Decode D=512", decode_d512[:2]),
    ):
        summary.append(f"### {label}")
        if not lines:
            summary.append("- MISSING")
        else:
            summary.extend(f"- `{line[:650]}`" for line in lines)
        summary.append("")

    (out / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return 0 if status == "GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
